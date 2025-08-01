"""
the DDPM model was originally based on
https://github.com/lucidrains/denoising-diffusion-pytorch
"""

import os
from scipy import linalg
import torchvision.transforms as transforms
import cv2 
# from pytorch_fid.inception import InceptionV3
from SinSEMI.functions import *
import math
import torch.autograd as autograd

from torch import nn
from torch.utils import data
from einops import rearrange
from functools import partial
import torch.nn.functional as F
from torchvision import utils
from matplotlib import pyplot as plt
from tqdm import tqdm
from SinSEMI.lpips_pytorch import LPIPS
from pytorch_fid.inception import InceptionV3

class Dataset(data.Dataset):
    def __init__(self, folder, image_size, blurry_img=False, exts=['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.blurry_img = blurry_img
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        if blurry_img:
            self.folder_recon = folder + '_recon/'
            self.paths_recon = [p for ext in exts for p in Path(f'{self.folder_recon}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x - x.mean()) / (x.std() + 1e-8))
        ])

    def __len__(self):
        return len(self.paths) * 128

    def __getitem__(self, index):
        path = self.paths[0]
        img = Image.open(path).convert('L')
        if self.blurry_img:
            path_recon = self.paths_recon[0]
            img_recon = Image.open(path_recon).convert('L')
            return self.transform(img), self.transform(img_recon)
        # else
        return self.transform(img)
    
def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)

def cosine_similarity(X, Y):
    '''
    Compute cosine similarity for each pair of images in the batch.
    Input shape: (batch, channel, H, W)
    Output shape: (batch, 1)
    '''
    # Flatten spatial dimensions for general cases
    b, c, h, w = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
    X = X.reshape(b, c, h * w)
    Y = Y.reshape(b, c, h * w)
    corr = norm(X)*norm(Y)#(B,C,H*W)
    similarity = corr.sum(dim=1).mean(dim=1)
    return similarity

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = x.float()[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# building block modules

class SinDDMConvBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),

        ) if exists(time_emb_dim) else None

        self.time_reshape = nn.Conv2d(time_emb_dim, dim, 1)
        self.ds_conv = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1)
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            condition = self.mlp(time_emb)
            condition = rearrange(condition, 'b c -> b c 1 1')
            condition = self.time_reshape(condition)
            h = h + condition
            # check_for_nan(h, 'h')

        h = self.net(h)
        return h + self.res_conv(x)


# denoiser model

class SinDDMNet(nn.Module):
    def __init__(
            self,
            dim,
            out_dim=None,
            channels=1,
            with_time_emb=True,
            multiscale=True,
            device=None
    ):
        super().__init__()
        self.device = device
        self.channels = channels
        self.multiscale = multiscale

        if with_time_emb:
            time_dim = 32

            if multiscale:
                self.SinEmbTime = SinusoidalPosEmb(time_dim)
                self.SinEmbScale = SinusoidalPosEmb(time_dim)
                self.time_mlp = nn.Sequential(
                    nn.Linear(time_dim * 2, time_dim * 4),
                    nn.GELU(),
                    nn.Linear(time_dim * 4, time_dim)
                )
            else:
                self.time_mlp = nn.Sequential(
                    SinusoidalPosEmb(time_dim),
                    nn.Linear(time_dim, time_dim * 4),
                    nn.GELU(),
                    nn.Linear(time_dim * 4, time_dim)
                )
        else:
            time_dim = None
            self.time_mlp = None

        half_dim = int(dim/2)

        self.l1 = SinDDMConvBlock(channels, half_dim, time_emb_dim=time_dim)
        self.l2 = SinDDMConvBlock(half_dim, dim, time_emb_dim=time_dim)
        self.l3 = SinDDMConvBlock(dim, dim, time_emb_dim=time_dim)
        self.l4 = SinDDMConvBlock(dim, half_dim, time_emb_dim=time_dim)

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            nn.Conv2d(half_dim, out_dim, 1)
        )

    def forward(self, x, time, scale=None):

        if exists(self.multiscale):
            time = time.view(-1)
            scale_tensor = torch.ones(size=time.shape).to(device=self.device) * scale
            t = self.SinEmbTime(time)
            s = self.SinEmbScale(scale_tensor)
            t_s_vec = torch.cat((t, s), dim=1)
            cond_vec = self.time_mlp(t_s_vec)
        else:
            t = self.time_mlp(time) if exists(self.time_mlp) else None
            cond_vec = t

        # check_for_nan(x, 'x')
        x = self.l1(x, cond_vec)
        # check_for_nan(x, 'l1')
        x = self.l2(x, cond_vec)
        # check_for_nan(x, 'l2')
        x = self.l3(x, cond_vec)
        # check_for_nan(x, 'l3')
        x = self.l4(x, cond_vec)

        return self.final_conv(x)


def compute_gaussian_product_coef(sigma1, sigma2, s=1):
    """ Given p1 = N(x_t|x_0, sigma_1**2) and p2 = N(x_t|x_1, sigma_2**2)
        return p1 * p2 = N(x_t| coef1 * x0 + coef2 * x1, var) """

    denom = sigma1**2 + sigma2**2
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1**2 * sigma2**2) / denom * s
    # var = 2 * (coef1 - coef1**2)
    return coef1, coef2, var

def unsqueeze_xdim(z, xdim):
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]

def space_indices(num_steps, count):
    assert count <= num_steps

    if count <= 1:
        frac_stride = 1
    else:
        frac_stride = (num_steps - 1) / (count - 1)

    cur_idx = 0.0
    taken_steps = []
    for _ in range(count):
        taken_steps.append(round(cur_idx))
        cur_idx += frac_stride

    return taken_steps

def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    # return np.linspace(linear_start, linear_end, n_timestep)
    betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return betas.numpy()


lpips = LPIPS(net_type="squeeze", version="0.1").to("cuda:0")
class MultiScaleGaussianDiffusion(nn.Module):
    def __init__(
            self,
            denoise_fn,
            folder,
            *,
            save_interm=False,
            results_folder = '/Results',
            n_scales,
            scale_factor,
            image_sizes,
            scale_mul=(1, 1),
            channels=1,
            train_full_t=False,
            scale_losses=None,
            loss_factor=1,
            loss_type='l2',
            s = 0,
            device=None,
            reblurring=True,
            sample_limited_t=False,
            omega=0,
    ):
        super().__init__()
        self.device = device
        self.save_interm = save_interm
        self.results_folder = Path(results_folder)
        self.channels = channels
        self.n_scales = n_scales
        self.scale_factor = scale_factor
        self.image_sizes = image_sizes
        self.scale_mul = scale_mul
        self.min_sigma = 0.3          # floor (tune)
        self.reg_lambda = 0.5

        self.sample_limited_t = sample_limited_t
        self.reblurring = reblurring

        self.img_prev_upsample = None
        # omega tests
        self.omega = omega

        for i in range(n_scales):  # flip xy->hw
            self.image_sizes += ((image_sizes[i][1], image_sizes[i][0]),)

        self.denoise_fn = denoise_fn

        self.loss_type = loss_type

        self.vairance_s = s
        self.input_paths = []
        self.ds_list = []
        self.dl_list = []
        self.data_list = []
        self.num_timesteps_trained = []
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)

        if scale_losses is not None:
            for i in range(n_scales - 1):
                if train_full_t:
                    self.num_timesteps_trained.append(
                        100)
                else:
                    self.num_timesteps_trained.append(100)
        

        for i in range(n_scales):
            self.input_paths.append(folder + 'scale_' + str(i))
            blurry_img = True if i > 0 else False
            self.ds_list.append(Dataset(self.input_paths[i], image_sizes[i], blurry_img))
            if i > 0:
                Data = self.ds_list[i][0]
                self.data_list.append((Data[0].to(self.device), Data[1].to(self.device)))
            else:
                cur_image = self.ds_list[i][0].to(self.device)
                # x_recon = F.interpolate(cur_image, size=(20, 20), mode='bilinear')
                # x_recon = F.interpolate(x_recon, size=(cur_image.shape[2], cur_image.shape[3]), mode='bilinear')
                x_recon = torch.randn_like(cur_image) 
                self.data_list.append(
                    (cur_image, x_recon))  # just duplicate orig over blurry_img for scale 0
        
        
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.energy_model = InceptionV3([block_idx]).to(device)
        # self.lpips = LPIPS(net_type="squeeze", version="0.1").to(self.device)
    
    @torch.no_grad()
    def sde_sampling(self, steps, x1, s, sigma_s, nfe=None, lpips_guidance=False):
        xt = x1.detach().to(self.device)

        xs = []
        pred_x0s = []

        if nfe is not None:
            # create a range with steps and nfe
            nfe += 1 
            steps_list = space_indices(steps, nfe)
        else:
            steps_list = space_indices(steps, steps-1)

        log_count = min(len(steps_list)-1, 10)
        log_steps = [steps_list[i] for i in space_indices(len(steps_list)-1, log_count)]
        
        steps_list = steps_list[::-1]

        pair_steps = zip(steps_list[1:], steps_list[:-1])
        # pair_steps = tqdm(pair_steps, desc='euler maruyama sampling', total=len(steps_list)) 
        step_size = 1 / (len(steps_list)+1)
        t = 1
        x1 = x1.to(self.device)
        for prev_step in tqdm(range(len(steps_list)+1)):

            step_tensor = torch.full((xt.shape[0],), t, device=self.device).float()
            with torch.no_grad():
                out = self.denoise_fn(xt, step_tensor, s)
            # check_for_nan(pred_x0, 'pred_x0')
            sigma = 1
            diffusion_coefficient = (t * (1-t))**(1/2) * sigma
            # diffusion_coefficient = 0.1
            grad = 0
            if lpips_guidance:
                grad = self.fid_energy(xt, t, s, sigma)
                g_norm = grad.flatten(1).norm(dim=1).mean()
                o_norm = out.flatten(1).norm(dim=1).mean()
                grad = grad / (g_norm + 1e-8) * o_norm
            
            grad_coefficient = 1

            xt = xt - step_size * (out + grad * grad_coefficient) + torch.randn_like(xt) * np.sqrt(step_size) * diffusion_coefficient
        
            t = t - step_size

            # print(t)
            xs.append(xt.detach().cpu())

        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack_bwd_traj(xs), stack_bwd_traj(xs)
    
    def p_losses(self, x1, x0, t, s, sigma_s, noise=None):
        # var = (t**2 * (1-t)**2) / (t**2 + (1-t)**2)
        var = t * (1-t) 
        rand = torch.randn_like(x0) * 0.35
        xt = (1-t) * x0 + t * x1 + rand * var.sqrt()
        dvar = (1-2*t) / (2*var.sqrt() + 1e-2)
        dvar = torch.where(dvar.abs() > 2,
                   torch.ones_like(dvar) * 2,
                   dvar)        
             
        # label = (xt-x0) / (t + 1e-2) # (xt - x0) / (t * (1-t))**(1/2)
        label = x1 - x0 + rand * dvar
        # label /= (1e-2 + var.sqrt())    
        pred = self.denoise_fn(xt, t, s)

        # reg = torch.mean(torch.relu(self.min_sigma - sigma_s))
        reg = 0
        if self.loss_type == 'l1':
            loss = F.l1_loss(pred, label)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(pred, label)
        return loss + self.reg_lambda * reg

    def forward(self, x, s, sigma_s, *args, **kwargs):
        x_orig = x[0]
        x_recon = x[1]
        if s == 0:
            x_recon = torch.randn_like(x_orig)
        b, c, h, w = x_orig.shape
        device = x_orig.device
        img_size = self.image_sizes[s]
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
        # t = torch.randint(0, self.num_timesteps_trained[s], (b,), device=device).long() / self.num_timesteps_trained[s]
        t = torch.rand((x_orig.shape[0],), device=device)
        t = t.view(-1, 1, 1, 1)
        return self.p_losses(x_recon, x_orig, t, s, sigma_s, *args, **kwargs)
    
    def fid_energy(self, y, t, s, sigma):
        y.requires_grad_(True)
        data = self.data_list[s]
        x_orig = data[0]
        x_recon = data[1]
        var = t * (1-t)
        x_t = (1-t) * x_orig + t * x_recon + torch.randn_like(x_orig) * var**0.5 * sigma
        x_t = x_t.unsqueeze(0)

        with torch.enable_grad():
            # make x_t and y 3d
            y_3ch = y.expand(-1, 3, -1, -1)            # ⚠ same storage as y  :contentReference[oaicite:0]{index=0}
            x_3ch = x_t.expand(-1, 3, -1, -1)

            Y = self.energy_model(y_3ch)[0]
            X = self.energy_model(x_3ch)[0]
            # repeat X to match the batch size of Y
            X = X.repeat(Y.shape[0] // X.shape[0], 1, 1, 1)
 
            # energy = -torch.log(cosine_similarity(Y,X)) * 1
            x_t_expanded = x_t.expand(y.size(0), -1, -1, -1)

            energy = lpips(x_t_expanded, y).view(y.size(0))

            # mu_y = Y.mean(dim=(2, 3), keepdim=True)
            # mu_x = X.mean(dim=(2, 3), keepdim=True)

            # energy = ((X - Y)**2).sum(dim=1) 

            # energy = frechet_per_sample(Y, X, eps=1e-6) * 1  # (B, 2048)
            
            grad = autograd.grad(outputs=energy, 
                                 inputs=y, 
                                 grad_outputs=torch.ones_like(energy), 
                                 create_graph=False, 
                                 retain_graph=False, 
                                 allow_unused=True)[0]
        
        return grad.detach()

def _cov_batched(feats: torch.Tensor) -> torch.Tensor:
    """
    feats: (B, C, M)  where M = spatial dimension product (e.g. H*W)
    returns: (B, C, C) covariance per batch item
    """
    mean = feats.mean(dim=-1, keepdim=True)                      # (B, C, 1)
    xc = feats - mean                                            # (B, C, M)
    # unbiased estimator (divide by M-1). Use einsum for clarity / speed.
    M = feats.shape[-1]
    cov = torch.matmul(xc, xc.transpose(0, 1)) / (M - 1 + 1e-8)   # (B, C, C)
    return cov

def _cov_single(feats: torch.Tensor) -> torch.Tensor:
    """
    feats: (N, C, M)  -> treat all N samples jointly
    returns: (C, C) covariance
    """
    mean = feats.mean(dim=(0, -1), keepdim=True)                 # (1, C, 1)
    xc = feats - mean                                            # (N, C, M)
    Mtot = feats.shape[0] * feats.shape[-1]                   # (C, N*M)
    cov = torch.matmul(xc, xc.T) / (Mtot - 1 + 1e-8)             # (C, C)
    return cov

def sqrtm_psd(A, eps=1e-12):
    """
    Matrix square root for (batched) symmetric PSD matrices.
    A: (..., C, C)
    returns: (..., C, C)
    """
    # eigh is batched & returns real eigenpairs for symmetric inputs
    w, V = torch.linalg.eigh(A)
    w = w.clamp_min(eps).sqrt()                     # avoid tiny negatives → 0
    return V @ torch.diag_embed(w) @ V.transpose(-1, -2)

def frechet_per_sample(Y, X_ref, eps: float = 1e-6):
    """
    X_ref : (N_ref, C, H, W)
    Y     : (B,     C, H, W)
    returns: (B,)  Fréchet distance of each Y_i to the X_ref distribution
    """
    # Flatten spatial dims
    B, C = Y.shape[:2]
    N_ref = X_ref.shape[0]
    M_y = Y.shape[2] * Y.shape[3]
    M_x = X_ref.shape[2] * X_ref.shape[3]

    Yf = Y.view(B, C)                 # (B, C, M_y)
    Xf = X_ref.view(N_ref, C)         # (N_ref, C, M_x)

    # Means
    mu_y = Yf.mean(dim=1)                 # (B, 1)
    mu_x = Xf.mean(dim=1)            # (B, 1)

    # Covariances
    cov_y = _cov_batched(Yf)               # (B, C, C)
    cov_x = _cov_single(Xf)                # (C, C)

    # Difference of means term
    diff = mu_y - mu_x                     # (B, C)
    mean_term = (diff ** 2)   # (B,)
# (B, C, C)
    # Matrix square root of cov_y * cov_x  (batched)
    prod = torch.matmul(cov_y, cov_x)            # (B, C, C)
    # torch.linalg.sqrtm is batched since PyTorch 2.2
    covmean = sqrtm_psd(prod)               # (B, C, C)
    # Numerical issues: sqrtm may return complex dtype with tiny imag parts
    if torch.is_complex(covmean):
        covmean = covmean.real

    trace_term = (
        torch.diagonal(cov_y + cov_x - 2.0 * covmean)
    )  # (B,)

    dist = mean_term + trace_term + eps
    return dist

def check_for_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
    # def fid_model_init(self):
    #     transform = transforms.Compose([
            
    #         transforms.ToTensor(),
    #         transforms.Lambda(lambda t: (t * 2) - 1)
    #     ])

    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     # scale0_path = '/home/Documents/image_generation/metric/data/euv_ls/scale_0/Lines_defect_85nm_x_polarization_0deg.png'
    #     scale0_path = '/home/Documents/image_generation/metric/data/euv_ch/scale_0/CH_defect_85nm_y_polarization_0deg.png'
    #     s0_image = cv2.imread(scale0_path, cv2.IMREAD_COLOR)
    #     s0_image = cv2.cvtColor(s0_image, cv2.COLOR_BGR2RGB)
    #     # permute to (B,C,H,W) and scale to [0, 1] to tensor, 
    #     s0_image = transform(s0_image)
    #     s0_image = s0_image.unsqueeze(0)
    #     self.s0_image = s0_image.to(device)

    #     scale1_path = '/home/Documents/image_generation/metric/data/euv_ch/scale_1_recon/CH_defect_85nm_y_polarization_0deg.png'
    #     s1_image = cv2.imread(scale1_path, cv2.IMREAD_COLOR)
    #     s1_image = cv2.cvtColor(s1_image, cv2.COLOR_BGR2RGB)
    #     # permute to (B,C,H,W) and scale to [0, 1] to tensor,
    #     s1_image = transform(s1_image)
    #     s1_image = s1_image.unsqueeze(0)
    #     self.s1_image = s1_image.to(device)

    #     scale2_path = '/home/Documents/image_generation/metric/data/euv_ch/scale_2_recon/CH_defect_85nm_y_polarization_0deg.png'
    #     s2_image = cv2.imread(scale2_path, cv2.IMREAD_COLOR)
    #     s2_image = cv2.cvtColor(s2_image, cv2.COLOR_BGR2RGB)
    #     # permute to (B,C,H,W) and scale to [0, 1] to tensor,
    #     s2_image = transform(s2_image)
    #     s2_image = s2_image.unsqueeze(0)
    #     self.s2_image = s2_image.to(device)

    #     block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    #     self.energy_model = InceptionV3([block_idx]).to(device)
    
    # def fid_energy(self, y, t, s):
    #     y.requires_grad_(True)
    #     if s == 0:
    #         ref_image = self.s0_image
    #     elif s == 1:
    #         ref_image = self.s1_image
    #     else:
    #         ref_image = self.s2_image
    #     ref_image = self.q_sample(x_start=ref_image, t=t, noise=None)

    #     with torch.enable_grad():
    #         Y = self.energy_model(y)[0]
    #         X = self.energy_model(ref_image)[0]
    #         # repeat X to match the batch size of Y
    #         # X = X.repeat(Y.shape[0], 1, 1, 1)

    #         energy = cosine_similarity(Y,X).sum()
    #         grad = autograd.grad(energy, y, allow_unused=True)[0]
        
    #     return grad