"""
the DDPM trainer was originally based on
https://github.com/lucidrains/denoising-diffusion-pytorch
"""

import copy
import os
import datetime
from functools import partial

from SinSEMI.functions import *
from SinSEMI.models import EMA

from torch.utils import data
from torchvision import transforms, utils
from torch.optim import Adam
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR

from matplotlib import pyplot as plt
from skimage.exposure import match_histograms
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

try:
    from apex import amp

    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False


# dataset classes

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

class ScaleVarianceModel(nn.Module):
    """
    Learnable scalar variance per scale.
    Args
    ----
    num_scales : int      # how many different image-scales / U-Nets you train
    init_logvar : float   # start value (log σ²) – e.g. -4  → σ ≈ 0.018
    """
    def __init__(self, num_scales: int, init_logvar: float = -4.0):
        super().__init__()
        # store log-variance so that positivity is guaranteed after exp()
        self.log_var = nn.Embedding(num_scales, 1)
        nn.init.constant_(self.log_var.weight, init_logvar)

    def forward(self, s, ref) -> torch.Tensor:
        """
        Parameters
        ----------
        s   : int or (B,) tensor with scale indices
        ref : tensor whose shape we need to broadcast to (usually x0)

        Returns
        -------
        sigma : tensor shaped like (B,1,1,1) that broadcasts to `ref`
        """
        # make sure s is a LongTensor – nn.Embedding needs that
        s = torch.as_tensor(s, dtype=torch.long, device=ref.device).view(-1)
        # log σ²  → σ  (square-root of variance)
        log_var = self.log_var(s)           # (B,1)
        sigma   = torch.exp(0.5 * log_var)  # (B,1)
        # reshape to (B,1,1,1,…)
        while sigma.dim() < ref.dim():
            sigma = sigma.unsqueeze(-1)
        return sigma  

class MultiscaleTrainer(object):

    def __init__(
            self,
            ms_diffusion_model,
            folder,
            *,
            ema_decay=0.995,
            n_scales=None,
            scale_factor=1,
            image_sizes=None,
            train_batch_size=32,
            train_lr=2e-5,
            train_num_steps=100000,
            gradient_accumulate_every=2,
            fp16=False,
            step_start_ema=2000,
            update_ema_every=10,
            save_and_sample_every=25000,
            avg_window=100,
            sched_milestones=None,
            results_folder='./results',
            device=None,
            lpips_guidance=False    ):
        super().__init__()
        self.device = device
        if sched_milestones is None:
            self.sched_milestones = [10000, 30000, 60000, 80000, 90000]
        else:
            self.sched_milestones = sched_milestones
        if image_sizes is None:
            image_sizes = []
        self.model = ms_diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.lpips_guidance = lpips_guidance

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every
        self.avg_window = avg_window

        self.batch_size = train_batch_size
        self.n_scales = n_scales
        self.scale_factor = scale_factor
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.input_paths = []
        self.ds_list = []
        self.dl_list = []
        self.data_list = []
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)

        for i in range(n_scales):
            self.input_paths.append(folder + 'scale_' + str(i))
            blurry_img = True if i > 0 else False
            self.ds_list.append(Dataset(self.input_paths[i], image_sizes[i], blurry_img))
            self.dl_list.append(
                cycle(data.DataLoader(self.ds_list[i], batch_size=train_batch_size, shuffle=True, pin_memory=True)))

            if i > 0:
                Data = next(self.dl_list[i])
                self.data_list.append((Data[0].to(self.device), Data[1].to(self.device)))
            else:
                cur_image = next(self.dl_list[i]).to(self.device)
                # x_recon = F.interpolate(cur_image, size=(20, 20), mode='bilinear')
                # x_recon = F.interpolate(x_recon, size=(cur_image.shape[2], cur_image.shape[3]), mode='bilinear')
                x_recon = torch.randn_like(cur_image) 
                self.data_list.append(
                    (cur_image, x_recon))  # just duplicate orig over blurry_img for scale 0

        self.var_model = ScaleVarianceModel(n_scales).to(self.device)
        params = list(self.model.parameters()) + list(self.var_model.parameters())
        self.opt = torch.optim.Adam(params, lr=train_lr)

        self.scheduler = MultiStepLR(self.opt, milestones=self.sched_milestones, gamma=0.5)

        self.step = 0
        self.running_loss = []
        self.running_scale = []
        self.avg_t = []
        self.image_sizes = image_sizes

        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex must be installed in order for mixed precision training to be turned on'

        self.fp16 = fp16
        if fp16:
            (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt,
                                                                    opt_level='O1')
        
        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'sched': self.scheduler.state_dict(),
            'running_loss': self.running_loss,
            'running_scale': self.running_scale,
            'variance_model': self.var_model.state_dict(),
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
        plt.rcParams['figure.figsize'] = [16, 8]

        plt.plot(self.running_loss)
        plt.grid(True)
        plt.ylim((0, 2))
        plt.savefig(str(self.results_folder / 'running_loss'))
        plt.clf()

    def load(self, milestone):
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=self.device)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
        self.scheduler.load_state_dict(data['sched'])
        self.running_loss = data['running_loss']
        # self.var_model.load_state_dict(data['variance_model'])
    #    self.running_scale = data['running_scale']

    def train(self):

        backwards = partial(loss_backwards, self.fp16)
        loss_avg = 0
        s_weights = torch.tensor(self.model.num_timesteps_trained, device=self.device, dtype=torch.float)
        while self.step < self.train_num_steps:
            # t weighted multinomial sampling
            s = torch.multinomial(input=s_weights, num_samples=1)  # uniform when train_full_t = True
            for i in range(self.gradient_accumulate_every):
                data = self.data_list[s]
                sigma_s = self.var_model(s, data[0])  # (B,1,1,1)
                loss = self.model(data, s, sigma_s)
                loss_avg += loss.item()
                backwards(loss / self.gradient_accumulate_every, self.opt)

            clip_grad_norm_(self.model.parameters(), 1.0)
            if self.step % self.avg_window == 0:
                print(f'step:{self.step} loss:{loss_avg/self.avg_window}')
                # print(f'scale:{s.item()} sigma_s:{sigma_s.mean().item()}')
                self.running_loss.append(loss_avg/self.avg_window)
                loss_avg = 0

            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()
            self.scheduler.step()
            self.step += 1
            if self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                self.save(milestone)

        print('training completed')

    @torch.no_grad()
    def sample_scales(self, x1, mean_std,  batch_size=16, custom_image_size_idxs=None,
                      custom_scales=None, custom_t_list=None, desc=None, lpips_guidance=False):
        if desc is None:
            desc = f'sample_{str(datetime.datetime.now()).replace(":", "_")}'

        # sample with custom scale list
        if custom_scales is None:
            custom_scales = [*range(self.n_scales)]
            n_scales = self.n_scales
        else:
            n_scales = len(custom_scales)
        if custom_image_size_idxs is None:
            custom_image_size_idxs = [*range(self.n_scales)]

        samples_from_scales = []
        t_list = custom_t_list
        res_sub_folder = '_'.join(str(e) for e in t_list)
        x1_shape = x1.repeat(batch_size, 1, 1, 1)
        samples_from_scales.append(torch.randn_like(x1_shape))
        mean, std = mean_std
        for scale in range(n_scales):
            cur_x = samples_from_scales[-1]
            cur_x = F.interpolate(cur_x, size=(self.image_sizes[scale]), mode='bilinear')
            # save_cur_x = (cur_x * std) + mean
            # file_path = Path(str(self.results_folder / f'upscale_{scale}_9.png'))
            # utils.save_image(save_cur_x, str(file_path))
            # sigma_s = self.var_model(scale, cur_x.to(self.device))  # (B,1,1,1)
            sigma_s = 0
            # print(f'sampling scale {scale} with sigma {sigma_s.mean().item()}')
            xs, pred_x0 = self.ema_model.sde_sampling(t_list[scale], 
                                                cur_x, 
                                                scale,
                                                sigma_s=sigma_s,
                                                lpips_guidance=lpips_guidance)
            recon_img = xs[:, 0, ...]
            samples_from_scales.append(recon_img)

            # save xs[:, 3, ...] and xs[;, 6, ...] for each scale
            # for t in [0, 3, 6]:
            #     cur_x = pred_x0[:, t, ...]
            #     final_results_folder = Path(str(self.results_folder / f'{scale}_{t}.png'))
            #     cur_x = (cur_x * std) + mean
            #     utils.save_image(cur_x, str(final_results_folder))
        
        final_img = (samples_from_scales[-1] * std) + mean
        final_results_folder = Path(str(self.results_folder / f'final_samples_unbatched_{desc}'))
        final_results_folder.mkdir(parents=True, exist_ok=True)
        for b in range(batch_size):
            utils.save_image(final_img[b], str(final_results_folder / res_sub_folder) + f'_out_b{b}.png')


