
import torch
from skimage import morphology, filters
from inspect import isfunction
import numpy as np
from PIL import Image
from pathlib import Path



try:
    from apex import amp

    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


def create_img_scales(foldername, filename, scale_factor=1.411, image_size=None, create=False, auto_scale=None):
    """
    Receives path to the desired training image and scale_factor that defines the downsampling rate.
    optional argument image_size can be given to reshape the original training image.
    optional argument auto_scale - limits the training image to have a given #pixels.
    The function creates the downsampled and upsampled blurry versions of the training image.
    Calculates n_scales such that RF area is ~40% of the smallest scale area with the given scale_factor.
    Also calculates the MSE loss between upsampled/downsampled images for starting T calculation (see paper).


    returns:
            sizes: list of image sizes for each scale
            rescale_losses: list of MSE losses between US/DS images for each scale
            scale_factor: modified scale_factor to allow 40% area ratio
    """
    orig_image = Image.open(foldername + filename)
    # convert to PNG extension for lossless conversion
    filename = filename.rsplit( ".", 1 )[ 0 ] + '.png'
    if image_size is None:
        image_size = (orig_image.size)
    if auto_scale is not None:
        scaler = np.sqrt((image_size[0] * image_size[1])/auto_scale)
        if scaler > 1:
            image_size = (int(image_size[0]/scaler), int(image_size[1]/scaler))
    sizes = []
    downscaled_images = []
    recon_images = []
    rescale_losses = []

    # auto resize
    # rf_net = 35
    area_scale_0 = 3000  # defined such that rf_net^2/area_scale0 ~= 40%
    s_dim = min(image_size[0], image_size[1])
    l_dim = max(image_size[0], image_size[1])
    scale_0_dim = int(round(np.sqrt(area_scale_0*s_dim/l_dim)))
    # clamp between 42 and 55
    scale_0_dim = 42 if scale_0_dim < 42 else (55 if scale_0_dim > 55 else scale_0_dim )
    # scale_0_dim = 20
    small_val = scale_0_dim
    min_val_image = min(image_size[0], image_size[1])
    n_scales = int(round( (np.log(min_val_image/small_val)) / (np.log(scale_factor)) ) + 1)
    scale_factor = np.exp((np.log(min_val_image / small_val)) / (n_scales - 1))

    for i in range(n_scales):
        cur_size = (int(round(image_size[0] / np.power(scale_factor, n_scales - i - 1))),
                    int(round(image_size[1] / np.power(scale_factor, n_scales - i - 1))))
        cur_img = orig_image.resize(cur_size, Image.BILINEAR)
        path_to_save = foldername + 'scale_' + str(i) + '/'
        if create:
            Path(path_to_save).mkdir(parents=True, exist_ok=True)
            cur_img.save(path_to_save + filename)
        downscaled_images.append(cur_img)
        sizes.append(cur_size)
    for i in range(n_scales - 1):
        recon_image = downscaled_images[i].resize(sizes[i + 1], Image.BILINEAR)
        recon_images.append(recon_image)
        rescale_losses.append(
                np.linalg.norm(np.subtract(downscaled_images[i + 1], recon_image)) / np.asarray(recon_image).size)
        if create:
            path_to_save = foldername + 'scale_' + str(i + 1) + '_recon/'
            Path(path_to_save).mkdir(parents=True, exist_ok=True)
            recon_image.save(path_to_save + filename)

    return sizes, rescale_losses, scale_factor, n_scales

