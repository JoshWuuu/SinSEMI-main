from html import parser
import cv2
import torch
import numpy as np
import argparse
import os
import torchvision
from SinSEMI.functions import create_img_scales
from SinSEMI.models import SinDDMNet, MultiScaleGaussianDiffusion
from SinSEMI.trainer import MultiscaleTrainer
import torch.nn.functional as F

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--scope", help='choose training scope.', default='SDE_x1_b32_var_035sigma_randn_grad_captwo_ones')
    parser.add_argument("--mode", default='train', help='choose mode: train, sample, clip_content, clip_style_gen, clip_style_trans, clip_roi, harmonization, style_transfer, roi')
    # Dataset
    parser.add_argument("--dataset_folder", help='choose dataset folder.', default='/home/Documents/SinSEMI-main/data/training_data/Line_Pair/')
    parser.add_argument("--image_name", help='choose image name.', default='line_pair_bridge_defect.png')
    parser.add_argument("--results_folder", help='choose results folder.', default='/home/Documents/SinSEMI-main/results/')
    parser.add_argument("--lpips_guidance", default=True, action='store_true',
                        help='use LPIPS guidance during sampling. If true, the model will sample with LPIPS guidance, otherwise it will sample without it.')
    # Net
    parser.add_argument("--dim", help='widest channel dimension for conv blocks.', default=160, type=int)
    # diffusion params
    parser.add_argument("--scale_factor", help='downscaling step for each scale.', default=1.4, type=float)
    # training params
    parser.add_argument("--train_batch_size", help='batch size during training.', default=32, type=int)
    parser.add_argument("--grad_accumulate", help='gradient accumulation (bigger batches).', default=1, type=int)
    parser.add_argument("--train_num_steps", help='total training steps.', default=120001, type=int)
    parser.add_argument("--save_and_sample_every", help='n. steps for checkpointing model.', default=10000, type=int)
    parser.add_argument("--avg_window", help='window size for averaging loss (visualization only).', default=100, type=int)
    parser.add_argument("--train_lr", help='starting lr.', default=1e-4, type=float)
    parser.add_argument("--sched_k_milestones", nargs="+", help='lr scheduler steps x 1000.',
                        default=[20, 40, 70, 80, 90, 110], type=int)
    parser.add_argument("--load_milestone", help='load specific milestone.', default=12, type=int)
    # sampling params
    parser.add_argument("--sample_batch_size", help='batch size during sampling.', default=100, type=int)
    parser.add_argument("--scale_mul", help='image size retargeting modifier.', nargs="+", default=[1, 1], type=float)
    parser.add_argument("--sample_t_list", nargs="+", help='Custom list of timesteps corresponding to each scale (except scale 0).', type=int)
    # device num
    parser.add_argument("--device_num", help='use specific cuda device.', default=0, type=int)

    # DEV. params - do not modify
    parser.add_argument("--sample_limited_t", help='limit t in each scale to stop at the start of the next scale', action='store_true')
    parser.add_argument("--omega", help='sigma=omega*max_sigma.', default=0, type=float)
    parser.add_argument("--loss_factor", help='ratio between MSE loss and starting diffusion step for each scale.', default=1, type=float)

    args = parser.parse_args()

    print('num devices: '+ str(torch.cuda.device_count()))
    device = f"cuda:{args.device_num}"
    scale_mul = (args.scale_mul[0], args.scale_mul[1])
    sched_milestones = [val * 1000 for val in args.sched_k_milestones]
    results_folder = args.results_folder + '/' + args.scope

    # set to true to save all intermediate diffusion timestep results
    save_interm = False

    sizes, rescale_losses, scale_factor, n_scales = create_img_scales(args.dataset_folder, args.image_name,
                                                                                  scale_factor=args.scale_factor,
                                                                                  create=True,
                                                                                  auto_scale=50000, # limit max number of pixels in image
                                                                                  )

    model = SinDDMNet(
        dim=args.dim,
        multiscale=True,
        device=device
    )
    model.to(device)

    ms_diffusion = MultiScaleGaussianDiffusion(
        folder=args.dataset_folder,
        denoise_fn=model,
        save_interm=save_interm,
        results_folder=results_folder, # for debug
        n_scales=n_scales,
        scale_factor=scale_factor,
        image_sizes=sizes,
        scale_mul=scale_mul,
        channels=3,
        train_full_t=True,
        scale_losses=rescale_losses,
        loss_factor=args.loss_factor,
        loss_type='l2',
        device=device,
        reblurring=True,
        sample_limited_t=args.sample_limited_t,
        omega=args.omega,

    ).to(device)

    ScaleTrainer = MultiscaleTrainer(
            ms_diffusion,
            folder=args.dataset_folder,
            n_scales=n_scales,
            scale_factor=scale_factor,
            image_sizes=sizes,
            train_batch_size=args.train_batch_size,
            train_lr=args.train_lr,
            train_num_steps=args.train_num_steps,  # total training steps
            gradient_accumulate_every=args.grad_accumulate,  # gradient accumulation steps
            ema_decay=0.995,  # exponential moving average decay
            fp16=False,  # turn on mixed precision training with apex
            save_and_sample_every=args.save_and_sample_every,
            avg_window=args.avg_window,
            sched_milestones=sched_milestones,
            results_folder=results_folder,
            device=device,
            lpips_guidance=args.lpips_guidance,

        )
    if args.load_milestone > 0:
        ScaleTrainer.load(milestone=args.load_milestone)
    if args.mode == 'train':
        ScaleTrainer.train()
        # Sample after training is complete
        # ScaleTrainer.sample_scales(scale_mul=(1, 1),    # H,W
        #                            custom_sample=True,
        #                            image_name=args.image_name,
        #                            batch_size=args.sample_batch_size,
        #                            custom_t_list=sample_t_list
        #                            )
        print('train complete')
        print('result is saved in: ', args.results_folder)
    elif args.mode == 'sample':
        # # Sample
        max_scale = ScaleTrainer.n_scales - 1
        scale_name = f'scale_{max_scale}'
        orig_x_path = os.path.join(args.dataset_folder, scale_name, args.image_name)
        orig_x = cv2.imread(orig_x_path)
        orig_x = cv2.cvtColor(orig_x, cv2.COLOR_BGR2GRAY)
        orig_x = orig_x / 255.0
        # orig_x = orig_x * 2.0 - 1.0
        mean, std = orig_x.mean(), orig_x.std()
        print(mean, std)

        x1 = ScaleTrainer.data_list[0][1][1]

        ScaleTrainer.sample_scales(x1=x1,
                                   mean_std=(mean, std),   # H,W
                                   batch_size=args.sample_batch_size,
                                   custom_t_list=[300, 300, 300],
                                   lpips_guidance=args.lpips_guidance,
                                   )

if __name__ == '__main__':
    main()
    quit()
