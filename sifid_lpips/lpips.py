import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path 
import cv2
import numpy as np
from lpips_pytorch import LPIPS, lpips
import torch


# /home/Documents/image_generation/oneshot_models/SinDiffusion/second_results/euv_ls/euv_lsscale_6
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
# parser.add_argument('--path2real', type=str, default='/home/Documents/SinSEM-main/data/training_data/Line_Pair/line_pair_bridge_defect.png', help=('Path to the real images'))
parser.add_argument('--path2real', type=str, default='/home/Documents/SinSEM-main/data/training_data/Contact_Hole/contact_hole_open_defect.png')
parser.add_argument('--path2fake', type=str, default='/home/Documents/SinSEM-main/results/ch_SDE_x1_b32_var_1sigma_clip10/final_samples_unbatched_sample_2025-07-29 07_23_24.554788', help=('Path to generated images'))
parser.add_argument('-c', '--gpu', default='0', type=str, help='GPU to use (leave blank for CPU only)')
parser.add_argument('--images_suffix', default='png', type=str, help='image file suffix')

def read_rgb(path: Path):
    """Read image as RGB tensor in [-1,1]."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (torch.from_numpy(img).permute(2, 0, 1).float() / 127.5) - 1.0
    
if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    path1 = args.path2real
    path2 = args.path2fake
    ds = []

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = LPIPS(net_type="squeeze", version="0.1").to(device).eval()

    # 1) read the reference once
    ref = read_rgb(path1).unsqueeze(0).to(device)   # (1,3,H,W)

    # 2) enumerate candidate images
    paths = list(sorted(Path(args.path2fake).glob("*.png")))
    assert paths, f"No PNGs in {path2}"

    # 3) loop in large mini-batches to amortize GPU launches
    scores = []
    with torch.inference_mode():
        for b in range(0, len(paths), 1):
            batch_paths = paths[b : b + 1]
            ims   = torch.stack([read_rgb(p) for p in batch_paths]).to(device)  # (B,3,H,W)
            refs  = ref.expand(ims.size(0), -1, -1, -1)                        # broadcast
            scores.append(criterion(refs, ims).flatten())                      # (B,)
    ds = torch.cat(scores).cpu().numpy()

    print(f"dir: {path2}")
    print(f"LPIPS = {np.mean(ds):.5f} ± {np.std(ds):.5f}  (N={len(ds)})")
