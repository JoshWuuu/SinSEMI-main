import os
import cv2
from matplotlib import pyplot as plt
import numpy as np

origin_path = "/home/Documents/SinSEM-main/data/training_data/Line_Pair/line_pair_bridge_defect.png"
origin_path = "/home/Documents/SinSEM-main/data/training_data/Contact_Hole/contact_hole_open_defect.png"
origin_image = cv2.imread(origin_path, cv2.IMREAD_GRAYSCALE)

folder_path = "/home/Documents/SinSEM-main/results/ch_SDE_x1_b32_var_1sigma_clip3/final_samples_unbatched_sample_2025-07-29 09_07_48.394271"

for image_path in sorted(os.listdir(folder_path)):
    print("Image Path:", image_path)
    image_path = os.path.join(folder_path, image_path)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Calculate common vmin and vmax values across both images     # 15 % on each side → total 30 %
    # vmin_adj, vmax_adj = 50, 80
    vmin_adj, vmax_adj = 110, 140
    fig = plt.figure(figsize=(10, 5))

    # left panel – original
    plt.subplot(1, 2, 1)                # rows, cols, index
    plt.title('Original Image')
    plt.imshow(origin_image, cmap='gray', vmin=vmin_adj, vmax=vmax_adj)                   # hide tick marks (optional)

    # right panel – generated
    plt.subplot(1, 2, 2)
    plt.title('Generated Image')
    plt.imshow(image, cmap='gray', vmin=vmin_adj, vmax=vmax_adj)
                 # tidy up spacing
    plt.show()
    print()


print("Origin Image Shape:", origin_image.shape)
print("Generated Image Shape:", image.shape)

