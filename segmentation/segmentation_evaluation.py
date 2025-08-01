import os 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--path2real-defect', type=str, default='/home/Documents/SinSEM-main/augment_data/lp/line_pair_defect', help=('Path to the real images'))
parser.add_argument('--path2real-nodefect', type=str, default='/home/Documents/SinSEM-main/augment_data/lp/line_pair_no_defect', help=('Path to the real images'))
parser.add_argument('--path2fake', type=str, default='/home/Documents/SinSEM-main/data/evaluation_data/evaluation_data/line_pair/sinsem', help=('Path to generated images'))
parser.add_argument('-c', '--gpu', default='0', type=str, help='GPU to use (leave blank for CPU only)')
parser.add_argument('--images_suffix', default='png', type=str, help='image file suffix')


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx] 
        mask = np.array(mask)  # Convert mask to a NumPy array
        if self.transform:
            image = self.transform(image)
            #mask = self.transform(mask)
        return image, mask

def calculate_iou(scores, targets):
    scores = torch.sigmoid(scores)
    predicted_masks = (scores > 0.5).float()
    intersection = (predicted_masks * targets).sum()
    union = predicted_masks.sum() + targets.sum() - intersection
    iou = torch.where(union > 0, intersection / (union + 1e-7), torch.ones_like(union))
    return iou

def find_connected_component(x1, x2, energy_threshold=50, circle_threshold=0.6):
    diff_image = cv2.absdiff(x1, x2)
    _, thresh_image = cv2.threshold(diff_image, 5, 1, cv2.THRESH_BINARY)
    # _, thresh_image = cv2.threshold(diff_image, 5, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_image, connectivity=8)
    
    output_mask = np.zeros_like(thresh_image, dtype=np.uint8)
    for i in range(1, num_labels):
        component_mask = (labels==i)
        sum_pixel_value = np.sum(thresh_image[component_mask])

        if sum_pixel_value > energy_threshold and is_circular(component_mask, threshold=circle_threshold):
            output_mask[component_mask] = 1
    return output_mask

def is_circular(mask, threshold=0.6):
    """
    Check if a binary mask is close to a circle.

    Args:
        mask (np.ndarray): Binary mask of the component.
        threshold (float): Threshold for circularity.

    Returns:
        bool: True if the component is close to a circle, False otherwise.
    """
    if np.sum(mask) == 0:
        return False  # Empty mask

    # Calculate the area of the component
    area = np.sum(mask)

    # Find the contour of the component
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False  # No contours found

    # Calculate the perimeter of the component
    perimeter = cv2.arcLength(contours[0], True)

    # Calculate the circularity
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    # Return True if the circularity is above the threshold
    return circularity > threshold

def rand_noise(x, sigma_min=0.0, sigma_max=1.0):
    sigma = torch.empty(1).uniform_(sigma_min, sigma_max).item()
    return x + torch.randn_like(x) * sigma/255.0

def main():
    args = parser.parse_args()
    # construct the segmentation mask first
    train_data = []
    train_mask = []
    model_name = 'gpdm'
    save_folder = '/home/Documents/SinSEM-main/segmentation/line_pair'
    save_folder = os.path.join(save_folder, model_name)
    os.makedirs(save_folder, exist_ok=True)
    gen_image_path = '/home/Documents/SinSEM-main/data/evaluation_data/evaluation_data/line_pair/gpdm'
    fdtd_image_path_defect = '/home/Documents/SinSEM-main/augment_data/lp/line_pair_defect'
    fdtd_image_path_nodefect = '/home/Documents/SinSEM-main/augment_data/lp/line_pair_no_defect'

    raw_fdtd_image = cv2.imread('/home/Documents/SinSEM-main/data/training_data/Line_Pair/line_pair_no_defect.png', cv2.IMREAD_GRAYSCALE)
    denoised_fdtd_image = cv2.GaussianBlur(raw_fdtd_image, (5,5), 1)
    all_files = os.listdir(gen_image_path)
    all_files = [f for f in all_files if f.lower().endswith('.png')]

    for idx, file in enumerate(all_files):
        gen_image = cv2.imread(os.path.join(gen_image_path, file), cv2.IMREAD_GRAYSCALE)
        denoised_gen_image = cv2.GaussianBlur(gen_image, (5,5), 1)
        thresh_image = find_connected_component(denoised_fdtd_image, denoised_gen_image)
        train_data.append(gen_image)
        train_mask.append(thresh_image)
    
    print(len(train_data))
    test_data = []
    test_mask = []

    test_files_path_defect = [os.path.join(fdtd_image_path_defect, f) for f in sorted(os.listdir(fdtd_image_path_defect))]
    test_files_path_no_defect = [os.path.join(fdtd_image_path_nodefect, f) for f in sorted(os.listdir(fdtd_image_path_nodefect))]
    for (defect_path, no_defect_path) in zip(test_files_path_defect, test_files_path_no_defect):
        test_defect_image = cv2.imread(defect_path, cv2.IMREAD_GRAYSCALE)
        test_no_defect_image = cv2.imread(no_defect_path, cv2.IMREAD_GRAYSCALE)
        test_data.append(test_defect_image)
        test_data.append(test_no_defect_image)
        denoise_defect_image = cv2.GaussianBlur(test_defect_image, (5,5), 1)
        denoise_no_defect_image = cv2.GaussianBlur(test_no_defect_image, (5,5), 1)
        thresh_image = find_connected_component(denoise_defect_image, denoise_no_defect_image, 100, 0.6)
        test_mask.append(thresh_image)
        test_mask.append(np.zeros_like(thresh_image, dtype=np.uint8))  # No defect mask is all zeros

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.RandomHorizontalFlip(p=0.7),
        # transforms.RandomVerticalFlip(p=0.7),
        # transforms.RandomRotation(degrees=2),                        # ±2°
        transforms.Lambda(lambda x: rand_noise(x, 0.0, 1.0))
        # transforms.Lambda(lambda x: x * 2 - 1)
        # transforms.Normalize((0.5,), (0.5,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    split_train_data = train_data[:int(0.95*len(train_data))]
    split_train_mask = train_mask[:int(0.95*len(train_data))]
    val_data = train_data[int(0.95*len(train_data)):]
    val_mask = train_mask[int(0.95*len(train_data)):]

    train_dataset = SegmentationDataset(split_train_data, split_train_mask, transform=transform)
    val_dataset = SegmentationDataset(val_data, val_mask, transform=transform)
    test_dataset = SegmentationDataset(test_data, test_mask, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = UNet(n_channels=1, n_classes=1, bilinear=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    num_epochs = 100
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # Learning rate scheduler

    best_loss = float('inf')
    patience = 20  # Number of epochs to wait for improvement
    early_stop_counter = 0  # Counter for early stopping

    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
        total_iou = 0
        count = 0
        
        for data, targets in train_loader:
            data = data.to(device=device)
            targets = targets.to(device=device, dtype=torch.float)

            # forward
            scores = model(data)
            loss = criterion(scores.squeeze(1), targets.squeeze(1).float())

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            iou = calculate_iou(scores.squeeze(1), targets.squeeze(1).float())

            total_iou += iou.item()
            total_loss += loss.item()
            count += 1
        average_loss = total_loss / len(train_loader)
        average_iou = total_iou / len(train_loader)

        print(f"Epoch {epoch+1}: Training Loss = {average_loss:.4f}, Training IoU = {average_iou:.4f}")
        
        scheduler.step()  # Update learning rate

        # add validation here
        total_loss = 0
        total_iou = 0
        count = 0
        for data, targets in val_loader:
            data = data.to(device=device)
            targets = targets.to(device=device, dtype=torch.float)

            scores = model(data)
            loss = criterion(scores.squeeze(1), targets.squeeze(1).float())

            iou = calculate_iou(scores.squeeze(1), targets.squeeze(1).float())

            total_iou += iou.item()
            total_loss += loss.item()
            count += 1

        average_loss = total_loss / len(val_loader)
        average_iou = total_iou / len(val_loader)

        print(f"Epoch {epoch+1}: Validation Loss = {average_loss:.4f}, Validation IoU = {average_iou:.4f}")

        if average_loss < best_loss:
            best_loss = average_loss
            save_path = os.path.join(save_folder, 'best_segmentation_model.pth')
            torch.save(model.state_dict(), save_path)
            early_stop_counter = 0  # Reset the counter
        else:
            early_stop_counter += 1  # Increment the counter

        if early_stop_counter >= patience:
            print("Early stopping triggered. No improvement in validation loss.")
            break

    load_path = os.path.join(save_folder, 'best_segmentation_model.pth')
    model.load_state_dict(torch.load(load_path))
    model.eval()
    model.to(device)

    total_iou = 0
    count = 0
    for data, targets in test_loader:
        data = data.to(device=device)
        targets = targets.to(device=device)

        scores = model(data)
        iou = calculate_iou(scores, targets)

        total_iou += iou.item()
        count += 1

        # plot the image, mask, and prediction
        image = data.squeeze(0).permute(1, 2, 0).cpu().numpy()
        mask = targets.permute(1, 2, 0).cpu().numpy()
        scores = F.sigmoid(scores)
        prediction = scores.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        prediction = (prediction > 0.5).astype(np.uint8)
            
        plot = True
        if plot:
            temp = plt.figure()
            # Plot the image
            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.title("Image")

            # Plot the mask
            plt.subplot(1, 3, 2)
            plt.imshow(mask, alpha=0.5)
            plt.title("Mask")

            # Plot the prediction
            plt.subplot(1, 3, 3)
            plt.imshow(prediction, alpha=0.5)
            plt.title("Prediction")
            plt.show()
            image_save_path = os.path.join(save_folder, f"test_{count}.png")
            plt.savefig(image_save_path)
            plt.close()

    # Calculate the average IoU for the entire test set
    average_iou = total_iou / len(test_loader)
    print('path to save the model:', load_path)
    print(f"Average IoU for the test set: {average_iou:.4f}")

if __name__ == '__main__':
    main()