# %%
# IMPORTANT: SOME KAGGLE DATA SOURCES ARE PRIVATE
# RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES.
# import kagglehub
# kagglehub.login()


# %%
# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

# ethz_cil_monocular_depth_estimation_2025_path = kagglehub.competition_download('ethz-cil-monocular-depth-estimation-2025')

# print('Data source import complete.')


# %% [markdown]
# # Prep

# %%
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from stopit import ThreadingTimeout as Timeout, threading_timeoutable as timeoutable  #doctest: +SKIP
from canny_edge_detector import canny_edge_detector
from enum import Enum

# Add custom transform to append Canny edge channel (legacy, not used for model input)
class ProcessImageAndAddCanny:
    def __init__(self, canny_low_thresh=30, canny_high_thresh=100):
        self.low_thresh = canny_low_thresh
        self.high_thresh = canny_high_thresh
        self.to_tensor = transforms.ToTensor()
    def __call__(self, pil_image):
        rgb_tensor = self.to_tensor(pil_image)
        gray_np = np.array(pil_image.convert("L"))
        edge_np = canny_edge_detector(gray_np, self.low_thresh, self.high_thresh)
        edge_tensor = torch.from_numpy(edge_np.astype(np.float32) / 255.0).unsqueeze(0)
        combined_tensor = torch.cat([rgb_tensor, edge_tensor], dim=0)
        return combined_tensor

# New transform to generate Canny edges for loss regularization
class GenerateCannyEdges:
    def __init__(self, canny_low_thresh=30, canny_high_thresh=100):
        self.low_thresh = canny_low_thresh
        self.high_thresh = canny_high_thresh
    def __call__(self, pil_image):
        gray_np = np.array(pil_image.convert("L"))
        edge_np = canny_edge_detector(gray_np, self.low_thresh, self.high_thresh)
        edge_tensor = torch.from_numpy(edge_np.astype(np.float32) / 255.0).unsqueeze(0)
        return edge_tensor
    
def load_params():
    import argparse
    parser = argparse.ArgumentParser(description='Monocular Depth Estimation with EfficientNet-B3')

    # parser.add_argument('-env', '--env', type=str, default='local', help='Environment: local or studentcluster', required=False)
    parser.add_argument('-n', '--train_first_n', type=int, help='Number of training samples to use (0 for all)', required=False)
    parser.add_argument('-e', '--epochs', type=int, help='number of epochs to train', required=False)
    parser.add_argument('-lr', '--learning_rate', type=float, help='Learning rate', required=False)
    parser.add_argument('-fff', '--fixed_for_first_n', type=int, help='Number of epochs to freeze the encoder', required=False)
    parser.add_argument('-cannyreg', '--use_canny_regularizer', action='store_true', help='Use Canny edge regularization', required=False)
    parser.add_argument('-cannyregweight', '--canny_reg_weight', type=float, help='Weight for Canny edge regularization', required=False)
    parser.add_argument('-cannyinc', '--canny_inc', action='store_true', help='Increase Canny edge regularization weight over epochs', required=False)
    parser.add_argument('-cannysmooth', '--canny_smoothening', type=int, nargs=2, help='Canny edge smoothening parameters', required=False)
    parser.add_argument('-goodloss', '--canny_goodloss_weight', type=float, help='Weight for good loss term', required=False)
    
    
    args = parser.parse_args()

    global TRAIN_FIRST_N, NUM_EPOCHS, LEARNING_RATE, FIXED_FOR_FIRST_N
    global USE_CANNY_REGULARIZER, CANNY_REG_WEIGHT, CANNY_INC, CANNY_SMOOTHENING, CANNY_GOODLOSS_WEIGHT
    global output_dir 

    if args.train_first_n is not None:
        TRAIN_FIRST_N = args.train_first_n
    if args.epochs is not None:
        NUM_EPOCHS = args.epochs
    if args.learning_rate is not None:
        LEARNING_RATE = args.learning_rate
    if args.fixed_for_first_n is not None:
        FIXED_FOR_FIRST_N = args.fixed_for_first_n
    if args.use_canny_regularizer is not None:
        USE_CANNY_REGULARIZER = args.use_canny_regularizer
    if args.canny_reg_weight is not None:
        CANNY_REG_WEIGHT = args.canny_reg_weight
    if args.canny_inc is not None:
        CANNY_INC = args.canny_inc
    if args.canny_smoothening is not None:
        CANNY_SMOOTHENING = args.canny_smoothening
    if args.canny_goodloss_weight is not None:
        CANNY_GOODLOSS_WEIGHT = args.canny_goodloss_weight

    print(f"Training with {TRAIN_FIRST_N} samples, {NUM_EPOCHS} epochs, learning rate {LEARNING_RATE}, fixed for first {FIXED_FOR_FIRST_N} epochs")
    print(f"USE_CANNY_REGULARIZER={USE_CANNY_REGULARIZER}, CANNY_REG_WEIGHT={CANNY_REG_WEIGHT}, CANNY_INC={CANNY_INC}, CANNY_SMOOTHENING={CANNY_SMOOTHENING}, CANNY_GOODLOSS_WEIGHT={CANNY_GOODLOSS_WEIGHT}")

    output_dir = os.getcwd()+f"eb3-cr-n{TRAIN_FIRST_N}-e{NUM_EPOCHS}-lr{LEARNING_RATE}-fff{FIXED_FOR_FIRST_N}-crw{CANNY_REG_WEIGHT}-crinc{CANNY_INC}-csg{CANNY_SMOOTHENING[0]}-{CANNY_SMOOTHENING[1]}-cglw{CANNY_GOODLOSS_WEIGHT}"

# %%
class ENV(Enum):
    STUDENTCLUSTER = 1
    LOCAL = 2

RUN_ON = ENV.LOCAL  # or RUNON.STUDENTCLUSTER, depending on your environment
data_dir = os.getcwd() if RUN_ON == ENV.LOCAL else '/work/scratch/timrieder/cil_monocular_depth'
output_dir = os.getcwd()
train_dir = os.path.join(data_dir, 'train/train')
test_dir = os.path.join(data_dir, 'test/test')
train_list_file = os.path.join(data_dir, 'train_list.txt')
test_list_file = os.path.join(data_dir, 'test_list.txt')
results_dir = os.path.join(output_dir, 'output/results')
predictions_dir = os.path.join(output_dir, 'output/predictions')

# %% [markdown]
# ### Hyperparameters
TRAIN_FIRST_N = 0  # Set to 0 to use all data, or set to a positive integer to limit the number of samples
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4 * 0
FIXED_FOR_FIRST_N = 0 # Set nr. of epochs to freeze the encoder
BATCH_SIZE = 4
USE_CANNY_REGULARIZER = True  # New flag to control edge-aware loss regularization
CANNY_REG_WEIGHT = 1        # Weight for the Canny regularization term
CANNY_INC = False
CANNY_SMOOTHENING = [80, 5]
CANNY_GOODLOSS_WEIGHT = .5
NUM_WORKERS = 0

WEIGHT_DECAY = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_SIZE = (426, 560)
PERSISTENT_WORKERS = True if NUM_WORKERS > 0 else False
PIN_MEMORY = True



output_dir_name_hyperparam = f"mon_depth_enb3_n{TRAIN_FIRST_N}_e{NUM_EPOCHS}_lr{LEARNING_RATE}_fff{FIXED_FOR_FIRST_N}"
output_dir = os.path.join(os.getcwd(), output_dir_name_hyperparam)
results_dir = os.path.join(output_dir, 'results')
predictions_dir = os.path.join(output_dir, 'predictions')

# %%
print (f"Using device: {DEVICE}")

# %% [markdown]
# ### Helper functions

# %%
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def target_transform(depth):
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(0).unsqueeze(0),
        size=INPUT_SIZE,
        mode='bilinear',
        align_corners=True
    ).squeeze()
    depth = depth.unsqueeze(0)
    return depth

# %% [markdown]
# # Dataset

# %%
class DepthDataset(Dataset):
    def __init__(self, data_dir, list_file, transform=None, target_transform=None, canny_transform=None, has_gt=True, use_canny_regularizer=False):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.canny_transform = canny_transform
        self.has_gt = has_gt
        self.use_canny_regularizer = use_canny_regularizer
        with open(list_file, 'r') as f:
            if has_gt:
                self.file_pairs = [line.strip().split() for line in f]
            else:
                self.file_list = [line.strip() for line in f]
    def __len__(self):
        return len(self.file_pairs if self.has_gt else self.file_list)
    def __getitem__(self, idx):
        if self.has_gt:
            rgb_path = os.path.join(self.data_dir, self.file_pairs[idx][0])
            depth_path = os.path.join(self.data_dir, self.file_pairs[idx][1])
            pil_rgb_image = Image.open(rgb_path).convert('RGB')
            depth = np.load(depth_path).astype(np.float32)
            depth = torch.from_numpy(depth)
            # print (f"Type and shape of depth: {type(depth)}, {depth.shape}")
            # print (f"Type and shape of pil_rgb_image: {type(pil_rgb_image)}, {pil_rgb_image.size}")
            rgb_tensor_transformed = self.transform(pil_rgb_image) if self.transform else transforms.ToTensor()(pil_rgb_image)
            if self.target_transform:
                depth = self.target_transform(depth)
            else:
                depth = depth.unsqueeze(0)
            if self.use_canny_regularizer and self.canny_transform:
                canny_edges = self.canny_transform(pil_rgb_image)
                return rgb_tensor_transformed, depth, canny_edges, self.file_pairs[idx][0]
            else:
                return rgb_tensor_transformed, depth, self.file_pairs[idx][0]
        else:
            rgb_path = os.path.join(self.data_dir, self.file_list[idx].split(' ')[0])
            pil_rgb_image = Image.open(rgb_path).convert('RGB')
            rgb_tensor_transformed = self.transform(pil_rgb_image) if self.transform else transforms.ToTensor()(pil_rgb_image)
            return rgb_tensor_transformed, self.file_list[idx]

# %% [markdown]
# # Model - EfficientNet-B3

# %%
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

# %%
class EfficientNetB3Depth(nn.Module):
    def __init__(self):
        super(EfficientNetB3Depth, self).__init__()
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1
        base_model = efficientnet_b3(weights=weights)
        # Modify first conv to accept 4 channels (RGB + Canny)
        original_conv = base_model.features[0][0]
        new_conv = nn.Conv2d(4, original_conv.out_channels,
                             kernel_size=original_conv.kernel_size,
                             stride=original_conv.stride,
                             padding=original_conv.padding,
                             bias=(original_conv.bias is not None))
        # new_conv.weight.data[:, :3, :, :] = original_conv.weight.data.clone()
        # new_conv.weight.data[:, 3, :, :] = original_conv.weight.data.mean(dim=1)
        with torch.no_grad():
            new_conv.weight[:, :3] = original_conv.weight.data.clone()
            new_conv.weight[:, 3:] = 0.0

        base_model.features[0][0] = new_conv
        self.encoder = base_model.features
        self.decoder_conv_initial = nn.Conv2d(1536, 512, kernel_size=1)
        self.upconv5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = UNetBlock(512, 256)
        self.upconv4 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec4 = UNetBlock(256, 128)
        self.upconv3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec3 = UNetBlock(128, 64)
        self.upconv2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(64, 32)
        self.upconv1 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(32, 16)
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)
    def forward(self, x):
        enc_out = self.encoder(x)
        d = self.decoder_conv_initial(enc_out)
        d = self.upconv5(d)
        d = self.dec5(d)
        d = self.upconv4(d)
        d = self.dec4(d)
        d = self.upconv3(d)
        d = self.dec3(d)
        d = self.upconv2(d)
        d = self.dec2(d)
        d = self.upconv1(d)
        if d.shape[2:] != x.shape[2:]:
            d = nn.functional.interpolate(d, size=x.shape[2:], mode='bilinear', align_corners=True)
        d = self.dec1(d)
        out = self.final_conv(d)
        return torch.sigmoid(out) * 10
    
def odd(x):
    return int(x//2) * 2 + 1

# Add custom edge-aware loss functions
print_one = True
def gradient_loss(pred_depth, canny_edge_map, device, factor):
    global print_one

    # thicken the edges of canny_edge_map to 3 pixels
    thickened_canny = nn.functional.max_pool2d(canny_edge_map, kernel_size=3, stride=1, padding=1)
    # smoothen the edges
    gauss_size = factor * CANNY_SMOOTHENING[1] + (1 - factor) * CANNY_SMOOTHENING[0]
    # gaussian_kernel = transforms.GaussianBlur(kernel_size=(odd(gauss_size), odd(gauss_size)), sigma=gauss_size/5)
    # smoothened_canny = gaussian_kernel(canny_edge_map)
    smoothened_canny = nn.functional.avg_pool2d(thickened_canny, kernel_size=odd(gauss_size), stride=1, padding=odd(gauss_size)//2)
    # smoothed_canny cv2.GaussianBlur(thickened_canny, (5, 5), 1)
    # gaussian_kernel = torch.tensor([[1, 4, 7, 4, 1],
    #                                  [4, 16, 26, 16, 4],
    #                                  [7, 26, 41, 26, 7],
    #                                  [4, 16, 26, 16, 4],
    #                                  [1, 4, 7, 4, 1]], dtype=torch.float32, device=canny_edge_map.device)
    # gaussian_kernel /= gaussian_kernel.sum()
    # gaussian_kernel = gaussian_kernel.view(1, 1, 5, 5)  # Reshape to 4D for convolution
    # smoothened_canny = nn.functional.conv2d(thickened_canny, gaussian_kernel, padding=2)
    # canny_edge_map_resized = nn.functional.interpolate(canny_edge_map, size=pred_depth.shape[-2:], mode='nearest')
    if print_one:
        print(f"Gradient loss: {pred_depth.shape}, {canny_edge_map.shape}")
        plt.imshow(smoothened_canny[0].cpu().squeeze(), cmap='gray')
        plt.title("Canny Edge Map")
        plt.axis('off')
        plt.show()
        print_one = False
    
    sobel_x_kernel = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=device).reshape(1,1,3,3)
    sobel_y_kernel = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=device).reshape(1,1,3,3)

    replication_pad = nn.ReplicationPad2d(1)
    pred_depth_padded = replication_pad(pred_depth)
    depth_grad_x = nn.functional.conv2d(pred_depth_padded, sobel_x_kernel, padding=0)
    depth_grad_y = nn.functional.conv2d(pred_depth_padded, sobel_y_kernel, padding=0)

    weights = (1-smoothened_canny)
    loss = torch.mean(torch.abs(depth_grad_x) * weights) + torch.mean(torch.abs(depth_grad_y) * weights)
    good_loss = torch.mean(1 - torch.abs(depth_grad_x * (1-weights))) + torch.mean(1 - torch.abs(depth_grad_y * (1-weights)))
    loss = loss + CANNY_GOODLOSS_WEIGHT * good_loss
    return loss

class CombinedLoss(nn.Module):
    def __init__(self, canny_reg_weight, use_canny_regularizer=False):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.canny_reg_weight = canny_reg_weight
        self.use_canny_regularizer = use_canny_regularizer
    def forward(self, predictions, targets, canny_edges=None, device='cpu', current_iter=NUM_EPOCHS):
        reconstruction_loss = self.mse_loss(predictions, targets)
        edge_loss_val = torch.tensor(0.0, device=device)
        total_loss = reconstruction_loss
        if self.use_canny_regularizer and canny_edges is not None and self.canny_reg_weight > 0:
            canny_edges_on_device = canny_edges.to(device)
            factor = current_iter / NUM_EPOCHS
            # print (f"Factor: {factor}")
            edge_loss = gradient_loss(predictions, canny_edges_on_device, device, factor) * (1 if not CANNY_INC else factor)
            total_loss = reconstruction_loss + self.canny_reg_weight * edge_loss
            edge_loss_val = edge_loss
        return total_loss, reconstruction_loss, edge_loss_val

# %% [markdown]
# # Training loop

# %%
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, use_canny_regularizer_flag):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    sample_batch = None  # For storing a batch for visualization
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        # Freeze encoder for first epoch only, unfreeze from second epoch onward.
        if epoch < FIXED_FOR_FIRST_N:
            # For DataParallel, access encoder via model.module.encoder.
            for param in model.module.encoder.parameters():
                param.requires_grad = False
            print(f"Encoder frozen for epoch {epoch+1}.")
        elif epoch == FIXED_FOR_FIRST_N:
            for param in model.module.encoder.parameters():
                param.requires_grad = True
            print(f"Encoder unfrozen from epoch {epoch+1}.")
        model.train()
        train_loss = 0.0
        train_recon_loss_epoch = 0.0
        train_edge_loss_epoch = 0.0
        for batch_idx, data_items in enumerate(tqdm(train_loader, desc="Training")):
            if use_canny_regularizer_flag:
                inputs, targets, canny_edges, _ = data_items
            else:
                inputs, targets, _ = data_items
                canny_edges = None
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss, recon_loss, edge_loss = criterion(outputs, targets, canny_edges, device, epoch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_recon_loss_epoch += recon_loss.item() * inputs.size(0)
            if use_canny_regularizer_flag:
                train_edge_loss_epoch += edge_loss.item() * inputs.size(0)
            # Save the first batch for visualization
            if batch_idx == 0 and sample_batch is None:
                # Detach and move to cpu for visualization
                sample_batch = (inputs.detach().cpu(), targets.detach().cpu(), outputs.detach().cpu())
        train_loss /= len(train_loader.dataset)
        train_recon_loss_epoch /= len(train_loader.dataset)
        if use_canny_regularizer_flag:
            train_edge_loss_epoch /= len(train_loader.dataset)
        train_losses.append(train_loss)
        model.eval()
        val_loss = 0.0
        val_recon_loss_epoch = 0.0
        val_edge_loss_epoch = 0.0
        with torch.no_grad():
            for data_items in tqdm(val_loader, desc="Validation"):
                if use_canny_regularizer_flag:
                    inputs, targets, canny_edges, _ = data_items
                else:
                    inputs, targets, _ = data_items
                    canny_edges = None
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss, recon_loss, edge_loss = criterion(outputs, targets, canny_edges, device, epoch)
                val_loss += loss.item() * inputs.size(0)
                val_recon_loss_epoch += recon_loss.item() * inputs.size(0)
                if use_canny_regularizer_flag:
                    val_edge_loss_epoch += edge_loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        val_recon_loss_epoch /= len(val_loader.dataset)
        if use_canny_regularizer_flag:
            val_edge_loss_epoch /= len(val_loader.dataset)
            print(f"Train Loss: {train_loss:.4f} (Recon: {train_recon_loss_epoch:.4f}, Edge: {train_edge_loss_epoch:.4f}) | "
                  f"Val Loss: {val_loss:.4f} (Recon: {val_recon_loss_epoch:.4f}, Edge: {val_edge_loss_epoch:.4f})")
        else:
            print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        # --- Save 1 sample image per epoch ---
        if sample_batch is not None:
            input_np = sample_batch[0][0].permute(1, 2, 0).numpy()
            target_np = sample_batch[1][0].squeeze().numpy()
            output_np = sample_batch[2][0].squeeze().numpy()
            # If input has 4 channels, drop the last (Canny) for visualization
            if input_np.shape[2] == 4:
                input_np = input_np[:, :, :3]
            input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min() + 1e-6)
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(input_np)
            plt.title("RGB Input")
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(target_np, cmap='plasma')
            plt.title("Ground Truth Depth")
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(output_np, cmap='plasma')
            plt.title("Predicted Depth")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"epoch_sample_{epoch+1}.png"))
            plt.close()
            sample_batch = None  # Reset for next epoch
        # --- End sample image save ---
        if True or val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(results_dir, 'best_model.pth'))
            print(f"New best model saved at epoch {epoch+1} with validation loss: {val_loss:.4f}")
            # print(f"New best (ANY!!!!!!!!!!!!!) model saved at epoch {epoch+1} with validation loss: {val_loss:.4f}")
    print(f"\nBest model with validation loss: {best_val_loss:.4f}")
    model.load_state_dict(torch.load(os.path.join(results_dir, 'best_model.pth')))
    return model

# %% [markdown]
# # Model evaluation

# %%
def evaluate_model(model, val_loader, device):
    model.eval()
    mae = 0.0
    rmse = 0.0
    rel = 0.0
    delta1 = 0.0
    delta2 = 0.0
    delta3 = 0.0
    sirmse = 0.0
    total_samples = 0
    target_shape = None
    with torch.no_grad():
        for data in tqdm(val_loader, desc="Evaluating"):
            # If canny regularizer is enabled, the dataset returns 4 values.
            if len(data) == 4:
                inputs, targets, _, filenames = data
            else:
                inputs, targets, filenames = data
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            total_samples += batch_size
            if target_shape is None:
                target_shape = targets.shape
            outputs = model(inputs)
            outputs = nn.functional.interpolate(
                outputs,
                size=targets.shape[-2:],
                mode='bilinear',
                align_corners=True
            )
            abs_diff = torch.abs(outputs - targets)
            mae += torch.sum(abs_diff).item()
            rmse += torch.sum(torch.pow(abs_diff, 2)).item()
            rel += torch.sum(abs_diff / (targets + 1e-6)).item()
            for i in range(batch_size):
                pred_np = outputs[i].cpu().squeeze().numpy()
                target_np = targets[i].cpu().squeeze().numpy()
                EPSILON = 1e-6
                valid_target = target_np > EPSILON
                if not np.any(valid_target):
                    continue
                target_valid = target_np[valid_target]
                pred_valid = pred_np[valid_target]
                log_target = np.log(target_valid)
                pred_valid = np.where(pred_valid > EPSILON, pred_valid, EPSILON)
                log_pred = np.log(pred_valid)
                diff = log_pred - log_target
                diff_mean = np.mean(diff)
                sirmse += np.sqrt(np.mean((diff - diff_mean) ** 2))
            max_ratio = torch.max(outputs / (targets + 1e-6), targets / (outputs + 1e-6))
            delta1 += torch.sum(max_ratio < 1.25).item()
            delta2 += torch.sum(max_ratio < 1.25**2).item()
            delta3 += torch.sum(max_ratio < 1.25**3).item()
            # if total_samples <= 5 * batch_size:
            #     for i in range(min(batch_size, 5)):
            #         idx = total_samples - batch_size + i
            #         input_np = inputs[i].cpu().permute(1, 2, 0).numpy()
            #         target_np = targets[i].cpu().squeeze().numpy()
            #         output_np = outputs[i].cpu().squeeze().numpy()
            #         input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min() + 1e-6)
            #         plt.figure(figsize=(15, 5))
            #         plt.subplot(1, 3, 1)
            #         plt.imshow(input_np)
            #         plt.title("RGB Input")
            #         plt.axis('off')
            #         plt.subplot(1, 3, 2)
            #         plt.imshow(target_np, cmap='plasma')
            #         plt.title("Ground Truth Depth")
            #         plt.axis('off')
            #         plt.subplot(1, 3, 3)
            #         plt.imshow(output_np, cmap='plasma')
            #         plt.title("Predicted Depth")
            #         plt.axis('off')
            #         plt.tight_layout()
            #         # plt.savefig(os.path.join(results_dir, f"sample_{idx}.png"))
            #         plt.close()
            del inputs, targets, outputs, abs_diff, max_ratio
        torch.cuda.empty_cache()
    total_pixels = target_shape[1] * target_shape[2] * target_shape[3]  # type: ignore
    mae /= total_samples * total_pixels
    rmse = np.sqrt(rmse / (total_samples * total_pixels))
    rel /= total_samples * total_pixels
    sirmse = sirmse / total_samples
    delta1 /= total_samples * total_pixels
    delta2 /= total_samples * total_pixels
    delta3 /= total_samples * total_pixels
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'siRMSE': sirmse,
        'REL': rel,
        'Delta1': delta1,
        'Delta2': delta2,
        'Delta3': delta3
    }
    return metrics

# %% [markdown]
# # Generate test predictions

# %%
def generate_test_predictions(model, test_loader, device):
    model.eval()
    ensure_dir(predictions_dir)
    with torch.no_grad():
        for inputs, filenames in tqdm(test_loader, desc="Generating Test Predictions"):
            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            outputs = model(inputs)
            outputs = nn.functional.interpolate(
                outputs,
                size=(426, 560),
                mode='bilinear',
                align_corners=True
            )
            for i in range(batch_size):
                filename = filenames[i].split(' ')[1]
                depth_pred = outputs[i].cpu().squeeze().numpy()
                np.save(os.path.join(predictions_dir, f"{filename}"), depth_pred)
            del inputs, outputs
        torch.cuda.empty_cache()

# %% [markdown]
# # Putting it all together

# %%
def main():
    # load_params()
    # ensure_dir(results_dir)
    # ensure_dir(predictions_dir)
    train_rgb_transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ProcessImageAndAddCanny(canny_low_thresh=30, canny_high_thresh=100),
        transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5],
                             std=[0.229, 0.224, 0.225, 0.5])
    ])
    test_rgb_transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        ProcessImageAndAddCanny(canny_low_thresh=30, canny_high_thresh=100),
        transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5],
                             std=[0.229, 0.224, 0.225, 0.5])
    ])
    canny_generator_transform = None
    if USE_CANNY_REGULARIZER:
        canny_generator_transform = transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            GenerateCannyEdges(canny_low_thresh=30, canny_high_thresh=100)
        ])
    train_full_dataset = DepthDataset(
        data_dir=train_dir,
        list_file=train_list_file,
        transform=train_rgb_transform,
        target_transform=target_transform,
        canny_transform=canny_generator_transform,
        has_gt=True,
        use_canny_regularizer=USE_CANNY_REGULARIZER
    )
    test_dataset = DepthDataset(
        data_dir=test_dir,
        list_file=test_list_file,
        transform=test_rgb_transform,
        canny_transform=canny_generator_transform,
        has_gt=False,
        use_canny_regularizer=USE_CANNY_REGULARIZER
    )
    total_size = len(train_full_dataset) if TRAIN_FIRST_N == 0 else TRAIN_FIRST_N
    train_size = int(0.85 * total_size)
    val_size = total_size - train_size
    unused_size = len(train_full_dataset) - total_size
    torch.manual_seed(0)
    train_dataset, val_dataset, _ = torch.utils.data.random_split(
        train_full_dataset, [train_size, val_size, unused_size]
    )
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     num_workers=NUM_WORKERS,
    #     pin_memory=PIN_MEMORY,
    #     drop_last=True,
    #     persistent_workers=PERSISTENT_WORKERS
    # )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    # print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
        print(f"Initially allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")


    results_dir = os.path.join(os.getcwd(), 'output/results-eb3-cr1inc805avg-0-40-.1-5-.5goodloss')
    # results_dir = os.path.join(os.getcwd(), 'output/results-eb3-cr-0-20-.2-2')
    loadModelFrom = os.path.join(results_dir, 'best_model.pth')
    if os.path.exists(loadModelFrom):
        print(f"Loading model from {loadModelFrom}")
        model = EfficientNetB3Depth()
        state_dict = torch.load(loadModelFrom, map_location=DEVICE)
        # Remove 'module.' prefix if present
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict)
        model = model.to(DEVICE)
    # if os.path.exists(loadModelFrom):
    #     print(f"Loading model from {loadModelFrom}")
    #     model = EfficientNetB3Depth()
    #     model.load_state_dict(torch.load(loadModelFrom, map_location=DEVICE))
    else:
        print(f"No pre-trained model found at {loadModelFrom}, initializing a new model.")

    
    # model = EfficientNetB3Depth()
    # model = nn.DataParallel(model)
    # model = model.to(DEVICE)
    # print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"Memory allocated after model init: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
    # criterion = CombinedLoss(canny_reg_weight=CANNY_REG_WEIGHT, use_canny_regularizer=USE_CANNY_REGULARIZER).to(DEVICE)
    # optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # print("Starting training...")
    # model = train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, DEVICE, USE_CANNY_REGULARIZER)
    # print("Evaluating model on validation set...")
    print ("No training, just evaluating model on validation set...")
    metrics = evaluate_model(model, val_loader, DEVICE)
    print("\nValidation Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    # with open(os.path.join(results_dir, 'validation_metrics.txt'), 'w') as f:
    #     for name, value in metrics.items():
    #         f.write(f"{name}: {value:.4f}\n")
    print("Generating predictions for test set...")
    generate_test_predictions(model, test_loader, DEVICE)
    print(f"Results saved to {results_dir}")
    print(f"All test depth map predictions saved to {predictions_dir}")

# %%
if __name__ == "__main__":
    main()



