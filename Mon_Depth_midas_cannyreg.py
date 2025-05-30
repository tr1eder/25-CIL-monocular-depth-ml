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
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from stopit import ThreadingTimeout as Timeout, threading_timeoutable as timeoutable  #doctest: +SKIP
from enum import Enum
import argparse
from canny_edge_detector import canny_edge_detector

def load_params():
    pass
    # parser = argparse.ArgumentParser(description='Monocular Depth Estimation with MiDaS (Minimal Version)')

    # # parser.add_argument('-env', '--env', type=str, default='local', help='Environment: local or studentcluster', required=False)
    # parser.add_argument('-n', '--train_first_n', type=int, help='Number of training samples to use (0 for all)', required=False)
    # parser.add_argument('-e', '--epochs', type=int, help='number of epochs to train', required=False)
    # parser.add_argument('-lr', '--learning_rate', type=float, help='Learning rate', required=False)
    # parser.add_argument('-fff', '--fixed_for_first_n', type=int, help='Number of epochs to freeze the encoder', required=False)
    # parser.add_argument('-cannyreg', '--use_canny_regularizer', action='store_true', help='Use Canny edge regularization', required=False)
    # parser.add_argument('-cannyregweight', '--canny_reg_weight', type=float, help='Weight for Canny edge regularization', required=False)
    # parser.add_argument('-cannyinc', '--canny_inc', action='store_true', help='Increase Canny edge regularization weight over epochs', required=False)
    # parser.add_argument('-cannysmooth', '--canny_smoothening', type=int, nargs=2, help='Canny edge smoothening parameters', required=False)
    # parser.add_argument('-goodloss', '--canny_goodloss_weight', type=float, help='Weight for good loss term', required=False)
    
    # args = parser.parse_args()

    # global TRAIN_FIRST_N, NUM_EPOCHS, LEARNING_RATE, FIXED_FOR_FIRST_N
    # global USE_CANNY_REGULARIZER, CANNY_REG_WEIGHT, CANNY_INC, CANNY_SMOOTHENING, CANNY_GOODLOSS_WEIGHT
    # global output_dir 

    # if args.train_first_n is not None:
    #     TRAIN_FIRST_N = args.train_first_n
    # if args.epochs is not None:
    #     NUM_EPOCHS = args.epochs
    # if args.learning_rate is not None:
    #     LEARNING_RATE = args.learning_rate
    # if args.fixed_for_first_n is not None:
    #     FIXED_FOR_FIRST_N = args.fixed_for_first_n
    # if args.use_canny_regularizer is not None:
    #     USE_CANNY_REGULARIZER = args.use_canny_regularizer
    # if args.canny_reg_weight is not None:
    #     CANNY_REG_WEIGHT = args.canny_reg_weight
    # if args.canny_inc is not None:
    #     CANNY_INC = args.canny_inc
    # if args.canny_smoothening is not None:
    #     CANNY_SMOOTHENING = args.canny_smoothening
    # if args.canny_goodloss_weight is not None:
    #     CANNY_GOODLOSS_WEIGHT = args.canny_goodloss_weight

    # print(f"Training with {TRAIN_FIRST_N} samples, {NUM_EPOCHS} epochs, learning rate {LEARNING_RATE}, fixed for first {FIXED_FOR_FIRST_N} epochs")

    # output_dir_name = f"midas_minimal_n{TRAIN_FIRST_N}_e{NUM_EPOCHS}_lr{LEARNING_RATE}_fff{FIXED_FOR_FIRST_N}"
    # output_dir = os.path.join(os.getcwd(), output_dir_name)


# %%
class ENV(Enum):
    STUDENTCLUSTER = 1
    LOCAL = 2

RUN_ON = ENV.LOCAL  # or RUNON.STUDENTCLUSTER, depending on your environment
data_dir = os.getcwd() if RUN_ON == ENV.LOCAL else '/work/scratch/timrieder/cil_monocular_depth'


# %% [markdown]
# ### Hyperparameters
TRAIN_FIRST_N = 4000  # Set to 0 to use all data, or set to a positive integer to limit the number of samples
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4 * .2
FIXED_DEC_FOR_FIRST_N = 1
FIXED_ENC_FOR_FIRST_N = max(2, FIXED_DEC_FOR_FIRST_N) # Set nr. of epochs to freeze the encoder
BATCH_SIZE = 4
USE_CANNY_REGULARIZER = True  # Set to False to disable Canny edge regularization
CANNY_REG_WEIGHT = .1  # Weight for the Canny edge regularization loss
CANNY_SMOOTHENING = [80, 5]
CANNY_GOODLOSS_WEIGHT = .5
NUM_WORKERS = 4

WEIGHT_DECAY = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_SIZE = (426, 560)  # Adjusted to user's requirement
PERSISTENT_WORKERS = True if NUM_WORKERS > 0 else False
PIN_MEMORY = True

# Set directories
train_dir = os.path.join(data_dir, 'train/train')
test_dir = os.path.join(data_dir, 'test/test')
train_list_file = os.path.join(data_dir, 'train_list.txt')
test_list_file = os.path.join(data_dir, 'test_list.txt')
# This ensures results_dir and predictions_dir are correctly formed if this cell defines the params
output_dir_name_hyperparam = f"midas_cannyreg_n{TRAIN_FIRST_N}_e{NUM_EPOCHS}_lr{LEARNING_RATE}_fff{FIXED_DEC_FOR_FIRST_N}-{FIXED_ENC_FOR_FIRST_N}"
output_dir = os.path.join(os.getcwd(), output_dir_name_hyperparam)
results_dir = os.path.join(output_dir, 'results')
predictions_dir = os.path.join(output_dir, 'predictions')


# %%
print (f"Spawning worker: {os.getpid()}")
# print (f"Output directory: {output_dir}")
# print (f"Results directory: {results_dir}")
# print (f"Predictions directory: {predictions_dir}")

# %% [markdown]
# ### Helper functions

# %%
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

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

class GenerateCannyEdges:
    def __init__(self, canny_low_thresh=30, canny_high_thresh=100):
        self.low_thresh = canny_low_thresh
        self.high_thresh = canny_high_thresh
    def __call__(self, pil_image):
        gray_np = np.array(pil_image.convert("L"))
        edge_np = canny_edge_detector(gray_np, self.low_thresh, self.high_thresh)
        edge_tensor = torch.from_numpy(edge_np.astype(np.float32) / 255.0).unsqueeze(0)
        return edge_tensor
    

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
# # Model - MiDaS Minimal

# %%

# class MiDaSDepth(nn.Module):
#     def __init__(self, in_channels=4): # Added in_channels argument
#         super(MiDaSDepth, self).__init__()
#         # Load MiDaS model - DPT_Hybrid
#         # Adding trust_repo=True for potentially custom code in the MiDaS hub model
#         self.midas_model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True)
#         self.input_channels_configured = in_channels # Store configured input channels

#         if self.input_channels_configured == 4:
#             # The DPT_Hybrid model uses a Vision Transformer (ViT) backbone.
#             # The first layer to modify is the projection convolution within the patch embedding layer.
#             # Path: self.midas_model.pretrained.model.patch_embed.proj
            
#             original_conv_layer = self.midas_model.pretrained.model.patch_embed.proj # type: ignore
            
#             # Get parameters from the original convolutional layer
#             original_in_chans = original_conv_layer.in_channels # This should be 3
#             out_channels = original_conv_layer.out_channels
#             kernel_size = original_conv_layer.kernel_size
#             stride = original_conv_layer.stride
#             padding = original_conv_layer.padding
#             dilation = original_conv_layer.dilation
#             groups = original_conv_layer.groups
#             has_bias = (original_conv_layer.bias is not None)

#             # Create the new convolutional layer with 4 input channels
#             new_conv_layer = nn.Conv2d(
#                 in_channels=4, # New number of input channels
#                 out_channels=out_channels,
#                 kernel_size=kernel_size,
#                 stride=stride,
#                 padding=padding,
#                 dilation=dilation,
#                 groups=groups,
#                 bias=has_bias
#             )

#             # Copy weights from the original layer
#             with torch.no_grad():
#                 # Copy weights for the first 3 channels (RGB)
#                 new_conv_layer.weight.data[:, :original_in_chans, :, :] = original_conv_layer.weight.data.clone()
                
#                 # Initialize weights for the 4th channel (Canny)
#                 # Following your EfficientNet example, initializing to zero
#                 new_conv_layer.weight.data[:, original_in_chans:, :, :].zero_()

#                 if has_bias:
#                     new_conv_layer.bias.data = original_conv_layer.bias.data.clone() # type: ignore
            
#             # Replace the original layer in the MiDaS model's backbone
#             self.midas_model.pretrained.model.patch_embed.proj = new_conv_layer # type: ignore
#             print("MiDaS model's first conv layer (patch_embed.proj) modified to accept 4 input channels.")

#         # These lines from your snippet seem unused if the model processes tensors directly from the DataLoader,
#         # as the ProcessImageAndAddCanny transform already produces tensors.
#         # If they are needed for a different workflow, they can be uncommented.
#         # midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
#         # self.midas_transform = midas_transforms.dpt_transform # type: ignore
#         # self.to_pil = transforms.ToPILImage() # Convert tensor to PIL image for MiDaS transform

#     def forward(self, x: torch.Tensor):
#         # x: (B, 4, H, W) or (B, 3, H, W)
        
#         # IMPORTANT: This forward method needs to be updated if self.input_channels_configured == 4
#         # to correctly pass the 4-channel tensor to self.midas_model
#         # and handle cases where x might be 3-channel.
#         # The current implementation below will take x (potentially 4 channels),
#         # slice it to 3 channels, and then pass that to self.midas_model.
#         # If self.midas_model now expects 4 channels, this will cause a mismatch.

#         # Current logic from your snippet:
#         if x.shape[1] == 4 and self.input_channels_configured == 3: # If model expects 3 but gets 4
#             x_processed = x[:, :3, :, :]  # Use only RGB
#         elif x.shape[1] == 3 and self.input_channels_configured == 4: # If model expects 4 but gets 3
#             # Pad the 3-channel input to 4 channels (e.g., with zeros for Canny)
#             padding = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
#             x_processed = torch.cat((x, padding), dim=1)
#         else: # Channels match or model expects 3 and gets 3, or model expects 4 and gets 4
#             x_processed = x

#         B, C_actual, H_orig, W_orig = x_processed.shape # C_actual is now what midas_model expects

#         # Resize
#         # MiDaS DPT_Hybrid typically expects 384x384 input
#         x_resized = nn.functional.interpolate(x_processed, size=(384, 384), mode='bilinear', align_corners=False)

#         # Run MiDaS
#         # self.midas_model now processes x_resized which has C_actual channels
#         depth_from_midas = self.midas_model(x_resized) # type: ignore
#         # DPT_Hybrid output is typically (B, 384, 384), needs unsqueeze for channel dim

#         # Upsample back to original size
#         depth_resized = nn.functional.interpolate(
#             depth_from_midas.unsqueeze(1), size=(H_orig, W_orig), mode='bicubic', align_corners=False
#         )

#         # Normalization and inversion logic from your snippet:
#         depth = depth_resized
#         # It's generally better to normalize per-sample if batch items have very different depth ranges
#         # For simplicity, using batch-wise min/max as in the snippet, but this can be problematic.
#         # Consider: current_min = depth.amin(dim=(1,2,3), keepdim=True)
#         #           current_max = depth.amax(dim=(1,2,3), keepdim=True)
#         current_min = depth.min() 
#         current_max = depth.max()
        
#         depth = depth - current_min
#         depth = depth / (current_max - current_min + 1e-6)  # Normalize to [0, approx 1]
        
#         # This inversion makes the output not strictly in [0,1] and might affect loss/metrics
#         # if they expect a normalized [0,1] range where 0 is far and 1 is near (or vice-versa).
#         depth = 1.0 / (depth + 0.14)

#         return depth

class MiDaSDepth(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained MiDaS model and transform
        # self.midas_model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        self.midas_model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.midas_transform = midas_transforms.dpt_transform # type: ignore

        # Small refinement head to merge depth and canny
        self.refine = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor):
        """
        x: Tensor of shape (B, 4, H, W) = RGB + Canny edge map
        """
        B, C, H, W = x.shape
        rgb = x[:, :3, :, :]
        canny = x[:, 3:, :, :]  # shape (B,1,H,W)

        # Resize input for MiDaS
        rgb_resized = nn.functional.interpolate(rgb, size=(384, 384), mode='bilinear', align_corners=False)
        depth = self.midas_model(rgb_resized)  # (B, H', W') # type: ignore
        depth = nn.functional.interpolate(depth.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False)

        # Normalize MiDaS depth prediction
        # print(f"Depth values between min and max: {depth.min().item()} and {depth.max().item()}")
        depth_norm = depth - depth.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        # print(f"Depth values after normalization: {depth_norm.min().item()} and {depth_norm.max().item()}")
        depth_norm = depth_norm / (depth_norm.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] + 1e-6)
        # print(f"Depth values after normalization and division: {depth_norm.min().item()} and {depth_norm.max().item()}")
        depth_norm = 1 / (depth_norm + 0.14)  # Inverse depth
        # print(f"Depth values after inversion: {depth_norm.min().item()} and {depth_norm.max().item()}")

        # Concatenate with Canny edge map and refine
        combined = torch.cat([depth_norm, canny], dim=1)  # (B,2,H,W)
        refined_depth = self.refine(combined)  # (B,1,H,W)

        return refined_depth

# class MiDaSDepth(nn.Module):
#     def __init__(self):
#         super(MiDaSDepth, self).__init__()
#         # Load MiDaS model - "MiDaS" for DPT-Hybrid (large), or "MiDaS_small" for a smaller version
#         self.midas_model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid") # Using MiDaS_small for potentially faster/lighter version
#         midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
#         self.midas_transform = midas_transforms.dpt_transform # type: ignore
#         self.to_pil = transforms.ToPILImage()  # Convert tensor to PIL image for MiDaS transform


#     def forward(self, x: torch.Tensor):
#         # x: (B, 4, H, W) or (B, 3, H, W)
#         if x.shape[1] == 4:
#             x = x[:, :3, :, :]  # Use only RGB for MiDaS
#         B, C, H, W = x.shape

#         # Resize
#         x_resized = nn.functional.interpolate(x, size=(384, 384), mode='bilinear', align_corners=False)

#         # Run MiDaS
#         depth = self.midas_model(x_resized) # type: ignore

#         # Upsample back to original size
#         depth_resized = nn.functional.interpolate(
#             depth.unsqueeze(1), size=(H, W), mode='bicubic', align_corners=False
#         )

#         # depth = 1 / (depth_resized + 1e-6)  # Invert depth to get distance

#         depth = depth_resized
#         depth = depth - depth.min()  # Normalize to [0, 1]
#         depth = depth / (depth.max() + 1e-6)  # Avoid division by zero
#         depth = 1 / (depth + .14)  # Inverse depth for visualization

#         return depth

def odd(x): return int(x//2) * 2 + 1

def gradient_loss(pred_depth, canny_edge_map, device, factor):
    gauss_size = odd(factor * CANNY_SMOOTHENING[1] + (1-factor) * CANNY_SMOOTHENING[0])
    # throw error if gauss_size is not an odd integer >=3
    if gauss_size < 3 or gauss_size % 2 != 1: 
        raise ValueError(f"Invalid gauss_size: {gauss_size}. It must be an odd integer >= 3.")
    # print if pred_depth contains nans
    # if torch.isnan(pred_depth).any():
    #     print("Warning: pred_depth contains NaNs.")

    thickened_canny = nn.functional.max_pool2d(canny_edge_map, kernel_size=3, stride=1, padding=1) # thicken
    smoothened_canny = nn.functional.avg_pool2d(thickened_canny, kernel_size=gauss_size, stride=1, padding=gauss_size//2) # smoothen

    sobel_x_kernel = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=device).reshape(1,1,3,3)
    sobel_y_kernel = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=device).reshape(1,1,3,3)
    replication_pad = nn.ReplicationPad2d(1)
    pred_depth_padded = replication_pad(pred_depth)
    depth_grad_x = nn.functional.conv2d(pred_depth_padded, sobel_x_kernel, padding=0)
    depth_grad_y = nn.functional.conv2d(pred_depth_padded, sobel_y_kernel, padding=0)

    weights = (1-smoothened_canny)
    loss = torch.mean(torch.abs(depth_grad_x) * weights) + torch.mean(torch.abs(depth_grad_y) * weights)
    goodloss = torch.mean(1 - torch.abs(depth_grad_x * (1-weights))) + torch.mean(1 - torch.abs(depth_grad_y * (1-weights)))
    return loss + CANNY_GOODLOSS_WEIGHT * goodloss

class BerHuLoss(nn.Module):
    def __init__(self):
        super(BerHuLoss, self).__init__()

    def forward(self, prediction, target):
        diff = prediction - target
        abs_diff = torch.abs(diff)
        c = 0.2 * abs_diff.max().item()  # scalar threshold

        # Apply piecewise function
        l1_mask = abs_diff <= c
        l2_mask = abs_diff > c

        loss = torch.zeros_like(abs_diff)
        loss[l1_mask] = abs_diff[l1_mask]
        loss[l2_mask] = (diff[l2_mask]**2 + c**2) / (2 * c)

        return loss.mean()
class CombinedLoss(nn.Module):
    def __init__(self, canny_reg_weight=1.0, use_canny_regularizer=False):
        super(CombinedLoss, self).__init__()
        # self.mse_loss = nn.MSELoss()
        self.berhu_loss = BerHuLoss()
        self.canny_reg_weight = canny_reg_weight
        self.use_canny_regularizer = use_canny_regularizer

    def forward(self, predictions, targets, canny_edges=None, device='cpu', current_iter=NUM_EPOCHS):
        reconstruction_loss = self.berhu_loss(predictions, targets)
        edge_loss = torch.tensor(0.0, device=predictions.device)
        total_loss = reconstruction_loss

        if self.use_canny_regularizer and canny_edges is not None and self.canny_reg_weight > 0:
            factor = min(1, current_iter / NUM_EPOCHS)  # Gradually increase factor from 0 to 1
            canny_edges_on_device = canny_edges.to(device) if isinstance(canny_edges, torch.Tensor) else canny_edges
            edge_loss = gradient_loss(predictions, canny_edges_on_device, device, factor)
            total_loss = reconstruction_loss + self.canny_reg_weight * edge_loss
        return total_loss, reconstruction_loss, edge_loss

# %% [markdown]
# # Training loop

# %%
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    sample_batch = None  # For storing a batch for visualization
    scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None # type: ignore

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        # Freeze encoder for first epoch only, unfreeze from second epoch onward.
        
        # Determine the actual model to access parameters, handling DataParallel case
        actual_model = model.module if hasattr(model, 'module') else model

        if epoch < FIXED_DEC_FOR_FIRST_N:
            print(f"Freezing both encoder and decoder, but not refine for epoch {epoch+1}.")
            for param in actual_model.midas_model.parameters():
                param.requires_grad = False
            for param in actual_model.refine.parameters():
                param.requires_grad = True
        elif epoch < FIXED_ENC_FOR_FIRST_N:
            print(f"Freezing encoder, but not decoder or refine for epoch {epoch+1}.")
            for param in actual_model.midas_model.pretrained.parameters():
                param.requires_grad = False
            # Unfreeze all decoder parameters in MiDaS model
            for param in actual_model.midas_model.scratch.parameters():
                param.requires_grad = True
            for param in actual_model.refine.parameters():
                param.requires_grad = True
        else:
            print(f"Unfreezing all parameters for epoch {epoch+1}.")
            for param in actual_model.midas_model.parameters():
                param.requires_grad = True
            for param in actual_model.refine.parameters():
                param.requires_grad = True

        # if epoch < FIXED_FOR_FIRST_N:
        #     # Ensure self.midas_model is the correct attribute name in MiDaSDepth
        #     for param in actual_model.midas_model.pretrained.parameters():
        #         param.requires_grad = False
        #     print(f"MiDaS backbone frozen for epoch {epoch+1}.")
        # elif epoch == FIXED_FOR_FIRST_N:
        #     for param in actual_model.midas_model.pretrained.parameters():
        #         param.requires_grad = True
        #     print(f"MiDaS backbone unfrozen from epoch {epoch+1}.")

        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params at epoch {epoch+1}: {num_trainable}")
        
        model.train()
        train_loss = 0.0
        train_recon_loss_epoch = 0.0
        train_edge_loss_epoch = 0.0 # Will remain 0
        for batch_idx, data_items in enumerate(tqdm(train_loader, desc="Training")):
            # Simplified data unpacking
            if USE_CANNY_REGULARIZER:
                inputs, targets, canny_edges, _ = data_items
            else:
                inputs, targets, _ = data_items
                canny_edges = None
            inputs, targets = inputs.to(device), targets.to(device) # targets are at INPUT_SIZE (426,560)
            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16): # type: ignore
                    outputs = model(inputs)
                    outputs = nn.functional.interpolate(outputs, size=targets.shape[-2:], mode='bilinear', align_corners=True)
                    loss, recon_loss, edge_loss = criterion(outputs, targets, canny_edges, device, epoch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                outputs = nn.functional.interpolate(outputs, size=targets.shape[-2:], mode='bilinear', align_corners=True)
                loss, recon_loss, edge_loss = criterion(outputs, targets, canny_edges, device, epoch)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_recon_loss_epoch += recon_loss.item() * inputs.size(0)
            train_edge_loss_epoch += edge_loss.item() * inputs.size(0)
            # Save the first batch for visualization
            if batch_idx == 0 and sample_batch is None:
                # Detach and move to cpu for visualization
                sample_batch = (inputs.detach().cpu(), targets.detach().cpu(), outputs.detach().cpu())
        train_loss /= len(train_loader.dataset)
        train_recon_loss_epoch /= len(train_loader.dataset)
        train_edge_loss_epoch /= len(train_loader.dataset) # Will be 0
        train_losses.append(train_loss)
        model.eval()
        val_loss = 0.0
        val_recon_loss_epoch = 0.0
        val_edge_loss_epoch = 0.0 # Will remain 0
        with torch.no_grad():
            for data_items in tqdm(val_loader, desc="Validation"):
                # Simplified data unpacking
                if USE_CANNY_REGULARIZER:
                    inputs, targets, canny_edges, _ = data_items
                else:
                    inputs, targets, _ = data_items
                    canny_edges = None
                inputs, targets = inputs.to(device), targets.to(device) # targets are at INPUT_SIZE (426,560)
                outputs = model(inputs) # outputs are at MiDaS internal resolution (e.g., 384,384)

                # Resize model outputs to match target size before loss calculation for validation
                outputs = nn.functional.interpolate(outputs, size=targets.shape[-2:], mode='bilinear', align_corners=True)
                
                loss, recon_loss, edge_loss = criterion(outputs, targets, canny_edges, device, epoch)
                val_loss += loss.item() * inputs.size(0)
                val_recon_loss_epoch += recon_loss.item() * inputs.size(0)
                val_edge_loss_epoch += edge_loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        val_recon_loss_epoch /= len(val_loader.dataset)
        val_edge_loss_epoch /= len(val_loader.dataset) 
        print(f"Train Loss: {train_loss:.4f} (Recon: {train_recon_loss_epoch:.4f}, Edge: {train_edge_loss_epoch:.4f}) | "
                  f"Val Loss: {val_loss:.4f} (Recon: {val_recon_loss_epoch:.4f}, Edge: {val_edge_loss_epoch:.4f})")

        if sample_batch is not None:
            input_np = sample_batch[0][0].permute(1, 2, 0).numpy()
            target_np = sample_batch[1][0].squeeze().numpy()
            output_np = sample_batch[2][0].squeeze().numpy()
            # If input has 4 channels, drop the last (Canny) for visualization - This is no longer needed as input is 3 channels
            # if input_np.shape[2] == 4: # Removed Canny
            #     input_np = input_np[:, :, :3] # Removed Canny
            input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min() + 1e-6) # Normalize for display
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
            ensure_dir(results_dir) # Ensure results_dir exists before saving
            plt.savefig(os.path.join(results_dir, f"epoch_sample_{epoch+1}.png"))
            plt.close()
            sample_batch = None  # Reset for next epoch
        # --- End sample image save ---
        if val_loss < best_val_loss: # Original logic kept for saving "best" model (currently saves every epoch due to True)
            best_val_loss = val_loss
            ensure_dir(results_dir) # Ensure results_dir exists before saving
            torch.save(model.state_dict(), os.path.join(results_dir, 'best_model.pth'))
            # print(f"New best (ANY!!!!!!!!!!!!!) model saved at epoch {epoch+1} with validation loss: {val_loss:.4f}")
            print(f"New best model saved at epoch {epoch+1} with validation loss: {val_loss:.4f}")
    print(f"\nBest model with validation loss: {best_val_loss:.4f}")
    ensure_dir(results_dir) # Ensure results_dir exists before loading
    model.load_state_dict(torch.load(os.path.join(results_dir, 'best_model.pth')))
    return model

# %% [markdown]
# # Model evaluation

def saveThisImage(input_np, target_np, output_np, idx, text='bad_sample'):
    ensure_dir(results_dir)  # Ensure results_dir exists before saving
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
    plt.savefig(os.path.join(results_dir, f"{text}_{idx}.png"))
    plt.close()

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
            # print (f"Outputs shape: {outputs.shape}, Targets shape: {targets.shape}")
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
                    continue # Skip if no valid target pixels
                target_valid = target_np[valid_target]
                pred_valid = pred_np[valid_target]
                log_target = np.log(target_valid)
                pred_valid = np.where(pred_valid > EPSILON, pred_valid, EPSILON) # Ensure pred_valid is positive for log
                log_pred = np.log(pred_valid)
                diff = log_pred - log_target
                diff_mean = np.mean(diff)
                sirmse += np.sqrt(np.mean((diff - diff_mean) ** 2))
                if np.sqrt(np.mean((diff - diff_mean) ** 2)) > .3:
                    print (f"Sample {total_samples}: sirmse_total={sirmse:.4f}, sirmse_sample={np.sqrt(np.mean((diff - diff_mean) ** 2)):.4f}")
                    saveThisImage(
                        inputs[i].cpu().permute(1, 2, 0).numpy(),
                        target_np,
                        pred_np,
                        total_samples - batch_size + i,
                        text='bad_sample')
            max_ratio = torch.max(outputs / (targets + 1e-6), targets / (outputs + 1e-6))
            delta1 += torch.sum(max_ratio < 1.25).item()
            delta2 += torch.sum(max_ratio < 1.25**2).item()
            delta3 += torch.sum(max_ratio < 1.25**3).item()
            # if total_samples <= 5 * batch_size: # This condition seems for debugging, can be kept or removed
            if total_samples <= 5 * batch_size:
                for i in range(min(batch_size, 5)):
                    idx = total_samples - batch_size + i
                    input_np = inputs[i].cpu().permute(1, 2, 0).numpy()
                    target_np = targets[i].cpu().squeeze().numpy()
                    output_np = outputs[i].cpu().squeeze().numpy()
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
                    plt.savefig(os.path.join(results_dir, f"sample_{idx}.png"))
                    plt.close()
            del inputs, targets, outputs, abs_diff, max_ratio

        torch.cuda.empty_cache()
    
    if total_samples == 0: # Avoid division by zero if val_loader is empty
        print("Warning: No samples found for evaluation.")
        return { 'MAE': 0, 'RMSE': 0, 'siRMSE': 0, 'REL': 0, 'Delta1': 0, 'Delta2': 0, 'Delta3': 0 }

    # Ensure target_shape is not None before using its dimensions
    if target_shape is None:
        print("Warning: target_shape not determined during evaluation (val_loader might be empty).")
        # Fallback or error, here returning zeros.
        return { 'MAE': 0, 'RMSE': 0, 'siRMSE': 0, 'REL': 0, 'Delta1': 0, 'Delta2': 0, 'Delta3': 0 }

    total_pixels_per_sample = target_shape[1] * target_shape[2] * target_shape[3] # C * H * W (depth is 1 channel)
    
    mae /= (total_samples * total_pixels_per_sample)
    rmse = np.sqrt(rmse / (total_samples * total_pixels_per_sample))
    rel /= (total_samples * total_pixels_per_sample)
    sirmse = sirmse / total_samples # sirmse is already averaged per sample in the loop
    delta1 /= (total_samples * total_pixels_per_sample)
    delta2 /= (total_samples * total_pixels_per_sample)
    delta3 /= (total_samples * total_pixels_per_sample)
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
        for inputs, filenames in tqdm(test_loader, desc="Generating Test Predictions"): # Unpack directly
            inputs = inputs.to(device)
            outputs = model(inputs) # MiDaS output (e.g. 384x384)
            
            # Resize to original INPUT_SIZE for saving, if necessary, or save at MiDaS resolution
            # The problem usually expects predictions at a certain resolution.
            # Assuming INPUT_SIZE is the desired output resolution for predictions.
            outputs = nn.functional.interpolate(outputs, size=INPUT_SIZE, mode='bilinear', align_corners=True)

            for i in range(outputs.size(0)):
                filename = filenames[i].split(' ')[1]
                # Prediction filename format might need adjustment based on competition/task requirements
                # e.g., test_000000_depth.npy or similar
                # base_name = os.path.splitext(os.path.basename(filename))[0]
                pred_path = os.path.join(predictions_dir, filename)
                
                depth_pred_np = outputs[i].squeeze().cpu().numpy()
                # np.save(pred_path, depth_pred_np)
                np.save(pred_path, depth_pred_np)
        print(f"Test predictions saved to {predictions_dir}")
        

# %% [markdown]
# # Putting it all together

# %%
def target_transform(depth):
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(0).unsqueeze(0),
        size=INPUT_SIZE,
        mode='bilinear',
        align_corners=True
    ).squeeze()
    depth = depth.unsqueeze(0)
    return depth

def main():

    

    # def target_transform(depth):
    #     # depth is a PIL Image or tensor, convert to PIL if needed
    #     if isinstance(depth, torch.Tensor):
    #         depth = transforms.ToPILImage()(depth)
        
    #     depth = shared_augmentations(depth)  # apply augmentations
        
    #     depth = transforms.ToTensor()(depth)  # convert back to tensor (C=1, H, W)
        
    #     return depth

    

    # Call load_params() if you want to use command-line arguments
    # If not, hyperparameters from the cell above will be used.
    # load_params() # Uncomment to use command-line args

    # Ensure directories exist (results_dir and predictions_dir are now defined based on output_dir)
    ensure_dir(output_dir) # Main output directory
    ensure_dir(results_dir)
    ensure_dir(predictions_dir)

    # 1. Shared augmentations (before Canny or conversion)
    shared_augmentations = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.Resize((int(INPUT_SIZE[0] * 1.2), int(INPUT_SIZE[1] * 1.2))),  # upscale
        # transforms.RandomRotation(degrees=5, expand=False, fill=0),
        # transforms.CenterCrop(INPUT_SIZE),
    ])

    # 2. Final processing
    if USE_CANNY_REGULARIZER:
        image_to_tensor = ProcessImageAndAddCanny(canny_low_thresh=30, canny_high_thresh=100)
        edge_extractor = GenerateCannyEdges(canny_low_thresh=30, canny_high_thresh=100)
    else:
        image_to_tensor = transforms.ToTensor()
        edge_extractor = lambda x: torch.zeros(1, *INPUT_SIZE)  # dummy edge map if unused

    # 3. Normalization (4 channels if Canny used)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406, 0.5] if USE_CANNY_REGULARIZER else [0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225, 0.5] if USE_CANNY_REGULARIZER else [0.229, 0.224, 0.225],
    )


    # -- Final transforms --
    train_rgb_transform = transforms.Compose([
        shared_augmentations,
        image_to_tensor,
        normalize
    ])

    test_rgb_transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        image_to_tensor,
        normalize
    ])

    canny_generator_transform = transforms.Compose([
        shared_augmentations,
        edge_extractor
    ]) if USE_CANNY_REGULARIZER else None


    # train_rgb_transform = transforms.Compose([
    #     transforms.Resize(INPUT_SIZE),
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.Resize((int(INPUT_SIZE[0] * 1.2), int(INPUT_SIZE[1] * 1.2))),  # scale up
    #     transforms.RandomRotation(degrees=5, expand=False, fill=0),  # rotate
    #     transforms.CenterCrop(INPUT_SIZE),  # crop back to original
    #     ProcessImageAndAddCanny(canny_low_thresh=30, canny_high_thresh=100) if USE_CANNY_REGULARIZER else transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5],
    #                          std=[0.229, 0.224, 0.225, 0.5])
        
    # ])
    # test_rgb_transform = transforms.Compose([
    #     transforms.Resize(INPUT_SIZE),
    #     ProcessImageAndAddCanny(canny_low_thresh=30, canny_high_thresh=100) if USE_CANNY_REGULARIZER else transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5],
    #                          std=[0.229, 0.224, 0.225, 0.5])
    # ])
    # canny_generator_transform = None
    # if USE_CANNY_REGULARIZER:
    #     canny_generator_transform = transforms.Compose([
    #         transforms.Resize(INPUT_SIZE),
    #         GenerateCannyEdges(canny_low_thresh=30, canny_high_thresh=100)
    #     ])
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
        persistent_workers=PERSISTENT_WORKERS
    )
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
    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
        print(f"Initially allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
    model = MiDaSDepth()  # Initialize the MiDaS model
    model = nn.DataParallel(model)
    model = model.to(DEVICE)
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"Memory allocated after model init: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
    criterion = CombinedLoss(CANNY_REG_WEIGHT, USE_CANNY_REGULARIZER).to(DEVICE)  # Use the combined loss function
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    print("Starting training...")
    model = train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, DEVICE)
    print("Evaluating model on validation set...")
    metrics = evaluate_model(model, val_loader, DEVICE)
    print("\nValidation Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    with open(os.path.join(results_dir, 'validation_metrics.txt'), 'w') as f:
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\n")
    print("Generating predictions for test set...")
    generate_test_predictions(model, test_loader, DEVICE)
    print(f"Results saved to {results_dir}")
    print(f"All test depth map predictions saved to {predictions_dir}")

if __name__ == '__main__':
    main()

# %%



