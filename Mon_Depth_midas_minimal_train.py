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

def load_params():
    parser = argparse.ArgumentParser(description='Monocular Depth Estimation with MiDaS (Minimal Version)')

    # parser.add_argument('-env', '--env', type=str, default='local', help='Environment: local or studentcluster', required=False)
    parser.add_argument('-n', '--train_first_n', type=int, help='Number of training samples to use (0 for all)', required=False)
    parser.add_argument('-e', '--epochs', type=int, help='number of epochs to train', required=False)
    parser.add_argument('-lr', '--learning_rate', type=float, help='Learning rate', required=False)
    parser.add_argument('-fff', '--fixed_for_first_n', type=int, help='Number of epochs to freeze the encoder', required=False)
    
    args = parser.parse_args()

    global TRAIN_FIRST_N, NUM_EPOCHS, LEARNING_RATE, FIXED_FOR_FIRST_N
    global output_dir 

    if args.train_first_n is not None:
        TRAIN_FIRST_N = args.train_first_n
    if args.epochs is not None:
        NUM_EPOCHS = args.epochs
    if args.learning_rate is not None:
        LEARNING_RATE = args.learning_rate
    if args.fixed_for_first_n is not None:
        FIXED_FOR_FIRST_N = args.fixed_for_first_n

    print(f"Training with {TRAIN_FIRST_N} samples, {NUM_EPOCHS} epochs, learning rate {LEARNING_RATE}, fixed for first {FIXED_FOR_FIRST_N} epochs")

    output_dir_name = f"midas_minimal_n{TRAIN_FIRST_N}_e{NUM_EPOCHS}_lr{LEARNING_RATE}_fff{FIXED_FOR_FIRST_N}"
    output_dir = os.path.join(os.getcwd(), output_dir_name)


# %%
class ENV(Enum):
    STUDENTCLUSTER = 1
    LOCAL = 2

RUN_ON = ENV.LOCAL  # or RUNON.STUDENTCLUSTER, depending on your environment
data_dir = os.getcwd() if RUN_ON == ENV.LOCAL else '/work/scratch/timrieder/cil_monocular_depth'
# output_dir is set by load_params or hyperparameter cell
# Ensure output_dir is initialized if load_params is not called before this cell
if 'output_dir' not in globals():
    output_dir = os.path.join(os.getcwd(), "output_default_minimal") # Default if load_params not run

train_dir = os.path.join(data_dir, 'train/train')
test_dir = os.path.join(data_dir, 'test/test')
train_list_file = os.path.join(data_dir, 'train_list.txt')
test_list_file = os.path.join(data_dir, 'test_list.txt')
results_dir = os.path.join(output_dir, 'results') # results_dir will be inside the specific output_dir
predictions_dir = os.path.join(output_dir, 'predictions') # predictions_dir will be inside the specific output_dir


# %% [markdown]
# ### Hyperparameters
TRAIN_FIRST_N = 10000  # Set to 0 to use all data, or set to a positive integer to limit the number of samples
NUM_EPOCHS = 2
LEARNING_RATE = 1e-4 * .3
FIXED_FOR_FIRST_N = 1 # Set nr. of epochs to freeze the encoder
BATCH_SIZE = 4
NUM_WORKERS = 4

WEIGHT_DECAY = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_SIZE = (426, 560)  # Adjusted to user's requirement
PERSISTENT_WORKERS = True if NUM_WORKERS > 0 else False
PIN_MEMORY = True

# Update output_dir based on hyperparameters if load_params() is not the primary way of setting them
# This ensures results_dir and predictions_dir are correctly formed if this cell defines the params
output_dir_name_hyperparam = f"midas_minimal_n{TRAIN_FIRST_N}_e{NUM_EPOCHS}_lr{LEARNING_RATE}_fff{FIXED_FOR_FIRST_N}"
output_dir = os.path.join(os.getcwd(), output_dir_name_hyperparam)
results_dir = os.path.join(output_dir, 'results')
predictions_dir = os.path.join(output_dir, 'predictions')


# %%
print (f"Using device: {DEVICE}")
print (f"Output directory: {output_dir}")
print (f"Results directory: {results_dir}")
print (f"Predictions directory: {predictions_dir}")

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
    def __init__(self, data_dir, list_file, transform=None, target_transform=None, has_gt=True):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.has_gt = has_gt
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
            # print (f"Info on self.transform: {self.transform}")
            
            rgb_tensor_transformed = self.transform(pil_rgb_image) if self.transform else transforms.ToTensor()(pil_rgb_image)
            if self.target_transform:
                depth = self.target_transform(depth)
            else:
                depth = depth.unsqueeze(0)
            return rgb_tensor_transformed, depth, self.file_pairs[idx][0]
        else:
            rgb_path = os.path.join(self.data_dir, self.file_list[idx].split(' ')[0])
            pil_rgb_image = Image.open(rgb_path).convert('RGB')
            rgb_tensor_transformed = self.transform(pil_rgb_image) if self.transform else transforms.ToTensor()(pil_rgb_image)
            return rgb_tensor_transformed, self.file_list[idx]

# %% [markdown]
# # Model - MiDaS Minimal

# %%

class MiDaSDepth(nn.Module):
    def __init__(self):
        super(MiDaSDepth, self).__init__()
        # Load MiDaS model - "MiDaS" for DPT-Hybrid (large), or "MiDaS_small" for a smaller version
        self.midas_model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid") # Using MiDaS_small for potentially faster/lighter version
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.midas_transform = midas_transforms.dpt_transform # type: ignore
        self.to_pil = transforms.ToPILImage()  # Convert tensor to PIL image for MiDaS transform


    def forward(self, x: torch.Tensor):
        # x: (B, 3, H, W), values in [0, 1], normalized to ImageNet stats
        B, C, H, W = x.shape

        # Resize
        x_resized = nn.functional.interpolate(x, size=(384, 384), mode='bilinear', align_corners=False)

        # Run MiDaS
        depth = self.midas_model(x_resized) # type: ignore

        # Upsample back to original size
        depth_resized = nn.functional.interpolate(
            depth.unsqueeze(1), size=(H, W), mode='bicubic', align_corners=False
        )

        # depth = 1 / (depth_resized + 1e-6)  # Invert depth to get distance

        depth = depth_resized
        depth = depth - depth.min()  # Normalize to [0, 1]
        depth = depth / (depth.max() + 1e-6)  # Avoid division by zero
        depth = 1 / (depth + .14)  # Inverse depth for visualization

        return depth

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions, targets):
        reconstruction_loss = self.mse_loss(predictions, targets)
        total_loss = reconstruction_loss
        edge_loss_val = torch.tensor(0.0, device=predictions.device) 
        return total_loss, reconstruction_loss, edge_loss_val

# %% [markdown]
# # Training loop

# %%
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    sample_batch = None  # For storing a batch for visualization
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        # Freeze encoder for first epoch only, unfreeze from second epoch onward.
        
        # Determine the actual model to access parameters, handling DataParallel case
        actual_model = model.module if hasattr(model, 'module') else model

        if epoch < FIXED_FOR_FIRST_N:
            # Ensure self.midas_model is the correct attribute name in MiDaSDepth
            for param in actual_model.midas_model.pretrained.parameters():
                param.requires_grad = False
            print(f"MiDaS backbone frozen for epoch {epoch+1}.")
        elif epoch == FIXED_FOR_FIRST_N:
            for param in actual_model.midas_model.pretrained.parameters():
                param.requires_grad = True
            print(f"MiDaS backbone unfrozen from epoch {epoch+1}.")

        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params at epoch {epoch}: {num_trainable}")
        
        model.train()
        train_loss = 0.0
        train_recon_loss_epoch = 0.0
        train_edge_loss_epoch = 0.0 # Will remain 0
        for batch_idx, data_items in enumerate(tqdm(train_loader, desc="Training")):
            # Simplified data unpacking
            inputs, targets, _ = data_items # _ was filenames
            inputs, targets = inputs.to(device), targets.to(device) # targets are at INPUT_SIZE (426,560)
            optimizer.zero_grad()
            outputs = model(inputs) # outputs are at MiDaS internal resolution (e.g., 384,384)
            
            # Resize model outputs to match target size before loss calculation
            outputs = nn.functional.interpolate(outputs, size=targets.shape[-2:], mode='bilinear', align_corners=True)
            
            loss, recon_loss, edge_loss = criterion(outputs, targets) # Removed canny_edges, device, epoch
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_recon_loss_epoch += recon_loss.item() * inputs.size(0)
            # if use_canny_regularizer_flag: # Removed Canny
            train_edge_loss_epoch += edge_loss.item() * inputs.size(0) # edge_loss is 0
            # Save the first batch for visualization
            if batch_idx == 0 and sample_batch is None:
                # Detach and move to cpu for visualization
                sample_batch = (inputs.detach().cpu(), targets.detach().cpu(), outputs.detach().cpu())
        train_loss /= len(train_loader.dataset)
        train_recon_loss_epoch /= len(train_loader.dataset)
        # if use_canny_regularizer_flag: # Removed Canny
        train_edge_loss_epoch /= len(train_loader.dataset) # Will be 0
        train_losses.append(train_loss)
        model.eval()
        val_loss = 0.0
        val_recon_loss_epoch = 0.0
        val_edge_loss_epoch = 0.0 # Will remain 0
        with torch.no_grad():
            for data_items in tqdm(val_loader, desc="Validation"):
                # Simplified data unpacking
                inputs, targets, _ = data_items # _ was filenames
                inputs, targets = inputs.to(device), targets.to(device) # targets are at INPUT_SIZE (426,560)
                outputs = model(inputs) # outputs are at MiDaS internal resolution (e.g., 384,384)

                # Resize model outputs to match target size before loss calculation for validation
                outputs = nn.functional.interpolate(outputs, size=targets.shape[-2:], mode='bilinear', align_corners=True)
                
                loss, recon_loss, edge_loss = criterion(outputs, targets) # Removed canny_edges, device, epoch
                val_loss += loss.item() * inputs.size(0)
                val_recon_loss_epoch += recon_loss.item() * inputs.size(0)
                # if use_canny_regularizer_flag: # Removed Canny
                val_edge_loss_epoch += edge_loss.item() * inputs.size(0) # edge_loss is 0
        val_loss /= len(val_loader.dataset)
        val_recon_loss_epoch /= len(val_loader.dataset)
        # if use_canny_regularizer_flag: # Removed Canny
        # Always print in the simplified format
        print(f"Train Loss: {train_loss:.4f} (Recon: {train_recon_loss_epoch:.4f}) | "
              f"Val Loss: {val_loss:.4f} (Recon: {val_recon_loss_epoch:.4f})")
        # else: # Removed Canny
            # print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}") # Removed Canny
        # --- Save 1 sample image per epoch ---
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
            # If canny regularizer is enabled, the dataset returns 4 values. # This condition is now always false.
            # if len(data) == 4: # Removed Canny
            #     inputs, targets, _, filenames = data # Removed Canny
            # else: # Removed Canny
            inputs, targets, filenames = data # Simplified data unpacking
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
def main():
    # Call load_params() if you want to use command-line arguments
    # If not, hyperparameters from the cell above will be used.
    # load_params() # Uncomment to use command-line args

    # Ensure directories exist (results_dir and predictions_dir are now defined based on output_dir)
    ensure_dir(output_dir) # Main output directory
    ensure_dir(results_dir)
    ensure_dir(predictions_dir)

    train_rgb_transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    ])
    test_rgb_transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
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
        # canny_transform=canny_generator_transform,
        has_gt=True,
        # use_canny_regularizer=USE_CANNY_REGULARIZER
    )
    test_dataset = DepthDataset(
        data_dir=test_dir,
        list_file=test_list_file,
        transform=test_rgb_transform,
        has_gt=False,
        # use_canny_regularizer=False
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
    criterion = CombinedLoss()  # Use the combined loss function
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



