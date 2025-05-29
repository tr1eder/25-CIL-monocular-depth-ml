import torch
import torchvision.transforms as T
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def siRMSE(pred, target, idx, text):
    """
    Computes the scaled inverse Root Mean Square Error (siRMSE) between predicted and target depth maps.
    
    Args:
        pred (np.ndarray): Predicted depth map.
        target (np.ndarray): Ground truth depth map.
        idx (int): Index for logging or display purposes.
        text (str): Text label for the output.
    
    Returns:
        float: The siRMSE value.
    """
    # pred = pred.astype(np.float32)
    # target = target.astype(np.float32)

    # Avoid division by zero
    target[target == 0] = 1e-6

    rmse = np.sqrt(np.mean((pred - target) ** 2))
    si_rmse = rmse / np.mean(target)

    print(f"siRMSE for {text} at index {idx}: {si_rmse:.4f}")
    return si_rmse

# Load MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")  # or "DPT_Large"
midas.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

# Load transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

# For DPT models
transform = midas_transforms.dpt_transform

# Load and resize image to 426x560
img = Image.open("train/train/sample_000000_rgb.png").convert("RGB")
gt  = (np.load("train/train/sample_000000_depth.npy").astype(np.float32))
# img_resized = img.resize((560, 426))  # (width, height)

# Transform input
img_np = np.array(img) # Convert PIL image to NumPy array
input_tensor = transform(img_np).to(device)

# Convert to tensor before transform
# img_tensor = torch.from_numpy(np.array(img_resized)).float() / 255.0  # shape: (H, W, 3)
# img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # shape: (1, 3, H, W)

# input_tensor = img_tensor.to(device)

# Inference
with torch.no_grad():
    prediction = midas(input_tensor)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=(426, 560),  # match original
        mode="bicubic",
        align_corners=False,
    ).squeeze()

# Normalize for visualization
depth_map = prediction.cpu().numpy()
# depth = depth_map.copy()
depth_map = depth_map - depth_map.min()  # Normalize to [0, 1]
depth_map = depth_map / (depth_map.max() + 1e-6)  # Avoid division by zero
depth_map = 1 / (depth_map + .14)  # Inverse depth for visualization
print (f"Depth map shape: {depth_map.shape}")
print (f"Depth map min: {depth_map.min()}, max: {depth_map.max()}")
# depth_map -= depth_map.min()
# depth_map /= depth_map.max()

siRMSE(depth_map, gt, 0, "sample_000000")

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Original image
axs[0].imshow(img)
axs[0].set_title("Original Image")
axs[0].axis("off")

# Predicted depth map
axs[2].imshow(depth_map, cmap="plasma")
axs[2].set_title("Predicted Depth")
axs[2].axis("off")

# Ground truth depth map (normalize for visualization)
gt_np = gt.cpu().numpy() if torch.is_tensor(gt) else gt
# gt_vis = gt_np - gt_np.min()
# gt_vis = gt_vis / gt_vis.max() if gt_vis.max() > 0 else gt_vis
axs[1].imshow(gt_np, cmap="plasma")
axs[1].set_title("Ground Truth Depth")
axs[1].axis("off")

plt.tight_layout()
plt.savefig("midas_depth_prediction.png")
# plt.show()
