import cv2
import numpy as np
from PIL import Image  # added to support PIL image conversion
from time import time
from tqdm import tqdm


def canny_edge_detector1(image, low_threshold, high_threshold):
    """
    Applies the Canny edge detection algorithm to an image.
    This is a custom implementation THAT IS MISSING SOME STEPS (e.g., non-maximum suppression).

    Parameters:
    - image: Input image (grayscale).
    - low_threshold: Lower threshold for hysteresis.
    - high_threshold: Upper threshold for hysteresis.

    Returns:
    - edges: Binary image with detected edges.
    """

    # Step 1: Gaussian Blur
    blurred = cv2.GaussianBlur(image, (5, 5), 1.4)

    # Step 2: Gradient Calculation
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    # Step 3: Non-maximum Suppression
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x) * (180 / np.pi)
    
    # Step 4: Hysteresis Thresholding
    strong_edges = (magnitude > high_threshold)
    weak_edges = ((magnitude >= low_threshold) & (magnitude <= high_threshold))
    
    edges = np.zeros_like(magnitude)
    edges[strong_edges] = 255
    edges[weak_edges] = 75
    
    return edges

def canny_edge_detector2(image, low_threshold, high_threshold):
    """
    Applies a Canny-like edge detection algorithm to an image based on provided logic.
    This is a custom implementation THAT STILL CONTAINS ERRORS.

    Parameters:
    - image: Input image (grayscale).
    - low_threshold: Lower threshold for hysteresis.
    - high_threshold: Upper threshold for hysteresis.

    Returns:
    - edges: Binary image with detected edges (0 or 255).
    """
    if image is None:
        raise ValueError("Input image is None.")
    if image.ndim != 2:
        raise ValueError("Input image must be grayscale.")

    # Step 1: Gaussian Blur (parameters from pseudo-code: G = gaussian_filter(9,2))
    # OpenCV's GaussianBlur ksize must be odd. (9,9) is fine. Sigma is 2.
    blurred = cv2.GaussianBlur(image, (9, 9), 2)

    # Step 2: Gradient Calculation using Sobel
    # cv2.CV_64F for high precision gradients
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # Step 3: Gradient Magnitude and Direction
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x) # Angle in radians, range [-pi, pi]

    # Step 4: Non-maximum Suppression
    # Define thinning kernels from pseudo-code
    thinning_h1 = np.array([[0,-1,0],[0,1,0],[0,0,0]], dtype=np.float64) # Compare with North
    thinning_h2 = np.array([[0,0,0],[0,1,0],[0,-1,0]], dtype=np.float64) # Compare with South

    thinning_pd1 = np.array([[-1,0,0],[0,1,0],[0,0,0]], dtype=np.float64) # Compare with NW
    thinning_pd2 = np.array([[0,0,0],[0,1,0],[0,0,-1]], dtype=np.float64) # Compare with SE

    thinning_v1 = np.array([[0,0,0],[-1,1,0],[0,0,0]], dtype=np.float64) # Compare with West
    thinning_v2 = np.array([[0,0,0],[0,1,-1],[0,0,0]], dtype=np.float64) # Compare with East

    thinning_sd1 = np.array([[0,0,-1],[0,1,0],[0,0,0]], dtype=np.float64) # Compare with NE
    thinning_sd2 = np.array([[0,0,0],[0,1,0],[-1,0,0]], dtype=np.float64) # Compare with SW

    # Quantize angles to 4 directions (0: H, 1: D/, 2: V, 3: D\)
    # Angle sectors in degrees for [-180, 180]:
    # H: [-22.5, 22.5] U [157.5, 180] U [-180, -157.5]
    # D/: (22.5, 67.5] U (-157.5, -112.5]
    # V: (67.5, 112.5] U (-112.5, -67.5]
    # D\: (112.5, 157.5] U (-67.5, -22.5]
    
    angle_deg = angle * 180.0 / np.pi
    
    # Create masks for each direction
    # Horizontal edge (gradient is vertical) -> compare N/S
    filter_v_mask = (((angle_deg >= -22.5) & (angle_deg <= 22.5)) | \
                     ((angle_deg > 157.5) | (angle_deg < -157.5)))
    
    # Primary Diagonal edge / (gradient is NW/SE) -> compare NW/SE
    filter_pd_mask = (((angle_deg > 22.5) & (angle_deg <= 67.5)) | \
                      ((angle_deg < -112.5) & (angle_deg >= -157.5)))

    # Vertical edge (gradient is horizontal) -> compare E/W
    filter_h_mask = (((angle_deg > 67.5) & (angle_deg <= 112.5)) | \
                     ((angle_deg < -67.5) & (angle_deg >= -112.5)))

    # Secondary Diagonal edge \ (gradient is NE/SW) -> compare NE/SW
    filter_sd_mask = (((angle_deg > 112.5) & (angle_deg <= 157.5)) | \
                      ((angle_deg < -22.5) & (angle_deg >= -67.5)))

    # Perform thinning based on direction
    # convolve(mag, thinning_kernel) > 0 means mag_center > mag_neighbor
    is_max_v = (cv2.filter2D(magnitude, -1, thinning_h1, borderType=cv2.BORDER_REPLICATE) > 0) & \
               (cv2.filter2D(magnitude, -1, thinning_h2, borderType=cv2.BORDER_REPLICATE) > 0)
    
    is_max_pd = (cv2.filter2D(magnitude, -1, thinning_pd1, borderType=cv2.BORDER_REPLICATE) > 0) & \
                (cv2.filter2D(magnitude, -1, thinning_pd2, borderType=cv2.BORDER_REPLICATE) > 0)

    is_max_h = (cv2.filter2D(magnitude, -1, thinning_v1, borderType=cv2.BORDER_REPLICATE) > 0) & \
               (cv2.filter2D(magnitude, -1, thinning_v2, borderType=cv2.BORDER_REPLICATE) > 0)

    is_max_sd = (cv2.filter2D(magnitude, -1, thinning_sd1, borderType=cv2.BORDER_REPLICATE) > 0) & \
                (cv2.filter2D(magnitude, -1, thinning_sd2, borderType=cv2.BORDER_REPLICATE) > 0)

    nonmax_suppressed_magnitude = np.zeros_like(magnitude)
    
    nonmax_suppressed_magnitude[filter_v_mask & is_max_v] = magnitude[filter_v_mask & is_max_v]
    nonmax_suppressed_magnitude[filter_pd_mask & is_max_pd] = magnitude[filter_pd_mask & is_max_pd]
    nonmax_suppressed_magnitude[filter_h_mask & is_max_h] = magnitude[filter_h_mask & is_max_h]
    nonmax_suppressed_magnitude[filter_sd_mask & is_max_sd] = magnitude[filter_sd_mask & is_max_sd]

    # Step 5: Hysteresis Thresholding
    strong_mask = (nonmax_suppressed_magnitude >= high_threshold)
    weak_mask = (nonmax_suppressed_magnitude >= low_threshold) & (nonmax_suppressed_magnitude < high_threshold)

    # Iterative edge linking
    # final_edges_mask starts with strong edges
    final_edges_mask = strong_mask.copy()
    
    # Kernel for checking 8-connectivity (from pseudo-code: expand_high)
    connectivity_kernel = np.ones((3,3), dtype=np.uint8)

    while True:
        prev_final_edges_mask = final_edges_mask.copy()
        # Dilate current strong edges to find connected regions
        dilated_strong_edges = cv2.dilate(final_edges_mask.astype(np.uint8), connectivity_kernel).astype(bool)
        
        # Promote weak edges that are connected to strong edges and not already in final_edges_mask
        promotable_weak_edges = dilated_strong_edges & weak_mask & (~final_edges_mask)
        
        if not np.any(promotable_weak_edges): # No new edges were promoted
            break
            
        final_edges_mask = final_edges_mask | promotable_weak_edges

    edges = final_edges_mask.astype(np.uint8) * 255
    return edges

def canny_edge_detector3(image, low_threshold, high_threshold):
    """
    Applies the Canny edge detection algorithm to an image using OpenCV's implementation.
    Accepts a PIL RGB image or a NumPy grayscale image.
    
    Parameters:
    - image: Input image (PIL RGB or NumPy grayscale).
    - low_threshold: Lower threshold for hysteresis (used as threshold1 in cv2.Canny).
    - high_threshold: Upper threshold for hysteresis (used as threshold2 in cv2.Canny).

    Returns:
    - edges: Binary image with detected edges (0 or 255) as a NumPy array.
    """
    # If input is a PIL image, convert it to grayscale NumPy array
    if isinstance(image, Image.Image):
        image = np.array(image.convert('L'))
    
    if image is None:
        raise ValueError("Input image is None.")
    if image.ndim != 2:
        raise ValueError("Input image must be grayscale after conversion.")

    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            if np.max(image) <= 1.0 and np.min(image) >= 0.0:
                image = (image * 255).astype(np.uint8)
            else:
                normalized_image = np.zeros_like(image, dtype=np.float32)
                cv2.normalize(image, normalized_image, 0, 255, cv2.NORM_MINMAX)
                image = normalized_image.astype(np.uint8)
        elif image.dtype == np.uint16:
             image = (image / 256).astype(np.uint8)
        else:
            try:
                image = image.astype(np.uint8)
            except ValueError as e:
                raise ValueError(f"Image dtype {image.dtype} cannot be directly converted to uint8 for Canny. Please preprocess. Error: {e}")

    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, low_threshold, high_threshold)
    
    return edges

canny_edge_detector = canny_edge_detector3

def preprocess_images(image_list_input, image_list_output, low_threshold, high_threshold):
    """
    Preprocesses images by applying Canny edge detection and saving the results.

    Parameters:
    - image_list_input: list of input image paths.
    - image_list_output: list of output image paths.
    - low_threshold: lower threshold for Canny edge detection.
    - high_threshold: upper threshold for Canny edge detection.
    """

    # print (f"Images are {image_list_input} to {image_list_output}")

    for i, (input_path, output_path) in enumerate(tqdm(zip(image_list_input, image_list_output), total=len(image_list_input), desc="Processing images")):
        # print(f"Processing image {i+1}/{len(image_list_input)}: {input_path} -> {output_path}")
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error reading image: {input_path}")
            continue

        edges = canny_edge_detector(image, low_threshold, high_threshold)
        cv2.imwrite(output_path, edges)

    print (f"Successfully processed {len(image_list_input)} images.")

def preprocess_images_from_file(filename, prefix, n=100000, low_threshold=30, high_threshold=100):
    """
    Preprocesses images listed in a file by applying Canny edge detection and saving the results.
    Parameters:
    - filename: Path to the file containing image paths.
    - prefix: Prefix for the image paths.
    - n: Number of images to process (default: 100000).
    - low_threshold: Lower threshold for Canny edge detection.
    - high_threshold: Upper threshold for Canny edge detection.
    """
    first_n = lambda folder, lines, ext='', n=n: [f"{folder}/{''.join(x.split()[0].split('.')[:-1])}{ext}.{x.split()[0].split('.')[-1]}" for x in lines[:n]]
    with open(filename, 'r') as f:
        images = f.readlines()
        image_list_input = first_n(prefix, images)
        image_list_output = first_n(prefix, images, ext='_canny')
        preprocess_images(image_list_input, image_list_output, low_threshold, high_threshold)


def display_image(image, edges):
    # Convert the grayscale image to BGR for color overlay
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    edges_uint8 = edges.astype(np.uint8)

    # Create an overlay by copying the original image
    overlay = image_bgr.copy()

    # Create a mask where any edge (weak or strong) is detected
    mask = edges_uint8 > 0

    # Set those pixels to red (BGR: [0, 0, 255])
    # overlay[mask] = [0, 0, 255]
    overlay[mask] = [0, 0, 0]
    overlay[mask, 2] = edges_uint8[mask]
    height, width = overlay.shape[:2]

    # Create a resizable window and display the overlaid image proportionally
    cv2.namedWindow('Canny Edge Detector', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Canny Edge Detector', width, height)
    cv2.imshow('Canny Edge Detector', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == "__main__":

    preprocess_images_from_file('train_list.txt', 'train/train')
    preprocess_images_from_file('test_list.txt', 'test/test')
    



    # image = cv2.imread('train/train/sample_000000_rgb.png', cv2.IMREAD_GRAYSCALE)

    # image = cv2.imread('output/results-b-c-600-8/sample_14.png', cv2.IMREAD_GRAYSCALE)
    # edges = canny_edge_detector(image, 30, 100)
    # display_image(image, edges)

    # image = cv2.imread('train/train/sample_000000_rgb.png', cv2.IMREAD_GRAYSCALE)
    # edges = canny_edge_detector(image, 30, 100)
    # time_start = time()
    # with open('train_list.txt', 'r') as f:
    #     lines = f.readlines()[0:600]  # Read the first 600 lines
    #     for line in lines:
    #         image = cv2.imread(f'train/train/{line.split()[0]}', cv2.IMREAD_GRAYSCALE)
    #         edges = canny_edge_detector(image, 30, 100)
    # time_end = time()
    # print(f"Time taken for 600 images: {time_end - time_start:.2f} seconds")

    