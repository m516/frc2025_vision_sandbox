import cv2
import numpy as np

def adaptive_normalize(image, kernel_size=15, sigma=3.0):
    """
    Adaptively normalize an image by dividing each pixel by the 
    Gaussian-weighted local average of its neighborhood.
    
    :param image: Input image (NumPy array). Can be grayscale or color.
    :param kernel_size: Size of the Gaussian kernel (must be odd).
    :param sigma: Standard deviation for the Gaussian kernel.
    :return: Adaptively normalized image (NumPy array, uint8).
    """
    
    # Ensure image is in float format for division operations
    image_float = image.astype(np.float32)
    
    # If image has multiple channels (e.g., BGR color),
    # apply the operation to each channel separately.
    if len(image_float.shape) == 3:
        # Split channels
        channels = cv2.split(image_float)
        normalized_channels = []
        
        for ch in channels:
            # Gaussian blur for local average
            blurred = cv2.GaussianBlur(ch, (kernel_size, kernel_size), sigma)
            # Add a small epsilon to avoid division by zero
            eps = 1e-8
            # Perform per-pixel division
            norm_ch = ch / (blurred + eps)
            normalized_channels.append(norm_ch)
        
        # Merge channels back
        normalized = cv2.merge(normalized_channels)
    
    else:
        # Single-channel (grayscale) case
        blurred = cv2.GaussianBlur(image_float, (kernel_size, kernel_size), sigma)
        eps = 1e-8
        normalized = image_float / (blurred + eps)
    
    # Optionally rescale to [0, 255]
    normalized = cv2.normalize(normalized, None, alpha=0, beta=255, 
                               norm_type=cv2.NORM_MINMAX)
    
    return normalized.astype(np.uint8)

# Example of usage
if __name__ == "__main__":
    img = cv2.imread("renders/0001.jpg", cv2.IMREAD_COLOR)
    result = adaptive_normalize(img, kernel_size=101, sigma=3.0)
    cv2.imshow("image", result)
    cv2.waitKey(5000)