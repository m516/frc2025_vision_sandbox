import numpy as np
import cv2
import timeit


from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap

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
    
    return normalized

i = cv2.imread("renders/0001.jpg")

@timing
def process(i):
    i = adaptive_normalize(i, kernel_size=31, sigma=11.0)
    i = i.astype(np.float32)
    i = i/256
    gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)

    mean_vals = cv2.mean(i)[:3]  # Extract the first 3 values (ignoring alpha if present)
    mean_vals = np.array(mean_vals, dtype=np.float32).reshape(1, 1, 3)
    a = i / mean_vals

    r=a[:,:,2]; g=a[:,:,1]; b=a[:,:,0]

    a = r + b - 2*g-abs(r-b)
    a = cv2.normalize(a, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    a = cv2.max(0, a-0.5)*2
    return a


cv2.imshow("img", process(i))
cv2.waitKey(5000)