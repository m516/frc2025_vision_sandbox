import numpy as np
import cv2
from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % 
              (f.__name__, args, kw, te - ts))
        return result
    return wrap

@timing
def detect_coral(image):
    image = image.astype(np.float32) / 255.0
    blurred = cv2.boxFilter(image, ddepth=-1, ksize=(11, 11))
    eps = 1e-8
    image = cv2.divide(image, blurred + eps)
    a = image
    r = a[..., 2]
    g = a[..., 1]
    b = a[..., 0]

    a = r + b
    cv2.imshow("a = r + b", a); cv2.waitKey(0)
    b = a - 2*g
    b = b * 5
    cv2.imshow("b = a - 2*g", b); cv2.waitKey(0)
    c = np.abs(r - b)
    cv2.imshow("c = abs(r - b)", c); cv2.waitKey(0)
    d = b - c
    cv2.imshow("d = b = c", d); cv2.waitKey(0)

    a = (r + b - 2*g) - np.abs(r - b)
    return a

if __name__ == "__main__":
    img = cv2.imread("renders/0131.jpg")
    out = detect_coral(img)
    # If you want to show the result (not recommended for pure performance timing):
    cv2.imshow("img", out)
    cv2.waitKey(0)
