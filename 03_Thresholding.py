import cv2
import numpy as np

if __name__ == "__main__":
    img = cv2.imread("renders/0131.jpg")
    
    # For division and multiplication, we generally want to use a floats or large ints.
    # In OpenCV, 
    #   int images range from 0 (black) to 255 (white)
    #   float images range from 0 (black) to 1 (white)
    img_f = img.astype(np.float32) / 255

    # Split the image into channels
    red_channel = img_f[...,2] # extract channel 2, the red channel
    green_channel = img_f[...,1] # extract channel 1, the green channel
    blue_channel = iimg_fmg[...,0] # extract channel 0, the blue channel

    is_purple = (
        # Filter out dark pixels
        (red_channel > 32) &
        (blue_channel > 32) &
        # Filter out blue and red pixels
        (red_channel / blue_channel < 1.5) &
        (blue_channel / red_channel < 1.5) &
        # Filter out gray, white, and green pixels
        (green_channel < 0.7 * np.minimum(red_channel, blue_channel))
    )

    # is_purple is now a boolean matrix. Convert it to uint8 so we can cv2.imshow it
    is_purple_uint8 = is_purple.astype(np.uint8) * 255

    # Displaying the results
    cv2.imshow("img", img)
    cv2.waitKey(2000) # wait 2000 milliseconds or until the user presses a keyboard button
    cv2.imshow("img", is_purple_uint8)
    cv2.waitKey(60000)
