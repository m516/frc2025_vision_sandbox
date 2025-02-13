import cv2

if __name__ == "__main__":
    img = cv2.imread("renders/0131.jpg")

    '''
    img is a 800 rows x 1280 columns x 3 channels matrix (NumPy array).
    Each channel is a color. 
        Usually in web design, the order is (red, green, blue)
        In OpenCV, the order is (0=blue, 1=green 2=red)
    For example, suppose pixel at row 64 and column 65 (x=65,y=64) is #BE5AFE,
                 RED    |   GREEN   |    BLUE
         Hex    B    E  |    5    A |    F    E
      Binary 1011 1110  | 0101 1010 | 1111 1110
     Decimal       190  |        90 |       254

    So 
        the red   brightness of that pixel is 190/255.
        the green brightness of that pixel is  90/255.
        the blue  brightness of that pixel is 254/255.

    Then,
        img[64, 65, 0] = 254
        img[64, 65, 1] = 90
        img[64, 65, 2] = 190
    
    In other words,
        img[64, 65] = [254, 90, 190]
    '''

    # Let's just show the red channel
    red_channel = img[...,2] # extract channel 2, the red channel
    cv2.imshow("img", red_channel)
    cv2.waitKey(0)
    # Let's try the other channels. Notice a couple of things:
    # 1. Calling cv2.imshow with the same image name string updates the image. That's a way to render animations.
    # 2. Each channel looks slightly different, highlighting different colors.
    cv2.imshow("img", img[...,1]) # channel 1, the green channel
    cv2.waitKey(0)
    cv2.imshow("img", img[...,0]) # channel 0, the blue channel
    cv2.waitKey(0)
