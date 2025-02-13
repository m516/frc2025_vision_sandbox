import cv2

if __name__ == "__main__":
    img = cv2.imread("renders/0131.jpg")
    cv2.imshow("img", img)
    cv2.waitKey(0)
