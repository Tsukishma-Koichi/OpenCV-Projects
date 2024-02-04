
import cv2

image = cv2.imread("resources/opencv_logo.jpg")

crop = image[10:170, 40:200]

cv2.imshow("crop", crop)
cv2.waitKey()
