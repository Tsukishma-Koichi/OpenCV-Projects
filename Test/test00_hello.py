
import cv2

print(cv2.getVersionString())

image = cv2.imread("resources/opencv_logo.jpg")
print(image.shape)  # 显示图片数据 (h, w, 3) 3: 3通道BGR


cv2.imshow("image", image)
cv2.waitKey()

