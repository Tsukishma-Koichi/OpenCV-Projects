'''
 *******************************************************************************************
 ** 功    能: 单帧调试
 ** 参    数:
 ** 返 回 值: 二值图像
 ********************************************************************************************
 '''

import cv2
import numpy as np

lower_yellow = np.array([20, 43, 46])  # 20,43,46
upper_yellow = np.array([50, 255, 255])  # 50,255,255

frame = cv2.imread("demo1.png")

# 高斯模糊
frame1 = cv2.GaussianBlur(frame, (13, 13), 10, 20)
# 转换成hsv格式
hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
# 得到二值化图像bin
bin = cv2.inRange(hsv, lower_yellow, upper_yellow)
# 腐蚀图像img
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
close = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, kernel, iterations=2)

# cv2.circle(frame,(200,200),5,(255,0,0))
# cv2.line()

# 显示原始图像
cv2.imshow("frame", frame)
# 显示二值图像
cv2.imshow("close", close)

print(close[0][0])

cv2.waitKey(0)

cv2.destroyAllWindows()
