# -*- coding:utf-8 -*-

import cv2
import numpy as np

capture = cv2.VideoCapture(0)

hsv_low = np.array([0, 0, 0])
hsv_high = np.array([0, 0, 0])


# 下面几个函数，写得有点冗余

def h_low(value):
    hsv_low[0] = value


def h_high(value):
    hsv_high[0] = value


def s_low(value):
    hsv_low[1] = value


def s_high(value):
    hsv_high[1] = value


def v_low(value):
    hsv_low[2] = value


def v_high(value):
    hsv_high[2] = value


cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 640, 480)

# H low：
#    0：指向整数变量的可选指针，该变量的值反映滑块的初始位置。
#  179：表示滑块可以达到的最大位置的值为179，最小位置始终为0。
# h_low：指向每次滑块更改位置时要调用的函数的指针，指针指向h_low元组，有默认值0。

cv2.createTrackbar('H low', 'image', 0, 179, h_low)
cv2.createTrackbar('H high', 'image', 0, 179, h_high)
cv2.createTrackbar('S low', 'image', 0, 255, s_low)
cv2.createTrackbar('S high', 'image', 0, 255, s_high)
cv2.createTrackbar('V low', 'image', 0, 255, v_low)
cv2.createTrackbar('V high', 'image', 0, 255, v_high)

while capture.isOpened():
    ret, frame = capture.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # BGR转HSV
    dst = cv2.inRange(hsv, hsv_low, hsv_high)  # 通过HSV的高低阈值，提取图像部分区域up_width = 600

    down_width = 640
    down_height = 480
    down_points = (down_width, down_height)
    dst = cv2.resize(dst, down_points, interpolation=cv2.INTER_LINEAR)

    cv2.imshow('dst', dst)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

h_l = cv2.getTrackbarPos('H low', 'image')
h_h = cv2.getTrackbarPos('H high', 'image')
s_l = cv2.getTrackbarPos('S low', 'image')
s_h = cv2.getTrackbarPos('S high', 'image')
v_l = cv2.getTrackbarPos('V low', 'image')
v_h = cv2.getTrackbarPos('V high', 'image')

lower = [h_l, s_l, v_l]
upper = [h_h, s_h, v_h]

print("lower: %s" % lower)
print("upper: %s" % upper)

cv2.destroyAllWindows()
