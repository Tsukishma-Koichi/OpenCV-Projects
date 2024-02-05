import cv2
import numpy as np

capture = cv2.VideoCapture(0)

capture.set(3, 640)
capture.set(4, 360)
# 设置捕获图像尺寸 “3”：宽度；“4”：高度


def choose_color(color):
    global lower, upper
    if color == "red":
        lower = np.array([0, 100, 100])
        upper = np.array([10, 255, 255])  # 设定红色的hsv阈值
        return lower, upper
    elif color == "green":
        lower = np.array([40, 40, 40])
        upper = np.array([80, 255, 255])  # 设定绿色的hsv阈值
        return lower, upper
    elif color == "blue":
        lower = np.array([100, 40, 40])
        upper = np.array([140, 255, 255])  # 设定蓝色的hsv阈值
        return lower, upper


while capture.isOpened():
    ret, frame = capture.read()
    # cap.read()返回两个值，第一个值为布尔值，如果视频正确，那么就返回true,  第二个值代表图像三维像素矩阵

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # rgb通道难以分离颜色 需要先转化到hsv色彩空间

    lower, upper = choose_color('red')  # 选择识别颜色

    mask1 = cv2.inRange(hsv, lower, upper)  # 设置掩模 只保留所选颜色部分
    res = cv2.bitwise_and(frame, frame, mask=mask1)  # 利用掩模与原图像做“与”操作 过滤出特定色
    mask = np.stack([mask1] * 3, axis=2)  # mask矩阵拓展

    kernel = np.ones((10, 10), np.uint8)  # 设置开运算所需核

    mask0 = cv2.dilate(mask1, kernel, iterations=1)

    opening = cv2.morphologyEx(mask0, cv2.MORPH_OPEN, kernel)  # 对得到的mask进行开运算
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 找到轮廓
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # 获取边界框的坐标和尺寸
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)  # 绘制矩形框

    cv2.imshow('frame', frame)  # 显示原视频
    cv2.imshow('hsv', hsv)  # 显示hsv视频
    cv2.imshow('mask', mask)  # 显示mask视频
    cv2.imshow('mask', mask0)  # 显示mask0视频
    cv2.imshow('res', res)  # 显示最终视频

    key = cv2.waitKey(1)

    if key == ord('s'):
        print('Size:')
        print(capture.get(3))
        print(capture.get(4))
    elif key == ord('q'):
        print('Over.')
        capture.release()
        cv2.destroyAllWindows()
        break
