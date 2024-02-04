import cv2
import numpy as np

# 定义红、绿、蓝三种颜色的阈值范围
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
lower_green = np.array([40, 50, 50])
upper_green = np.array([90, 255, 255])
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([140, 255, 255])

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取摄像头捕获的图像
    ret, frame = cap.read()

    # 将图像转换为 HSV 颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 根据颜色阈值创建掩模
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # 在原始图像上标记红色区域
    contours, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, 'Red', (x+w-70, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        print("Red.")
    # 在原始图像上标记绿色区域
    contours, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, 'Green', (x+w-70, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        print("Green.")
    # 在原始图像上标记蓝色区域
    contours, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, 'Blue', (x+w-70, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        print("Blue.")
    # 显示标记后的图像
    cv2.imshow('Color Detection', frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()