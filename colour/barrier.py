import cv2
import numpy as np


lower_green = np.array([58, 90, 67])
upper_green = np.array([99, 255, 214])
green = (0, 0, 255)
cv2.namedWindow('video', cv2.WINDOW_AUTOSIZE)


class Detector:
    def __init__(self, img, lower, upper, color):
        self.img = img
        self.lower = lower
        self.upper = upper
        self.color = color
    def img_process(self):
        kernel = np.ones((35, 35), np.uint8)
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        opening = cv2.morphologyEx(hsv, cv2.MORPH_OPEN, kernel)
        mask = cv2.inRange(opening, self.lower, self.upper)
        res = cv2.bitwise_and(self.img, self.img, mask=mask)
        return res

    def cnts_draw(self):
        obj = Detector(self.img, self.lower, self.upper, self.color)
        res = obj.img_process()
        canny = cv2.Canny(res, 300, 450)  # Canny边缘检测算法，用来描绘图像中物体的边缘，（100，200为此函数的两个阈值，该阈值越小轮廓的细节越丰富）
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:  # 传递到max函数中的轮廓不能为空
            cv2.imshow('video', self.img)
            return
        else:
            max_cnt = max(contours, key=cv2.contourArea)
            cv2.drawContours(self.img, max_cnt, -1, self.color, 2)
            (x, y, w, h) = cv2.boundingRect(max_cnt)
            # cv2.rectangle(self.img, (x, y), (x + w, y + h), self.color, 3)
            area = cv2.contourArea(max_cnt)
            print(area)
            cv2.imshow('video', self.img)


cap = cv2.VideoCapture('1.mp4')

new_width = 640
new_height = 480

while cap.isOpened():
    flag, frame = cap.read()
    if not flag:
        print("无法读取摄像头！")
        break
    else:
        if frame is not None:
            resized_frame = cv2.resize(frame, (new_width, new_height))
            Green = Detector(resized_frame, lower_green, upper_green, green)
            Green.img_process()
            Green.cnts_draw()
            key = cv2.waitKey(10)
            if key == 27:  # ESC退出程序
                break
        else:
            print("无画面")
            break

cap.release()
cv2.destroyAllWindows()
