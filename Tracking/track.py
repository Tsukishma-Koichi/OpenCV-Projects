import numpy as np
import cv2
import math


lower_yellow = np.array([20, 43, 46])  # 20,43,46
upper_yellow = np.array([55, 255, 255])  # 50,255,255

# cap = cv2.VideoCapture(0)
vedio = 'demo2.avi'
cap = cv2.VideoCapture(vedio)

count_cir = 0
count_thr = 0
count_ten = 0

while True:
	ret, frame = cap.read()
	if not ret:
		break
	# 显示原始图像
	cv2.imshow("frame", frame)
	# 高斯模糊
	frame = cv2.GaussianBlur(frame, (13, 13), 10, 20)
	# 转换成hsv格式
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	# 得到二值化图像bin
	bin = cv2.inRange(hsv, lower_yellow, upper_yellow)
	# 腐蚀图像img
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	close = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, kernel, iterations=2)

	# 中线提取
	img = cv2.resize(close, dsize=(200, 200))
	oriX = 0
	oriY = 160
	staX = 100
	endY = 135  # 中线起止
	staY = oriY
	LEFT = staX  # 起始行寻白使用
	RIGHT = staX
	OriRight = 0  # 找跳变使用
	OriLeft = 0
	left = 0  # 道路边线的左右
	right = 0

	left_line = []  # 用于记录边线x坐标
	right_line = []

	while True:
		if LEFT >= img.shape[1]:  # 检查是否超出图像宽度范围
			break

		if img[staY, LEFT] == 255:
			staX = LEFT
			break
		if img[staY, RIGHT] == 255:
			staX = RIGHT
			break
		LEFT = LEFT + 1
		RIGHT = RIGHT - 1

	while staY >= endY:
		right = staX
		left = staX
		while img[staY, right] == 255:
			right += 1
		right_line.append(right)  # 记录坐标
		while img[staY, left] == 255:
			left -= 1
		left_line.append(left)

		staX = int((left + right) / 2)

		# 环岛识别
		if staY == endY and count_cir <= 2:
			if len(left_line) >= 2 and len(right_line) >= 2:
				diff_left = left_line[-1] - left_line[-25]  # 检测边线坐标差值
				diff_right = right_line[-1] - right_line[-25]
				if diff_left < -40 and diff_right < -8:  # 左边线突变，右边线变化不大则为左环岛
					print("左环岛")
					count_cir = count_cir + 1

				# 三叉
				if count_cir == 1 and count_thr == 0:
					length_end = abs(left_line[-1] - right_line[-1])
					length_sta = abs(left_line[0] - right_line[0])
					diff = abs(length_end - length_sta)
					if diff > 70:
						if count_thr == 0:
							print("三叉")
		# 十字
		if staY == endY and count_cir >= 2:
			delt_left = left_line[-1] - left_line[-10]
			delt_right = right_line[-1] - right_line[-10]
			if delt_left < -10 and delt_right > 10:
				print("十字")


		if staY == oriY:
			oriX = staX
		staY -= 1

	cv2.line(img, (oriX, oriY), (staX, endY), (0, 0, 255), 2)

	cv2.imshow("img", img)
	if cv2.waitKey(10) & 0xFF == ord(' '):
		break





