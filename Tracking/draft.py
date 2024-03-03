# import cv2
# import numpy as np
#
# img = cv2.imread('cornor.png')
# img = cv2.resize(img, dsize=(600, 600))
# gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# gray = np.float32(gray)
#
# dst = cv2.cornerHarris(gray, 7, 9, 0.06)
# indices = np.where(dst == dst.max())
# points = list(zip(indices[1], indices[0]))
#
# cv2.circle(img, points[0], 10, (0, 0, 255))
# cv2.imshow('img', img)
# cv2.waitKey(0)

import cv2
import numpy as np


# def detect_sharp_angles(image_path):
# 	# 读取图像
# 	image = cv2.imread(image_path)
# 	image = cv2.resize(image, dsize=(600, 600))
#
# 	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# 	# 边缘检测
# 	edges = cv2.Canny(gray, 100, 200)
#
# 	# 寻找轮廓和顶点
# 	contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# 	# 遍历每个轮廓
# 	for contour in contours:
# 		# 获取轮廓的近似多边形
# 		epsilon = 0.02 * cv2.arcLength(contour, True)
# 		approx = cv2.approxPolyDP(contour, epsilon, True)
#
# 		# 如果近似多边形的边数大于等于3，则计算夹角
# 		if len(approx) >= 3:
# 			for i in range(len(approx)):
# 				p1 = approx[i][0]
# 				p2 = approx[(i + 1) % len(approx)][0]
# 				p3 = approx[(i + 2) % len(approx)][0]
#
# 				# 计算向量
# 				v1 = p1 - p2
# 				v2 = p3 - p2
#
# 				# 计算向量之间的夹角
# 				angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
#
# 				# 转换为度数
# 				angle_deg = np.degrees(angle)
#
# 				# 判断是否为锐角
# 				if angle_deg < 90:
# 					print("锐角:", angle_deg)
#
# 					# 标出顶点
# 					cv2.circle(image, tuple(p2), 5, (255, 0, 0), 2)
#
# 	# 显示图像和边缘
# 	cv2.imshow("Image", image)
# 	cv2.imshow("Edges", edges)
# 	cv2.waitKey(0)
# 	cv2.destroyAllWindows()
#
# image_path = "cornor.png"  # 图像路径
# detect_sharp_angles(image_path)

# # 中线提取
# 	#  中线提取
# 	img = cv2.resize(close, dsize=(600, 600))
# 	oriX = 0
# 	staX = 300
# 	endY = 440  # 中线起止
# 	staY = 500
# 	LEFT = staX  # 起始行寻白使用
# 	RIGHT = staX
# 	OriRight = 0  # 找跳变使用
# 	OriLeft = 0
# 	left = 0  # 道路边线的左右
# 	right = 0
#
# 	while True:
# 		if img[staY, LEFT] == 255:
# 			staX = LEFT
# 			break
# 		if img[staY, RIGHT] == 255:
# 			staX = RIGHT
# 			break
# 		LEFT = LEFT + 1
# 		RIGHT = RIGHT - 1
#
# 	while staY >= endY:
# 		right = staX
# 		left = staX
# 		while img[staY, right] == 255:
# 			right += 1
# 		while img[staY, left] == 255:
# 			left -= 1
# 		staX = int((left + right) / 2)
#
# 		# 环岛识别
# 		# delta = 100
# 		LeftFlag = 0
# 		RightFlag = 0
# 		division_line = staX
# 		if staY == endY:
#
# 			# 设置边线丢失阈值
# 			threshold = 280
#
# 			# 统计图像中左侧和右侧白色像素的数量
# 			left_white_pixel_count = 0
# 			right_white_pixel_count = 0
#
# 			for pixel in img[endY]:
# 				if pixel == 255:
# 					if pixel < division_line:
# 						left_white_pixel_count += 1
# 					else:
# 						right_white_pixel_count += 1
#
#
#
# 			if left_white_pixel_count > threshold:
# 				LeftFlag = 1
# 			elif right_white_pixel_count > threshold:
# 				RightFlag = 1
#
# 			if RightFlag == 1 and """90 > angle_deg > 0""":
# 				print("右环岛")
# 				# print(angle_deg)
# 			if LeftFlag == 1 and """0 > angle_deg > -90""":
# 				print("左环岛")
# 				# print(angle_deg)
#
# 		if staY == 500:
# 			oriX = staX
# 		staY -= 1

	# cv2.line(img, (oriX, 500), (staX, endY), (0, 0, 0), 2)
	# if oriX - staX != 0:
	# 	# 计算斜率
	# 	m = (500 - endY) / (oriX - staX)
	# 	# 计算倾斜角（弧度）
	# 	angle_rad = math.atan(m)
	# 	# 将弧度转换为角度
	# 	angle_deg = math.degrees(angle_rad)

# #  三岔路与十字环识别
		# ThreeMode = 0
		# if OriLeft > left and OriRight < right:
		# 	ThreeMode = 1
		# Sum = 0
		# I = 540
		# J = 290
		# while ThreeMode == 1:
		# 	if img[I, J] == 255:
		# 		Sum += 1
		# 	I += 1
		# 	J += 1
		# 	if J >= 310:
		# 		break
		# if Sum >= 200:
		# 	print("十字")
		# if Sum != 0 and Sum < 200:
		# 	pass
		# 	# print("三叉")
		# OriLeft = left
		# OriRight = right




import numpy as np
import cv2
import math


lower_yellow = np.array([20, 43, 46])  # 20,43,46
upper_yellow = np.array([55, 255, 255])  # 50,255,255

# cap = cv2.VideoCapture(0)
vedio = 'demo2.avi'
cap = cv2.VideoCapture(vedio)

count_cir = 0  # 记录环岛个数
count_thr = 0  # 记录三叉

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
						print("三叉")
						print(diff)


		if staY == oriY:
			oriX = staX
		staY -= 1

	cv2.line(img, (oriX, oriY), (staX, endY), (0, 0, 255), 2)

	cv2.imshow("img", img)
	if cv2.waitKey(10) & 0xFF == ord(' '):
		break