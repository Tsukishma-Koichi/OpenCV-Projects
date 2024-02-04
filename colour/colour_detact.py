import cv2
import numpy as np


# 设置三种颜色hsv阈值 以三原色为例
lower_red = np.array([0, 43, 46])
upper_red = np.array([10, 255, 255])
lower_green = np.array([37, 27, 58])
upper_green = np.array([107, 255, 255])
lower_blue = np.array([100, 43, 46])
upper_blue = np.array([124, 255, 255])
# 实际操作中根据 colour_hsv.py 确定颜色阈值

# 设置线框颜色
red = (0, 0, 225)
green = (0, 255, 0)
blue = (225, 0, 0)

cv2.namedWindow('video', cv2.WINDOW_AUTOSIZE)  # 设置窗口'video'，大小为自适应模式
cv2.resizeWindow('video', 640, 480)  # 为窗口设置宽度（640）和高度（480）


def img_process(img, lower, upper):
	"""
	根据阈值处理图像，提取阈值内的颜色。返回处理后只留下指定颜色的图像（其余为黑色）
	:param img: 原图像
	:param lower: 最高阈值
	:param upper: 最低阈值
	:return: 指定颜色的图像
	"""
	kernel = np.ones((35, 35), np.uint8)  # 创建一个35x35卷积核，卷积核内元素全为1
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 将BGR图像转化为HSV图像，方便颜色提取
	opening = cv2.morphologyEx(hsv, cv2.MORPH_OPEN, kernel)  # 用卷积核对图像进行形态学开运算操作，去除噪声
	mask = cv2.inRange(opening, lower, upper)  # 开运算得到的图像用阈值进行二值化处理（处理后的结果为在阈值内的部分变为白色，不在阈值内的部分为黑色）
	res = cv2.bitwise_and(img, img, mask=mask)  # 二值化处理后的图像与原图进行位与运算（处理后在阈值内的颜色变为原颜色，不在阈值内的部分仍为黑色）
	return res  # 该函数的返回值为位与运算之后的图像，此图像只保留了在阈值内的图像，其余部分为黑色


def cnts_draw(img, res, color):
	"""
	在原图像上绘出指定颜色的轮廓
	:param img: 原图像
	:param res: 只剩某颜色的位与运算后的图像
	:param color: 轮廓的颜色
	:return: 无
	"""
	canny = cv2.Canny(res, 100, 200)  # Canny边缘检测算法，用来描绘图像中物体的边缘，（100，200为此函数的两个阈值，该阈值越小轮廓的细节越丰富）
	contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# 寻找图像轮廓的函数，这里先用Canny算法得到只保留轮廓的图像方便轮廓的找寻
	if len(contours) == 0:  # 传递到max函数中的轮廓不能为空
		cv2.imshow('video', img)
		return
	else:
		max_cnt = max(contours, key=cv2.contourArea)  # 找到轮廓中最大的一个
		cv2.drawContours(img, max_cnt, -1, color, 2)  # 在原图上绘制这个最大轮廓
		(x, y, w, h) = cv2.boundingRect(max_cnt)
		# 找到这个最大轮廓的最大外接矩形，返回的（x，y）为这个矩形右下角的顶点，w为宽度，h为高度
		cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)  # 在原图上绘制这个矩形
		cv2.imshow('video', img)  # 展示原图


def color_find(img):
	"""
	找到原图像最多的颜色，打印出来该颜色的名称
	:param img: 原图像
	:return: 无
	"""
	kernel = np.ones((35, 35), np.uint8)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	opening = cv2.morphologyEx(hsv, cv2.MORPH_OPEN, kernel)  # 以上为图像处理
	h_list = cv2.calcHist([opening], [0], None, [180], [0, 180])  # 对Open图像的H通道进行直方图统计
	h_list_max = np.where(h_list == np.max(h_list))  # 找到直方图h_list中列方向最大的点h_list_max
	if 0 < h_list_max[0] < 10:  # H在0~10为红色
		print('red')
	elif 35 < h_list_max[0] < 77:  # H在35~77为绿色
		print('green')
	elif 100 < h_list_max[0] < 124:  # H在100~124为蓝色
		print('blue')
	else:
		return


if __name__ == "__main__":
	cap = cv2.VideoCapture(0)  # 调用摄像
	while cap.isOpened():
		flag, frame = cap.read()
		if not flag:
			print("无法读取摄像头！")
			break
		else:
			if frame is not None:
				res_blue = img_process(frame, lower_blue, upper_blue)
				res_green = img_process(frame, lower_green, upper_green)
				res_red = img_process(frame, lower_red, upper_red)
				cnts_draw(frame, res_blue, blue)
				cnts_draw(frame, res_green, green)
				cnts_draw(frame, res_red, red)
				color_find(frame)
				key = cv2.waitKey(10)
				if key == 27:  # ESC退出程序
					break
			else:
				print("无画面")
				break

	cap.release()
	cv2.destroyAllWindows()
