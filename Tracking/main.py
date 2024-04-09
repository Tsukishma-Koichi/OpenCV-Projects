import cv2
import numpy as np
import time
import math
import colour_detect as dc
#  import os
#  import pandas as pd
#  from tqdm import tqdm

start_time = time.time()

# 二值化阈值
lower_yellow = np.array([8, 43, 46])  # 20,43,46
upper_yellow = np.array([24, 255, 255])  # 50,255,255

lower_red = np.array([142, 55, 56])
upper_red = np.array([179, 200, 122])
red = [0, 0, 255]

# 计数器
count_thr = 0
count_cir = 0
count_cross = 0
Count = 0           # 总计数


# cap = cv2.VideoCapture(0)  # 读取摄像头
cap = cv2.VideoCapture("output1.mp4")  # 读取视频文件
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # 二值化

    # 高斯模糊
    frame = cv2.resize(frame, dsize=(120, 120))
    # 识别启停区
    Red = dc.Detector(frame, lower_red, upper_red, red)
    Red.img_process()
    Red.cnts_draw()  # 输出”停止“

    frame = cv2.GaussianBlur(frame, (13, 13), 6)
    h = frame.shape[0]
    w = frame.shape[1]
    #  img = average(img)

    # frame = frame[100: 200, 0: 200]
    # 转换成hsv格式
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 得到二值化图像bin
    bin = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # 腐蚀图像img
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, kernel, iterations=2)

    m = int(h/2)
    n = s = 0
    while m < h:
        n = 0
        while n < w:
            if img[m, n] == 255:
                s += 1
            n += 3
        m += 3
    # print(s)    # 调参: threshold

    # 获取边界数组
    oriX = staX = int(w/2)
    staY = oriY = 115
    endY = EndY = 85   # 真中线起止
    Len = staY - endY  # 中线长即数组长
    LEFT = RIGHT = staX  # 起始行寻白使用
    left = right = 0  # 道路边线的左右
    left_line = []  # 用于记录边线x坐标
    right_line = []

    # 注释此处开启计数器
    # count_thr = 0
    # count_cir = 0
    # count_cross = 0
    # Count = 0

    # 寻白
    f = 0
    while True:
        if RIGHT >= img.shape[1]:
            f = 1
            break
        if img[staY, LEFT] == 255:
            staX = LEFT
            break
        if img[staY, RIGHT] == 255:
            staX = RIGHT
            break
        LEFT = LEFT - 1
        RIGHT = RIGHT + 1
    # 中点继承法
    # while staY > 0:
    while staY >= endY:
        right = staX
        left = staX
        while img[staY, right] == 255 and right < w-1:
            right += 1
        right_line.append(right)
        while img[staY, left] == 255 and left >= 0:
            left -= 1
        left_line.append(left)
        '''
        if len(left_line) >= 18:
            if abs(right_line[-1] - right_line[-2]) > 2 or abs(left_line[-1] - left_line[-2]) > 2:
                EndY = staY
                break
        '''
        staX = int((left + right) / 2)
        if staY == oriY:
            oriX = staX

        #  元素识别
        threshold = 550    # 下部白色像素点过少就不进行元素识别
        # threshold = 0  # threshold开关
        if staY == endY and s > threshold and f == 0:
            #  环岛
            if len(left_line) >= 2 > count_cir and (Count == 0 or Count == 2):
                '''
                cir_down_delta = 30
                cir_up_delta = 30
                diff_down_left = left_line[3] - left_line[8]  # 检测边线坐标差值
                diff_down_right = abs(right_line[3] - right_line[8])
                diff_up_left = left_line[-5] - left_line[-1]
                diff_up_right = abs(right_line[-1] - right_line[-5])
                # print(diff_up_left,diff_up_right)			# 调参: cir_up_delta
                # print(diff_down_left,diff_down_right) 	# 调参: cir_down_delta
                if diff_down_left > cir_down_delta and diff_down_right <= 1:  # 左边线突变，右边线变化不大则为左环岛
                    print("下环岛")
                    # count_cir += 1
                    # Count += 1
                if diff_up_left > cir_up_delta and diff_up_right <= 2:
                    print("上环岛")
                    # count_cir += 1
                    # Count += 1
                '''
                # 算法二
                isJump = 1
                for u in range(5, Len-10):
                    if left_line[u-1] - left_line[u] >= 8:
                        for v in range(5,u-1):
                            if left_line[v-4] - left_line[v] > 1:
                                isJump = 0
                                break
                        if isJump == 1 and img[h-u, u] == 255:
                            p = 18
                            while p < Len:
                                if -1 > right_line[p] - right_line[p-2] or right_line[p] - right_line[p-2] > 1:
                                    break
                                p += 1
                            if p == Len:
                                print("环岛")
            # 三叉
            '''
            i = 21
            while i < Len-5:
                if (right_line[i-3] < right_line[i] or right_line[i-3] == right_line[i] == w) and (left_line[i-3] > left_line[i] or left_line[i-3] == left_line[i] == 0):
                    i += 1
                else:
                    break
            if i == Len-1 and count_thr == 0:
                print("三叉")
                # count_thr += 1
                # Count += 1'''
            if s >= 550:
                Black_staX = int(w/2)
                r = staY
                Flag = 0
                while r > staY - 50:
                    if img[r, staX] == 0:
                        break
                    r -= 1
                if r != staY-50:
                    while r > 0:
                        Black_Left = Black_Right = Black_staX
                        while img[r, Black_Right] == 0:
                            Black_Right += 1
                            if Black_Right == w-10:
                                Flag = 1
                                break
                        while img[r, Black_Left] == 0:
                            Black_Left -= 1
                            if Black_Left == 10:
                                Flag = 1
                                break
                        Black_staX = int((Black_Right + Black_Left)*0.5)
                        Black_Left = Black_Right = Black_staX
                        ll = Black_Right - Black_Left
                        if ll >= 80:
                            Flag = 1
                            break
                        r -= 3
                    if Flag == 0 and 53 <= oriX <= 67:
                        print("三岔")

            # 十字
            if Count >= 0 and s >= 700:  # and count_cross < 2:
                '''
                # 算法一
                cross_delta = 40
                midpoint_delta = 5
                diff_cross_up = right_line[Len-1] - left_line[Len-1]
                diff_cross_down = right_line[0] - left_line[0]
                diff_UpDown = diff_cross_up - diff_cross_down
                diff_midpoint = abs(int(((right_line[Len-1] + left_line[Len-1]) - (right_line[0] + left_line[0])) * 0.5))
                # print(diff_UpDown, diff_midpoint)		# 调参: cross_delta, midpoint_delta
                if diff_UpDown >= cross_delta and diff_midpoint <= midpoint_delta and img[endY + 20, staX] == 255:
                    print("十字")
                    count_cross += 1'''
                # 算法二
                k = 0
                while k < Len:
                    if right_line[k] - left_line[k] != 120:
                        break
                    k += 1
                if k == Len:
                    for o in range(0, staY):
                        if img[o, 60] == 0:
                            k = -1
                            break
                if k == Len:
                    print("十字")
        staY -= 1

    dis = oriX - w/2
    arg = math.atan2(oriY - endY, staX - oriX)
    arg = 90 - math.degrees(arg)
    cv2.putText(frame, f"dis={dis:.2f}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
    cv2.putText(frame, f"arg={arg:.2f}de", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
    cv2.line(img, (oriX, oriY), (staX, endY), (0, 0, 255), 1)
    # 显示原始图像
    cv2.imshow("frame", frame)
    # 显示二值图像
    img = cv2.resize(img, dsize=(300, 300))
    cv2.imshow("img", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

end_time = time.time()
print(end_time - start_time)
cap.release()














'''    
# 算法三
if right_line[-1] >= right_line[-2]+2 and left_line[-2] >= left_line[-1]:
    if right_line[-2] >= right_line[-3] +2 and left_line[-2] <= left_line[-3]:
        print("十字3")
        count_cross += 1
''''''
# 算法四
z = -4
while z < -1:
    if left_line[z] != 0:
        break
    z += 1
if z == -1:
    if right_line[-2] - left_line[-2] >= 35:
        print("十字4")
'''
'''
image = cv2.imread("peng.png")
def rgb2gray(image):
    h = image.shape[0]
    w = image.shape[1]
    grayimage = np.zeros((h, w), np.uint8)
    for i in tqdm(range(h)):
        for j in range(w):
            grayimage[i, j] = 0.144 * image[i, j, 0] + 0.587 * image[i, j, 1] + 0.299 * image[i, j, 1]
    return grayimage

# 大津法
def otsu(image):
    # 高和宽
    h = image.shape[0]
    w = image.shape[1]
    # 求总像素
    m = h * w

    otsuimg = np.zeros((h, w), np.uint8)
    # 初始阈值
    initial_threshold = 0
    # 最终阈值
    final_threshold = 0
    # 初始化各灰度级个数统计参数
    histogram = np.zeros(256, np.int32)
    # 初始化各灰度级占图像中的分布的统计参数
    probability = np.zeros(256, np.float32)

    # 各个灰度级的个数统计
    for i in tqdm(range(h)):
        for j in range(w):
            s = image[i, j]
            histogram[s] = histogram[s] + 1
    # 各灰度级占图像中的分布的统计参数
    for i in tqdm(range(256)):
        probability[i] = histogram[i] / m

    for i in tqdm(range(255)):
        w0 = w1 = 0  # 前景和背景的灰度数
        fgs = bgs = 0  # 定义前景像素点灰度级总和背景像素点灰度级总和
        for j in range(256):
            if j <= i:  # 当前i为分割阈值
                w0 += probability[j]  # 前景像素点占整幅图像的比例累加
                fgs += j * probability[j]
            else:
                w1 += probability[j]  # 背景像素点占整幅图像的比例累加
                bgs += j * probability[j]
        u0 = fgs / w0  # 前景像素点的平均灰度
        u1 = bgs / w1  # 背景像素点的平均灰度
        G = w0 * w1 * (u0 - u1) ** 2
        if G >= initial_threshold:
            initial_threshold = G
            final_threshold = i
    print(final_threshold)

    for i in range(h):
        for j in range(w):
            if image[i, j] > final_threshold:
                otsuimg[i, j] = 255
            else:
                otsuimg[i, j] = 0
    return otsuimg


grayimage = rgb2gray(image)
otsuimage = otsu(grayimage)

cv2.waitKey()
# print(new_image)
'''
'''
# 高斯模糊
	fra = cv2.GaussianBlur(frame, (13, 13), 10, 20)
	# 转换成hsv格式
	hsv = cv2.cvtColor(fra, cv2.COLOR_BGR2HSV)
	# 得到二值化图像bin
	bin = cv2.inRange(hsv, lower_yellow, upper_yellow)
	# 腐蚀图像img
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	close = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, kernel, iterations=2)
	#  中线提取
	img = cv2.resize(close, dsize=(200, 200))  # 把原图缩小一下方便计算
	oriX = 0
	oriY = 160
	staX = 100
	endY = 135  # 中线起止
	staY = oriY
	LEFT = staX  # 起始行寻白使用
	RIGHT = staX
	left = 0  # 道路边线的左右
	right = 0
	left_line = []  # 用于记录边线x坐标
	right_line = []
	count_cir = 0		# 计数器
	count_cross = 0
	count_thr = 0
	while True:
		if LEFT >= img.shape[1]:
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
		while img[staY, right] == 255 and right < 199:
			right += 1
		right_line.append(right)  # 记录坐标
		while img[staY, left] == 255:
			left -= 1
		left_line.append(left)
		staX = int((left + right) / 2)
		if staY == oriY:
			oriX = staX
			
			
		if staY == endY:
			if len(left_line) >= 2 and len(right_line) >= 2:
				#  环岛
				diff_left = left_line[-1] - left_line[-25]  # 检测边线坐标差值
				diff_right = right_line[-1] - right_line[-25]
				if diff_left < -40 and diff_right < -8:  # 左边线突变，右边线变化不大则为左环岛
					cv2.putText(frame, "左环岛", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
					print("左环岛")
					Count_leftRound += 1
				if diff_left < 8 and diff_right > 40:
					print("右环岛")
					cv2.putText(frame, "右环岛", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
					Count_rightRound += 1
				#  三岔路与十字环
				DeltaLeft = left_line[-1] - left_line[-8]
				DeltaRight = right_line[-1] - right_line[-8]
				if (DeltaLeft >= 57) or (DeltaRight <= -57):
					print("三叉")'''
'''right = staX
	left = staX
	while img[staY, right] == 255:
		right += 1
	right_line.append(right)   # 记录坐标
	while img[staY, left] == 255:
		left -= 1
	left_line.append(left)
	staY -= 1
	OriX = int((right + left) / 2
	while staY >= endY:
		if img[staY,right] == 255:
			while img[staY,right] == 255:
				right += 1
		else:
			while img[staY,right] == 0:
				right -= 1
		right_line.append(right)  # 记录坐标
		if img[staY,left] == 255:
			while img[staY,left] == 255:
				left -= 1
		else:
			while img[staY,right] == 0:
				left += 1
		left_line.append(left)
		staX = int((left + right) / 2)
		staY -= 1
	'''
'''
img = cv2.resize(close, dsize=(600, 600))
	oriX = 0
	staX = 300
	endY = 440          # 中线起止
	staY = 500
	LEFT = staX         # 起始行寻白使用
	RIGHT = staX
	OriRight = 0        # 找跳变使用
	OriLeft = 0
	left = 0            # 道路边线的左右
	right = 0
	while True:
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
		while img[staY, left] == 255:
			left -= 1
		staX = int((left + right) / 2)
		# 环岛识别
		if staY == endY:
			if len(left_line) >= 2 and len(right_line) >= 2:
				diff_left = left_line[-1] - left_line[-25]  # 检测边线坐标差值
				diff_right = right_line[-1] - right_line[-25]
				if diff_left < -40 and diff_right < -8:  # 左边线突变，右边线变化不大则为左环岛
					# print(diff_left)
					# print(diff_right)
					print("左环岛")
				if diff_left > 8 and diff_right > 40:
					print("右环岛")
'''

'''
ThreeMode = 0
		if OriLeft > left and OriRight < right:
			ThreeMode = 1

		I = 540
		J = 290
		while ThreeMode == 1:
			Sum = 0
			if img[I,J] == 255:
				Sum += 1
			I += 1
			J += 1
			if J >= 310:
				break
		if Sum >= 200:
			print("十字")
		if Sum != 0 and Sum < 200:
			print("三叉")
		OriLeft = left
		OriRight = right



import cv2
import numpy as np
import math

# 计算横向位移和偏角
def calculate_dis_arg(oriX, staX, adjusted_500, adjusted_endY, frame_width):
	# 假设图像的中心为道路宽度的中心
	image_center = frame_width / 2
	# 计算横向偏移：道路中心线的位置与图像中心的差值
	dis = (oriX + staX) / 2 - image_center
	# 计算偏角
	arg = math.atan2(adjusted_endY - adjusted_500, staX - oriX)
	# 将弧度转换为度
	arg = math.degrees(arg)
	return dis, arg


def process_frame(frame, lower_yellow, upper_yellow):
	# 应用高斯模糊减少噪声
	blurred = cv2.GaussianBlur(frame, (13, 13), 0)
	# 转换到HSV颜色空间
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	# 创建黄色阈值的掩码
	mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
	# 应用闭运算形态学操作清除噪点
	kernel = np.ones((5, 5), np.uint8)
	close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

	return close

# 检测三岔路口
def calculate_slope(point1, point2):
	# 计算斜率
	return (point2[1] - point1[1]) / (point2[0] - point1[0]) if point2[0] != point1[0] else float('inf')


def detect_v_turning_points(binary_img, slope_threshold, row_step=20):
	rows, cols = binary_img.shape
	left_slopes = []
	right_slopes = []
	left_turning_point = None
	right_turning_point = None

	# 从下往上检测左、右边界点
	for row in range(rows - row_step, 0, -row_step):
		left_idx = np.argmax(binary_img[row])
		right_idx = np.argmax(binary_img[row, ::-1])

		# 计算斜率并检测变化
		if row + row_step < rows:
			prev_left_idx = np.argmax(binary_img[row + row_step])
			prev_right_idx = np.argmax(binary_img[row + row_step, ::-1])

			left_slope = calculate_slope((prev_left_idx, row + row_step), (left_idx, row))
			right_slope = calculate_slope((cols - prev_right_idx, row + row_step), (cols - right_idx, row))

			left_slopes.append(left_slope)
			right_slopes.append(right_slope)

			# 检测斜率变化
			if len(left_slopes) > 1 and abs(left_slope - left_slopes[-2]) > slope_threshold:
				left_turning_point = (left_idx, row)
			if len(right_slopes) > 1 and abs(right_slope - right_slopes[-2]) > slope_threshold:
				right_turning_point = (cols - right_idx, row)
	# 检测中间V拐点
	center_v_point = detect_center_v_point(binary_img, left_turning_point, right_turning_point, row_step)

	return left_turning_point, right_turning_point, center_v_point


def detect_center_v_point(binary_img, left_tp, right_tp, width_threshold=100):
	if left_tp is None or right_tp is None:
		return None

	# 获取左右拐点的行索引
	row_min = min(left_tp[1], right_tp[1])
	row_max = max(left_tp[1], right_tp[1])

	# 初始化变量来存储V拐点的位置
	v_point = None

	# 从左右拐点之间的最低行开始向上扫描，寻找V拐点
	for row in range(row_max, row_min, -1):  # 从低到高进行扫描
		row_pixels = binary_img[row, :]
		transitions = np.where(row_pixels[:-1] != row_pixels[1:])[0]  # 找到黑白转换的位置

# 判断环岛
def detect_roundabout(img, delta=50, min_size=180):
	debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 为了调试创建的彩色图像
	height, width = img.shape
	center_x = width // 2
	left_flag, right_flag = False, False
	for y in range(height-1, 0, -1):
		left_size = right_size = 0
		# 检测左侧和右侧白色区域的宽度
		for x in range(center_x, -1, -1):
			if img[y, x] == 255: left_size += 1
			else: break
		for x in range(center_x, width):
			if img[y, x] == 255: right_size += 1
			else: break
		# 检测是否满足环岛的条件
		if left_size >= min_size or right_size >= min_size:
			cv2.line(debug_img, (0, y), (width, y), (0, 0, 255), 1)
			if left_size >= min_size: left_flag = True
			if right_size >= min_size: right_flag = True

	return left_flag, right_flag, debug_img

def main(video_path, lower_yellow, upper_yellow):
	cap = cv2.VideoCapture(video_path)
	while True:
		ret, frame = cap.read()
		if not ret: break

		close = process_frame(frame, lower_yellow, upper_yellow)
		left_detected, right_detected, debug_img = detect_roundabout(close)
		#  中线提取
		img = cv2.resize(close, dsize=(600, 600))
		oriX = 0
		staX = 300
		endY = 440  # 中线起止
		staY = 500
		LEFT = staX  # 起始行寻白使用
		RIGHT = staX
		while True:
			if LEFT >= img.shape[1]:  # 检查LEFT是否超出右边界
				break
			if RIGHT < 0:  # 检查RIGHT是否超出左边界
				break
			if img[staY, LEFT] == 255:
				staX = LEFT
				break
			if img[staY, RIGHT] == 255:
				staX = RIGHT
				break
			LEFT += 1
			RIGHT -= 1
		while staY >= endY:
			right = staX
			left = staX
			while 0 <= left < img.shape[1] and img[staY, left] == 255:
				left -= 1
			while 0 <= right < img.shape[1] and img[staY, right] == 255:
				right += 1
			staX = int((left + right) / 2)
			if staY == 500:
				oriX = staX
			staY -= 1
		# 计算原始帧大小的中线坐标
		height, width = frame.shape[:2]  # 使用原始帧的高度和宽度
		scale_y = height / 600  # 计算高度缩放因子
		scale_x = width / 600  # 计算宽度缩放因子
		# 调整中线的起点和终点坐标
		adjusted_oriX = int(oriX * scale_x)
		adjusted_staX = int(staX * scale_x)
		adjusted_500 = int(500 * scale_y)
		adjusted_endY = int(endY * scale_y)

		# 在 debug_img 上绘制中线
		cv2.line(debug_img, (adjusted_oriX, adjusted_500), (adjusted_staX, adjusted_endY), (255, 0, 0), 2)

		# 横向位移和偏角
		dis, arg = calculate_dis_arg(adjusted_oriX, adjusted_staX, adjusted_500, adjusted_endY, width)
		# 将横向位移和偏角显示在处理过的图像上
		cv2.putText(debug_img, f"dis={dis:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
		cv2.putText(debug_img, f"arg={arg:.2f}deg", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

		# 如果检测到三岔路口
		# 在主循环中调用 detect_three_way
		left_point, right_point, center_v_point = detect_v_turning_points(close, 10, 10)
		if(left_point and right_point and center_v_point ) :
			print("检测到三岔路口")


		# 如果检测到环岛
		if left_detected or right_detected:
			detection_info = f"Find a corner"
			print(detection_info)
			cv2.putText(frame, detection_info, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

		cv2.imshow("Debug", debug_img)
		cv2.imshow("Original", frame)

		if cv2.waitKey(60) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

# 黄色的HSV阈值
lower_yellow = np.array([20, 43, 46])
upper_yellow = np.array([55, 255, 255])

# 视频文件路径
video_path = "demo2.avi"
main(video_path, lower_yellow, upper_yellow)
'''
