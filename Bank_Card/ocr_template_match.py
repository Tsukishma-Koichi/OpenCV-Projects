# Import tools
import numpy as np
import argparse
import cv2
import myutils

# Set up parameters
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
				help="Path to input the image")
ap.add_argument("-t", "--template", required=True,
				help="Path to template OCR-A image")
args = vars(ap.parse_args())


# Drawing display
def cv_show(name, img):
	cv2.imshow(name, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


# 读取模板
img = cv2.imread(args["template"])
# cv_show("img", img)

ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv_show("ref", ref)

ref = cv2.threshold(ref, 150, 255, cv2.THRESH_BINARY_INV)[1]
# cv_show("ref", ref)

# 计算轮廓
refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, refCnts, -1, (0, 0, 255), 3)
# cv_show("img", img)
refCnts = myutils.sort_contours(refCnts, method="left-to-right")[0]
digits = {}

# 遍历每一个轮廓
for i, c in enumerate(refCnts):
	x, y, w, h = cv2.boundingRect(c)
	roi = ref[y:y + h, x:x + w]
	roi = cv2.resize(roi, (57, 88))

	digits[i] = roi

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# 读取输入图像 预处理
image = cv2.imread(args["image"])
# cv_show("Original", image)
image = myutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv_show("Grayscale", gray)

# 礼貌操作 突出明亮区域
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, sqKernel)
# cv_show("tophat", tophat)

tophat = cv2.threshold(tophat, 100, 255, cv2.THRESH_BINARY)[1]
# cv_show("thresh", tophat)

# sobel算子 + 闭运算 将数字连在一起
sobelx = cv2.Sobel(tophat, cv2.CV_32F, 1, 0, ksize=-1)  # "-1": 自动选择一个合适的高斯滤波核大小作为阈值处理的参数
sobelx = cv2.convertScaleAbs(sobelx)
# cv_show("Sobel X", sobelx)
sobely = cv2.Sobel(tophat, cv2.CV_32F, 0, 1, ksize=-1)
sobely = cv2.convertScaleAbs(sobely)
# cv_show("Sobel Y", sobely)
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
# cv_show("Sobel XY", sobelxy)

grad = cv2.morphologyEx(sobelxy, cv2.MORPH_CLOSE, rectKernel)
# cv_show("Gradient", grad)

# OTSU自动寻找合适阈值
thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# cv_show("thresh", thresh)

grad1 = cv2.erode(thresh, rectKernel, iterations=1)  # 腐蚀一次
grad0 = cv2.dilate(grad1, sqKernel, iterations=3)  # 膨胀两次
# cv_show("grad", grad0)

# 在原图像上绘制轮廓
threshCnts, hierarchy = cv2.findContours(grad0.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = threshCnts
cur_image = image.copy()
cv2.drawContours(cur_image, cnts, -1, (0, 0, 255), 3)
# cv_show("cur_image", cur_image)

# 筛选出所需轮廓
locs = list()
for i, c in enumerate(cnts):
	x, y, w, h = cv2.boundingRect(c)
	ar = w / h
	# cur = image.copy()
	# cv2.drawContours(cur, [c], -1, (0, 0, 255), 3)
	# cv_show(f"{i}", cur)
	# print(f"{i} w={w} h={h}", ar)

	if 2.4 < ar < 2.5:
		if 45 < w < 48 and 18 < h < 20:
			locs.append([x, y, w, h])

locs = sorted(locs, key=lambda x: x[0])

#
output = []
for i, (gX, gY, gW, gH) in enumerate(locs):
	groupOutput = []

	group = gray[gY-2:gY+gH+2, gX-2:gX+gW+2]
	# cv_show("group", group)

	# 提取数字
	group = cv2.threshold(group, 100, 255, cv2.THRESH_BINARY)[1]
	# group = cv2.morphologyEx(group, cv2.MORPH_CLOSE, sqKernel)
	# cv_show("group", group)

	digitCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	digitCnts = myutils.sort_contours(digitCnts, method="left-to-right")[0]

	for c in digitCnts:
		x, y, w, h = cv2.boundingRect(c)
		roi = group[y:y+h, x:x+w]
		# cur = group.copy()
		# cv2.drawContours(cur, [c], -1, (0, 0, 255), 3)
		# cv_show("cur", cur)
		print(roi.shape)
		roi = cv2.resize(roi, (57, 88))
		# cv_show("roi", roi)

		# 匹配
		scores = []
		for digit, digitROI in digits.items():
			result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF_NORMED)
			_, score, _, _ = cv2.minMaxLoc(result)
			scores.append(score)

		# 找出最适值
		groupOutput.append(str(np.argmax(scores)))  # 求出索引

	cv2.rectangle(image, (gX-2, gY-2), (gX+gW+2, gY+gH+2), (0, 0, 255), 2)
	cv2.putText(image, "".join(groupOutput), (gX, gY-15),
				cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

	output.extend(groupOutput)

cv_show("Image", image)
