import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image to scanned")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
ratio = image.shape[0] / 500
orig = image.copy()


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
	dim = None
	(h, w) = image.shape[:2]
	if width is None and height is None:
		return image
	if width is None:
		r = height / float(h)
		dim = (int(w * r), height)
	else:
		r = width / float(width)
		dim = (width, int(h * r))
	resized = cv2.resize(image, dim, interpolation=inter)
	return resized


def order_points(pts):
	rect = np.zeros((4, 2), dtype="float32")
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect


def four_point_transform(image, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	max_width = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	max_height = max(int(heightA), int(heightB))

	dst = np.array([
					[0, 0],
					[max_width-1, 0],
					[max_width-1, max_height-1],
					[0, max_height-1]], dtype="float32")

	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (max_width, max_height))

	return warped


image = resize(orig, height=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
canny = cv2.Canny(blur, 75, 200)

cnts = cv2.findContours(canny.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]  # [0]: 轮廓列表
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

for c in cnts:
	peri = cv2.arcLength(c, True)  # 计算轮廓近似
	approx = cv2.approxPolyDP(c, 0.02*peri, True)
	# epsilon: 原始轮廓到近似轮廓最大距离 准确度参数
	if len(approx) == 4:
		scr_cnt = approx
		break

cv2.drawContours(image, [scr_cnt], -1, (0, 255, 0), 2)

# 透视变换
warped = four_point_transform(orig, scr_cnt.reshape(4, 2)*ratio)

warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
# ref = cv2.threshold(warped_gray, 155, 255, cv2.THRESH_BINARY)[1]
ref = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
							cv2.THRESH_BINARY, 9, 8)
cv2.imwrite("scan.png", ref)

cv2.imshow("Outline", ref)
cv2.waitKey(0)
cv2.destroyAllWindows()