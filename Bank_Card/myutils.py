import cv2


def sort_contours(cnts, method="left-to-right"):
	reverse = True
	i = 0

	if method == "left-to-right" or method == "bottom-to-top":
		reverse = False

	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1

	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	cnts, boundingBoxes = zip(*sorted(zip(cnts, boundingBoxes),
										key=lambda b: b[1][i], reverse=reverse))

	return cnts, boundingBoxes

def resize(image, width):
	h, w = image.shape[:2]
	height = int(h * width / w)
	resized_image = cv2.resize(image, (width, height))

	return resized_image
