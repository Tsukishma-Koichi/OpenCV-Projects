from PIL import Image
import pytesseract
import cv2
import os

preprocess = 'blur'

image = cv2.imread('scan.png')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

if preprocess=='blur':
	gray = cv2.medianBlur(gray, 3)

if preprocess=='thresh':
	gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

text = pytesseract.image_to_string(cv2.imread(filename))
print(text)
os.remove(filename)

cv2.imshow('image', image)
cv2.imshow('Gray Image', gray)
cv2.waitKey(0)