from stitcher import Stitcher
import cv2

imageA = cv2.imread('asset/left.png')
imageB = cv2.imread('asset/right.png')

stitcher = Stitcher()
result, vis = stitcher.stitch([imageA, imageB], show_matches=True)

cv2.imshow('Image A', imageA)
cv2.imshow('Image B', imageB)
cv2.imshow('KeyPoint Matcher', vis)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()