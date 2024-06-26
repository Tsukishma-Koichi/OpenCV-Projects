import numpy as np
import cv2


class Stitcher:

	def stitch(self, images, ratio=0.75, reprojThresh=4.0, show_matches=False):
		imageB, imageA = images[0], images[1]

		kpsA, featuresA = self.detectAndDescribe(imageA)
		kpsB, featuresB = self.detectAndDescribe(imageB)

		M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

		if M is None:
			return None

		matches, H, status = M

		result = cv2.warpPerspective(imageA, H, (imageA.shape[1]+imageB.shape[1], imageA.shape[0]))
		# self.cv_show('result', result)
		result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
		# self.cv_show('result', result)

		if show_matches:
			vis = self.drawMatches(imageA, kpsA, imageB, kpsB, matches, status)
			return (result, vis)  # 返回拼接图像和匹配结果图像

		return result



	def detectAndDescribe(self, image):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		descriptor = cv2.SIFT_create()
		kps, features = descriptor.detectAndCompute(image, None)
		kps = np.float32([kp.pt for kp in kps])

		return kps, features


	def matchKeypoints(self, kpA, kpB, featuresA, featuresB, ratio, reprojThresh):
		matcher = cv2.BFMatcher()

		raw_matches = matcher.knnMatch(featuresA, featuresB, k=2)

		matches = []
		for m in raw_matches:
			if len(m) == 2 and m[0].distance < ratio * m[ 1].distance:
				matches.append((m[0].trainIdx, m[0].queryIdx))

			if len(matches) > 4:
				ptsA = np.float32([kpA[i] for _, i in matches])
				ptsB = np.float32([kpB[i] for i, _ in matches])

				H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

		return matches, H, status

	def drawMatches(self, imgA, kpsA, imgB, kpsB, matches, status):
		"""
		绘制匹配结果
		"""
		(hA, wA) = imgA.shape[:2]
		(hB, wB) = imgB.shape[:2]
		vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
		vis[0:hA, 0:wA] = imgA
		vis[0:hB, wA:] = imgB

		for ((trainIdx, queryIdx), s) in zip(matches, status):
			if s == 1:
				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
				cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

		return vis


	def cv_show(self, name, img):
		cv2.imshow(name, img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()