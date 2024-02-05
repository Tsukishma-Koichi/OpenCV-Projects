# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# img = cv2.imread('opencv_logo.jpg', cv2.IMREAD_GRAYSCALE)
# cv2.IMREAD_GRAYSCALE 显示灰度图
# part = img[0:50, 0:200, 2]  #图像切片

# 保存
# cv2.imwrite('opencv_logo.png', img)

# type(img)  # 格式
# img.size  # 像素点个数
# img.dtype  # 数据类型


# cap = cv2.VideoCapture(0)  # 参数为数字则调用对应摄像头，为路径参数则调用对应视频
#
# while cap.isOpened():
# 	ret, frame = cap.read()
# 	if frame is not None:
# 		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# 		cv2.imshow('frame', gray)
# 	if cv2.waitKey(1) & 0xFF == ord('q'):
# 		break
# 	# cv2.waitKey(1): 这个函数会等待指定的毫秒数（在这里是1毫秒），来检测用户是否按下了键盘上的某个按键。
# 	# 				如果没有按键被按下，则函数会返回 - 1。
# 	# & 0xFF: 这个操作是为了确保我们只考虑按下键的ASCII码的最低8位。这是因为在一些系统中，cv2.waitKey()
# 	# 		的返回值可能包含更高位的信息，而我们只关心最低8位的键盘输入。
# 	# ord('q'): 这个函数返回字符'q'的ASCII码值。
#
# cap.release()  # 释放资源，关闭摄像头或视频文件
# cv2.destroyAllWindows()  # 销毁所有OpenCV窗口


# b, g, r = cv2.split()  # 三通道分离
# img = cv2.merge((b, g, r))


# 边界填充
# top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)
# img = cv2.imread('panda.png')
#
# replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REPLICATE)
# reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT)
# reflect101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT_101)
# wrap = cv2.copyMakeBorder(img, top_size, bottom_size, right_size, left_size, cv2.BORDER_WRAP)
# constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_CONSTANT, value=[0])
#
# plt.subplot(231), plt.imshow(img, cmap='gray'), plt.title('Original')
# plt.subplot(232), plt.imshow(replicate, cmap='gray'), plt.title('Replicate')
# plt.subplot(233), plt.imshow(reflect, cmap='gray'), plt.title('Reflect')
# plt.subplot(234), plt.imshow(reflect101, cmap='gray'), plt.title('Reflect101')
# plt.subplot(235), plt.imshow(wrap, cmap='gray'), plt.title('Wrap')
# plt.subplot(236), plt.imshow(constant, cmap='gray'), plt.title('Constant')
#
# plt.show()


# 数值计算
# img = cv2.imread('panda.png')
# res = img + 10  # 对矩阵中每个值加10
# 如果超过最大值（256）取余

# 图像融合
# img1 = cv2.imread('panda.png')
# img2 = cv2.imread('opencv_logo.jpg')
#
# resize1 = cv2.resize(img1, (250, 250))
#
# result = cv2.addWeighted(img2, 0.4, resize1, 0.6, 0)
# plt.imshow(result)
# plt.show()


# import numpy as np
# import cv2 as cv
# cap = cv.VideoCapture(0)
# # 声明编码器和创建 VideoWrite 对象
# fourcc = cv.VideoWriter_fourcc(*'XVID')
# out = cv.VideoWriter('output.avi',fourcc, 20.0, (640,480))
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret==True:
#         frame = cv.flip(frame,0)
#         # 写入已经翻转好的帧
#         out.write(frame)
#         cv.imshow('frame',frame)
#         if cv.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break
# # 释放已经完成的工作
# cap.release()
# out.release()
# cv.destroyAllWindows()


# 傅里叶变换 滤波

# 以下为使用NumPy进行变换 高通滤波
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv2.imread('Jean.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# f = np.fft.fft2(gray)  # 傅里叶变换
# fshift = np.fft.fftshift(f)  # 将左上角频率移至中间位置
# magnitude_spectrum = 20*np.log(np.abs(fshift))  # 映射变换 提高可视化
#
# rows, cols = gray.shape
# crow, ccol = rows//2 , cols//2  # 计算图像的中心点坐标
# fshift[crow-30:crow+31, ccol-30:ccol+31] = 0  # 设置掩码
# f_ishift = np.fft.ifftshift(fshift)  # 将零频率分量移动到频率域图像的中心
# img_back = np.fft.ifft2(f_ishift)  # 对处理后的频率域图像进行逆傅里叶变换，得到空域中的图像
# img_back = np.real(img_back)  # 获取逆傅里叶变换后图像的实部
#
# plt.subplot(221),plt.imshow(gray, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(222),plt.imshow(magnitude_spectrum, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.subplot(223),plt.imshow(img_back, cmap = 'gray')
# plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
# plt.subplot(224),plt.imshow(img_back)
# plt.title('Result in JET'), plt.xticks([]), plt.yticks([])
# plt.show()

# 以下为使用OpenCV进行变换 低通滤波
# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt
#
# img = cv.imread('Jean.png')
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# img_M = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#
# dft = cv.dft(np.float32(gray), flags=cv.DFT_COMPLEX_OUTPUT)
# dft_shift = np.fft.fftshift(dft)
# magnitude_spectrum = 20*np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
#
# rows, cols = gray.shape
# crow, ccol = rows//2, cols//2
# # create a mask first, center square is 1, remaining all zeros
# mask = np.zeros((rows,cols,2),np.uint8)
# mask[crow-30:crow+30, ccol-30:ccol+30] = 1
# # apply mask and inverse DFT
# fshift = dft_shift * mask
# f_ishift = np.fft.ifftshift(fshift)
# img_back = cv.idft(f_ishift)
# img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])
#
# plt.subplot(131), plt.imshow(img_M)
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(132),plt.imshow(magnitude_spectrum, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.subplot(133),plt.imshow(img_back, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()


# import cv2
# import numpy as np
#
# img = cv2.imread('../resources/Jean.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# sift = cv2.SIFT_create()
# kp, des = sift.detectAndCompute(gray, None)
# img = cv2.drawKeypoints(gray, kp, img)
# cv2.imshow('img', img)
# cv2.waitKey(0)



