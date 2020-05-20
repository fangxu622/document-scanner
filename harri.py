# import the necessary packages
from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = False,
				default=r"F:\SZU-Prj\jiaonang\2019_310_capsule_prj\document-scaner\faildata\9.jpg")
args = vars(ap.parse_args())

# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)


#参数如下：

# img 输入图像，数据类型为float32
# blockSize 角点检测当中的邻域值。
# ksize 使用Sobel函数求偏导的窗口大小
# k 角点检测参数，取值为0.04到0.06
# convert the image to grayscale, blur it, and find edges
# in the image
#meanimage = cv2.pyrMeanShiftFiltering(image, 15, 30, termcrit=(cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, 5, 1))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

edged = cv2.Canny(gray, 75, 200)
#edged = np.float32(gray)

corners = cv2.cornerHarris(edged, 10, 5, 0.05)

# for i in corners:
#     x,y = i.ravel()
#     cv2.circle(img,(x,y),2,255,-1)

image[corners>0.01*corners.max()]=[0,0,255]

cv2.imshow("img",image)
cv2.imshow("edged",edged)

cv2.waitKey(0)
cv2.destroyAllWindows()