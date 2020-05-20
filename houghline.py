from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = False,
				default=r"F:\SZU-Prj\jiaonang\2019_310_capsule_prj\document-scaner\faildata\17.jpg")
args = vars(ap.parse_args())

# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)

meanimage = cv2.pyrMeanShiftFiltering(image, 5, 10, termcrit=(cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, 5, 1))
img = cv2.GaussianBlur(meanimage,(3,3),0)
edges = cv2.Canny(img, 50, 150, apertureSize = 3)
lines = cv2.HoughLines(edges,1,np.pi/180,118) #这里对最后一个参数使用了经验型的值
result = img.copy()
for line in lines:
	rho = line[0][0]  #第一个元素是距离rho
	theta= line[0][1] #第二个元素是角度theta
	print (rho)
	print (theta)
	if  (theta < (np.pi/4. )) or (theta > (3.*np.pi/4.0)): #垂直直线
		pt1 = (int(rho/np.cos(theta)),0)               #该直线与第一行的交点
		#该直线与最后一行的焦点
		pt2 = (int((rho-result.shape[0]*np.sin(theta))/np.cos(theta)),result.shape[0])
		cv2.line( result, pt1, pt2, (255))             # 绘制一条白线
	else:                                                  #水平直线
		pt1 = (0,int(rho/np.sin(theta)))               # 该直线与第一列的交点
		#该直线与最后一列的交点
		pt2 = (result.shape[1], int((rho-result.shape[1]*np.cos(theta))/np.sin(theta)))
		cv2.line(result, pt1, pt2, (255), 1)           # 绘制一条直线

cv2.imshow('Canny', edges )
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

