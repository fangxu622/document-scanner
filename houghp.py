
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
				default=r"F:\SZU-Prj\jiaonang\2019_310_capsule_prj\document-scaner\data\37.jpg")
args = vars(ap.parse_args())

# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)

# in the image
meanimage = cv2.pyrMeanShiftFiltering(image, 10, 50, maxLevel=2,termcrit=(cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, 5, 1))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)
#gray = cv2.medianBlur(gray, (3, 3))

#eqh_img_gray=cv2.equalizeHist(gray)

edged = cv2.Canny(gray,5,60 )#75, 200)

minLineLength = 10
maxLineCap = 5
 
lines = cv2.HoughLinesP(edged, 1, np.pi/180, 100, minLineLength, maxLineCap)
edges = cv2.cvtColor(edged,cv2.COLOR_GRAY2BGR)
for x1,y1,x2,y2 in lines[:,0,:]:
    cv2.line(edges, (x1,y1),(x2,y2),(0,0,255),1)

# show the original image and the edge detected image
print("STEP 1: Edge Detection")

kernel = np.ones((3, 3), np.uint8)
dilation_edge = cv2.dilate(edged, kernel) 
#dilation_edge = cv2.dilate(edged, kernel)
erosion = cv2.erode(dilation_edge, kernel)
# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
#cnts = cv2.findContours(dilation_edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)#cv2.RETR_LIST
cnts = cv2.findContours(dilation_edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)#cv2.RETR_LIST
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

#cv2.imshow("Image", meanimage)
cv2.imshow("Edged", edged)
cv2.imshow("dilation", dilation_edge)
cv2.imshow("meanshift", meanimage)
cv2.imshow("line_img", edges)
#cv2.imshow("erosion", erosion)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# loop over the contours
index_p=[]
screenCnt_tem=[]
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	
	print(len(approx))
	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx)>5:#> 4 and len(approx) < 9:
		index_p.append(peri)
		screenCnt_tem.append(approx)
		
screenCnt=screenCnt_tem[index_p.index(max(index_p))]
print(len(screenCnt_tem))
	#screenCnt.append(approx)
		#break
 
# show the contour (outline) of the piece of paper
print("STEP 2: Find contours of paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# apply the four point transform to obtain a top-down
# view of the original image
# warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

 
# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect

# warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
# T = threshold_local(warped, 11, offset = 10, method = "gaussian")
# warped = (warped > T).astype("uint8") * 255
 
# show the original and scanned imagesq
#print("STEP 3: Apply perspective transform")
#cv2.imshow("Original", imutils.resize(orig, height = 650))
#cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)

#cv2.destroyAllWindows()
# python scan.py --image F:\SZU-Prj\jiaonang\2019_310_capsule_prj\document-scaner\data\1.jpg
#python scan.py --image F:\SZU-Prj\jiaonang\2019_310_capsule_prj\document-scaner\data\1.jpg