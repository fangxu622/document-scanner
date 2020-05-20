import numpy as np
import cv2
import imutils
cv2.namedWindow("images")

def nothing():
    pass
kernel = np.ones((3,3),np.uint8)
path=r"F:\SZU-Prj\jiaonang\2019_310_capsule_prj\document-scaner\data\30.jpg"
cv2.createTrackbar("s1","images",0,255,nothing)
cv2.createTrackbar("s2","images",0,255,nothing)
image = cv2.imread(path)
img = imutils.resize(image, height = 800)
img = cv2.pyrMeanShiftFiltering(img, 10, 50, maxLevel=2,termcrit=(cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, 5, 1))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
while(1):
    s1 = cv2.getTrackbarPos("s1","images")
    s2 = cv2.getTrackbarPos("s2","images")
    
    out_img_tem = cv2.Canny(img,s1,s2)

    out_img = cv2.morphologyEx(out_img_tem, cv2.MORPH_OPEN, kernel)
    dilation_edge = cv2.dilate(out_img_tem, kernel) 
    cv2.imshow("img",out_img)
    cv2.imshow("img1",out_img_tem)
    cv2.imshow("img2",dilation_edge)
    k = cv2.waitKey(1)&0xFF
    if k==27:
        break
cv2.destroyAllWindows()