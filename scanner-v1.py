# import the necessary packages
from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils


def main(video_path,freq):
    cap = cv2.VideoCapture(video_path)
    c=0
    while (cap.isOpened()):
        c = c + 1
        ret, image = cap.read()
        if not ret:
            break
        if c % freq==0:#and c > 1800
            #image = cv2.imread(args["image"])
            #ratio = image.shape[0] / 500.0
            #orig = image.copy()
            image = imutils.resize(image, height = 500)

            edge_point=np.array([np.array([[0,0]]),np.array([[0,50]]),np.array([[50,50]]),np.array([[50,0]])])
            meanimage = cv2.pyrMeanShiftFiltering(image, 15, 30, termcrit=(cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, 5, 1))
            gray = cv2.cvtColor(meanimage, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(gray, 75, 200)

            kernel = np.ones((3, 3), np.uint8)
            dilation_edge = cv2.dilate(edged, kernel) 

            cnts = cv2.findContours(dilation_edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)#cv2.RETR_LIST
            cnts = imutils.grab_contours(cnts)
            cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:4]
            index_p=[]
            screenCnt_tem=[]
            for ci in cnts:
                #peri = cv2.arcLength(ci, True)
                peri = cv2.arcLength(ci, True)
                approx = cv2.approxPolyDP(ci, 0.02* peri,True)
                if len(approx)==4:
                    index_p.append(peri)
                    screenCnt_tem.append(approx)
            if len(index_p) != 0:
                screenCnt=screenCnt_tem[index_p.index(max(index_p))]
            else:
                screenCnt=edge_point

            cv2.drawContours(image, [screenCnt], -1, (255, 0, 255), 2)
            cv2.imshow("Outline", image)
        #print("已读完")
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
    cap.release()
    return 0

if __name__ == "__main__":
    #path=r"F:\FFOutput\00058te.mp4"
    #path=r"H:\input\00084.mp4"
    path=r"F:\SZU-Prj\jiaonang\2019_310_capsule_prj\document-scaner\data\VID_20190802_191816.mp4"
    main(path,5)

#cv2.destroyAllWindows()
# python scan.py --image F:\SZU-Prj\jiaonang\2019_310_capsule_prj\document-scaner\data\1.jpg
#python scan.py --image F:\SZU-Prj\jiaonang\2019_310_capsule_prj\document-scaner\data\1.jpg

#	        #    if len(approx)==4:
            #        index_p.append(peri)
		    #        screenCnt_tem.append(approx)	
            #screenCnt=screenCnt_tem[index_p.index(max(index_p))]