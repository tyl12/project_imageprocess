import os
import sys
import time
import copy
import shutil
import re
import cv2
from cv2 import cv
import numpy as np

need_update = True
def update(_=None):
    global need_update
    need_update = True

def decodeFile(source, cap, idx, docvt):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("ERROR:Failed to open video %s" % source)
        return None, None

    ret = True
    frame = None

    cap.set(cv.CV_CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret or frame is None:
        print("ERROR:fail to read frame %d" % i)
    else:
        if not docvt:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return ret, gray

def main():
    input_dir = os.path.abspath(os.getcwd())
    tmplist = os.listdir(input_dir)
    fList = [f for f in tmplist if f.split('.')[-1].lower() in ["avi", "mpeg", "rmvb"]]

    for f in fList:

        source=os.path.join(os.getcwd(), f)
        print("Decode stream file %s" %source)

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("ERROR:Failed to open video %s" % source)
            continue

        totalFrameNumber = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))
        print("INFO:Total frame count=%d" % totalFrameNumber)

        width = cap.get(cv.CV_CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv.CV_CAP_PROP_FPS)
        codec = cap.get(cv.CV_CAP_PROP_FOURCC)
        docvt = cap.get(cv.CV_CAP_PROP_CONVERT_RGB)
        print("INFO:frame size=%dx%d, fps=%d, codec=%s, docvt=%d" %(width, height, fps, codec, docvt))

        cv2.namedWindow("track",1)
        cv2.createTrackbar('pos', 'track', 0, totalFrameNumber, update)
        cv2.setTrackbarPos('pos', 'track', 0)

        i=0
        flag=""
        while True:
            if i<=0:
                i=0
            elif i >= totalFrameNumber:
                i=totalFrameNumber-1

            ret, img = decodeFile(source, cap, i, docvt)
            if not ret or img is None:
                print("Fail to get frame %d" %i)
                break

            text=f+ " " + flag
            ret, textsize = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.0, thickness=1)
            #row,col=img.shape[0:2]
            #x, y = 5, (row-2*textsize)
            x, y = 5, (2*textsize)
            cv2.putText(img, text, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), thickness = 1)
            cv2.imshow("track", img)

            ch = 0xFF & cv2.waitKey(5)
            #print ch

            if ch == 27:
                break

            global need_update
            if ch >= 255: #timeout
                if need_update: #mouse
                    #print "update"
                    need_update=False
                    i_new=cv2.getTrackbarPos('pos','track')
                    if i_new != i: #check whether real mouse event happened
                        flag=""
                    i=i_new
                else: #no mouse, only timeout
                    #print "timeout", flag
                    if flag == "B":
                        print("Process file %s"%(f))
                        i -= 1
                    elif flag == "F":
                        print("Process file %s"%(f))
                        i += 1
                    cv2.setTrackbarPos('pos', 'track', i)
                continue

            need_update = False #notes: once key pressed, mouse event will also be triggered. so clear beforehand.
            flag=""
            if ch == 98: ##b
                print("Process file %s"%(f))
                i -= 1
            elif ch == 102: ##f
                print("Process file %s"%(f))
                i += 1
            elif ch == 70: ##F
                print "F pressed"
                flag="F"
            elif ch == 66: #B
                print "B pressed"
                flag="B"
            cv2.setTrackbarPos('pos', 'track', i)

        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

