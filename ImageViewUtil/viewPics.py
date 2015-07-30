import os
import sys
import time
import copy
import shutil
import re
from os import getcwd
import cv2
from cv2 import cv
import numpy as np
from functools import cmp_to_key


def cmpfunc(af, bf):
    a=af.split('.')[0]
    b=bf.split('.')[0]
    cmp_list1 = a.split('-')
    cmp_list2 = b.split('-')
    if len(cmp_list1) != len(cmp_list2):
        raise Exception
    cmpLen=len(cmp_list1)
    for i in range(cmpLen):
        try:
            node1=int(cmp_list1[i]);
            node2=int(cmp_list2[i]);
        except Exception, e:
            continue
        ret=node1-node2
        if ret != 0:
            return ret
    print("Same file? %s, %s" %(af, bf))
    return 0

need_update = True
def update(_=None):
    global need_update
    need_update = True


def main():
    input_dir = os.path.abspath(os.getcwd())
    tmplist = os.listdir(input_dir)
    fList_orig = [f for f in tmplist if f.split('.')[-1].lower() in ["gif", "bmp", "jpg"]]

    sorted_key = None
    for f in fList_orig:
        if f.split('.')[-1].lower() in ["gif", "bmp", "jpg"]:
            nameonly = f.split('.')[0]
            fileNum = re.search('-(\d+)\.', f)
            if fileNum is not None:
                sorted_key = cmp_to_key(cmpfunc)
                break
    fSorted = sorted(fList_orig, key=sorted_key)

    cv2.namedWindow("track",1)

    cv2.createTrackbar('pos', 'track', 0, len(fSorted), update)
    cv2.setTrackbarPos('pos', 'track', 0)

    i=0
    flag=""
    while True:
        if i<=0:
            i=0
        elif i >= len(fSorted):
            i=len(fSorted)-1

        f=fSorted[i]
        aFile = os.path.join(input_dir,f)
        if os.path.isfile(aFile):
            img = cv2.imread(aFile,1)

            text=f+ " " + flag + " "+fSorted[0]+","+fSorted[-1]
            ret, textsize = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.0, thickness=1)
            #row,col=img.shape[0:2]
            #x, y = 5, (row-2*textsize)
            x, y = 5, (2*textsize)
            cv2.putText(img, text, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), thickness = 1, lineType=cv2.CV_AA)
            cv2.imshow("track", img)

            ch = 0xFF & cv2.waitKey(5)
            #print ch

            if ch == 27:
                break

            global need_update
            if ch >= 255: #timeout
                if need_update: #mouse
                    print "update"
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

