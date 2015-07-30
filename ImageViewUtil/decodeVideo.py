import os
import sys
import re
import cv2
from cv2 import cv

DEBUG=False

def decodeFile(fPath, dstpath):
    source=fPath
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("ERROR:Failed to open video %s" % source)
        return False

    totalFrameNumber = cap.get(cv.CV_CAP_PROP_FRAME_COUNT);
    print("INFO:Total frame count=%d" % totalFrameNumber)

    width = cap.get(cv.CV_CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv.CV_CAP_PROP_FPS)
    codec = cap.get(cv.CV_CAP_PROP_FOURCC)
    docvt = cap.get(cv.CV_CAP_PROP_CONVERT_RGB)
    print("INFO:frame size=%dx%d, fps=%d, codec=%s, docvt=%d" %(width, height, fps, codec, docvt))

    if DEBUG:
        cv2.namedWindow("frame",1)

    ret = True
    for i in range(int(totalFrameNumber)):
        dst = "%05d.bmp" % i
        #print("DEBUG: decode frame %s to %s" % (i, dst))
        outputFile = os.path.join(dstpath, dst)

        ret, frame = cap.read()
        if not ret or frame is None:
            print("ERROR:fail to read frame %d" % i)
            break

        if not docvt:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(outputFile, gray)

        if DEBUG:
            cv2.imshow("frame", frame);
            cv2.waitKey(0)

    if DEBUG:
        cv2.destroyWindow("frame")

    return ret


if __name__ == "__main__":
    fList = os.listdir('.')
    curdir = os.getcwd()
    for f in fList:
        fPath = os.path.join(curdir, f)
        if os.path.isdir(fPath):
            print("Skip dir %s" % fPath)
            continue
        factorList = f.strip().split('.')
        if len(factorList) != 2:
            print("Skip invalid file %s" % f)
            continue
        pre,post = factorList
        if post not in ['avi', 'rmvb']:
            print("Skip invalid fmt %s" % post)
            continue

        dstpath = os.path.join(curdir, pre)
        if os.path.isdir(dstpath):
            ans=raw_input("Delete existing output dir %s ?" % output_path)
            if ans in ["Y", "y"]:
                __import__('shutil').rmtree(dstpath)

        if not os.path.isdir(dstpath):
            os.mkdir(dstpath)
        if decodeFile(fPath, dstpath):
            print("Decode file %s -> %s done!" % (fPath, dstpath))

