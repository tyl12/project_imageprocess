
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
from localUtils import *
from collections import OrderedDict


gCFG=None
#AUTO_SET_ROI = True
AUTO_SET_ROI = False

gTrackROILeft = 0
gTrackROITop = 0
gTrackROIRight = 0
gTrackROIBot = 0

drag_start= None
sel = (0,0,0,0)
img_sel = None
def onmouse(event, x, y, flags, param):
    global drag_start, sel, img_sel
    if event == cv2.EVENT_LBUTTONDOWN:
        drag_start = x, y
        sel = 0,0,0,0
    elif event == cv2.EVENT_LBUTTONUP:
        drag_start = None
    elif drag_start:
        if flags & cv2.EVENT_FLAG_LBUTTON:
            minpos = min(drag_start[0], x), min(drag_start[1], y)
            maxpos = max(drag_start[0], x), max(drag_start[1], y)
            sel = minpos[0], minpos[1], maxpos[0], maxpos[1]
            img = img_sel.copy()
            cv2.rectangle(img, (sel[0], sel[1]), (sel[2], sel[3]), (0,0,255), 2)
            cv2.imshow("track", img)
            cv2.waitKey(1)
        else:
            print("DEBUG:selection is complete")
            drag_start = None

#
#     -------------------
#    |   ------------    |   2
#    |   |           |   |
#    |   |           |   |
#    |   |           |   |
#    |   |           |   |
#    |   |           |   |   10
#    |   |           |   |
#    |   |           |   |
#    |   |           |   |
#    |   |           |   |
#    |   |-----------|   |
#    |                   |    2
#     -------------------
#      1        8      1
#

def getTrackRegion(input_dir, output_file):
    global img_sel, sel

    picgen=PicGenerator(input_dir, ["bmp","gif","jpeg"])
    ret, aFile = picgen.readFilePath()
    if not ret:
        return (0,0,0,0)
    img_sel = cv2.imread(aFile,1)
    if img_sel is None:
        print("ERROR:Fail to read img from %s" % input_dir)
        return (0,0,0,0)

    if AUTO_SET_ROI is not True:
        cv2.namedWindow("track",1)
        cv2.setMouseCallback("track", onmouse)
        cv2.imshow("track", img_sel)

        while(True):
            if (cv2.waitKey(10) & 255) == 27:
                break
            if sel != (0,0,0,0) and drag_start is None:
                break
        cv2.destroyWindow("track")
    else:
        row, col = img_sel.shape[0:2]
        global gCFG
        direct = gCFG.get("GESTURE_TYPE")
        if  direct == "FLING_UP":
            left  = int(col*1/10)
            right = int(col*9/10)
            top   = int(row*9/14)
            bot   = int(row*12/14)
        elif direct == "FLING_DOWN":
            left  = int(col*1/10)
            right = int(col*9/10)
            top   = int(row*2/14)
            bot   = int(row*5/14)
        elif direct == "FLING_LEFT":
            left  = int(col*7/10)
            right = int(col*9/10)
            top   = int(row*2/14)
            bot   = int(row*12/14)
        elif direct == "FLING_RIGHT":
            left  = int(col*1/10)
            right = int(col*3/10)
            top   = int(row*2/14)
            bot   = int(row*12/14)
        sel = (left, top, right, bot)

    print("Select Track Region(left, top, right, bot): %s" % sel)
    return sel

##**********************************************************************##
#{{{
class ObjTrack(object):
    def __init__(self,left, top, right, bot, input_dir, output_file):
        self.picgen = PicGenerator(input_dir, ["bmp","jpg","gif"])

        cv2.namedWindow('track')
        self.OrderPos=OrderedDict() # filename:[posx,posy]
        self.dir = input_dir
        self.output_file = output_file
        self.trackROI = (left, top, right-left, bot-top)

    def __del__(self):
        cv2.destroyAllWindows()
        self.OrderPos=None
        self.picgen.reset()

    def getTrace(self):
        return self.OrderPos


    # specify the trackWin according to trackROI.
    #
    #        .trackWin(left, top, width, height)
    #        .
    #        .    .trackROI(left, top, width, height)
    #        .    .
    #        .----.----------.
    #        |    .          |
    #        |    .-----.    |
    #        |    |     |    |
    #        |    |_____|    |
    #        |               |
    #        |---------------|       
    #
    #
    def updateTrackWin(self):
        row_extRatio = 0.5
        col_extRatio = 0.5
        roi = self.trackROI
        roi_width = roi[2]
        roi_height = roi[3]

        row_ext = int(row_extRatio * roi_height)
        col_ext = int(col_extRatio * roi_width)

        left, top, right, bot = roi[0], roi[1], roi[0]+roi[2], roi[1]+roi[3]
        left_margin     = (left - col_ext >0) and (left - col_ext) or 0
        top_margin      = (top - row_ext >0) and (top - row_ext) or 0
        right_margin    = (right + col_ext >self.image_col) and (self.image_col) or (right + col_ext)
        bot_margin      = (bot + row_ext >self.image_row) and (self.image_row) or (bot + row_ext)

        self.trackWin   = [left_margin, top_margin, right_margin-left_margin, bot_margin-top_margin]

    def start_with_h_wholeimg(self):#{{{
        isFirst = True
        point_list=[]
        for f in self.picgen:
            self.frame = cv2.imread(f,1)
            if self.frame is None:
                break
            image_row, image_col=self.frame.shape[0:2]
            vis = self.frame.copy()
            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

            #8bit hsv, 0<H<180, 0<S<255, 0<V<255
            mask = cv2.inRange(hsv, np.array((0., 10., 10.)), np.array((180., 255., 255.))) 

            if isFirst:
                roi = self.trackROI
                x0, y0, x1, y1 = roi[0], roi[1], roi[0]+roi[2], roi[1]+roi[3]
                self.track_window = (x0, y0, x1-x0, y1-y0)

                hsv_roi = hsv[y0:y1, x0:x1]
                mask_roi = mask[y0:y1, x0:x1]

                hist = cv2.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX) #min to 0, max to 255(which could be expressed by 1byte)
                self.hist = hist.reshape(-1)

                vis_roi = vis[y0:y1, x0:x1]
                cv2.bitwise_not(vis_roi, vis_roi)
                vis[mask == 0] = 0

                self.image_row = image_row
                self.image_col = image_col
                isFirst = False

            self.updateTrackWin() ##TODO

            prob = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
            prob &= mask
            term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

            ## track_window-> (x, y, width, height)
            track_box, self.track_window = cv2.CamShift(prob, self.track_window, term_crit)

            try:
                cv2.ellipse(vis, track_box, (0, 0, 255), 2)
                mt =self.track_window
                cv2.rectangle(vis, (mt[0], mt[1]), (mt[0]+mt[2], mt[1]+mt[3]), (0,255,0), 2)
                #debug
                cv2.imshow('prob', prob)
            except:
                print(track_box)

            point_list.append(track_box[0])

            cv2.imshow('track', vis)

            ch = 0xFF & cv2.waitKey(5)
            if ch == 27:
                break
        # plot trace result
        try:
            point_x=[i[0] for i in point_list]
            point_y=[image_row - i[1] for i in point_list]
            import matplotlib.pyplot as plt
            plt.plot(point_x,point_y,'r-o')
            plt.show()
        except Exception,e:
            raise Exception()

        cv2.destroyAllWindows()#}}}

    def start_with_h_window(self):#{{{
        isFirst = True
        point_list=[]
        flag_quit = False
        for f in self.picgen:
            self.frame = cv2.imread(f,1)
            if self.frame is None:
                break
            image_row, image_col=self.frame.shape[0:2]
            self.image_row = image_row
            self.image_col = image_col

            vis = self.frame.copy()
            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

            ## filter out pixels with too low S & V
            #8bit hsv, 0<H<180, 0<S<255, 0<V<255; mask will be 255 for bit in range, otherwise 0
            mask = cv2.inRange(hsv, np.array((0., 10., 10.)), np.array((180., 255., 255.))) 

            if isFirst:
                roi = self.trackROI
                x0, y0, x1, y1 = roi[0], roi[1], roi[0]+roi[2], roi[1]+roi[3]
                self.track_window = (x0, y0, x1-x0, y1-y0)
                hsv_roi = hsv[y0:y1, x0:x1]
                mask_roi = mask[y0:y1, x0:x1]

                #TODO:add contour detect before calchist
                hist = cv2.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX) #min to 0, max to 255(which could be expressed by 1byte)
                self.hist = hist.reshape(-1)

                vis_roi = vis[y0:y1, x0:x1]
                cv2.bitwise_not(vis_roi, vis_roi)
                vis[mask == 0] = 0

                isFirst = False

            self.updateTrackWin() ##TODO

            #----------------------------
            ## empty all other regions around trackWin
            mask_win = np.zeros((self.image_row,self.image_col), np.uint8)
            mask_win[self.trackWin[1]:(self.trackWin[1]+self.trackWin[3]),
                    self.trackWin[0]:(self.trackWin[0]+self.trackWin[2])] = np.ones((self.trackWin[3], self.trackWin[2]), np.uint8) * 255

            mask &= mask_win
            #----------------------------

            ##calcBackProject(images, channels, hist, ranges, scale)
            ##  images: input image arrays, could have multi-channels
            ##  channels: count from 0
            ##  hist: input hist
            ##  ranges: Array of arrays of the histogram bin boundaries in each dimension
            prob = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
            prob &= mask
            term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
            ## track_window-> (x, y, width, height)
            track_box, self.track_window = cv2.CamShift(prob, tuple(self.trackWin), term_crit)

            self.trackROI =[int(track_box[0][0] - self.trackROI[2]/2),
                    int(track_box[0][1] - self.trackROI[3]/2),
                    self.trackROI[2],
                    self.trackROI[3]]

            try:
                #debug: show mask window
                test = vis.copy()
                test[mask == 0] = 0
                cv2.imshow("mask",test)
                #show track box from camshift
                cv2.ellipse(vis, track_box, (0, 0, 255), 2)
                #show trackWin
                mt =self.trackWin
                cv2.rectangle(vis, (mt[0], mt[1]), (mt[0]+mt[2], mt[1]+mt[3]), (0,255,0), 2)
                #show trackROI
                mt =self.trackROI
                cv2.rectangle(vis, (mt[0], mt[1]), (mt[0]+mt[2], mt[1]+mt[3]), (255,0,0), 2)
                #debug
                cv2.imshow('prob', prob)
            except:
                print("ERROR:"+str(track_box))

            point_list.append(track_box[0])

            cv2.imshow('track', vis)

            ch = 0xFF & cv2.waitKey(5)
            if ch == 27:
                flag_quit = True
                break

        # plot trace result
        if not flag_quit:
            try:
                point_x=[i[0] for i in point_list]
                point_y=[image_row - i[1] for i in point_list]
                import matplotlib.pyplot as plt
                plt.plot(point_x,point_y,'r-o')
                plt.show()
            except Exception,e:
                raise Exception()
        cv2.destroyAllWindows()#}}}

    def start_with_patch(self):#{{{
        isFirst = True
        point_list=[]
        for f in self.picgen:
            self.frame = cv2.imread(f,1)
            if self.frame is None:
                break
            image_row, image_col=self.frame.shape[0:2]
            self.image_row = image_row
            self.image_col = image_col

            vis = self.frame.copy()
            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

            #8bit hsv, 0<H<180, 0<S<255, 0<V<255
            mask = cv2.inRange(hsv, np.array((0., 10., 10.)), np.array((180., 255., 255.))) 

            if isFirst:
                roi = self.trackROI
                x0, y0, x1, y1 = roi[0], roi[1], roi[0]+roi[2], roi[1]+roi[3]
                self.track_window = (x0, y0, x1-x0, y1-y0)
                hsv_roi = hsv[y0:y1, x0:x1]
                mask_roi = mask[y0:y1, x0:x1]

                #TODO:add contour detect before calchist
                hist = cv2.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX) #min to 0, max to 255(which could be expressed by 1byte)
                self.hist = hist.reshape(-1)

                vis_roi = vis[y0:y1, x0:x1]
                cv2.bitwise_not(vis_roi, vis_roi)
                vis[mask == 0] = 0

                isFirst = False

            self.updateTrackWin() ##TODO

            ## cv.CalcBackProjectPatch(images, dst, patch_size, hist, method, factor) -> None
            ##cv.ShowImage("ttt", cv.fromarray(prob))
            #prob = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
            prob = hsv.copy()
            backproject = cv.CreateImage(cv.GetSize(cv.fromarray(hsv)), 8, 1)
            cv.CalcBackProjectPatch(cv.fromarray(hsv), prob, backproject, self.hist, cv.CV_COMP_CORREL, 1)

            prob &= mask
            term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
            track_box, self.track_window = cv2.CamShift(prob, self.track_window, term_crit)

            try:
                cv2.ellipse(vis, track_box, (0, 0, 255), 2)
                cv2.imshow('track', vis)
                #debug
                cv2.imshow('prob', prob)
            except:
                print("ERROR:"+str(track_box))

            point_list.append(track_box[0])

            ch = 0xFF & cv2.waitKey(5)
            if ch == 27:
                break
        # plot trace result
        try:
            point_x=[i[0] for i in point_list]
            point_y=[image_row - i[1] for i in point_list]
            import matplotlib.pyplot as plt
            plt.plot(point_x,point_y,'r-o')
            plt.show()
        except Exception,e:
            raise Exception()

        cv2.destroyAllWindows()#}}}

    def start_with_hs_window(self):#{{{
        isFirst = True
        point_list=[]
        flag_quit = False
        for f in self.picgen:
            self.frame = cv2.imread(f,1)
            if self.frame is None:
                break
            image_row, image_col=self.frame.shape[0:2]
            self.image_row = image_row
            self.image_col = image_col

            vis = self.frame.copy()
            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

            ## filter out pixels with too low S & V
            #8bit hsv, 0<H<180, 0<S<255, 0<V<255; mask will be 255 for bit in range, otherwise 0
            mask = cv2.inRange(hsv, np.array((0., 10., 10.)), np.array((180., 255., 255.))) 

            if isFirst:
                roi = self.trackROI
                x0, y0, x1, y1 = roi[0], roi[1], roi[0]+roi[2], roi[1]+roi[3]
                self.track_window = (x0, y0, x1-x0, y1-y0)
                hsv_roi = hsv[y0:y1, x0:x1]
                mask_roi = mask[y0:y1, x0:x1]

                #TODO:add contour detect before calchist
                hist_hs = cv2.calcHist( [hsv_roi], [0,1], mask_roi, [16,16], [0,180, 0, 255] )

                cv2.normalize(hist_hs, hist_hs, 0, 255, cv2.NORM_MINMAX) #min to 0, max to 255(which could be expressed by 1byte)

                self.hist_hs = hist_hs

                vis_roi = vis[y0:y1, x0:x1]
                cv2.bitwise_not(vis_roi, vis_roi)
                vis[mask == 0] = 0

                isFirst = False

            self.updateTrackWin() ##TODO

            #----------------------------
            ## empty all other regions around trackWin
            mask_win = np.zeros((self.image_row,self.image_col), np.uint8)
            mask_win[self.trackWin[1]:(self.trackWin[1]+self.trackWin[3]),
                    self.trackWin[0]:(self.trackWin[0]+self.trackWin[2])] = np.ones((self.trackWin[3], self.trackWin[2]), np.uint8) * 255

            mask &= mask_win
            #----------------------------

            ##calcBackProject(images, channels, hist, ranges, scale)
            ##  images: input image, could have multi-channels
            ##  channels: count from 0
            ##  hist: input hist
            ##  ranges: Array of arrays of the histogram bin boundaries in each dimension
            prob_hs = cv2.calcBackProject([hsv], [0,1], self.hist_hs, [0, 180, 0, 255], 2)

            prob_hs &= mask
            term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
            ## track_window-> (x, y, width, height)
            track_box, self.track_window = cv2.CamShift(prob_hs, tuple(self.trackWin), term_crit)

            self.trackROI =[int(track_box[0][0] - self.trackROI[2]/2),
                    int(track_box[0][1] - self.trackROI[3]/2),
                    self.trackROI[2],
                    self.trackROI[3]]

            try:
                #debug: show mask window
                test = vis.copy()
                test[mask == 0] = 0
                cv2.imshow("mask",test)
                #show track box from camshift
                cv2.ellipse(vis, track_box, (0, 0, 255), 2)
                #show trackWin
                mt =self.trackWin
                cv2.rectangle(vis, (mt[0], mt[1]), (mt[0]+mt[2], mt[1]+mt[3]), (0,255,0), 2)
                #show trackROI
                mt =self.trackROI
                cv2.rectangle(vis, (mt[0], mt[1]), (mt[0]+mt[2], mt[1]+mt[3]), (255,0,0), 2)
                #debug
                cv2.imshow('prob_hs', prob_hs)
            except:
                print("ERROR:"+str(track_box))

            point_list.append(track_box[0])

            cv2.imshow('track', vis)

            ch = 0xFF & cv2.waitKey(5)
            if ch == 27:
                flag_quit = True
                break

        # plot trace result
        if not flag_quit:
            try:
                point_x=[i[0] for i in point_list]
                point_y=[image_row - i[1] for i in point_list]
                import matplotlib.pyplot as plt
                plt.plot(point_x,point_y,'r-o')
                plt.show()
            except Exception,e:
                raise Exception()
        cv2.destroyAllWindows()#}}}

    def start_with_hsv_window(self):#{{{
        isFirst = True
        point_list=[]
        flag_quit = False
        for f in self.picgen:
            self.frame = cv2.imread(f,1)
            if self.frame is None:
                break
            image_row, image_col=self.frame.shape[0:2]
            self.image_row = image_row
            self.image_col = image_col

            vis = self.frame.copy()
            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

            ## filter out pixels with too low S & V
            #8bit hsv, 0<H<180, 0<S<255, 0<V<255; mask will be 255 for bit in range, otherwise 0
            mask = cv2.inRange(hsv, np.array((0., 10., 10.)), np.array((180., 255., 255.))) 

            if isFirst:
                roi = self.trackROI
                x0, y0, x1, y1 = roi[0], roi[1], roi[0]+roi[2], roi[1]+roi[3]
                self.track_window = (x0, y0, x1-x0, y1-y0)
                hsv_roi = hsv[y0:y1, x0:x1]
                mask_roi = mask[y0:y1, x0:x1]

                #TODO:add contour detect before calchist
                hist_hsv = cv2.calcHist( [hsv_roi], [0,1,2], mask_roi, [16,16,8], [0,180, 0, 255, 0, 255] )

                cv2.normalize(hist_hsv, hist_hsv, 0, 255, cv2.NORM_MINMAX) #min to 0, max to 255(which could be expressed by 1byte)

                self.hist_hsv = hist_hsv

                vis_roi = vis[y0:y1, x0:x1]
                cv2.bitwise_not(vis_roi, vis_roi)
                vis[mask == 0] = 0

                isFirst = False

            self.updateTrackWin() ##TODO

            #----------------------------
            ## empty all other regions around trackWin
            mask_win = np.zeros((self.image_row,self.image_col), np.uint8)
            mask_win[self.trackWin[1]:(self.trackWin[1]+self.trackWin[3]),
                    self.trackWin[0]:(self.trackWin[0]+self.trackWin[2])] = np.ones((self.trackWin[3], self.trackWin[2]), np.uint8) * 255

            mask &= mask_win
            #----------------------------

            ##calcBackProject(images, channels, hist, ranges, scale)
            ##  images: input image, could have multi-channels
            ##  channels: count from 0
            ##  hist: input hist
            ##  ranges: Array of arrays of the histogram bin boundaries in each dimension
            prob_hsv = cv2.calcBackProject([hsv], [0,1,2], self.hist_hsv, [0, 180, 0, 255, 0, 255], 1)

            prob_hsv &= mask
            term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
            ## track_window-> (x, y, width, height)
            track_box, self.track_window = cv2.CamShift(prob_hsv, tuple(self.trackWin), term_crit)

            self.trackROI =[int(track_box[0][0] - self.trackROI[2]/2),
                    int(track_box[0][1] - self.trackROI[3]/2),
                    self.trackROI[2],
                    self.trackROI[3]]

            try:
                #debug: show mask window
                test = vis.copy()
                test[mask == 0] = 0
                cv2.imshow("mask",test)
                #show track box from camshift
                cv2.ellipse(vis, track_box, (0, 0, 255), 2)
                #show trackWin
                mt =self.trackWin
                cv2.rectangle(vis, (mt[0], mt[1]), (mt[0]+mt[2], mt[1]+mt[3]), (0,255,0), 2)
                #show trackROI
                mt =self.trackROI
                cv2.rectangle(vis, (mt[0], mt[1]), (mt[0]+mt[2], mt[1]+mt[3]), (255,0,0), 2)
                #debug
                cv2.imshow('prob_hsv', prob_hsv)
            except:
                print("ERROR:"+str(track_box))

            point_list.append(track_box[0])

            cv2.imshow('track', vis)

            ch = 0xFF & cv2.waitKey(5)
            if ch == 27:
                flag_quit = True
                break

        # plot trace result
        if not flag_quit:
            try:
                point_x=[i[0] for i in point_list]
                point_y=[image_row - i[1] for i in point_list]
                import matplotlib.pyplot as plt
                plt.plot(point_x,point_y,'r-o')
                plt.show()
            except Exception,e:
                raise Exception()
        cv2.destroyAllWindows()#}}}

    def start_with_corre_window_gray(self):#{{{
        isFirst = True
        point_list=[]
        flag_quit = False
        for f in self.picgen:
            self.frame = cv2.imread(f,0)
            if self.frame is None:
                break
            image_row, image_col=self.frame.shape[0:2]
            self.image_row = image_row
            self.image_col = image_col

            vis = self.frame.copy()

            if isFirst:
                roi = self.trackROI
                x0, y0, x1, y1 = roi[0], roi[1], roi[0]+roi[2], roi[1]+roi[3]
                self.frameROI = self.frame[y0:y1, x0:x1]

                isFirst = False

            self.updateTrackWin() ##TODO

            win = self.trackWin
            x0, y0, x1, y1 = win[0], win[1], win[0]+win[2], win[1]+win[3]
            self.frameWin = self.frame[y0:y1, x0:x1]

            #----------------------------
            ## empty all other regions around trackWin
            mask_win = np.zeros((self.image_row,self.image_col), np.uint8)
            mask_win[self.trackWin[1]:(self.trackWin[1]+self.trackWin[3]),
                    self.trackWin[0]:(self.trackWin[0]+self.trackWin[2])] = np.ones((self.trackWin[3], self.trackWin[2]), np.uint8) * 255
            #----------------------------
            result = cv2.matchTemplate(self.frameWin, self.frameROI, cv2.TM_CCOEFF_NORMED)
            result8 = cv2.normalize(result,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
            cv2.imshow("result", result8)

            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)

            self.trackROI =[int(self.trackWin[0]+maxLoc[0]),
                    int(self.trackWin[1]+maxLoc[1]),
                    self.trackROI[2],
                    self.trackROI[3]]

            vis=self.frame.copy()

            try:
                #show trackWin
                mt =self.trackWin
                cv2.rectangle(vis, (mt[0], mt[1]), (mt[0]+mt[2], mt[1]+mt[3]), 1, 2)
                #show trackROI
                mt =self.trackROI
                cv2.rectangle(vis, (mt[0], mt[1]), (mt[0]+mt[2], mt[1]+mt[3]), 255, 2)
                cv2.imshow('track', vis)
            except:
                print("ERROR:"+str(track_box))

            mt =self.trackROI
            point_list.append([int(mt[0]+mt[2]/2), int(mt[1]+mt[3]/2)])

            ch = 0xFF & cv2.waitKey(5)
            if ch == 27:
                flag_quit = True
                break

        # plot trace result
        if not flag_quit:
            try:
                point_x=[i[0] for i in point_list]
                point_y=[image_row - i[1] for i in point_list]
                import matplotlib.pyplot as plt
                plt.plot(point_x,point_y,'r-o')
                plt.show()
            except Exception,e:
                raise Exception()
        cv2.destroyAllWindows()#}}}
#}}}

#{{{
class ScrollMotion(object):
    def __init__(self,left, top, right, bot, input_dir, output_file):
        print "##@@##", left, top, right, bot
        self.picgen = PicGenerator(input_dir, ["bmp","jpg","gif"])
        self.OrderPos=OrderedDict() # filename:[posx,posy]
        self.dir = input_dir
        self.output_file = output_file
        self.initialROI = (left, top, right-left, bot-top)

    def __del__(self):
        cv2.destroyAllWindows()
        self.OrderPos=None
        self.picgen.reset()

    def getMotion(self):
        return self.OrderPos

    # specify the trackWin according to trackROI.
    #
    #        .trackWin(left, top, width, height)
    #        .
    #        .    .trackROI(left, top, width, height)
    #        .    .
    #        .----.----------.
    #        |    .          |
    #        |    .-----.    |
    #        |    |     |    |
    #        |    |_____|    |
    #        |               |
    #        |---------------|       
    #
    #
    def updateMatchWin(self):
        row_extRatio = 6
        col_extRatio = 0.5
        roi = self.matchedROI
        roi_width = roi[2]
        roi_height = roi[3]

        row_ext = int(row_extRatio * roi_height)
        col_ext = int(col_extRatio * roi_width)

        left, top, right, bot = roi[0], roi[1], roi[0]+roi[2], roi[1]+roi[3]
        left_margin     = (left - col_ext >0) and (left - col_ext) or 0
        top_margin      = (top - row_ext >0) and (top - row_ext) or 0
        right_margin    = (right + col_ext >self.image_col) and (self.image_col) or (right + col_ext)
        bot_margin      = (bot + row_ext >self.image_row) and (self.image_row) or (bot + row_ext)

        self.trackWin   = [left_margin, top_margin, right_margin-left_margin, bot_margin-top_margin]
        self.frameWin   = self.frame[top_margin:bot_margin, left_margin:right_margin].copy() #copy out

    def updateTemplate(self):
        # select pos for next ROI search
        roi = self.initialROI
        x0, y0, x1, y1 = roi[0], roi[1], roi[0]+roi[2], roi[1]+roi[3]
        self.frameTemplate = self.frame[y0:y1, x0:x1].copy() #copy out but not refer

    def run(self):#{{{
        #final data container
        data=Container()
        data.file=[]
        data.delta_x=[]
        data.delta_y=[]
        data.pos_x=[]
        data.pos_y=[]
        data.correlation_coef=[]

        isFirst = True
        flag_quit = False
        retry = 0
        for f in self.picgen:
            self.frame = cv2.imread(f,0)
            if self.frame is None:
                break
            image_row, image_col=self.frame.shape[0:2]
            self.image_row, self.image_col = image_row, image_col

            src = self.frame
            ret, gray_bin = cv2.threshold(src, 10, 255, cv2.THRESH_OTSU|cv2.THRESH_TOZERO)
            self.frame=gray_bin

            if isFirst:
                #self.trackROI = copy.deepcopy(self.initialROI)
                self.pre_matchedROI = copy.deepcopy(self.initialROI)
                self.matchedROI = copy.deepcopy(self.initialROI)
                self.updateTemplate() # select pos for next ROI search
                isFirst = False

            self.updateMatchWin() # select window pos to be searched, based on previous matchedROI

            # do template match
            result = cv2.matchTemplate(self.frameWin, self.frameTemplate, cv2.TM_CCOEFF_NORMED)
            result8 = cv2.normalize(result,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
            cv2.imshow("result", result8)

            #if maxVal < 0.5:
            #    if retry >=3:
            #        retry = 0
            #        self.updateTemplate()
            #    else:
            #        retry += 1
            #    continue
            #else:
            #    retry = 0

            self.matchedROI =[int(self.trackWin[0]+maxLoc[0]),
                    int(self.trackWin[1]+maxLoc[1]),
                    self.matchedROI[2],
                    self.matchedROI[3]]
            lastTemplate = self.frameTemplate.copy() # for display

            #parcel data. notes, row/col count from left-top
            data.file.append(name)
            data.delta_x.append(self.matchedROI[0]-self.pre_matchedROI[0])
            data.delta_y.append(self.pre_matchedROI[1]-self.matchedROI[1])
            data.correlation_coef.append(maxVal)

            if self.matchedROI[1] > self.image_row *1.0/3 and maxVal > 0.5:
                #ROI following state, donot update template
                self.pre_matchedROI = self.matchedROI
                #print "follow"
            else:
                #need to reset ROI, update template
                self.updateTemplate() # select pos for next ROI search based on matched motionROI
                self.pre_matchedROI = copy.deepcopy(self.initialROI)
                #print "reset"

            try:
                global gCFG
                debug_showmotion = gCFG.get("DEBUG_SHOWMOTION")
                if debug_showmotion:
                    #show lastTemplate
                    cv2.imshow('lastTemplate', lastTemplate)

                    vis=self.frame.copy()
                    #show trackWin
                    mt =self.trackWin
                    cv2.rectangle(vis, (mt[0], mt[1]), (mt[0]+mt[2], mt[1]+mt[3]), 100, 2)

                    #show matched motionROI
                    mt =self.matchedROI
                    cv2.rectangle(vis, (mt[0], mt[1]), (mt[0]+mt[2], mt[1]+mt[3]), 255, 2)
                    cv2.imshow('track', vis)
            except:
                print("ERROR:"+str(track_box))

            ch = 0xFF & cv2.waitKey(2)
            if ch == 27:
                flag_quit = True
                break

        #prepare the output date
        sx, sy=0,0
        for i in data.delta_x:
            sx+=i
            data.pos_x.append(sx)
        for i in data.delta_y:
            sy+=i
            data.pos_y.append(sy)

        # plot trace result
        if not flag_quit:
            try:
                import matplotlib.pyplot as plt

                index = [i for i in range(len(data.file))]
                entry_list=[
                        # frame index for x axis, y axis, saved png file prefix name, title, x label, y label
                        [r'index', r'data.pos_x'           , r'track_t_x'     , r'frame index', r'position x']              ,
                        [r'index', r'data.pos_y'           , r'track_t_y'     , r'frame index', r'position y']              ,
                        [r'index', r'data.delta_x'         , r'track_t_deltax', r'frame index', r'delta x']                 ,
                        [r'index', r'data.delta_y'         , r'track_t_deltay', r'frame index', r'delta y']                 ,
                        [r'index', r'data.correlation_coef', r'track_t_coef'  , r'frame index', r'correltation coefficient']
                        ]
                for i in range(len(entry_list)):
                    ent= entry_list[i]
                    ##plot results
                    plt.figure(i)
                    eval("plt.plot(" + ent[0] +','+ ent[1] + ',' + '\'r-o\'' + ")")

                    plt.title(ent[2])
                    plt.xlabel(ent[3])
                    plt.ylabel(ent[4])

                    ##save images
                    tmp_file = os.path.join(os.getcwd(), ent[2]+'.png')
                    plt.savefig(tmp_file,dpi=80)

                debug_showmotion = gCFG.get("DEBUG_SHOWMOTION")
                if debug_showmotion:
                    plt.show() #block ops

                plt.close()

            except Exception,e:
                raise Exception()

        #save the result to output file
        outFd = open(self.output_file, 'w+')
        outFd.write("# motion track using template match")
        outFd.write("@TRACK\n")
        outFd.write("INDEX, FILE, POS_X, POS_Y, DELTA_X, DELTA_Y, CORRELATION_COEF\n")
        for i in range(len(data.file)):
            outFd.write("%5d, %15s, %5d, %5d, %5d, %5d, %5f\n" % (i, data.file[i], data.pos_x[i], data.pos_y[i], data.delta_x[i], data.delta_y[i], data.correlation_coef[i]))
        outFd.write("@END\n\n")

        cv2.destroyAllWindows()#}}}
#}}}

@TimeTag
def local_trackInDir(input_dir, left, top, right, bot, output_file,cfg):

    global gCFG
    gCFG = cfg

    if gCFG.get("GESTURE_TYPE") in ["DRAG"]:
        #track one object over background
        track = ObjTrack(left, top, right, bot, input_dir, output_file)
        track.start_with_hs_window()
        # #track.start_with_h_window()
        # #track.start_with_h_wholeimg()
        #track.start_with_corre_window_gray()
        # ##track.start_with_hsv_window()

    if gCFG.get("GESTURE_TYPE") in ["FLING_UP", "FLING_DOWN", "FLING_LEFT", "FLING_RIGHT"]:
        track = ScrollMotion(left, top, right, bot, input_dir, output_file)
        track.run()


def trackInDir(working_dir, cfg, output_file):
    global gROIMargin
    global gCameraFPS
    global gLCDFPS
    global gUseColorSpace
    global gDiffThreshold
    global gLONGFRAME_THRESHOLD_FACTOR
    global gPHASE_THRESHOLD_FACTOR
    global gDEBUG_SHOWDIFF
    global gDEBUG_SHOWPHASE
    global gDEBUG_DUMPPHASE
    global gPHASE_FILTER_H_WINLEN

    #specify the track region by config file or manually
    try:
        left    = cfg.get("TrackROILeft")
        top     = cfg.get("TrackROITop")
        right   = cfg.get("TrackROIRight")
        bot     = cfg.get("TrackROIBot")
    except Exception,e:
        left, top, right, bot = 0,0,0,0
        print("Specify the track region manually, %s" % e)

    os.chdir(working_dir)

    #check input image dir: working_dir/dst/
    input_dir = os.path.join(working_dir, "dst")
    if not os.path.isdir(input_dir):
        raise Exception, "Invalid dir:" + input_dir

    print "##@@##", left, right
    if left == 0 and top == 0 and right == 0 and bot == 0:
        left, top, right, bot = getTrackRegion(input_dir, output_file)
    print "##@@##", left, right

    local_trackInDir(input_dir, left, top, right, bot, output_file, cfg)


