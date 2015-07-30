
import os
import sys
import copy
import re
import shutil
from localUtils import *
from collections import OrderedDict
import ConfigParser
import numpy as np
import cv2
from cv2 import cv
from math import ceil

gDEBUG_DUMPDIFF = 1
gDEBUG_SHOWDIFF = 0
gDEBUG_SHOWPHASE = 0

gFILTER_POPNOISE=True

gMoving_Direction="VERTICAL"
gROIMargin = 5
gCameraFPS = 1000.0
gResampleFactor = 1
gLCDFPS = 60.0

gLONGFRAME_THRESHOLD_FACTOR = 1.0

gUseColorSpace="BGR"
gPicNumCheckThreshold=-1
gNoiseInfoList = []
gSCALE_TO_DETECT_DIFF=1000
gNoiseBaseInited=False
gNoiseCutOffRatio=0.99
gNoiseDistribution=[]

gVERTICAL_BOX_CNT = 10
gPHASE_THRESHOLD_FACTOR=10;
gPHASE_FILTER_H_WINLEN=3


def getImageInfo(img):
    image_row, image_col = img.shape[0:2]
    ratio_row=[0.2,0.4,0.6,0.8]
    ratio_col=[0.2,0.4,0.6,0.8]
    vals=[]
    for i in ratio_row:
        row=int(i*image_row)
        for j in ratio_col:
            col=int(j*image_col)
            val=img[row, col]
            if len(val.shape) != 0:
                val=np.sum(val)/val.shape[0]
            vals.append(val)
    print("DEBUG: image sample info: %s" % vals)

    image_weight = sum(vals)/len(vals)
    return image_row, image_col, image_weight

def getNoiseThreshold(img):
    global gNoiseInfoList
    image_row, image_col, image_weight=getImageInfo(img)

    #FIXME
    sorted_list = sorted(gNoiseInfoList, key=lambda t:t.weight)
    noisethreshold = max(s.level for s in sorted_list)
    '''
    if len(sorted_list) > 1:
        low = sorted_list[0]
        high = sorted_list[-1]
        if image_weight <= low.weight:
            noisethreshold = low.level
        elif image_weight >= high.weight:
            noisethreshold = high.level
        else:
            noisethreshold = (image_weight - low.weight) * (high.level-low.level)\
                    /(high.weight - low.weight) + low.level
    else:
        noisethreshold = sorted_list[0].level
    '''

    print("DEBUG: image_weight= %s, noisethreshold= %s" % (image_weight, noisethreshold))
    return noisethreshold

def getRawImageDiffFrame(working_dir, img_cur, img_pre, crop_list):
    '''
    get the frame difference between two images. the specified ROI will be cleared. no noise depression or filter was applied.
    the returned value depends on the colorspace used.
        for "GRAY", 1 channel frame will be used.
        for "BGR,YUV,HLS", 3 channels will be used, and the caller may ignore the light channel by itself if necessary.
    '''
    global gUseColorSpace
    #check input files
    if gUseColorSpace in ["BGR","YUV","HLS"]:
        loadtype = 1
    elif gUseColorSpace in ["GRAY"]:
        loadtype = 0

    if img_cur.shape != img_pre.shape:
        print("WARNING:different image dimension, %s, %s, skip" %( img_cur.shape, img_pre.shape))
        return None
    image_row, image_col=img_cur.shape[0:2]

    # first blur on src img
    #notes: ensure NOT inplace ops

    if gUseColorSpace in ["HLS"]:
        img_cur = cv2.cvtColor(img_cur, cv2.COLOR_BGR2HLS)
        img_pre = cv2.cvtColor(img_pre, cv2.COLOR_BGR2HLS)
    elif gUseColorSpace in ["YUV"]:
        img_cur = cv2.cvtColor(img_cur, cv2.COLOR_BGR2YUV)
        img_pre = cv2.cvtColor(img_pre, cv2.COLOR_BGR2YUV)

    imgDiff =cv2.absdiff(img_cur,img_pre)

    #clean ROI
    for region in crop_list:
        print("DEBUG:clear crop region: %s, (%d, %d, %d, %d, %d)" % (region[1], region[0][0], region[0][1], region[0][2], region[0][3], region[0][4]))
        left , top, right, bot, ext = region[0]
        if (left, top, right, bot) != (0,0,0,0):
            sx, sy, ex, ey = calcCropROI([left,top], [right,bot], ext, image_row, image_col)
            if loadtype == 0:
                #notice the w/h sequence
                zeroreg = np.zeros((ey-sy,ex-sx), np.uint8)
                imgDiff[sy:ey, sx:ex] = zeroreg
            elif loadtype == 1:
                zeroreg = np.zeros((ey-sy,ex-sx,3), np.uint8)
                imgDiff[sy:ey, sx:ex, :] = zeroreg

    #show frame diff
    if gDEBUG_SHOWDIFF:
        cv2.imshow("img_cur", img_cur);
        cv2.imshow("imgDiff", imgDiff);
        cv2.waitKey(1)

    return imgDiff


#********************************************************************************************************#
#
# case1:
#       single image switch duration < vsyncperiod
#       visuable stable frame duration = vsyncperiod-imageswitchduration
#       visuable stable long frame duration = 2*vsyncperiod-imageswitchduration > vsyncperiod
#
#
#       vsync        image stable
#         |----------------|                |----------------|                 |----------------|
#         |                |                |                |                 |                |
#         |                |                |                |                 |                |
#         |                |                |                |                 |                |
#----------                ------------------                -------------------                ---
#         ^                ^                ^
#         |<-image switch->|                |
#         |<-----vsync period-------------->|
#                          |<-------------->|
#                   visuable stable frame duration
#
#
#         |----------------|                ..................                 |----------------|
#         |                |                .                .                 |                |
#         |                |                .                .                 |                |
#         |                |                .                .                 |                |
#----------                -----------------------------------------------------                ---
#         ^                ^                                                   ^
#                          |<------visuable stable long frame duration-------->|
#
#
# case2:
#       single image switch duration > vsyncperiod
#       visuable stable frame duration = 0
#       visuable stable long frame duration = 2*vsyncperiod-imageswitchduration < vsyncperiod
#
#       vsync                 single image stable
#         |----------------|----------------|.----------------|
#         |                |   .            |   .             |
#         |                |     .          |     .           |
#         |                |       .        |       .         |
#----------                ---------        ---------         ------
#         ^                         ^                ^
#         |<--single image switch-->|                |
#         |<-vsync period->|
#
#
#         |----------------.               |----------------|.----------------|
#         |                   .            |                |   .             |
#         |                     .          |                |     .           |
#         |                       .        |                |       .         |
#----------                        ---------                ---------         ------
#         |<-single image switch-->|
#                                  |<----->|
#                     visuable stable long frame duration
#
#
# case3:
#       special case: most backfound keep same and only partial image changed
#       single image switch duration < vsyncperiod
#       visuable stable frame duration < vsyncperiod
#       visuable stable long frame duration = 2*vsyncperiod-imageswitchduration < vsyncperiod
#
#         |---.--.---------|               |---.--.---------|
#         |   .     .      |               |   .     .      |
#         |   .       .    |               |   .       .    |
#         |   .         .  |               |   .         .  |
#----------   ............ -----------------   ............ ---------
#             |<-------->|                     |<-------->|
#         single image switch
#                        |<------------------->|
#                     visuable stable frame duration
#
#         |---.--.---------|               |----------------|             |---.--.---------|
#         |   .     .      |               |                |             |   .     .      |
#         |   .       .    |               |                |             |   .       .    |
#         |   .         .  |               |                |             |   .         .  |
#----------   ............ -----------------                ---------------   ............ ----------
#             |<-------->|
#         single image switch
#                        |<-------------------------------------------------->|
#                                   visuable stable long frame duration
#
#
#
#********************************************************************************************************#

def filterFrameDiff(imgDiff, img_cur, img_pre):
    '''
    input the raw image diff,
    output the filtered image, with one plannar(gray)
    '''
    global gUseColorSpace
    #ignore the light channel for YUV,HLS
    if gUseColorSpace == "BGR":
        imgDiff=((imgDiff[:,:,0]+imgDiff[:,:,1]+imgDiff[:,:,2])/3).copy()
    elif gUseColorSpace == "GRAY":
        imgDiff = imgDiff
    elif gUseColorSpace in ["HLS"]:
        imgDiff=((imgDiff[:,:,0]+imgDiff[:,:,2])/2).copy()
    elif gUseColorSpace in ["YUV"]:
        imgDiff=((imgDiff[:,:,1]+imgDiff[:,:,2])/2).copy()

    #FIXME
    noisethreshold = getNoiseThreshold(img_cur)
    print "##@@## use noisethreshold", noisethreshold

    # threshold on diff img
    #imgDiffThred = imgDiff.copy()
    #imgDiffThred = cv2.adaptiveThreshold(imgDiff, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 5)
    #ret, imgDiffThred = cv2.threshold(imgDiff, noisethreshold, 255, cv2.THRESH_BINARY);

    #get the pixel position that under noise base threshold
    #suppress noise base
    '''
    mask = cv2.inRange(imgDiff, np.array((0.0)), np.array((noisethreshold*1.0)))  #filter low<=I<=upper
    imgDiff[mask != 0]=0
    imgDiff_tmp=imgDiff
    '''
    imgDiff_tmp=imgDiff
    #ret, imgDiff_tmp = cv2.threshold(imgDiff, noisethreshold,  255, cv2.THRESH_TOZERO) #val<=noisethreshold will be zero

    #e_diff=cv2.erode(src=imgDiff_tmp,kernel=None,iterations=1)
    #d_diff=cv2.dilate(src=e_diff,kernel=None,iterations=1)
    d_diff = imgDiff_tmp
    return d_diff


##@@##
m=0

def compareFrameDiff(working_dir, img_cur, img_pre, direction, crop_list):
    '''
    get the image difference frame, with the same dimension like src frames
    imgDiff returned in gray/HLS/BGR... mode
    '''
    imgDiff = getRawImageDiffFrame(working_dir, img_cur, img_pre, crop_list)

    imgDiffFilter = filterFrameDiff(imgDiff, img_cur, img_pre)

    #use container() to store image statistics info
    data=Container()
    data.noneZeroCnt = cv2.countNonZero(imgDiffFilter) #number of non-zero pixels
    data.noneZeroSum = np.sum(imgDiffFilter) #sum of pixel values over whole image

    '''
    #TODO:use distribution info to increase precision
    image_row, image_col = imgDiffFilter.shape[0:2]
    ColDistribution = []
    for i in range(image_row):
        NzCnt = cv2.countNonZero(imgDiffFilter[i,:])
        if NzCnt == 0:
            continue
        NzSum = np.sum(imgDiffFilter[i,:])
        ColDistribution.append([i, NzCnt, NzSum])
    data.noneZeroDistribution = ColDistribution;
    '''
    disList = getDistributionList(imgDiffFilter, 255)
    pltLen = getNZLen(disList)

    ##@@##
    global m
    if True: #generate the noise distribution figures
        try:
            import matplotlib
            import matplotlib.pyplot as plt

            plt.figure(1)
            plt.subplot(111)

            global gNoiseDistribution
            bin_avg=gNoiseDistribution
            plt.figure(1)
            #plt.plot(range(1,len(bin_avg)), bin_avg[1:], 'r*-')
            #plt.plot(range(1,pltLen+1), disList[1:pltLen+1], 'b+-')
            print "##@@##1"
            plt.plot(range(1,len(bin_avg)), bin_avg[1:], 'r*-',\
                    range(1,pltLen+1), disList[1:pltLen+1], 'b+-')
            print "##@@##2"
            plt.axis([1,max([getNZLen(bin_avg)+1,pltLen+1]), \
                    min([min(bin_avg[1:]), min(disList[1:pltLen+1])]), max([max(bin_avg[1:]), max(disList[1:pltLen+1])])])
            print "##@@##3"
            plt.title("Noise Distribution")
            plt.xlabel("gray diff(0~255)")
            plt.ylabel("noise distribution")

            tmp_dir = os.path.join(working_dir,'tmpdir')
            if not os.path.isdir(tmp_dir):
                os.mkdir(tmp_dir)
            tmp_file = os.path.join(tmp_dir, "diffdistribution_"+ str(m) + ".png")
            m+=1
            plt.savefig(tmp_file,dpi=80)
            if gDEBUG_SHOWPHASE:
                plt.show() #block ops
            plt.close()
        except Exception,e:
            print("ERROR:generate matplotlib failed, %s" % e)

    return data, imgDiffFilter


def copy_seq(srcdir, dstdir, start_index, file_cnt, prefix=None):
    if prefix is None:
        prefix=""
    generator = PicGenerator(srcdir, ["gif", "bmp", "jpg"])
    for i in range(start_index, start_index+file_cnt):
        ret, aFile = generator.readFilePath(i)
        if not ret:
            print("ERROR:Invalid file index %s" % i)
        dstfile=os.path.join(dstdir, prefix + aFile.split(os.path.sep).pop())
        shutil.copy(aFile, dstfile)
        print("copy frame file: %s -> %s" % ( aFile, dstfile))
    return

def filter_phase(working_dir, OrderMap, image_row, image_col, output_file_phase, cfg, scale_threshold=None):
    '''
    post-process phase-correlation data to generate the phase switch points,
    the results will be output to output_file_phase
    '''

    '''
    generate the switch descriptor, with below format:
    [index, "direction", count, count]
            switch point index, start from 0
            image file name
            switch direction, "up" or "down"
            point count with same sign before switch
            point count with same sign after switch
    '''
    global gCameraFPS
    global gResampleFactor
    global gLCDFPS
    global gLONGFRAME_THRESHOLD_FACTOR
    global gPHASE_THRESHOLD_FACTOR
    global gSCALE_TO_DETECT_DIFF
    global gPHASE_FILTER_H_WINLEN

    gCameraFPS = cfg.get("CameraFPS")
    gResampleFactor = cfg.get("camera_resample_factor")
    gLCDFPS = cfg.get("LCDFPS")
    gPHASE_FILTER_H_WINLEN = cfg.get("PHASE_FILTER_H_WINLEN")
    gSCALE_TO_DETECT_DIFF=cfg.get("SCALE_TO_DETECT_DIFF")
    gPHASE_THRESHOLD_FACTOR = cfg.get("PHASE_THRESHOLD_FACTOR")

    #make a deep copy/backup of the original data before processing
    #Orig_OrderMap = copy.deepcopy(OrderMap)
    image_pnt = image_col*image_row
    scale = scale_threshold if scale_threshold is not None else gSCALE_TO_DETECT_DIFF
    area_scale = image_pnt / (scale)

    #prepare the working list beforehand, for speed optimization
    #modification may be applied
    key_list = OrderMap.keys()
    val_list = OrderMap.values()
    list_len = len(key_list)

    #******************************************************************************#
    #filter out 'single abnormal/sudden change pulse' value
    global gFILTER_POPNOISE
    if gFILTER_POPNOISE:
        filter_win_len = gPHASE_FILTER_H_WINLEN
        idx_list=[]
        for idx in range(list_len):
            left = idx - filter_win_len
            right = idx + filter_win_len
            if 0 <= left < list_len and 0 <= right < list_len:
                #print left, idx, right
                cnt = 0
                for i in range(left, right+1):
                    if i == idx:
                        continue
                    if val_list[i].noneZeroCnt > 0:
                        break
                    cnt += 1
                #if cnt == 2*filter_win_len and val_list[idx].noneZeroCnt > 0:
                if cnt == 2*filter_win_len and val_list[idx].noneZeroCnt > area_scale:
                    idx_list.append(idx)

        print( "\t" + str( [ [key_list[idx], val_list[idx].noneZeroCnt] for idx in idx_list ]))

        print("filter out sudden change point & val:")
        for idx in idx_list:
            print("\t%s" % str( [key_list[idx], val_list[idx].noneZeroCnt]))
            '''
            OrderMap[key_list[idx]].noneZeroCnt = 0
            OrderMap[key_list[idx]].noneZeroSum = 0
            '''
            val_list[idx].noneZeroCnt = 0
            val_list[idx].noneZeroSum = 0

    #******************************************************************************#

    #node.index, index in key_list/val_list
    #node.key, key to loopup OrderMap
    #node.direct, "up" or "down"
    #node.prev_cnt
    #node.next_cnt

    modList = [1 if val.noneZeroCnt>=area_scale else 0 for val in val_list]
    diffList = map(lambda x, y: y-x, modList[0:-1], modList[1:None])
    switchIdxList = [i+1 for i in range(len(diffList)) if diffList[i]!=0]

    switchLenList = map(lambda x, y: y-x, switchIdxList[0:-1], switchIdxList[1:None])
    if switchIdxList:
        switchLenList.insert(0, switchIdxList[0])
        switchLenList.append(len(modList)-switchIdxList[-1])

    switch_node_list = []
    for i,idx in enumerate(switchIdxList):

        preLen = switchLenList[i]
        postLen = switchLenList[i+1]
        key = key_list[idx]

        node = Container()
        if modList[idx] == 1:
            node.direct = "up"
        else:
            node.direct = "down"
        node.index = idx
        node.key = key
        node.prev_cnt = preLen
        node.next_cnt = postLen
        switch_node_list.append(node)

    '''
    #find the phase start/stop points, pay attention to below case:
    ---------    -------------
            |    |
            ------
    '''
    changeidx_list = [i for i in range(len(modList)) if modList[i] == 1]

    phase_start_index=phase_stop_index=0
    if not changeidx_list:
        print "ERROR: no change found in image list"
    elif len(changeidx_list) == 1:
        print "WARNING: only one changing point?"
    else:
        phase_start_index = changeidx_list[0]
        phase_stop_index = changeidx_list[-1]
        print "Phase start index =", phase_start_index
        print "Phase stop index =", phase_stop_index
    start_key = key_list[phase_start_index]
    stop_key = key_list[phase_stop_index]

    print("list all switch points: index, key, direction, prev_cnt, next_cnt.")
    for node in switch_node_list:
        print(" \t %5d, %10s, %10s, %5d, %5d" % (node.index, node.key, node.direct, node.prev_cnt, node.next_cnt))

    tmp_dir = os.path.join(working_dir,'tmpdir')
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)

    #*****************************************************************************#
    print("write switch points to file %s" % output_file_phase)
    # output wrapper
    with open(output_file_phase, 'w') as outFd:
        outFd.write("# phase switch: (imagesize=%dx%d, scale=%d, area_scale=%d, camerafps=%d, phase count threshold=%d)\n" \
                % (image_row, image_col, scale, area_scale, gCameraFPS, gPHASE_THRESHOLD_FACTOR*(gCameraFPS/gResampleFactor)/gLCDFPS))
        outFd.write("@SWITCH\n")
        outFd.write("INDEX, FILE, DIRECTION, PREV_CNT, NEXT_CNT\n")

        for node in switch_node_list:
            outFd.write("%5d, %10s, %10s, %5d, %5d\n" % (node.index, node.key, node.direct, node.prev_cnt, node.next_cnt))

        outFd.write("@END\n\n")

        #*****************************************************************************#
        print("write phase duration to file %s" % output_file_phase)
        outFd.write("@DURATION\n")
        stdStr = "BEGIN_IDX, BEGIN_FILE, END_IDX, END_FILE, PHASE_START_IDX, PHASE_START_FILE, PHASE_STOP_IDX, PHASE_STOP_FILE, FRAMES_CNT, DURATION(ms)"
        outFd.write("%s\n" % stdStr)
        # no switch detected in image sequence
        if phase_start_index != phase_stop_index:
            infoStr = "%5d, %10s, %5d, %10s, %5d, %10s, %5d, %10s, %5d, %5d\n" % (\
                    0, key_list[0], list_len-1, key_list[-1],\
                    phase_start_index, start_key, phase_stop_index, stop_key, \
                    (phase_stop_index-phase_start_index), (phase_stop_index-phase_start_index)*1000/(gCameraFPS/gResampleFactor))
            outFd.write(infoStr)

            print("phase transition:%s" % stdStr)
            print(infoStr)

            #copy the phase start/stop frames to tmpdir for quick review.
            if True:
                print("copy phase start/stop frame")
                srcdir = os.path.join(working_dir,'dst')
                copy_seq(srcdir, tmp_dir, phase_start_index, 1, "PhaseStart_")
                copy_seq(srcdir, tmp_dir, phase_stop_index, 1, "PhaseStop_")

        outFd.write("@END\n\n")


        #*******************************************************************************#
        #
        #write long frame to file, end with blank line
        #
        # below three cases will be taken as phase switch points:
        #
        # 1> switch, down followed by up
        # -----------|         |--------------
        #            |         |
        #            `---------` (gLONGFRAME_THRESHOLD_FACTOR*vsync ~ gPHASE_THRESHOLD_FACTOR*vsync)
        #
        #*******************************************************************************#
        outFd.write("# long frame: index, image, framecount (count threshold=%d ~ %d)\n" \
                % ((gLONGFRAME_THRESHOLD_FACTOR*gCameraFPS/gResampleFactor/gLCDFPS), (gPHASE_THRESHOLD_FACTOR*gCameraFPS/gResampleFactor/gLCDFPS)) )
        outFd.write("@LONGFRAME\n")
        outFd.write("INDEX, FILE, FRAMECOUNT, LF_PNTCNT_THRESHOLD\n")

        DOWN_List = [] #contain all up->down transition points
        LF_List = [] #contain all LF start points
        if len(switch_node_list) != 0:
            if False: #use hardcoding frame num threshold
                longframe_trans_idx = []
                for node in switch_node_list:
                    if (node.direct == "down" and node.next_cnt >= gLONGFRAME_THRESHOLD_FACTOR*(gCameraFPS/gResampleFactor)/gLCDFPS and \
                            node.next_cnt < gPHASE_THRESHOLD_FACTOR*(gCameraFPS/gResampleFactor)/gLCDFPS):
                        outFd.write("%5d, %20s, %5d\n" % (node.index, node.key, node.next_cnt))
                        longframe_trans_idx.append(node[0])
            else: #use adaptive threshold
                sample_cnt = int((gCameraFPS/gResampleFactor)/gLCDFPS)
                for node in switch_node_list[:-1]: #!! skip the last switch point for LF detection
                    if node.direct == "down":
                        #first, get the threshold for each node
                        if node.index > sample_cnt:
                            index_head = range(node.index-sample_cnt, node.index, 1)
                            #index_tail
                        else: #rarely get in
                            index_head = range(0, node.index, 1)
                        if index_head:
                            head_peak = max(val_list[i].noneZeroCnt for i in index_head)
                        else: #TODO: below branch should NEVER go into. as there should be at least one point before the 'switch' point
                            head_peak = val_list[0].noneZeroCnt

                        # FIXME:any other algorigthm?
                        '''
                        for large difference, peak will be high and we'll get a large rate, the final threshold will be about 5;
                        otherwise, for small difference, small peak will be generate a small rate and final threshould will be 16;

                        #experimental result for 1000fps camera
                        rate = head_peak*1.0/area_scale/2
                        rate = rate>10 and 10 or rate
                        LF_CntThreshold = 11*(10-rate)/10 + 5 + 1

                        #abstraction
                        threshold
                        |
                        |std_sample_pnt (16 for 1kfps camera) * coef
                        |.
                        | .
                        |  .
                        |   .
                        |    .
                        |     .
                        |      .
                        |       .
                        |        .....................
                        |       std_sample_pnt*0.3
                        |
                        |
                        |------------------------------------>rate
                                std_rate (10)             1
                        '''

                        std_sample_pnt = gCameraFPS/gResampleFactor/gLCDFPS
                        thre_bottom = std_sample_pnt*0.3
                        thre_top = std_sample_pnt*1.0  #which depends on the response time of LCD.
                        rate = head_peak*1.0/(area_scale) #rate should be larger than 1 if no error happend
                        std_rate = 20
                        if rate >= std_rate:
                            LF_CntThreshold = thre_bottom
                        else:
                            LF_CntThreshold = thre_bottom - (thre_bottom - thre_top)/std_rate*(std_rate-rate)

                        #temporarily record the data for log
                        DOWN_node = copy.deepcopy(node)
                        DOWN_node.LF_AdaptiveThreshold = LF_CntThreshold
                        DOWN_List.append(DOWN_node)

                        #second, get the LF according to threshold
                        if (node.next_cnt >= LF_CntThreshold and node.next_cnt < gPHASE_THRESHOLD_FACTOR*(gCameraFPS/gResampleFactor)/gLCDFPS):
                            outFd.write("%5d, %20s, %5d, %5d\n" % (node.index, node.key, node.next_cnt, int(LF_CntThreshold)))
                            LF_List.append(DOWN_node)
                print( "parse \"down\" index, adaptive threshold:")
                for node in DOWN_List:
                    print( "\t %5d, %10s, %5d, %5d" % (node.index, node.key, node.next_cnt, node.LF_AdaptiveThreshold))
        outFd.write("@END\n")

    #*******************************************************************************#
    #copy the LF sequence to tmpdir for quick review.
    if True:
        srcdir = os.path.join(working_dir,'dst')

        for node in LF_List:
            start_index = node.index
            file_cnt = node.next_cnt
            if file_cnt>100:
                print("WARNING:Skip too many LF file, startindex=%s, cnt=%s, copy only first 100 pics" % ( start_index, file_cnt))
                file_cnt = 100
            copy_seq(srcdir, tmp_dir, start_index, file_cnt, "LF_")

    #*******************************************************************************#
    print("")
    print("%s:" % output_file_phase)
    for line in file(output_file_phase):
        print(line.strip("\n"))
    print("")

    #*******************************************************************************#
    # show/dump phase image
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        '''
        orig_values = Orig_OrderMap.values()
        sum_list = [val.noneZeroSum for val in orig_values]
        pnt_list = [val.noneZeroCnt for val in orig_values]
        '''
        sum_list = [val.noneZeroSum for val in val_list]
        pnt_list = [val.noneZeroCnt for val in val_list]

        plt.figure(1)
        plt.subplot(211)
        plt.plot(range(len(pnt_list)), pnt_list, 'b+-', \
                idx_list,[pnt_list[i] for i in idx_list], 'ro', \
                range(len(pnt_list)), [area_scale for i in pnt_list], 'g-', \
                [LF_List[i].index for i in range(len(LF_List))], [0]*len(LF_List), 'yv', \
                [phase_start_index, phase_stop_index], [0,0], 'm^')
        plt.title("Frame Difference")
        plt.xlabel("frame index")
        plt.ylabel("frame difference")
        plt.legend(("diff", "popnoise", "diff threshold", "long frame", "phase transition"), \
                prop={'size':6})

        plt.subplot(212)
        plt.axis([0,len(pnt_list),-1,area_scale*5])
        plt.plot(range(len(pnt_list)), pnt_list, 'b+-', \
                idx_list,[pnt_list[i] for i in idx_list], 'ro', \
                range(len(pnt_list)), [area_scale for i in pnt_list], 'g-', \
                [LF_List[i].index for i in range(len(LF_List))], [0]*len(LF_List), 'yv', \
                [phase_start_index, phase_stop_index], [0,0], 'm^')
        plt.xlabel("frame index")
        plt.ylabel("frame difference")
        plt.legend(("diff", "popnoise", "diff threshold", "long frame", "phase transition"), \
                prop={'size':6})

        plt.text( len(pnt_list)/10, area_scale*3, "phase start=%d,\nphase stop=%d,\nphase duration=%dms"%(phase_start_index,\
                phase_stop_index, (phase_stop_index-phase_start_index)*1000/(gCameraFPS/gResampleFactor)) )

        tmp_dir = os.path.join(working_dir,'tmpdir')
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
        tmp_file = os.path.join(tmp_dir, "result_phase.png")
        plt.savefig(tmp_file,dpi=80)
        #plt.show() #block ops
        plt.close()

        ##summary all possible LF to output figure
        plt.figure(1)
        plt.subplot(211)
        plt.xlabel("LF distribution")
        plt.ylabel("True")
        left=[node.index for node in LF_List]
        width=[node.next_cnt for node in LF_List]
        height=[1 for node in LF_List]
        plt.bar(left=left, width=width, height=height, color="red")

        plt.subplot(212)
        plt.xlabel("LF index")
        plt.ylabel("LF pic number")
        left=[node.index for node in LF_List]
        width=[2 for node in LF_List]
        height=[node.next_cnt for node in LF_List]
        plt.bar(left=left, width=width, height=height, color="red")

        tmp_file = os.path.join(tmp_dir, "result_LF_summary.png")
        plt.savefig(tmp_file,dpi=80)
        if gDEBUG_SHOWPHASE:
            plt.show() #block ops
        plt.close()

    except Exception,e:
        print("ERROR:generate matplotlib failed, %s" % e)

@TimeTag
def local_detectDiffInDir(working_dir, crop_list, output_file):
    global gMoving_Direction

    direction = gMoving_Direction

    #check input image dir: working_dir/dst/
    input_dir = os.path.join(working_dir, "dst")
    if not os.path.isdir(input_dir):
        raise Exception, "Invalid dir:" + input_dir

    [image_row, image_col]=[0,0]

    OrderMap=OrderedDict()

    #check load type
    global gUseColorSpace
    if gUseColorSpace in ["BGR","YUV","HLS"]:
        loadtype = 1
    elif gUseColorSpace in ["GRAY"]:
        loadtype = 0

    aFile_pre = None

    generator = PicGenerator(input_dir, ["gif", "bmp", "jpg"])
    for aFile,img in ImageQueue(generator, loadtype):
        f=aFile.split(os.path.sep).pop()
        #first file, no comparison, skip
        if aFile_pre is None:
            #OrderMap[f]=None #TODO
            data=Container()
            data.noneZeroCnt=0
            data.noneZeroSum=0
            OrderMap[f]=data
            aFile_pre = aFile
            img_pre = img
            continue
        img_cur = img
        if img_cur is None or img_pre is None:
            raise Exception, "Fail to load file"+ aFile +","+ aFile_pre

        #imgDiff returned in gray mode
        data, imgDiff = compareFrameDiff(working_dir, img_cur, img_pre, direction, crop_list)
        aFile_pre = aFile
        img_pre = img_cur

        if data is None:
            print("WARNING:fail to process %s, %s" % (aFile, aFile_pre))
            continue
        else:
            print("DEBUG:compare frame diff done for %s, %s" %(aFile, aFile_pre))
            OrderMap[f]=data

        if [image_row, image_col] == [0,0]:
            image_row, image_col = imgDiff.shape[0:2]

        #******************************************************************************#
        #for debug
        global gDEBUG_SHOWDIFF
        global gDEBUG_DUMPDIFF
        if gDEBUG_SHOWDIFF or gDEBUG_DUMPDIFF:
            #used to illustrate the difference position, but not quantity
            imgDiffPos = imgDiff.copy()
            imgDiffPos[imgDiff == 0] = 0
            imgDiffPos[imgDiff != 0] = 255

            #show frame diff
            if gDEBUG_SHOWDIFF:
                cv2.imshow("imgDiffPos", imgDiffPos)
                cv2.waitKey(1)

            #dump the frame diff as jpeg
            if gDEBUG_DUMPDIFF:
                tmp_dir = os.path.join(working_dir,'tmpdir')
                if not os.path.isdir(tmp_dir):
                    os.mkdir(tmp_dir)
                filt_file = os.path.join(tmp_dir, os.path.split(f)[1].split('.')[0] + "_filt_diff.jpg")
                ##@@## posi_file = os.path.join(tmp_dir, os.path.split(f)[1].split('.')[0] + "_diff_pos.jpg")
                ##@@## cv2.imwrite(filt_file, imgDiff)
                global m
                posi_file = os.path.join(tmp_dir, "diffdistribution_"+ str(m-1) + "_.png")
                cv2.imwrite(posi_file, imgDiffPos)
        #******************************************************************************#

    ## create output file
    with open(output_file, 'a+') as outFd:
        outFd.write("#frame difference, point count & sum\n")
        outFd.write("@DIFF\n")
        outFd.write("INDEX, FILE, DIFFCOUNT_SUM, DIFFCOUNT_POINT\n")
        #prepare working list beforehand for performance optimization, RO
        ## for (index, (key,value)) in enumerate(OrderMap.items()):
        index_list = range(len(OrderMap))
        key_list = OrderMap.keys()
        val_list = OrderMap.values()
        for index in index_list:
            key = key_list[index]
            value = val_list[index]
            # key, row/col difference cnt, point difference cont
            outFd.write("%5d,  %10s,  %10d,  %10d\n" % (index, key, value.noneZeroSum, value.noneZeroCnt))
        outFd.write("@END\n")
    print("detectDiffDir %s Done!"%input_dir)
    print("Result in %s" % output_file)

    # show/dump phase image
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        #TODO: polish perf
        sum_list = [val.noneZeroSum for val in val_list]
        pnt_list = [val.noneZeroCnt for val in val_list]
        avg_list = [(val.noneZeroCnt !=0 and val.noneZeroSum/val.noneZeroCnt or 0) for val in val_list]

        plt.figure(1)
        plt.subplot(311)
        plt.plot(index_list, pnt_list, 'b+-')
        plt.title("Frame Difference")
        plt.xlabel("frame index")
        plt.ylabel("difference points")

        plt.subplot(312)
        plt.plot(index_list, sum_list, 'b+-')
        plt.xlabel("frame index")
        plt.ylabel("difference sum")

        plt.subplot(313)
        plt.plot(index_list, avg_list, 'b+-')
        plt.xlabel("frame index")
        plt.ylabel("difference avg")

        tmp_dir = os.path.join(working_dir,'tmpdir')
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
        tmp_file = os.path.join(tmp_dir, "frame_diff_overview.png")

        plt.savefig(tmp_file,dpi=80)
        if gDEBUG_SHOWPHASE:
            plt.show() #block ops
        plt.close()
    except Exception,e:
        print("ERROR:generate matplotlib failed, %s" %e)

def getNZLen(vallist):
    for i in range(len(vallist)-1,-1,-1):
        if int(vallist[i])>0:
            break
    return i

def getDistributionList(imgDiff, statistics_bin_num=255):
    #statistics_bin_num should be in range [1,255]
    if statistics_bin_num<1 or statistics_bin_num>255:
        raise Exception, "invalid statistics_bin_num" + str(statistics_bin_num)+", should be in range [1,255]"
    #bin_cnt_list=[0]*statistics_bin_num #eg. 0~9, 10~19, ..., 200~299
    hist =cv2.calcHist([imgDiff], [0], None, [statistics_bin_num], [0.0, 255.0])
    return hist.reshape(-1)

    '''
    steps=range(statistics_bin_num)
    binwidth=int(255/statistics_bin_num)
    image_row, image_col=imgDiff.shape[0:2]
    for i in range(image_row):
        for j in range(image_col):
            region = int(imgDiff[i,j]/binwidth)
            if region > 0:
                bin_cnt_list[region] += 1

    '''
    '''
    for s in steps:
        #print("DEBUG:bin %s, %s" %( 10*s, 10*(s+1)))
        low=s*binwidth+1 #NOTES: here we *skip* 0, which is not noise!
        up=(s+1)*binwidth
        mask = cv2.inRange(imgDiff, np.array((low)), np.array((up)))  #filter low<=I<=upper

        points_cnt=cv2.countNonZero(mask)
        bin_cnt_list[s] += points_cnt
    return bin_cnt_list
    '''

def compareFrameDiffNoise(working_dir, img_cur, img_pre, crop_list, statistics_bin_num):
    '''
    get noise distribution in form of: [0,bin_border), [bin_border, 2*bin_border), [2*bin_border, 3*bin_border), ....
    '''
    global gUseColorSpace

    imgDiff = getRawImageDiffFrame(working_dir, img_cur, img_pre, crop_list)
    image_row, image_col=imgDiff.shape[0:2]

    # second blur on diff img
    if gUseColorSpace == "BGR":
        imgDiff=((imgDiff[:,:,0]+imgDiff[:,:,1]+imgDiff[:,:,2])/3).copy()
    elif gUseColorSpace == "GRAY":
        imgDiff = imgDiff
    elif gUseColorSpace in ["HLS"]:
        imgDiff=((imgDiff[:,:,0]+imgDiff[:,:,2])/2).copy()
    elif gUseColorSpace in ["YUV"]:
        imgDiff=((imgDiff[:,:,1]+imgDiff[:,:,2])/2).copy()

    bin_cnt_list = getDistributionList(imgDiff, statistics_bin_num)
    return bin_cnt_list

def local_detectNoiseBase(calibrate_dir, working_dir, crop_list_input):
    calibrate_list={\
            calibrate_dir:[], \
            #os.path.join(working_dir, "dst"):crop_list_input, \
            }

    noise_info_list = [] # [[obj_weight, obj_noisebase],]
    for input_dir, crop_list in calibrate_list.items():
        #check input image dir: working_dir/dst/
        if not os.path.isdir(input_dir):
            raise Exception, "Invalid dir:" + input_dir

        global gUseColorSpace
        if gUseColorSpace in ["BGR","YUV","HLS"]:
            loadtype = 1
        elif gUseColorSpace in ["GRAY"]:
            loadtype = 0

        global gPicNumCheckThreshold
        picgen = PicGenerator(input_dir, ["gif", "bmp", "jpg"])
        num=picgen.getLen()
        if num < gPicNumCheckThreshold:
            raise Exception, "Not enough src images for frame difference threshold detection, %s, %s" %(num, gPicNumCheckThreshold)
        print("Number of images to check noise: %s"% gPicNumCheckThreshold)

        fList_sorted = [picgen.readFilePath(i)[1] for i in range(0,gPicNumCheckThreshold)]

        step_list=[1] #currently we simply loop the files by step 1
        statistics_bin_num = 255 #range from 1 to 255
        OrderMap_step=OrderedDict()
        OrderMapAvg_step=OrderedDict()
        aFile = None
        for step in step_list:

            OrderMap=OrderedDict()
            fIndex = range(0,gPicNumCheckThreshold-step)
            for i in fIndex[0:-1]: #last index not present here
                aFile_pre, aFile = fList_sorted[i], fList_sorted[i+step]

                cur_e=aFile.split(os.path.sep).pop()
                pre_e=aFile_pre.split(os.path.sep).pop()

                img_cur = cv2.imread(aFile, loadtype);
                img_pre = cv2.imread(aFile_pre, loadtype);
                if img_cur is None or img_pre is None:
                    raise Exception, "Fail to load file"+ aFile +","+ aFile_pre

                statistics_list = compareFrameDiffNoise(working_dir, img_cur, img_pre, crop_list, statistics_bin_num)

                if statistics_list is None:
                    print("ERROR: fail to process %s, %s" %( aFile, aFile_pre))
                    continue
                if statistics_bin_num != len(statistics_list):
                    raise Exception, "unmatched bin length"
                OrderMap[cur_e]=statistics_list
            if aFile is None:
                raise Exception, "Unknown error when process dir"

            #summary all results for output
            bin_sum = [0]*statistics_bin_num
            for key,value in OrderMap.items():
                bin_sum = map(lambda x,y:x+y, bin_sum, value)

            bin_avg = map(lambda n:n*1.0/len(OrderMap), bin_sum)
            print "INFO:Noise distribution for step %d: %s" %(step, value)

            #record result of this round run
            OrderMap_step[step]=OrderMap
            OrderMapAvg_step[step]=bin_avg

            #get image pixel info
            image_row, image_col, image_weight=getImageInfo(cv2.imread(aFile))

            ## create output file
            noise_file = os.path.join(working_dir,r'noisestep_'+ os.path.split(input_dir)[1] + '_' +str(step)+r'.txt')

            with open(noise_file, 'w+') as outFd:
                outFd.write("# noise\n")
                outFd.write("@NOISE\n")
                outFd.write("BIN_BOUNDERY(1~255), PIXEL_COUNT, PERCENTAGE\n")
                bin_step = int(255/statistics_bin_num)
                bin_boundary = range(0+1, 255+1, bin_step)
                for i in range(len(bin_boundary)):
                    outFd.write("%10d,  %10d,  %4d %%\n" % (bin_boundary[i], int(bin_avg[i]), int(bin_avg[i]*100.0/(image_col*image_row))))
                outFd.write("@END\n")
            print("detectNoiseBase %s Done!\nResult in file %s" % (input_dir,noise_file))

            if True: #generate the noise distribution figures
                try:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt

                    plt.figure(1)
                    colorList=['b', 'g', 'r', 'c', 'm', 'y', 'w']

                    valueList=[]
                    for key,value in OrderMap.items():
                        valueList.append(value)

                    pltLen = getNZLen(bin_avg)

                    for i,value in enumerate(valueList):
                        plt.plot(range(1,pltLen+1), value[1:pltLen+1], colorList[i%len(colorList)]+'+-')
                    plt.plot(range(1,pltLen+1), bin_avg[1:pltLen+1], 'k*-', linewidth=4.0)
                    plt.title("Noise Distribution (black for average)")
                    plt.xlabel("gray diff(0~255)")
                    plt.ylabel("noise distribution")

                    tmp_dir = os.path.join(working_dir,'tmpdir')
                    if not os.path.isdir(tmp_dir):
                        os.mkdir(tmp_dir)
                    tmp_file = os.path.join(tmp_dir, "noisedistribution_"+os.path.split(input_dir)[1] + '_' +str(step)+".png")
                    plt.savefig(tmp_file,dpi=80)
                    if gDEBUG_SHOWPHASE:
                        plt.show() #block ops
                    plt.close()
                except Exception,e:
                    print("ERROR:generate matplotlib failed, %s" % e)

            if False: #debug for each step
                try:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    plt.figure(1)
                    plt.plot(bin_boundary, bin_avg, 'b*-')
                    plt.title("Noise")
                    plt.xlabel("statistics bin")
                    plt.ylabel("pixel count")

                    tmp_dir = os.path.join(working_dir,'tmpdir')
                    if not os.path.isdir(tmp_dir):
                        os.mkdir(tmp_dir)
                    tmp_file = os.path.join(tmp_dir, "noise_"+ os.path.split(input_dir)[1] + '_'+str(step)+".png")
                    plt.savefig(tmp_file,dpi=80)
                    if gDEBUG_SHOWPHASE:
                        plt.show() #block ops
                    plt.close()
                except Exception,e:
                    print("ERROR:generate matplotlib failed, %s" % e)

        #****************************************************************************#
        #return the threshold to supress the noise
        step_used_to_update = step_list[0] #TODO
        bin_avg = OrderMapAvg_step[step_used_to_update]
        global gNoiseDistribution
        gNoiseDistribution=bin_avg

        #choose the threshold
        #ensure sum of all points in the bins above threshold should be less than the pixel difference count we wanna detect
        #FIXME: the noise threshold should be with the area_scale used in function filter_phase!
        sum_avg=sum(bin_avg)
        global gNoiseCutOffRatio
        for i in range(len(bin_avg)):
            if (sum(bin_avg[0:i+1])) > image_row*image_col*gNoiseCutOffRatio or bin_avg[i] == 0: # (1-noisecutoffratio) remained noise
                break
        if i == 0:
            i=1
        noise_level = (i)*step_used_to_update

        #show the statictics list used to determine the threshold
        if True: #debug final step used to determind noise
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                bin_step = int(255/statistics_bin_num)
                bin_boundary = range(0+1, 255+1, bin_step)

                plt.figure(1)
                plt.subplot(211)
                plt.plot(range(1,len(bin_avg)), bin_avg[1:], 'b*-')
                plt.title("Noise")
                plt.xlabel("statistics bin")
                plt.ylabel("pixel count")

                plt.subplot(212)
                plt.xlabel("statistics bin")
                plt.ylabel("pixel count")
                pltLen = getNZLen(bin_avg)
                plt.plot(range(1,pltLen+1), bin_avg[1:pltLen+1], 'b*-')

                tmp_dir = os.path.join(working_dir,'tmpdir')
                if not os.path.isdir(tmp_dir):
                    os.mkdir(tmp_dir)
                tmp_file = os.path.join(tmp_dir, "noiseavg_"+os.path.split(input_dir)[1] + '_' +str(step_used_to_update)+".png")
                plt.savefig(tmp_file,dpi=80)
                if gDEBUG_SHOWPHASE:
                    plt.show() #block ops
                plt.close()
            except Exception,e:
                print("ERROR:generate matplotlib failed, %s" %e)

        if noise_level > 30:
            raise Exception, "So noisy?"

        info = Container()
        info.weight = image_weight
        info.level = noise_level
        noise_info_list.append(info)

    return noise_info_list

    ##****************************************************************************##

@TimeTag
def updateNoiseBase(calibrate_dir, working_dir, cfg):
    global gPicNumCheckThreshold

    gPicNumCheckThreshold = cfg.get("PIC_NUM_CHECK_DIFF_THRESHOLD")
    noiseBaseThreshold = cfg.get('NOISE_BASE_THRESHOLD')

    crop_list=[]
    if cfg.has_key("crop_region_list"):
        crop_list=[[node.crop[:], node.desc]  for node in cfg.get("crop_region_list")]

    if gPicNumCheckThreshold > 1: #valid value in range [2, inf]
        noise_info_list = local_detectNoiseBase(calibrate_dir, working_dir, crop_list)
        print("INFO:Use local detected noise base threshold (from %s images):" % gPicNumCheckThreshold)
    else:
        info = Container()
        info.weight = 0
        info.level = noiseBaseThreshold
        noise_info_list = []
        noise_info_list.append(info)
        info = Container()
        info.weight = 255
        info.level = noiseBaseThreshold
        noise_info_list.append(info)
        print("INFO:Use user specified noise base threshold (from config file):")

    for info in noise_info_list:
        print("INFO:\t weight=%s, threshold=%s" % (info.weight, info.level))

    if len(noise_info_list) == 0:
        raise Exception, "Invalid noise base threshold %s" % noise_info_list

    global gNoiseInfoList
    gNoiseInfoList = copy.deepcopy(noise_info_list)

def detectDiffInDir(calibrate_dir, working_dir, cfg, output_file):
    global gMoving_Direction
    global gROIMargin
    global gCameraFPS
    global gResampleFactor
    global gLCDFPS
    global gUseColorSpace
    global gDEBUG_SHOWDIFF
    global gDEBUG_SHOWPHASE
    global gDEBUG_DUMPDIFF

    if cfg.get("GESTURE_TYPE") == "FLING_VERTICAL":
        gMoving_Direction = "VERTICAL"
    elif cfg.get("GESTURE_TYPE") == "FLING_HORIZONTAL":
        gMoving_Direction = "HORIZONTAL"
    elif cfg.get("GESTURE_TYPE") == "FLING_ZOOM":
        gMoving_Direction = "ZOOM"

    gROIMargin = cfg.get("ROIMargin")
    gCameraFPS = cfg.get("CameraFPS")
    gResampleFactor = cfg.get("camera_resample_factor")
    gLCDFPS = cfg.get("LCDFPS")
    gUseColorSpace = cfg.get("COLOR_SPACE")
    #gDEBUG_SHOWDIFF = cfg.get("DEBUG_SHOWDIFF")
    gDEBUG_SHOWPHASE = cfg.get("DEBUG_SHOWPHASE")
    #gDEBUG_DUMPDIFF = cfg.get("DEBUG_DUMPDIFF")

    global gNoiseBaseInited
    if not gNoiseBaseInited:
        updateNoiseBase(calibrate_dir, working_dir, cfg)
        gNoiseBaseInited=True

    if os.path.isfile(output_file):
        os.remove(output_file);

    crop_list=[]
    if cfg.has_key("crop_region_list"):
        crop_list=[[node.crop, node.desc]  for node in cfg.get("crop_region_list")]

    local_detectDiffInDir(working_dir, crop_list, output_file)
    #reset the flag
    gNoiseBaseInited=False

def _parseDiffFile(diff_file):
    OrderMap=OrderedDict()

    KEY_TUPLE=("INDEX", "FILE", "DIFFCOUNT_SUM", "DIFFCOUNT_POINT")
    cont = getContent(diff_file, "@DIFF")
    if cont is None:
        print "ERROR: invalid frame diff file, %s" % diff_file
        return None

    cont_map = getMap(cont)

    sum_list = cont_map['DIFFCOUNT_SUM']
    pnt_list = cont_map['DIFFCOUNT_POINT']
    fList = cont_map['FILE']
    for i,f in enumerate(fList):
        data=Container()
        data.noneZeroCnt=int(pnt_list[i]) #convert the str elements to int before using
        data.noneZeroSum=int(sum_list[i])
        OrderMap[f]=data
    return OrderMap

def calcLTF(working_dir, input_file_diff, output_file_phase, cfg, scale_threshold=None):
    '''
    input_file_diff: the file contains the frame difference result
    output_file_phase: output file contains the LTF calculation result
    scale_threshold: threshold value to calculate LTF, if not specified, it will be read from config
    '''
    input_dir = os.path.join(working_dir, 'dst')
    generator = PicGenerator(input_dir, ["gif", "bmp", "jpg"])
    image_row = image_col = 0
    for aFile,img in ImageQueue(generator):
        img = cv2.imread(aFile)
        image_row, image_col = img.shape[0:2]
        if image_row > 0  and image_col > 0:
            break
    if image_row <= 0 or image_col <= 0:
        print "ERROR: invalid image size info"
        return False
    OrderMap = _parseDiffFile(input_file_diff)
    filter_phase(working_dir, OrderMap, image_row, image_col, output_file_phase, cfg, scale_threshold)
    return True

##****************************************************************************##
def calcDist(pnt1, pnt2):
    x1, y1 = pnt1
    x2, y2 = pnt2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

#roiDiff in gray mode
def checkTouchPoint(roiDiff, roi_cur, roi_picPos, loadtype):
    #copy for debug draw
    roi_raw = roi_cur.copy()
    roi_gray = roiDiff.copy()

    ret, gray_bin = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_OTSU|cv2.THRESH_BINARY)
    allcnt = cv2.countNonZero(gray_bin)

    circles = cv2.HoughCircles(image=gray_bin, method=cv.CV_HOUGH_GRADIENT, dp=2, minDist=5, param1=255, param2=20, minRadius=8, maxRadius=30)
    #cannyimg=cv2.Canny(gray_bin, 100, 255)
    #circles = cv2.HoughCircles(image=cannyimg, method=cv.CV_HOUGH_GRADIENT, dp=2, minDist=5, param1=255, param2=20, minRadius=10, maxRadius=30)

    if circles is None:
        print("DEBUG:no hough circle found")
        return False,allcnt, -1, -1

    circleall = circles[0]
    circle=circleall[0]
    x, y, r = circle

    ##check circle area
    ## empty all other regions around trackWin
    roi_row = gray_bin.shape[0]
    roi_col = gray_bin.shape[1]
    gray_bin_copy = gray_bin.copy()
    mask_gray = np.zeros((roi_row,roi_col), np.uint8)

    cv2.circle(img=mask_gray, center=(x, y), radius=r, color=255, thickness=-1, lineType=8, shift=0)
    #gray_bin_copy[mask_gray == 0]=0
    gray_bin_copy &= mask_gray

    cnt = cv2.countNonZero(gray_bin_copy)
    if (cnt < 3.14*r*r/2):
        print("DEBUG:circle area not match, %s, skip" % cnt)
        return False, allcnt, -1, -1

    if r>50 or r<5:
        print("DEBUG:circle radius not match, %s, skip" % r)
        return False, allcnt, -1, -1

    if False:
        print("DEBUG:detected circle=%s" % circle)
        cv2.circle(roi_raw, (x,y), r, (0,0,255), 2)  # show search rectangle RED
        cv2.imshow("gray_bin", gray_bin)
        cv2.imshow("roi_gray", roi_gray)
        cv2.imshow("touch circle", roi_raw)
        cv2.imshow("gray_bin_masked", gray_bin_copy)
        #cv2.imshow("canny", cannyimg)
        cv2.waitKey(1)

    return True,allcnt, [int(x), int(y)], int(r)

def updateTouchCropList(working_dir, gesture_file, cfg):
    #touch info from gesture file
    if not os.path.isfile(gesture_file):
        print("ERROR:Invalid gesture file, %s" % gesture_file)
        return False

    '''
    single touch: onDown -> onSingleTapUp -> onSingleTapConfirmed
    Fling: onDown -> (onShowPress) -> onScroll -> ... -> onFling
    '''

    global gUseColorSpace
    gUseColorSpace = cfg.get("COLOR_SPACE")
    #check input files
    if gUseColorSpace in ["BGR","YUV","HLS"]:
        loadtype = 1
    elif gUseColorSpace in ["GRAY"]:
        loadtype = 0

    picdir = os.path.join(working_dir,"dst")
    generator = PicGenerator(picdir)
    ret, aFile = generator.readFilePath(0)
    if not ret:
        raise Exception()
    image_row, image_col = cv2.imread(aFile, loadtype).shape[0:2]
    cfg.set('pic_width', image_col)
    cfg.set('pic_height', image_row)

    touchSeq = getTouchSeq(gesture_file)
    print("INFO:get screen based touchSeq from gesture file, %s" % touchSeq)
    touchSeqPic = [screen2pic(cfg, gesturePos) for gesturePos in touchSeq]
    print("INFO:get pic based touchSeq from gesture file, %s" % touchSeqPic)
    touchRadius = 12 #TODO

    cropPntList = []
    #convert to form: [ [[x1,y1],[x2,y2]], [[x2,y2],[x3,y3]], [...], ...]
    compareList = [ [touchSeqPic[i], touchSeqPic[i+1]] for i in range(len(touchSeqPic)-1) ]
    for (pnt1, pnt2) in compareList:
        dist = calcDist(pnt1, pnt2)
        cnt = int(ceil(dist / touchRadius))+1
        for i in range(cnt):
            cropPntList.append( [pnt1[0]+i*(pnt2[0]-pnt1[0])/cnt, pnt1[1]+i*(pnt2[1]-pnt1[1])/cnt] )

    if not cfg.has_key("crop_region_list"):
        cfg.set("crop_region_list", [])
    crop_list=cfg.get("crop_region_list")
    for pnt in cropPntList:
        data=Container()
        data.crop=[int(pnt[0]),int(pnt[1]),int(pnt[0]),int(pnt[1]), touchRadius]
        data.desc="touch_trace"
        crop_list.append(data)
    cfg.set("crop_region_list", crop_list)
    print("INFO: updated crop list based on gesture file, %s" % crop_list)

    if True: ##dump crop region for debug purpose
        tmp_dir = os.path.join(working_dir, "tmpdir")
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
        frame = np.zeros((image_row, image_col), np.uint8)

        #clean ROI
        for data in crop_list:
            print("DEBUG:clear crop region: %s, (%s)" % (data.desc, data.crop))
            left , top, right, bot, ext = data.crop
            if (left, top, right, bot) != (0,0,0,0):
                sx, sy, ex, ey = calcCropROI([left,top], [right,bot], ext, image_row, image_col)
                zeroreg = np.ones((ey-sy,ex-sx), np.uint8)*255
                frame[sy:ey, sx:ex] = zeroreg
        cv2.imwrite(os.path.join(tmp_dir, "cropROIList.png"), frame)
    return True


@TimeTag
def detectTouchTrigger(calibrate_dir, working_dir, gesture_file, cfg, output_file_touch):
    if not os.path.isfile(gesture_file):
        print("ERROR:Invalid gesture file, %s, return" % gesture_file)
        return False

    gesturePos=gestureKeyDown(gesture_file)
    gesturePos=map(int, gesturePos)

    if gesturePos == []:
        print("WARNING:No gesture pos specified, return")
        return False
    print("INFO:detect screen based gesture pos, %s" % gesturePos)

    global gUseColorSpace
    gUseColorSpace = cfg.get("COLOR_SPACE")
    #check input files
    if gUseColorSpace in ["BGR","YUV","HLS"]:
        loadtype = 1
    elif gUseColorSpace in ["GRAY"]:
        loadtype = 0

    global gNoiseBaseInited
    if not gNoiseBaseInited:
        updateNoiseBase(calibrate_dir, working_dir, cfg)
        gNoiseBaseInited=True

    diffList=[]
    touchTriggerFile=None
    OrderMap=OrderedDict()
    picdir = os.path.join(working_dir,"dst")
    aFile_pre = None
    generator = PicGenerator(picdir)
    ## udpate pic size in cfg, which will be used when calculating gesture pos
    ret, aFile = generator.readFilePath(0)
    if not ret:
        raise Exception()
    image_row, image_col = cv2.imread(aFile, loadtype).shape[0:2]
    cfg.set('pic_width', image_col)
    cfg.set('pic_height', image_row)
    picPos = screen2pic(cfg, gesturePos)
    print("INFO:detect pic based gesture pos, %s" % str(picPos))

    generator.resetIndex(0)
    touch_step = int(gCameraFPS/gResampleFactor/gLCDFPS)
    if touch_step <= 1:
        touch_step = 1
        print("ERROR: camera fps too low!")

    detect_failed = False
    valid_touch_list=[]
    radius_avg=0
    center_x_avg=0
    center_y_avg=0
    image_row, image_col=0, 0
    for i in range(0, generator.getLen()-touch_step): #TODO
        ret_pre, aFile_pre = generator.readFilePath(i)
        img_pre            = cv2.imread(aFile_pre, loadtype)
        ret, aFile         = generator.readFilePath(i+touch_step)
        img_cur            = cv2.imread(aFile, loadtype)
        if ret is not True or ret_pre is not True:
            continue

        f_pre = aFile_pre.split(os.path.sep).pop()
        f     = aFile.split(os.path.sep).pop()

        if img_cur is None or img_pre is None:
            raise Exception, "Fail to load file"+ aFile +","+ aFile_pre

        image_row, image_col=img_cur.shape[0:2]
        #slice touch detection roi
        ext = 50 #FIXME
        x,y = picPos
        sx, sy, ex, ey = calcCropROI(picPos, picPos, ext, image_row, image_col)

        if loadtype == 0:
            roi_cur = img_cur[sy:ey, sx:ex]
            roi_pre = img_pre[sy:ey, sx:ex]
        elif loadtype == 1:
            roi_cur = img_cur[sy:ey, sx:ex, :]
            roi_pre = img_pre[sy:ey, sx:ex, :]

        crop_list=[[[0,0,0,0, 0],""],]
        #roiDiff returned in gray mode
        data, roiDiff = compareFrameDiff(working_dir, roi_cur, roi_pre, "up", crop_list)

        if data is None:
            print("WARNING:fail to process %s, %s" % (aFile, aFile_pre))
            continue
        else:
            print("DEBUG:compare frame diff done for %s, %s" %(aFile, aFile_pre))
            OrderMap[f]=data
        #diffList.append([data.noneZeroCnt, data.noneZeroSum])

        if i > touch_step*3:
            print("DEBUG:Check possible touch point on file %s" % f)
            roi_picPos = [x-sx, y-sy]
            ret, allcnt, center, radius = checkTouchPoint(roiDiff, roi_cur, roi_picPos, loadtype)

            if allcnt > roiDiff.shape[0]*roiDiff.shape[1]/2:
                detect_failed = True
                print("WARNING:stop threshold met on file %s, skip touch detection" % (f))
                break
            if not ret:
                continue

            print("DEBUG:double check touch point on file %s, center=%s, radius=%s" % (f,center,radius))
            if len(valid_touch_list)>=1 and valid_touch_list[-1][0] != i-1:
                valid_touch_list=[]
            valid_touch_list.append([i, f, center, radius])
            if len(valid_touch_list) >= touch_step/2 or len(valid_touch_list) >= 5:
                radius_avg = sum([s[3] for s in valid_touch_list])/len(valid_touch_list)
                center_x_avg = sum([s[2][0] for s in valid_touch_list])/len(valid_touch_list)
                center_y_avg = sum([s[2][1] for s in valid_touch_list])/len(valid_touch_list)
                ds=[1 for s in valid_touch_list if abs(s[3]-radius_avg)<radius_avg]
                dc=[1 for s in valid_touch_list if calcDist(s[2], [center_x_avg,center_y_avg]) < radius_avg]
                print("DEBUG:center=(%s, %s), radius_avg=%s"%(center_x_avg, center_y_avg, radius_avg))
                if len(ds) >= len(valid_touch_list) and len(dc) >= len(valid_touch_list):
                    touchTriggerFile = valid_touch_list[0][1]
                    print("INFO:Found touch point on file %s, center=%s, radius=%s" % (touchTriggerFile,[center_x_avg, center_y_avg], radius_avg))
                    break
                else:
                    del valid_touch_list[0]
                    continue

    if touchTriggerFile is None and detect_failed is False:
        print("WARNING:No touch point found after process all files, set fail")
        detect_failed = True

    if not detect_failed:
        multi = 3
        center = (sx+center_x_avg, sy+center_y_avg)
        ext = multi*radius_avg
        crop_x0, crop_y0, crop_x1, crop_y1 = calcCropROI(center, center, ext, image_row, image_col)
        crop_ext=0

        data=Container()
        data.crop=[int(crop_x0), int(crop_y0), int(crop_x1), int(crop_y1), int(crop_ext)]
        data.desc="touch_point"
        if not cfg.has_key("crop_region_list"):
            cfg.set("crop_region_list", [])
        crop_list=cfg.get("crop_region_list")
        crop_list.append(data)
        cfg.set("crop_region_list", crop_list)

    if cfg.get("debug_dumptouch"): #debug
        tmp_dir = os.path.join(working_dir, "tmpdir")
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)

        if detect_failed:
            dump_file = f
        else:
            dump_file=touchTriggerFile

        dump_touch = cv2.imread(os.path.join(picdir,dump_file),1) #NOTES: always use color image to make the touch area obviously.
        cv2.circle(dump_touch, picPos, 1, (0,0,255), 2)  # show search rectangle RED
        cv2.rectangle(dump_touch, (sx,sy), (ex,ey), (0,0,255), 2)

        if not detect_failed:
            cv2.circle(dump_touch, (int(sx+center_x_avg),int(sy+center_y_avg)), int(radius_avg), (0,255,0), 2)  # show detected touch circle GREEN

        print("DEBUG:dump debug touch point to file %s" % os.path.join(tmp_dir, "touch_"+dump_file))
        cv2.imwrite(os.path.join(tmp_dir, "touch_"+dump_file), dump_touch)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        value_list = OrderMap.values()

        sum_list = [val.noneZeroSum for val in value_list]
        pnt_list = [val.noneZeroCnt for val in value_list]

        plt.figure(1)
        plt.plot(range(len(pnt_list)), pnt_list, 'b+-')
        tmp_file = os.path.join(tmp_dir, "touch_detect.png")
        plt.savefig(tmp_file,dpi=80)
        if cfg.get("DEBUG_SHOWPHASE"):
            plt.show()
        plt.close()
    except Exception,e:
        print("ERROR:generate matplotlib failed, %s" % e)

    #*****************************************************************************#
    print("write touch trigger point to file %s" % output_file_touch)
    # output wrapper
    if os.path.isfile(output_file_touch):
        os.remove(output_file_touch)
    with open(output_file_touch, 'w+') as outFd:
        scr_w, scr_h = cfg.get("screen_width"), cfg.get("screen_height")
        outFd.write("# touch trigger, screen_size=%sx%s, pic_size=%sx%s\n" %(scr_w,scr_h, image_col, image_row))  #notice the size sequence
        outFd.write("@TOUCH\n")
        outFd.write("INDEX, FILE, SCREEN_X, SCREEN_Y, PIC_X, PIC_Y\n")

        if not detect_failed:
            outFd.write("%5d, %10s, %5d, %5d, %5d, %5d\n" % (generator.getIndexOfFileName(touchTriggerFile), \
                    touchTriggerFile,gesturePos[0],gesturePos[1],picPos[0],picPos[1]))

        outFd.write("@END\n\n")

    return True

