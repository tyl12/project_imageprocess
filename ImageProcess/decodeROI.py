import os
import sys
import re
from localUtils import *
from collections import OrderedDict
from os import getcwd

import cv2
from cv2 import cv
import numpy as np

gFrame_Ratio=[[0.7647,0.7647],[0.7647,0.2549],[0.2549,0.7647],[0.2549,0.2549]]

enum_gray2dec={
        0b0000:0,
        0b0001:1,
        0b0011:2,
        0b0010:3,
        0b0110:4,
        0b0111:5,
        0b0101:6,
        0b0100:7,
        0b1100:8,
        0b1101:9,
        0b1111:10,
        0b1110:11,
        0b1010:12,
        0b1011:13,
        0b1001:14,
        0b1000:15,
        };

# notes: as in most scenarios, the images will show "white" and the gray value is more 
# sensitive to black, here we just choose 100 instead of 255/2
threshold_graybin=100; #TODO

def decodeFileROI(aFile, src, left, top, right, bot):
    '''
    input: image file, ROI position.
    output: [graycode, decoded decimalcode, bits_weight array]
    '''

    global gFrame_Ratio

    (graycode, decimal, bits_weight)=(0,0,[0])

    #check input parameters
    width  = right-left
    height = bot-top
    if width<=0 or height<=0:
        print("ERROR:Fatal error, invalid rect position")
        return (graycode, decimal, bits_weight)

    #ret, gray_bin = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU|cv2.THRESH_BINARY)
    #roi = gray_bin[top:bot,left:right]

    #only consider the near by region but not whole image, which will decrease the false detection
    image_row, image_col=src.shape[0:2]
    extMargin = right-left
    left_margin, top_margin, right_margin, bot_margin = \
            calcCropROI([left, top], [right, bot], extMargin, image_row, image_col)

    extROI = src[top_margin:bot_margin, left_margin:right_margin]
    ret, extBinROI = cv2.threshold(extROI, 0, 255, cv2.THRESH_OTSU|cv2.THRESH_BINARY)
    roi = extBinROI[top-top_margin:bot-top_margin, left-left_margin:right-left_margin]

    if False:
        cv2.imshow("extBinROI",extBinROI)
        cv2.imshow("roi",roi)
        cv2.waitKey(200)
    sample_points=[]
    for ratio in gFrame_Ratio:
        sample_points.append([int(float(ratio[1])*height),int(float(ratio[0])*width)])

    vals=[]
    for pnt in sample_points:
        [row_idx, col_idx]=pnt
        val=roi[row_idx,col_idx]
        print("DEBUG:point pos=%s, value=%s" %( pnt, roi[row_idx,col_idx]))
        vals.append(val)

    graycode=0
    for val in vals[::-1]:
        if val<threshold_graybin:
            graycode =(graycode<<1)|1;
        else:
            graycode =(graycode<<1)|0;
    decimal = enum_gray2dec[graycode]

    if False:
        debugdir=os.path.join(curdir,"debug")
        if not os.path.isdir(debugdir):
            os.mkdir(debugdir)
        cv2.imwrite(debugdir + "/crop_" + fileName + ".bmp", roi)

    bits_weight=[int(abs(bw - 128)*100/128) for bw in vals]
    print("DEBUG:Decode file %s done, with ROI %s" %( aFile,[left,top,right,bot] ))
    print("DEBUG:\tGray code bit: %s" %( vals[::-1]))
    print("DEBUG:\tDecoded decimal code: %s" % (decimal))
    print("DEBUG:\tbits_weight:" % (bits_weight))
    return (graycode, decimal, bits_weight)


def local_decodeDir(working_dir, left, top, right, bot, output_file):
    print("DEBUG:decode working_dir %s"% working_dir)
    print("DEBUG:[left, top, right, bot]= %s" %([left, top, right, bot]))
    print("DEBUG:output_file %s" % output_file)

    global gFrame_Ratio

    #check input image dir: working_dir/dst/
    input_dir = os.path.join(working_dir, "dst")
    if not os.path.isdir(input_dir):
        raise Exception, "Invalid dir:" + input_dir

    OrderMap=OrderedDict()

    if os.path.isfile(output_file):
        os.remove(output_file);

    generator = PicGenerator(input_dir)
    for aFile, img in ImageQueue(generator, loadtype = 0):
        f=aFile.split(os.path.sep).pop()
        #decode file ROI
        (graycode, decimal, bits_weight)=decodeFileROI(aFile, img, left, top, right, bot)
        OrderMap[f]=[decimal,graycode,bits_weight];

    ## adjust increamental
    key_pre = None
    incr_sum=0
    BitMod=(1<<len(gFrame_Ratio))
    for key,value in OrderMap.items():
        print("DEBUG:Processing point  %s:%s" %( key, value))
        ## filter out jump points
        ## suppose all decoded num is correct with current weight value
        if key_pre is None:
            incr_sum = value[0]
        else:
            print("DEBUG:Adjust increamental point %s:%s" %( key, value[0] ))
            if value[0] < OrderMap[key_pre][0]:
                if (OrderMap[key_pre][0] == BitMod-1 or OrderMap[key_pre][0] == BitMod-2) \
                        and (value[0] == 0 or value[0] == 1):
                            incr_sum += value[0] + BitMod - OrderMap[key_pre][0]
            else:
                incr_sum += value[0] - OrderMap[key_pre][0]

        OrderMap[key][1] = incr_sum
        key_pre = key

    ## create output file
    with open(output_file, 'w+') as outFd:
        outFd.write("# ROI decode debug file\n")
        outFd.write("@FRAMECOUNT\n")
        outFd.write("FILE, DECIMAL_INCR, DECIMAL, WEIGHT\n")
        index_list = range(len(OrderMap))
        key_list = OrderMap.keys()
        val_list = OrderMap.values()
        for idx in index_list:
            # key, increamental num, decimal, weight
            key = key_list[idx]
            value = val_list[idx]
            outFd.write("%10s, %10d, %10d, %10d\n" % (key, value[1], value[0], min(value[2])))
        # key, incremental num, decoded num, weight
        outFd.write("@END\n")

    # show/dump phase image
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        pnt_list=[val[0] for val in val_list]
        plt.plot(index_list, pnt_list, 'o-')

        plt.title("Frame Count")
        plt.xlabel("frame index")
        plt.ylabel("frame count")

        tmp_dir = os.path.join(working_dir,'tmpdir')
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
        tmp_file = os.path.join(tmp_dir, "result_frame.png")
        plt.savefig(tmp_file,dpi=100)
        #plt.show() #block ops
        plt.close()
    except Exception,e:
        print("ERROR:import matplotlib failed, %s" % e)


    if False:
        print("DEBUG:")
        print("DEBUG:%s:"%output_file)
        for line in file(output_file):
            print("DEBUG:"+line.strip("\n"))
        print("DEBUG:")

    print("decodeDir %s done!\n Result in file %s" %( input_dir, output_file ))

@TimeTag
def decodeDir(working_dir, cfg, output_file):
    left    = cfg.get("ROILeft")
    top     = cfg.get("ROITop")
    right   = cfg.get("ROIRight")
    bot     = cfg.get("ROIBot")
    local_decodeDir(working_dir, left, top, right, bot, output_file)

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("ERROR:Usage: python " + sys.argv[0] + " <input dir> <left> <top> <right> <bot> <output file>")
        exit(0)
    local_decodeDir(input_dir, left, top, right, bot, output_file)

