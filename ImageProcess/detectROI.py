import os
import sys
import re
from localUtils import *
import ConfigParser
import numpy as np
import cv2
from cv2 import cv


(left_Ratio_Min,top_Ratio_Min)=(0,0)
(right_Ratio_Max,bot_Ratio_Max)=(0,0)

#filter the ROI region with farthest distance, it's just one assumption on surface display mode
def checkSquareAndSort(squares):
    if len(squares) == 0:
        return (0,0,0,0)
    idx=0
    dist_max=0;
    i=0
    for squ in squares:
        dist_L=max([(p[0]+p[1]) for p in squ])
        if dist_L>=dist_max:
            dist_max=dist_L
            idx=i
        i += 1
    rect=squares[idx]

    #pnts_len=([(p[0]+p[1]) for p in rect])
    #[left, top] = rect[ pnts_len.index(min(pnts_len)) ]
    #[right, bot] = rect[ pnts_len.index(max(pnts_len)) ]

    pnts_x  = ([p[0] for p in rect])
    pnts_y  = ([p[1] for p in rect])
    left    = min(pnts_x)
    top     = min(pnts_y)
    right   = max(pnts_x)
    bot     = max(pnts_y)

    print("DEBUG:Farthest rect=%s" % rect)
    return (left, top, right, bot)

def find_file_ROI(aFile, src):
    global left_Ratio_Min, top_Ratio_Min, right_Ratio_Max, bot_Ratio_Max

    if False:
        cv2.imshow("gray", src)
        cv2.waitKey(500)

    [image_row, image_col]=[src.shape[i] for i in xrange(2)]

    thresh = 0
    ret, gray_bin = cv2.threshold(src, thresh, 255, cv2.THRESH_OTSU|cv2.THRESH_BINARY)
    if False:
        cv2.imshow("gray_bin", gray_bin)
        cv2.waitKey(500)

    contours0, hierarchy = cv2.findContours( gray_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #contours0, hierarchy = cv2.findContours( gray_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #contours0, hierarchy = cv2.findContours( gray_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    #show all possbile contours
    if False:
        src_rgb = cv2.imread(aFile,1);
        if src_rgb is not None:
            cv2.drawContours( src_rgb, contours0, -1, (255,0,0), 3)  #show contour with blue
            cv2.namedWindow("contours", cv2.CV_WINDOW_AUTOSIZE)
            cv2.imshow("contours", src_rgb)
            cv2.waitKey(1)
            raw_input("press key to continue...")

    squares=[]
    for con in contours0:
        cnt_len = cv2.arcLength(con, True)
        con = cv2.approxPolyDP(con, 0.02*cnt_len, True)
        if len(con) == 4 and cv2.isContourConvex(con):
            cnt = con.reshape(-1,2)

            #show current countour
            if False:
                src_rgb = cv2.imread(aFile,1);
                if src_rgb is not None:
                    tmp=[]
                    tmp.append(cnt)
                    cv2.drawContours( src_rgb, tmp, -1, (255,0,0), 3)  #show contour with blue
                    cv2.namedWindow("squares", cv2.CV_WINDOW_AUTOSIZE)
                    cv2.imshow("squares", src_rgb)
                    cv2.waitKey(500)
                    raw_input("press key to continue...")

            print("DEBUG:Check 1st level candicated rect=%s" % cnt)
            if cv2.contourArea(con) < 100:
                print("DEBUG:Check contourArea fail")
                continue

            print("DEBUG:Check 2nd level candicated rect=%s" %cnt)

            #the ROI should be in right-bottom of the surface
            #it's one asssumption to increase the recognization possibility
            if (left_Ratio_Min,top_Ratio_Min) != (0,0) or (right_Ratio_Max,bot_Ratio_Max) != (0,0):
                position_right = True
                for xy in cnt:
                    [x,y]=[xy[0],xy[1]]
                    if x < image_col*left_Ratio_Min or y < image_row*top_Ratio_Min or \
                            x > image_col*right_Ratio_Max or y > image_row*bot_Ratio_Max:
                        position_right = False;
                        break
                if position_right == False:
                    print("DEBUG:Check position_right fail, image_col=%d, image_row=%d, x=%d, y=%d" %(image_col, image_row, x, y))
                    continue;

            #the ROI should be square, so the degree should be ~90.
            max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])

            #get the arc length of current ROI contour
            lenlist = [cv2.arcLength( np.array([con[i],con[i+1]]),True) for i in xrange(2)]
            print("DEBUG:Two near length=%s"%lenlist)

            #if (max_cos<0.1) and (abs(lenlist[0]-lenlist[1])<5): #the arc length difference should be small
            if (max_cos>=0.2): #0.2 means not less than 80 degree
                print("DEBUG:Check max_cos fail, max_cos=%s"%max_cos)
            else:
                print("DEBUG:Found square=%s" % cnt)
                squares.append(cnt)
    print("Final squares=%s" % squares)

    if False:
        src_rgb = cv2.imread(aFile,1);
        if src_rgb is None:
            print("ERROR:Fail to read image rgb" + aFile)
        cv2.drawContours( src_rgb, squares, -1, (255,0,0), 3)  #show contour with blue
        cv2.namedWindow("squares", cv2.CV_WINDOW_AUTOSIZE)
        cv2.imshow("squares", src_rgb)
        cv2.waitKey(500)
        raw_input("press key to continue...")
    (left, top, right, bot) = checkSquareAndSort(squares)
    return (left, top, right, bot)

def find_ROI_with_dir(working_dir, output_file):
    print("DEBUG:working dir %s"% working_dir)
    print("DEBUG:output file %s"% output_file)

    ##check input dir
    if not os.path.isdir(working_dir):
        raise Exception, "Invalid working dir:" + working_dir

    #check input image dir: working_dir/dst/
    input_dir = os.path.join(working_dir, "dst")
    if not os.path.isdir(input_dir):
        raise Exception, "Invalid dir:" + input_dir

    #check output file
    if os.path.isfile(output_file):
        os.remove(output_file)
    output_dir = os.path.split(output_file)[0]
    if output_dir == "":
        output_dir="."
    if not os.path.isdir(output_dir):
        print("No existing output path" + output_dir)
        print("Create output dir" + output_dir)
        os.mkdir(output_dir)

    # start to detect ROI region
    (left,top,right,bot)=(0,0,0,0)
    generator = PicGenerator(input_dir)
    for aFile,img in ImageQueue(generator, loadtype = 0):
        (left, top, right, bot)=find_file_ROI(aFile,img);
        if (left,top,right,bot)!=(0,0,0,0):
            print("ROI found with file %s, %s" %( aFile, [left, top, right, bot]))
            break

    # update result to output file
    print("generate the output file %s for result %s" % (output_file,[left, top, right, bot]))
    with open(output_file, "w") as result:
        result.write("# ROI info for" + str(output_dir) + "\n")
        result.write("@ROI\n")
        result.write(("%10s, "*4 % ("LEFT","TOP","RIGHT","BOT")) + "\n")
        result.write("%10s, "*4 % (str(left),str(top),str(right),str(bot)) +"\n")
        result.write("@END")

    print("")
    print(output_file+":")
    for line in file(output_file):
        print( line.strip("\n") )
    print("")

    return (left, top, right, bot)

@TimeTag
def update_ROI_with_dir(working_dir, cfg, output_file):
    global left_Ratio_Min, top_Ratio_Min, right_Ratio_Max, bot_Ratio_Max

    left    = cfg.get("ROILeft")
    top     = cfg.get("ROITop")
    right   = cfg.get("ROIRight")
    bot     = cfg.get("ROIBot")

    if (left,top,right,bot) != (0,0,0,0):
        left    = cfg.get("ROILeft")    - cfg.get("LEFT_CROP")
        top     = cfg.get("ROITop")     - cfg.get("TOP_CROP")
        right   = cfg.get("ROIRight")   - cfg.get("LEFT_CROP")
        bot     = cfg.get("ROIBot")     - cfg.get("TOP_CROP")

        if cfg.get("ROTATE") == 90:
            (left,top, right,bot) = (-top, left, -bot, right)
        elif cfg.get("ROTATE") == 270:
            (left,top, right,bot) = (top, -left, bot, -right)
        elif cfg.get("ROTATE") == 180:
            (left,top, right,bot) = (-left,-top, -right,-bot)

        print("Using specified ROI region %s" % ([left,top,right,bot]))
    else:
        left_Ratio_Min  = cfg.get("left_Ratio_Min")
        top_Ratio_Min   = cfg.get("top_Ratio_Min")
        right_Ratio_Max = cfg.get("right_Ratio_Max")
        bot_Ratio_Max   = cfg.get("bot_Ratio_Max")
        (left, top, right, bot) = find_ROI_with_dir(working_dir, output_file)

    cfg.set("ROILeft", left)
    cfg.set("ROITop",top)
    cfg.set("ROIRight",right)
    cfg.set("ROIBot",bot)

    ## add framecount roi region into crop region list
    data=Container()
    ext=5
    data.crop=[left, top, right, bot, ext]
    data.desc="framecount_roi"
    if not cfg.has_key("crop_region_list"):
        cfg.set("crop_region_list", [])

    crop_list=cfg.get("crop_region_list")
    crop_list.append(data)
    cfg.set("crop_region_list", crop_list)

    return (left, top, right, bot)

