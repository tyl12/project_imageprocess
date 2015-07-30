import os
import sys
import re
from localUtils import *
import shutil
import numpy as np
from functools import cmp_to_key
import cv2
from multiprocessing import Process

USE_PIL=False


#-------------------------------------------------------------------------------------------------#
gCropMargin = 10

def find_file_Crop_Region(aFile):
    global gCropMargin
    src = cv2.imread(aFile,0);
    if src is None:
        print("ERROR:Fail to read image %s"%(aFile))
        return (0,0,0,0)

    image_row, image_col=src.shape

    src_blur = cv2.medianBlur(src, 3)

    gray_min = src_blur.min()
    gray_max = src_blur.max()
    crop_threshold = gray_min + (gray_max-gray_min)/10

    print("gray_min,gray_max=%s,%s"%(gray_min,gray_max))

    ret, gray_bin = cv2.threshold(src_blur, crop_threshold, 255, cv2.THRESH_BINARY)

    left, top, right, bot=0,0,0,0

    for j in range(image_col):
        col = gray_bin[:,j]
        cnt = cv2.countNonZero(col)
        if cnt > 0:
            left = j
            break;
    for j in range(image_col)[::-1]:
        col = gray_bin[:,j]
        cnt = cv2.countNonZero(col)
        if cnt > 0:
            right = j
            break;
    for i in range(image_row):
        row = gray_bin[i,:]
        cnt = cv2.countNonZero(row)
        if cnt > 0:
            top = i
            break;
    for i in range(image_row)[::-1]:
        row = gray_bin[i,:]
        cnt = cv2.countNonZero(row)
        if cnt > 0:
            bot = i
            break;

    if (left, top, right, bot) != (0,0,0,0):
        sx = (left - gCropMargin >0) and (left - gCropMargin) or 0
        sy = (top - gCropMargin >0) and (top - gCropMargin) or 0
        ex = (right + gCropMargin >image_col-1) and (image_col-1) or (right + gCropMargin)
        ey = (bot + gCropMargin >image_row-1) and (image_row-1) or (bot + gCropMargin)

    return sx, sy, ex, ey


@TimeTag
def find_Crop_Region_with_dir(input_dir, output_file):
    print("input dir=%s"%(input_dir))
    print("output file=%s"%(output_file))

    #check output file
    if os.path.isfile(output_file):
        os.remove(output_file)
    output_dir = os.path.split(output_file)[0]
    if output_dir == "":
        output_dir="."
    if not os.path.isdir(output_dir):
        print("No existing output path %s"%(output_dir))
        print("create output dir %s"%(output_dir))
        os.mkdir(output_dir)

    # start to detect ROI region
    (left,top,right,bot)=(0,0,0,0)
    if not os.path.isdir(input_dir):
        print("ERROR:Invalid input dir %s"%(input_dir))
        return (left, top, right, bot)

    for aFile in PicGenerator(input_dir, ["bmp", "gif", "jpg"]):
        (left, top, right, bot)=find_file_Crop_Region(aFile);
        if (left,top,right,bot)!=(0,0,0,0):
            print("CropRegion found with file %s, %s)"%(aFile, (left, top, right, bot)))
            break

    # update result to output file
    print("generate the output file %s for result %s"%(output_file, [left, top, right, bot]))
    with open(output_file, "w") as result:
        result.write("# Crop Region info for" + str(output_dir) + "\n")
        result.write("@CROP_REGION\n")
        result.write(("%10s, "*4 % ("LEFT","TOP","RIGHT","BOT")) + "\n")
        result.write("%10s, "*4 % (str(left),str(top),str(right),str(bot)) +"\n")
        result.write("@END")

    print("")
    print("%s:"%(output_file))
    for line in file(output_file):
        print(line.strip("\n"))
    print("")

    return (left, top, right, bot)

def update_Crop_Region_with_dir(input_dir, cfg, output_file):
    '''
    process input_dir/image_files, output result data to output_file, update crop config to cfg.
    '''
    left    = cfg.get("LEFT_CROP")
    top     = cfg.get("TOP_CROP")
    right   = cfg.get("RIGHT_CROP")
    bot     = cfg.get("BOT_CROP")

    if (left,top,right,bot) != (0,0,0,0):
        print("Using specified Crop region %s"%(left,top,right,bot))
    else:
        (left, top, right, bot) = find_Crop_Region_with_dir(input_dir, output_file)
        cfg.set("LEFT_CROP",left)
        cfg.set("TOP_CROP",top)
        cfg.set("RIGHT_CROP",right)
        cfg.set("BOT_CROP",bot)

    return (left, top, right, bot)

#-------------------------------------------------------------------------------------------------#

## angle: Rotation angle in degrees. Positive values mean counter-clockwise rotation (the
##        coordinate origin is assumed to be the center).
def rotate_scale_image(aFile, loadtype, angle, scale=1.0, expand=1):
    image = cv2.imread(aFile, loadtype)
    if abs(angle) <= 0.1: #in unit of degree
        print("DEBUG:return without rotation, too small rotation angle %s" % angle)
        return image

    if loadtype == 0:
        image_channel = 1
    else:
        image_channel = image.shape[2]
    image_row, image_col = image.shape[0:2]

    if expand:
        expandLen = int(max(image.shape[0:2])*math.sqrt(2)+2)
        image_expand = np.zeros((expandLen, expandLen, image_channel), np.uint8)

        row_start = (expandLen-image_row)/2
        col_start = (expandLen-image_col)/2
        row_stop = row_start + image_row
        col_stop = col_start + image_col

        if image_channel > 1:
            image_inter = np.zeros((expandLen, expandLen, image_channel), np.uint8)
            image_inter[row_start:row_stop, col_start:col_stop, :] = image[0:image_row,0:image_col, :]
        else:
            image_inter = np.zeros((expandLen, expandLen), np.uint8)
            image_inter[row_start:row_stop, col_start:col_stop] = image[0:image_row,0:image_col]
    else:
        image_inter = image

    inter_row, inter_col = image_inter.shape[0:2]

    center=(inter_col/2,inter_row/2)
    mat = cv2.getRotationMatrix2D(center, angle, scale)

    #dst = cv2.warpAffine(image_inter, mat, dsize=(inter_col, inter_row), borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.WARP_INVERSE_MAP)
    dst = cv2.warpAffine(image_inter, mat, dsize=(inter_col, inter_row))

    if False:
        cv2.namedWindow("orig", 1)
        cv2.namedWindow("rotate", 1)
        cv2.imshow("orig", image_inter)
        cv2.imshow("rotate", dst)
        cv2.waitKey(0)
    return dst

def rename_crop_rotate_File(aFile, outputFile, crop_left, crop_top, crop_right, crop_bot, rotate, loadtype):
    output_path = os.path.split(outputFile)[0]

    if not os.path.isdir(output_path):
        print("No existing output path, create %s"%(output_path))
        os.mkdir(output_path)
    if os.path.isfile(outputFile):
        print("Delete existing output file %s"%(outputFile))
        os.remove(outputFile)

    ## print("DEBUG:%s->%s"%(aFile,outputFile))

    global USE_PIL
    if USE_PIL:
        from PIL import Image
        img = Image.open(aFile)
        if rotate != 0:
            img = img.rotate(rotate, expand=1)

        w=crop_right-crop_left
        h=crop_bot-crop_top
        if w>0 and h>0 and crop_left>=0 and crop_top>=0:
            box=(crop_left, crop_top, crop_right, crop_bot)
            img = img.crop(box)

        img.save(outputFile, "BMP")
    else:
        dst = rotate_scale_image(aFile, loadtype, rotate, expand=1)

        w=crop_right-crop_left
        h=crop_bot-crop_top
        if w>0 and h>0 and crop_left>=0 and crop_top>=0:
            dst=dst[crop_top:crop_bot, crop_left:crop_right]

        cv2.imwrite(outputFile, dst)
    return True

def rename_crop_rotate_file_list(rename_map_list, left, top, right, bot, rotate, loadtype):
    for task in rename_map_list:
        aFile = task[0]
        outputFile = task[1]
        rename_crop_rotate_File(aFile, outputFile, left, top, right, bot, rotate, loadtype)

def rename_crop_rotate_file_multithread_dispatcher(rename_map_list, left, top, right, bot, rotate, loadtype):
    task_num = len(rename_map_list)
    print("handle file index 0 ~ %d with new thread" % (task_num-1))
    if task_num >= 20:
        thread_num = 4
    else:
        thread_num = 1
    task_per_thread = int(task_num/thread_num)
    thread_list=[]
    border_index=[]
    border_index = range(0, task_num, task_per_thread)
    #if border_index[-1] < task_num-1:
    #    border_index.append(task_num-1)
    border_index.append(task_num) ## Notes! one count exceeding, which will not cause exceeding exception when using slice
    for i in range(0, len(border_index)-1):
        print("create thread for file index %d ~ %d" % (border_index[i], border_index[i+1]))
        s = threading.Thread(target=rename_crop_rotate_file_list, \
                args=(rename_map_list[border_index[i]:border_index[i+1]], left, top, right, bot, rotate, loadtype))
        s.setDaemon(True)
        thread_list.append(s)
        s.start()
    #for i in thread_list:
    #    i.join()
    onTrigger_sec = time.time()
    while len(thread_list) > 0:
        if time.time() - onTrigger_sec > 20*60: # 20min timeout
            print("ERROR:image file processing thread blocked?!")
            exit(-1)
        else:
            try:
                # Join all threads using a timeout so it doesn't block
                # Filter out threads which have been joined or are None
                thread_list = [t.join(1000) for t in thread_list if t is not None and t.isAlive()]
            except KeyboardInterrupt:
                print("ERROR:Ctrl-c received! Exit!")
                exit(-1)
    return

def rename_crop_rotate_file_multiprocess_dispatcher(rename_map_list, left, top, right, bot, rotate, loadtype):
    task_num = len(rename_map_list)
    print("handle file index 0 ~ %d with new thread" % (task_num-1))
    if task_num >= 20:
        thread_num = 4
    else:
        thread_num = 1
    task_per_thread = int(task_num/thread_num)
    border_index=[]
    border_index = range(0, task_num, task_per_thread)
    border_index.append(task_num) ## Notes! one count exceeding, which will not cause exceeding exception when using slice

    p_list=[]
    for i in range(0, len(border_index)-1):
        print("create subprocess for file index %d ~ %d" % (border_index[i], border_index[i+1]))
        p=Process(target=rename_crop_rotate_file_list, args=(rename_map_list[border_index[i]:border_index[i+1]], left, top, right, bot, rotate, loadtype))
        p.start()
        p_list.append(p)
    #for p in p_list:
    #   p.join()
    onTrigger_sec = time.time()
    while len(p_list) > 0:
        if time.time() - onTrigger_sec > 20*60: # 20min timeout
            print("ERROR:image file processing thread blocked?!")
            exit(-1)
        else:
            try:
                # Join all processes using a timeout so it doesn't block
                # Filter out threads which have been joined or are None
                p_list = [p.join(1000) for p in p_list if p is not None and p.is_alive()]
            except KeyboardInterrupt:
                print("ERROR:Ctrl-c received! Exit!")
                exit(-1)
    return

def local_rename_crop_rotate_Dir(working_dir, left, top, right, bot, rotate_degree, loadtype, resample_factor, rename=False, summary_manual_file=None):
    print("working_dir %s"%working_dir)
    print("crop %s"%([left, top, right, bot]))
    print("rotate %s"%rotate_degree)
    ##check input dir
    if not os.path.isdir(working_dir):
        raise Exception, "Invalid working dir:" + working_dir
    #check input dir: working_dir/src/
    input_dir = os.path.join(working_dir, "src")
    if not os.path.isdir(input_dir):
        raise Exception, "Invalid dir:" + input_dir

    generator = None
    #summary_manual_file is not None, meaning manual split flag has been set
    if summary_manual_file:
        generator = PicGeneratorExtend(input_dir, summary_manual_file, ["gif", "bmp", "jpg"])
    #summary_manual_file is None, using auto split as usual
    else:
        generator = PicGenerator(input_dir, ["gif", "bmp", "jpg"])
    fList_orig = generator.getFileList()

    #check output dir: working_dir/dst/
    output_path = os.path.join(working_dir, "dst")

    try:
        left_crop = int(left)
        top_crop = int(top)
        right_crop = int(right)
        bot_crop = int(bot)
        rotate = float(rotate_degree)
        n = int(rotate/360)
        rotate = rotate - n*360
    except Exception,e:
        print(e)
        raise Exception()

    ##check output dir
    if os.path.isdir(output_path):
        print("*"*30)
        print("Delete existing output dir %s ?"%output_path)
        #ans=raw_input("Delete existing output dir " + output_path + "?")
        ans = "Y"
        if ans in ["Y", "y"]:
            shutil.rmtree(output_path)
        else:
            print("using existing output dir %s" %output_path)
    print("Update output dir %s" % output_path)
    os.mkdir(output_path)

    USE_MULTITHREAD=True

    rename_map_list = []
    orig_len = len(fList_orig)
    resample_list = [fList_orig[i] for i in range(0, orig_len, resample_factor)]
    for (i,f) in enumerate(resample_list):
        aFile = os.path.join(input_dir,f)
        ## print("DEBUG:Process file %s"%(aFile))
        if os.path.isfile(aFile):
            if rename:
                (curdir, fileName) = os.path.split(aFile)
                extendName = fileName.split('.').pop()

                # create outptu file name
                # choice 1> use origin file number
                #fileNum = re.search('(\d+)\.', fileName)
                #fileNewName = "%05d" % (int(fileNum.group(1)))
                # choice 2> use incremental number
                fileNewName = "%05d" % (i)
                fileNewName = fileNewName + "." + extendName

                # create output file
                outputFile = os.path.join(output_path, fileNewName)
            else:
                outputFile = os.path.join(output_path, f)
            rename_map_list.append([aFile, outputFile])
    if USE_MULTITHREAD:
        #rename_crop_rotate_file_multithread_dispatcher(rename_map_list, left, top, right, bot, rotate, loadtype)
        rename_crop_rotate_file_multiprocess_dispatcher(rename_map_list, left, top, right, bot, rotate, loadtype)
    else:
        for old, new in rename_map_list:
            rename_crop_rotate_File(old, new, left, top, right, bot, rotate, loadtype)
    return True

@TimeTag
def rename_rotate_crop_Dir(working_dir, cfg, summary_manual_file=None):
    left    = cfg.get("LEFT_CROP")
    top     = cfg.get("TOP_CROP")
    right   = cfg.get("RIGHT_CROP")
    bot     = cfg.get("BOT_CROP")

    gUseColorSpace = cfg.get("COLOR_SPACE")
    if gUseColorSpace in ["BGR","YUV","HLS"]:
        loadtype = 1
    elif gUseColorSpace in ["GRAY"]:
        loadtype = 0

    degree  = cfg.get("ROTATE")
    resample_factor = cfg.get("camera_resample_factor")
    return local_rename_crop_rotate_Dir(working_dir, left, top, right, bot, degree, loadtype, resample_factor, summary_manual_file=summary_manual_file)

#*******************************************************************************#

#local debug, specify square points manually by mouse
click_list=[]
click_tmp=None
right_butt_cancel=False

def onmouse(event, x, y, flags, param):
    global click_list, right_butt_cancel, click_tmp
    if event == cv2.EVENT_RBUTTONDOWN and flags & cv2.EVENT_FLAG_LBUTTON:
        print("R click, set cancel flag")
        right_butt_cancel=True
    elif event == cv2.EVENT_LBUTTONDOWN:
        click_tmp=(x,y)
        right_butt_cancel=False
        print("%d,%d" %(x,y))
    elif event == cv2.EVENT_LBUTTONUP:
        if right_butt_cancel is True:
            print("cancelled")
            return
        click_list.append(click_tmp)
        right_butt_cancel=False


#
#    --------------------> x
#   |
#   |     -----------
#   |    |           |
#   |    |           |
#   |    |   (tx,ty) |
#   |    |           |
#   |    |           |
#   |    |           |
#   |     -----------
#   |
#   |
#  \|/
#   y
#
# transform origin point to center (tx,ty), where tx=w/2 >0, ty=h/2 >0
#
#            y
#            ^
#            |
#            |
#     p1 ---------  p2
#       |    |    |
#       |    |    |
#       |    -----|----------->x
#       |         |
#       |         |
#     p4 ---------  p3
#
#   x'=x-tx
#   y'=ty-y
#
# then sort the squ points.

def sortSquClkwise(image, square):
    cvimg = cv2.imread(image)
    image_row, image_col = cvimg.shape[0:2]
    #tx, ty=int(image_col/2), int(image_row/2)
    tx=sum(p[0] for p in square)/4
    ty=sum(p[1] for p in square)/4

    transform=[[p[0]-tx,ty-p[1]] for p in square]
    print(transform)

    sortSqu=[]
    std=[[-1,1],[1,1],[1,-1],[-1,-1]]
    for i in std:
        cnt = 0
        for p in transform:
            if i[0]*p[0] > 0 and i[1]*p[1]>0:
                sortSqu.append(p)
                break;
            cnt += 1
        if  cnt >= 4: raise Exception
    print("Sorted Squ points in center axis: %s" % sortSqu)
    #transfrom back to origin axis
    transform=[[p[0]+tx,ty-p[1]] for p in sortSqu]
    print("Sorted Squ points in image coordinate: %s" % transform)
    return transform

def detectSquContour(image, colorpat):
    '''detect the edge (suppose it's square) of the input picture, output the four points,
    origin point in top-left'''

    img = cv2.imread(image, 1);
    img_row, img_col = img.shape[0:2]

    slide = None
    RO_pat_list = ([0, "B"], [1, "G"], [2, "R"])
    for i,pat in RO_pat_list:
        if pat == colorpat:
            slide=img[:,:,i].copy()
            break
    if slide is None:
        raise Exception

    ret, gray = cv2.threshold(slide, 0, 255, cv2.THRESH_OTSU|cv2.THRESH_BINARY)

    contours0, hierarchy = cv2.findContours( gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Num of detected contours: %d" % len(contours0))
    res=[]
    for con in contours0:
        cnt_len = cv2.arcLength(con, True)
        con = cv2.approxPolyDP(con, 0.02*cnt_len, True)
        if len(con) == 4 and cv2.isContourConvex(con):
            cnt = con.reshape(-1,2)
            print("Check candidate" + str([i for i in cnt]) + "...")
            #check area
            if cv2.contourArea(con) < (img_row * img_col / 10):
                print("DEBUG:Check contourArea fail, contourArea=%s, img_row=%d, img_col=%d" %( cv2.contourArea(con), img_row, img_col))
                continue

            #check angle. the ROI should be square, so the degree should be ~90.
            max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
            if max_cos >= 0.1:
                print("DEBUG:Check angle fail, cos= %s" % max_cos)
                continue

            res.append(cnt)
    else:
        print("Detected square: %s" % str( [i for i in res]))

    if len(res) != 1:
        raise Exception("Too many/less candidates found")

    if False:
        cv2.drawContours( img, res, -1, (0,0,255), 3)  #show contour with blue
        cv2.namedWindow("img",0)
        cv2.imshow("img",img)
        cv2.waitKey(5)

    return res[0]


def detectImageEdge(cali_dir, working_dir, cfg, colorpat):
    image=None

    gUseColorSpace = cfg.get("COLOR_SPACE")
    if gUseColorSpace in ["BGR","YUV","HLS"]:
        loadtype=1
    elif gUseColorSpace in ["GRAY"]:
        loadtype=0

    for aFile in PicGenerator(cali_dir, ["bmp", "gif", "jpg"]):
        try:
            if True:
                squ = detectSquContour(aFile, colorpat)
            else: #debug
                cvimg=cv2.imread(aFile)
                tmpimg = cvimg.copy()
                cv2.namedWindow("track",flags=cv2.WINDOW_AUTOSIZE)
                #cv2.namedWindow("track",flags=0)
                cv2.setMouseCallback("track", onmouse)
                cv2.imshow("track", cvimg)
                cv2.waitKey(10)
                while True:
                    for i in click_list:
                        cv2.circle(cvimg, (i[0], i[1]), 3, (0, 255, 0), -1)
                    cv2.imshow("track", cvimg)
                    if len(click_list) == 4:
                        cv2.waitKey(1000)
                        break
                    if (cv2.waitKey(200) & 255) == 27:
                        raise Exception
                squ = click_list
        except Exception,e:
            print("WARNING:Fail to detect squcontour from file %s. %s" % (aFile, e))
            continue
        image = aFile
        break
    if image is None:
        print("WARNING:Return without calibration from dir %s" % cali_dir)
        return

    #in cvimage coordinate. top-left should be the first point, and in clockwise sequence
    squ = sortSquClkwise(image,squ)

    # positive for counter-clockwise rotation, in unit degree
    deltaTheta = np.arctan( -(squ[1][1]-squ[0][1]) * 1.0 / (squ[1][0]-squ[0][0]) )
    deltaTheta = deltaTheta * 180.0 / math.pi
    rotate_counterclkwise = cfg.get('ROTATE')-deltaTheta

    tmp_dir = os.path.join(working_dir, "tmpdir")
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)

    if True:
        src = cv2.imread(image, 1)
        contour_orig = src.copy()
        cv2.drawContours( contour_orig, [np.array(squ, dtype=np.int32)], -1, (0,0,255), 3)  #show contour with blue
        contour_file = os.path.join(os.path.join(tmp_dir, "calibrated_degree_contour_"+image.split(os.sep)[-1].split(".")[-2]+".bmp"))
        cv2.imwrite(contour_file, contour_orig)
        print("calibrate file for degree detection. %s -> %s" % (image, contour_file))

    rim_file = os.path.join(os.path.join(tmp_dir, "calibrated_rotate_"+image.split(os.sep)[-1].split(".")[-2]+".bmp"))

    global USE_PIL
    if USE_PIL:
        from PIL import Image
        pil_img = Image.open(image)
        rim = pil_img.rotate(rotate_counterclkwise, expand=1)
        rim.save(rim_file, "BMP")
    else:
        dst = rotate_scale_image(image, loadtype, rotate_counterclkwise, expand=1)
        cv2.imwrite(rim_file, dst)

    print("calibrate file for rotation. %s -> %s" % (image, rim_file))

    #re-detect contour after rotation
    squ = detectSquContour(rim_file, colorpat)
    #in cvimage coordinate. top-left should be the first point, and in clockwise sequence
    squ = sortSquClkwise(image,squ)

    if True:
        src = cv2.imread(rim_file, 1)
        contour_crop_after_rotate = src.copy()
        cv2.drawContours( contour_crop_after_rotate, [np.array(squ, dtype=np.int32)], -1, (0,0,255), 3)  #show contour with blue
        contour_file = os.path.join(os.path.join(tmp_dir, "calibrated_rotate_crop_contour_"+image.split(os.sep)[-1].split(".")[-2]+".bmp"))
        cv2.imwrite(contour_file, contour_crop_after_rotate)
        print("calibrate file for rotate and crop. %s -> %s" % (image, contour_file))

    crop_rect = (squ[0][0], squ[0][1], squ[2][0], squ[2][1])
    print("Update cfg with calibration results: crop_rect=%s, rotate=%s" % (crop_rect, rotate_counterclkwise))

    cfg.set('left_crop', crop_rect[0])
    cfg.set('top_crop', crop_rect[1])
    cfg.set('right_crop', crop_rect[2])
    cfg.set('bot_crop', crop_rect[3])
    cfg.set('rotate', rotate_counterclkwise)

    return

def local_detectTouch(input_dir, cvpos):
    if not os.path.isdir(input_dir):
        raise Exception, "Invalid dir:" + input_dir

    imgDiffList=[]
    roi_prev = None
    generator = PicGenerator(input_dir)
    for aFile,img in ImageQueue(generator, loadtype=1):
        roi = img[cvpos[1]-10:cvpos[1]+2, cvpos[0]-10:cvpos[0]+10, :].copy()
        if roi_prev is None:
            roi_prev = roi.copy()
            continue

        imgDiff =cv2.absdiff(roi, roi_prev)

        imgDiffSum = np.sum(imgDiff) #sum of pixel values over whole image
        cv2.imshow("touch",roi)
        cv2.waitKey(2)
        roi_prev = roi.copy()
        imgDiffList.append(imgDiffSum)
    print("DEBUG:"+imgDiffList)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.figure(1)
        plt.plot(range(len(imgDiffList)), imgDiffList, 'b+-')
        plt.title("Frame Difference")
        plt.xlabel("frame index")
        plt.ylabel("frame difference")
        plt.show()
    except Exception,e:
        print("ERROR:import matplotlib failed, %s" % e)

    return 0

def detectTouch(input_dir, cfg, cvpos):
    coord_touch = cvpos[0], -cvpos[1]
    pos = coordinateTransform(cfg.get('RO_COORDINATE_OFFSET'), cfg.get('RO_COORDINATE_THETA'), coord_touch)
    startpos = local_detectTouch(input_dir, (pos[0], -pos[1]))
    return startpos


