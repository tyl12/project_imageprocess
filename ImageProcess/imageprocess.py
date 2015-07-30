import os
import sys
import preProcessing
import detectPhase
import detectROI
import decodeROI
import localUtils
import platform
#import motionTrack

def initImageLog(fd):
    output=localUtils.DupOutput()
    output.dup({sys.stdout:['critical','error','warning','info', 'notset'],\
        fd:['critical','error','warning','info','debug','notset']})
    return output

def recoverImageLog(output):
    output.recover()
    return

def main(testdir=None, need_preprocess=False, need_decode=False, need_phase=False, need_track=False, need_compress=False, manualSplitPhase=False):
    print("Launch ImageProcessing")
    # config log output
    LOG_FILENAME=os.path.join(testdir,"ImageProcessing.log")
    fd=open(LOG_FILENAME, 'w+')
    output=initImageLog(fd)
    try:
        do_process(testdir=testdir,
                   need_preprocess=need_preprocess,
                   need_decode=need_decode,
                   need_phase=need_phase,
                   need_track=need_track,
                   need_compress=need_compress,
                   manualSplitPhase=manualSplitPhase)
    except Exception, e:
        print "Fail to do process", e
    recoverImageLog(output)
    fd.close()
    print("ImageProcessing done!")

def do_process(testdir=None, need_preprocess=False, need_decode=False, need_phase=False, need_track=False, need_compress=False, manualSplitPhase=False):

    #***********************************************************#
    working_dir=testdir
    output_file_ROI=os.path.join(testdir,r'result_ROI.txt')
    output_file_Crop=os.path.join(testdir,r'result_Crop.txt')
    output_file_frame=os.path.join(testdir,r'result_frame.txt')
    output_file_diff=os.path.join(testdir,r'result_diff.txt')
    output_file_phase=os.path.join(testdir,r'phase_file.txt')
    output_file_touch=os.path.join(testdir,r'result_touch.txt')
    output_file_track=os.path.join(testdir,r'result_track.txt')
    input_file_config=os.path.join(testdir,r'input_config_file.txt')
    gesture_file=os.path.join(testdir,r'gesture.txt')
    src_image_dir=os.path.join(testdir,r'src')
    dst_image_dir=os.path.join(testdir,r'dst')
    calibrate_dir=os.path.join(testdir, r'calibrate')
    #***********************************************************#
    print("Launch ImageProcessing")

    tmp_dir = os.path.join(working_dir,'tmpdir')
    print("clean temporary dir, %s" % tmp_dir)
    if os.path.isdir(tmp_dir):
        generator=localUtils.PicGenerator(tmp_dir, ['jpg','bmp','gif'])
        for i in generator:
            os.remove(i)
    #***********************************************************#

    print("processing entry:" + testdir)

    #check local config
    cfg = localUtils.Config(input_file_config)
    if cfg is None:
        raise Exception, "parse config fail"

    if manualSplitPhase:
        summary_manual_file = os.path.join(src_image_dir, r'summary_manual.txt')
    else:
        summary_manual_file = None

    #process working_dir/calibrate/*
    if need_preprocess:
        #udpate cfg.crop/rotate
        preProcessing.detectImageEdge(calibrate_dir, working_dir, cfg, "G")
        preProcessing.rename_rotate_crop_Dir(working_dir, cfg, summary_manual_file=summary_manual_file)

    # process working_dir/dst/
    if need_decode == True or ( need_decode is None and cfg.get('UXTool')):
        #update cfg.framecountROI
        (left,top,right,bot)=detectROI.update_ROI_with_dir(working_dir, cfg, output_file_ROI)
        decodeROI.decodeDir(working_dir, cfg, output_file_frame)

    if need_phase:
        if not manualSplitPhase:
            detectPhase.detectTouchTrigger(calibrate_dir, working_dir, gesture_file, cfg, output_file_touch)
            detectPhase.updateTouchCropList(working_dir, gesture_file, cfg)
        detectPhase.detectDiffInDir(calibrate_dir, working_dir, cfg, output_file_diff)
        detectPhase.calcLTF(working_dir, output_file_diff, output_file_phase, cfg, None)

    if need_track:
        motionTrack.trackInDir(working_dir, cfg, output_file_track)

    if True:
        localUtils.createSummaryReport(working_dir, gesture_file, cfg)

    if need_compress:
        import time
        t = time.localtime()
        t_str = "_%d_%02d_%02d_%02d_%02d_%02d" % (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)
        compress_list = {\
                os.path.join(working_dir, 'src'):os.path.join(working_dir, 'src'+t_str),\
                os.path.join(working_dir, 'dst'):os.path.join(working_dir, 'dst'+t_str),\
                }
        clean_list  = {\
                os.path.join(working_dir, 'src'),\
                os.path.join(working_dir, 'dst'),\
                }
        for key, val in compress_list.items():
            print("INFO:compress content: %s -> %s" % (key, val))
            ret = localUtils.compressImgWithJPEG(key, val, cfg)
            if ret == False: #compress failure
                break
        if ret:# only do clean when everything work well
            for key in clean_list:
                if os.path.isdir(key):
                    print("INFO:remove content: %s" % key)
                    __import__('shutil').rmtree(key)


    print("ImageProcessing done!")
    return cfg

if __name__ == "__main__":

    sysarc = platform.system()
    if sysarc == "Linux":
        print("Running on Linux system")
        testlist=[ \
                 [0, r'../Q1',                                                       "preprocess", "decode", "phase", "-----", '-----'], \
                 [0, r'/home/ylteng/work/share/del_cameralink/case02_fling_home/output' ,             "------", "decode", "phase", "-----", '-----'], \
                 [0, r'/home/ylteng/work/share/del/data_2013_05_31_10_56_05/output/' ,             "preprocess", "decode", "phase", "-----", '-----'], \
                 [0, r'/home/mysamba/public/test_dec' ,             "preprocess", "-----", "phase", "-----", '-----'], \
                 [0, r'/home/mysamba/public/nexus7/' ,             "-----", "-----", "phase", "-----", '-----'], \
                 [1, r'/home/mysamba/public/jew/' ,             "preprocess", "decode", "phase", "-----", '-----'], \
                 [0, r'/home/mysamba/public/t1' ,             "preprocess", "-----", "-----", "-----", '-----'], \
                 ]
    else:
        print("Running on Windows system")
        testlist=[ \
                [0, r'c:\AOS Sequences\MFLD-case1-home pull icon',                     "preprocess", "decode", "phase", "-----", '-----'], \
                [0, r'c:\AOS Sequences\MFLD-case2-fling app pan',                      "preprocess", "decode", "phase", "-----", '-----'], \
                ]

    for entry in testlist:
        if entry[0] == 0:
            continue
        if len(entry) != 7:
            print "ERROR: invalid input parameters number %d" % len(entry)
            continue

        print("processing entry: %s", str(entry))
        testdir             = entry[1]
        need_preprocess     = (entry[2] == "preprocess")
        need_decode         = (entry[3] == "decode")
        need_phase          = (entry[4] == "phase")
        need_track          = (entry[5] == "track")
        need_compress       = (entry[6] == "compress")

        main(testdir=testdir, need_preprocess=need_preprocess, need_decode=need_decode, need_phase=need_phase, need_track=need_track, need_compress=need_compress)


def runAll():
    myproj_output = os.path.abspath(r'C:\cygwin\home\yiliangt\work\python_opencv\pub_MYPROJ\MYPROJOutput')
    print("INFO:process output dir = %s" % myproj_output)
    if not os.path.isdir(myproj_output):
        print("ERROR:invalid myproj output dir specified, %s" % myproj_output)
    else:
        for product in sorted(os.listdir(myproj_output)):
            productPath = os.path.join(myproj_output, product)
            if not os.path.isdir(productPath):
                print("ERROR:invalid productPath , %s" % productPath)
            else:
                print("INFO:process product = %s" % productPath)
                for case in sorted(os.listdir(productPath)):
                    casePath = os.path.join(productPath, case)
                    if not os.path.isdir(casePath):
                        print("ERROR:invalid casePath , %s" % casePath)
                    else:
                        print("INFO:    process case = %s" % case)
                        runProductCase(myproj_output, product, case)

def runProductCase(OutputFolder, product, case):
    casepath = os.path.join(OutputFolder, product, case)
    contents = []
    hist = getRunHistory(casepath)
    if len(hist) > 0:
        for i in hist:
            if i[1] == 'Y':
                contents.append("%s" %(i[0]))

    if len(contents) == 0:
        print('ERROR:No valid build existed for %s/%s' % (product, case))
        return
    for i in contents:
        testdir = os.path.join(OutputFolder, product, case, i, 'output')

        print("processing entry:" + str(testdir))

        need_preprocess     = False
        need_decode         = True
        need_phase          = True
        need_track          = False
        need_compress       = False

        main(testdir=testdir, need_preprocess=need_preprocess, need_decode=need_decode, need_phase=need_phase, need_track=need_track, need_compress=need_compress)

def getRunHistory(casepath):
    historylist=[]
    if not os.path.isdir(casepath):
        return historylist
    history = sorted(os.listdir(casepath))
    for i in history:
        if os.path.isdir(os.path.join(casepath, i, 'output', "src")):
            entry = [i, 'Y']
        else:
            entry = [i, 'N']
        historylist.append(entry)
    return historylist


