import os
import sys
import re
import inspect
import time
import copy
import ConfigParser
import math
import numpy as np
import cv2
import cv2.cv as cv
import threading
from collections import OrderedDict
from Queue import Queue
import subprocess


def TimeTag(func):
    def newFunc(*args, **args2):
        t0 = time.time()
        print(">"*50)
        print("@ %s, {%s} start" % (time.strftime("%X", time.localtime()), func.__name__))
        back = func(*args, **args2)
        print("@ %s, {%s} end" % (time.strftime("%X", time.localtime()), func.__name__))
        print("@ %.3fs taken for {%s}" % (time.time() - t0, func.__name__))
        print("<"*50)
        return back
    return newFunc

## NOTES: what it returns is a list of string, but not int/float. conversion should be done by enduser.
def getFileMap(infile, start, end):
    orderMap=OrderedDict()
    cont = file(infile).readlines()
    key_list = [s.split().pop() for s in cont[start+1].split(',')]
    for key in key_list:
        orderMap[key]=[]
    for row in range(start+2, end):
        line = cont[row]
        for i, col in enumerate(line.split(',')):
            col=col.split().pop()
            orderMap[key_list[i]].append(col)
    return orderMap

def createSummaryReport(working_dir, gesture_file, cfg):
    phase_file=os.path.join(working_dir, "phase_file.txt")
    touch_file=os.path.join(working_dir, "result_touch.txt")
    LF_file=os.path.join(working_dir, "phase_file.txt")

    #touch trigger info
    start_index = -1
    end_index   = -1
    touch_map   = None
    if os.path.isfile(touch_file):
        for idx, line in enumerate(file(touch_file)):
            pat = line.strip()
            if pat == r"@TOUCH":
                start_index = idx
            if start_index >= 0 and pat == r"@END":
                end_index = idx
                break
        if start_index >= 0 and end_index - start_index > 1:
            touch_map = getFileMap(touch_file, start_index, end_index)

    ##====================================================================================
    #touch info from gesture file
    touch_duration = 0
    if not os.path.isfile(gesture_file):
        print("ERROR:Invalid gesture file, %s" % gesture_file)
    else:
        touch_start=cfg.get('touch_start')
        isClick, clickDur = gestureIsSingleClick(gesture_file, touch_start)
        if isClick and clickDur >= 0:
            touch_duration = clickDur #NOTES: in unit *ms*
            print("INFO:Single click event, event duration = %d ms" % clickDur)

    ##====================================================================================
    #phase start/stop info
    start_index = -1
    end_index   = -1
    phase_map   = None
    if os.path.isfile(phase_file):
        for idx, line in enumerate(file(phase_file)):
            pat = line.strip()
            if pat == r"@DURATION":
                start_index = idx
            if start_index >= 0 and pat == r"@END":
                end_index = idx
                break
        if start_index >= 0 and end_index - start_index > 1:
            phase_map = getFileMap(phase_file, start_index, end_index)

    ##====================================================================================
    #L/F info
    start_index = -1
    end_index   = -1
    LF_map      = None
    if os.path.isfile(phase_file):
        for idx, line in enumerate(file(phase_file)):
            pat = line.strip()
            if pat == r"@LONGFRAME":
                start_index = idx
            if start_index >= 0 and pat == r"@END":
                end_index = idx
                break
        if start_index >= 0 and end_index - start_index > 1:
            LF_map = getFileMap(phase_file, start_index, end_index)

    ##====================================================================================
    if touch_map is not None and len(touch_map['INDEX']) >= 1:
        touch_index = int(touch_map['INDEX'][0] )
    else:
        touch_index = -1

    if phase_map is not None and len(phase_map['PHASE_START_IDX']) >= 1:
        phase_start = int(phase_map['PHASE_START_IDX'][0] )
    else:
        phase_start = -1

    if phase_map is not None and len(phase_map['PHASE_STOP_IDX']) >= 1:
        phase_stop  = int(phase_map['PHASE_STOP_IDX'][0] )
    else:
        phase_stop = -1

    print("touch_index=%s, touch_duration=%s, phase_start=%s, phase_stop=%s" % (touch_index, touch_duration, phase_start, phase_stop))

    ##====================================================================================
    '''
    convert the sampling count based LF number to surfaceframe/vsync based LF number.
    vsync:      ________|________|________|________|________|_____________
    samplecnt:
                 ---------________--------------------------------------- 1 LF

                 ---------_________________------------------------------ 2 LF

                 --------------____________------------------------------ 1 LF

            K = ceil(LF samplecnt/vsync samplecnt)
            rem = (LF samplecnt) - K*(vsync samplecnt)
            R = 1 if rem > threshold else 0
            surface(or vsync) based LF number = K + R
    '''


    LF_list=[]
    LF_threshold_list=[]
    if LF_map is not None and len(LF_map['FRAMECOUNT']) >= 1:
        LF_list = [int(lf) for lf in LF_map['FRAMECOUNT']]
        LF_threshold_list=[int(thre) for thre in LF_map['LF_PNTCNT_THRESHOLD']]

    frames_per_vsync = cfg.get('camerafps')*1.0/cfg.get('camera_resample_factor')/cfg.get('lcdfps')

    LF_VsyncCnt_list=[]
    for i in range(len(LF_list)):
        lf = LF_list[i]
        threshold = LF_threshold_list[i]
        K=int(lf/frames_per_vsync)
        rem=(1.0*lf) - K*frames_per_vsync
        R = 1 if rem >= threshold else 0
        surfaceCnt = K+R
        LF_VsyncCnt_list.append(surfaceCnt)

    surface_LF_samplecnt = [i*frames_per_vsync for i in LF_VsyncCnt_list]
    sum_surface_LF_samplecnt = sum([cnt for cnt in surface_LF_samplecnt])  #sum([])==0

    if (phase_stop - phase_start + 1 > 0):
        fps = cfg.get('lcdfps') * ( 1 - sum_surface_LF_samplecnt*1.0/(phase_stop - phase_start + 1))
    else:
        fps = -1

    ##====================================================================================
    ## output to summary report
    summary_file = os.path.join(working_dir, "summary.txt")
    with open(summary_file, 'w+') as outFd:
        #basic config
        outFd.write("Camera capture framerate = %s, resample factor = %s\n" % (cfg.get("camerafps"), cfg.get("camera_resample_factor")))

        #processing result
        if touch_index <= phase_start and touch_index >= 0:
            outFd.write("touch trigger file index = %s\n" % touch_index)
        else:
            outFd.write("touch trigger file index = %s, adjust to phase start index = %s\n" % (touch_index, phase_start))
            touch_index=phase_start

        touch_duration_framecount = touch_duration*0.001 * cfg.get("camerafps")/cfg.get("camera_resample_factor")
        if touch_duration > 0:
            outFd.write("Single touch event, event duration in framecount = %s => %0.2f ms\n" %(touch_duration_framecount, touch_duration))
        else:
            outFd.write("Not single touch event, event duration in framecount = %s => %0.2f ms\n" % (touch_duration_framecount, touch_duration))

        outFd.write("phase start file index = %s\n" % phase_start)
        outFd.write("phase stop file index = %s\n" % phase_stop)

        response_latency = phase_start-touch_index-touch_duration_framecount
        if response_latency < 0:
            print("ERROR: negative response_latency=%s found, sth. wrong, adjust to 0 by force" % response_latency)
            response_latency = 0
        outFd.write("response latency (phase start - touch trigger - touch duration) = %s => %.2f ms\n" % \
                (response_latency, (response_latency)*1000.0/(cfg.get("camerafps")/cfg.get("camera_resample_factor"))))

        complete_latency = phase_stop-touch_index-touch_duration_framecount
        if complete_latency < 0:
            print("ERROR: negative complete_latency=%s found, sth. wrong, adjust to 0 by force" % complete_latency)
            complete_latency = 0
        outFd.write("complete latency (phase stop - touch trigger - touch duration) = %s => %.2f ms\n" % \
                (complete_latency, (complete_latency)*1000.0/(cfg.get("camerafps")/cfg.get("camera_resample_factor"))))

        outFd.write("phase duration (phase stop - phase start) = %s => %.2f ms\n" % (phase_stop-phase_start, \
                (phase_stop-phase_start)*1000.0/(cfg.get("camerafps")/cfg.get("camera_resample_factor"))))
        outFd.write("equivalent fps during phase transition = %.2f fps\n" % fps)

        outFd.write("L/F list: (index, sampling count, threshould, surface/frame count)\n")
        LF_idx_list=LF_map['INDEX']
        LF_samplecnt_list=LF_map['FRAMECOUNT']
        LF_threshold_list=LF_map['LF_PNTCNT_THRESHOLD']
        for i in range(0, len(LF_idx_list)):
            outFd.write("\t%10s, \t%10s, \t%10s, \t%10s,\n" %(int(LF_idx_list[i]), int(LF_samplecnt_list[i]), \
                int(LF_threshold_list[i]), LF_VsyncCnt_list[i]))

    if True: ##
        for i in file(summary_file):
            print(i.split('\n')[0])

    ##====================================================================================

def getContent(inFile, tag):
    '''
    readin contents of inFile, according to the specified tag
    '''
    cont=[]
    flag=False
    if os.path.isfile(inFile):
        for line in file(inFile):
            line_stripped = line.strip()
            if line_stripped == tag:
                flag=True
                continue
            if line_stripped == "@END" and flag == True:
                flag=False
                return cont
            if flag == True and line_stripped != "" and line_stripped[0] != "#":
                cont.append(line_stripped)
    else:
        print("ERROR:Parse file not existing %s" % inFile)
        return None

    if flag==True:
        print("ERROR:Parse file format error %s" % inFile)
        return None

def getMap(content):
    '''
    get the contents of a file with special format.
    content: input string with format "@XXX".
    outut: elements leading with '@XXX' and end with '@END', in form of a string map.
    NOTES:all basic elements in parsed map are with 'str' type
    '''
    if content is None:
        return None
    key_line = content[0]
    cont_lines = content[1:]
    cont_map={}
    key_list=[key.strip() for key in key_line.split(",")]
    for key in key_list:
        cont_map[key]=[]

    for line in cont_lines:
        line_sep = [sep.strip() for sep in line.split(",")]
        for i,key in enumerate(key_list):
            cont_map[key].append(line_sep[i])
    return cont_map

#------------------------------------------------------------------------

@TimeTag
def parse_preprocessing(pre_processing_file):
    KEY_TUPLE=("LEFT_CROP", "TOP_CROP", "RIGHT_CROP", "BOT_CROP", "ROTATE")
    cont = getContent(pre_processing_file, "@INPUT_PRE_PROCESSING")
    if cont is None:
        return (0 for i in xrange(len(KEY_TUPLE)))
    cont_map = getMap(cont)
    for key,value in cont_map.items():
        print("DEBUG:%s,%s"%(key,value))

    ret=[]
    for key in KEY_TUPLE:
        if cont_map.has_key(key):
            ret.append(int(cont_map[key]))
        else:
            ret.append(0)
    return tuple(ret)

#------------------------------------------------------------------------

@TimeTag
class Config(object):
    def load_def_config(self):
        default_config_path = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(default_config_path, "default_config_file.txt")#default config file location

        print("Loading default config file, %s" % config_file)

        if not os.path.isfile(config_file):
            raise Exception, "Invalid default config file" + str(config_file)
        cf = ConfigParser.ConfigParser()
        cf.read(config_file)

        val = cf.get("config", "CONFIG_LIST")
        return eval(val)

    def __init__(self, config_file):
        self.__config_file = config_file
        self.__configDict = {}

        configDict={}
        #load default config
        def_config_list = self.load_def_config()

        #update config
        print( "Loading user specified config file, %s" % config_file)
        if not os.path.isfile(config_file):
            raise Exception, "Invalid config file" + str(config_file)
        update_cf = ConfigParser.ConfigParser()
        update_cf.read(config_file)

        #loop entry in default config file, search corresponding one in updated config file
        for node in def_config_list:
            key = node[0].lower()

            data = Container()
            data.type=node[2]
            data.ctrl= node[3]

            try:
                raw_val = update_cf.get("config", key)
            except ConfigParser.NoOptionError:
                #notes: for node in def_config_list, value type is already interpreted as in list
                data.val = node[1]
                print("DEBUG:Use default config node %s: %s" %(key, data.val))
            else:
                #nodes: for node in updated_cf, string conversion is required.
                data.val = eval(data.type + "(" + str(raw_val) + ")")
                print("DEBUG:Update default config node %s: (%s -> %s)" %(key, node[1], data.val ))

            #if data.type in ["str"]:
            #    data.val = raw_val
            #else:
            #    data.val = eval(data.type + "(" + str(raw_val) + ")")
            configDict[key] = data

        #finally check unrecognized entries in updated config file
        update_list = update_cf.options("config")

        for i in update_list:
            if not configDict.has_key(i.lower()):
                print("ERROR: Ignore unrecognized config key: %s, %s" % (i, update_cf.get("config",i)))

        self.__configDict = configDict
        self.showConfig()

    def showConfig(self):
        print("List all config entries:")
        for key,data in sorted(self.__configDict.items(), key=lambda t:t[0]):
            print("\t%-40s: %-12s, %-5s" % ( key, data.val, data.ctrl))

    def getConfig(self):
        #copy out, not reference!
        return dict(self.__configDict)

    def get(self, key):
        if not self.__configDict.has_key(key.lower()):
            print("ERROR:Invalid key: %s" % key)
            raise Exception, "Invalid key:" + key
            #return None
        else:
            #notes: deep copy but not reference to avoid config corruption
            return copy.deepcopy(self.__configDict[key.lower()].val)

    def has_key(self, key):
        if not self.__configDict.has_key(key.lower()):
            return False
        return True

    def set(self, key, val):
        if not self.__configDict.has_key(key.lower()):
            print("WARNING:add new entry. %s:%s"%(key, val))
            data = Container()
            data.val = val
            data.type="unknown"
            data.ctrl= "rw"
        else:
            data = self.__configDict[key.lower()]
            if data.ctrl not in ["rw", "unknown"]:
                print("ERROR: Update existing RO entry not permitted. %s:%s" % (key, val))
                raise Exception
            print("WARNING: update existing entry. %s:(%s->%s)" % (key, data.val, val))
            data.val = val
        self.__configDict[key.lower()]=data

#------------------------------------------------------------------------

class Container(object):
    pass

class PicGenerator(object):
    def matchType(self, f, typeList):
        if f.split('.').pop().lower() in typeList:
            return True
        return False

    def __init__(self, dirname, typeList=None, special_sort=False):
        self.index = -1
        self.len = 0
        self.dir = None
        self.fList = []

        if typeList is None:
            typeList=["bmp", "gif", "jpg"]
        if not os.path.isdir(dirname):
            print("ERROR:Invalid Dir %s"% dirname)
            return

        self.dir = dirname
        self.index = 0
        filelist = [f for f in os.listdir(dirname) if self.matchType(f, typeList)]
        # check input file name
        if not special_sort:
            self.fList = sorted(filelist)
        else:
            def cmpfunc(a, b):
                cmp_list1 = a.split('.')[0].split('-')
                cmp_list2 = b.split('.')[0].split('-')
                if len(cmp_list1) != len(cmp_list2):
                    raise Exception
                for i in range(len(cmp_list1)):
                    try:
                        node1=int(cmp_list1[i]);
                        node2=int(cmp_list2[i]);
                    except Exception, e:
                        continue
                    ret=node1-node2
                    if ret != 0:
                        return ret
                print("ERROR:Same file? %s, %s" %(a, b))
                return 0

            sorted_key = None
            for f in filelist:
                nameonly = f.split('.')[0]
                fileNum = re.search('-(\d+)\.', f)
                if fileNum is not None:
                    sorted_key = cmp_to_key(cmpfunc)
                    break
            self.fList = sorted(filelist, key=sorted_key)
        self.len = len(self.fList)

    def getFileList(self):
        return self.fList

    def forward(self):
        #if self.index < self.len:
        if self.checkIndexValid(self.index):
            self.index += 1
        return self.index

    def backward(self):
        #if self.index >= 0:
        if self.checkIndexValid(self.index):
            self.index -= 1
        return self.index

    #return whole filepath
    def readFilePath(self, i=None):
        if i is None:
            #if self.dir is None or self.index>=self.len or self.index<0:
            if self.dir is None or not self.checkIndexValid(self.index):
                print("ERROR:Invalid file pointer: %d out of range (%d,%d)" % (self.index, 0, self.len-1))
                raise StopIteration
            cur = self.index
        else:
            #if i<0 or i>=self.len:
            if not self.checkIndexValid(i):
                print("ERROR:Invalid file pointer: %d out of range (%d,%d)" % (i, 0, self.len-1))
                raise StopIteration
            cur = i
        f=self.fList[cur]
        print("DEBUG:read index %d, file %s" % (cur, f))
        f=os.path.join(self.dir, f)
        if os.path.isfile(f):
            return True, f
        print("ERROR:Fail to read file %s" % f)
        return False, None

    #use only the file name instead of complete file path
    def getIndexOfFileName(self, f):
        f=os.path.basename(f)
        try:
            i = self.fList.index(f)
        except Exception:
            return -1
        else:
            return i

    def checkIndexValid(self, i=None):
        if i is None:
            i=self.index
        if i<0 or self.len<=0 or i>=self.len:
            print("DEBUG:Invalid index %d" % i)
            return False
        return True

    def resetIndex(self, newpos=0):
        self.index = newpos

    def getLen(self):
        return self.len

    def getPos(self):
        return self.index

    def getList(self):
        absList = [os.path.join(dirname, self.fList[i]) for i in range(len(self.fList))]
        return absList

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    #return file path
    def next(self):
        #if self.index >= self.len:
        if not self.checkIndexValid(self.index):
            raise StopIteration
        else:
            #step forward
            fName = self.fList[self.index]
            f=os.path.join(self.dir, fName)
            self.index += 1
            return f

class PicGeneratorExtend(PicGenerator):
    def __init__(self, dirname, summary_manual_file, typeList=None, special_sort=False):
        super(PicGeneratorExtend, self).__init__(dirname, typeList, special_sort)
        if not os.path.isfile(summary_manual_file):
            print "Error: manual split file is not exist: " + summary_manual_file
            return

        self.readManualSplitPhase(summary_manual_file)
        self.fList = self.fList[self.pnt[0]:self.pnt[1]+1]
        self.len = len(self.fList)
        print "start:" + self.fList[0]
        print "end:" + self.fList[-1]

    def readManualSplitPhase(self, summary_manual_file):
        with open(summary_manual_file, 'r') as f:
            self.pnt = []
            for line in f:
                import re
                index = re.search('index-(\d+):', line)
                if index:
                    self.pnt.append(int(index.group(1)))
                else:
                    raise Exception, "Invalid summary_manual file: %s" % summary_manual_file
                if len(self.pnt) == 2:
                    break
            if(len(self.pnt) != 2):
                    raise Exception, "Invalid summary_manual file: %s" % summary_manual_file
            elif self.pnt[1] < self.pnt[0]:
                    raise Exception, "Invalid summary_manual file: %s" % summary_manual_file
            else:
                print "manual phase split point:" + str(self.pnt)

__DEBUG_ImageQueue__=False

class ImageQueue(object):
    def __init__(self, generator, loadtype=1 , depth=20):
        self.generator = generator
        self.cond = threading.Condition()
        self.Queue = Queue(depth)
        self.exit = False
        self.timeout = 0.01

        self.rdThread = threading.Thread(target=self.loadImage, args=(generator, self.Queue, loadtype))
        self.rdThread.setDaemon(True)
        self.rdThread.start()

    def loadImage(self, generator, queue, loadtype):
        timePre = time.time()
        for f in generator:
            img = cv2.imread(f, loadtype)
            item = [f, img]

            if __DEBUG_ImageQueue__:
                print("DEBUG:thread acquire")
            self.cond.acquire()
            while True:
                if time.time() - timePre > 10 or self.exit:
                    if __DEBUG_ImageQueue__:
                        print("WARNING:loadimage watchdog timeout or exit notified...")
                        print("DEBUG:thread release")
                    self.cond.release()
                    return
                try:
                    queue.put(item, 0)
                except Exception:
                    if __DEBUG_ImageQueue__:
                        print("DEBUG:thread wait")
                    self.cond.wait(self.timeout)
                    continue
                else:
                    break
            if __DEBUG_ImageQueue__:
                print("DEBUG:thread notify")
            self.cond.notify()
            if __DEBUG_ImageQueue__:
                print("DEBUG:thread release")
            self.cond.release()
            timePre = time.time()

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if __DEBUG_ImageQueue__:
            print("DEBUG:main acquire")
        self.cond.acquire()
        while True:
            if self.Queue.empty() and not self.rdThread.isAlive():
                if __DEBUG_ImageQueue__:
                    print("DEBUG:main release")
                self.cond.release()
                raise StopIteration

            try:
                item = self.Queue.get(0)
            except Exception:
                if __DEBUG_ImageQueue__:
                    print("DEBUG:main wait")
                self.cond.wait(self.timeout)
                continue
            else:
                break
        if __DEBUG_ImageQueue__:
            print("DEBUG:main notify")
        self.cond.notify()
        if __DEBUG_ImageQueue__:
            print("DEBUG:main release")
        self.cond.release()
        return item

    def __del__(self):
        self.exit = True
        if __DEBUG_ImageQueue__:
            print("DEBUG:main acquire")
        self.cond.acquire()
        if __DEBUG_ImageQueue__:
            print("DEBUG:main notify")
        self.cond.notify()
        if __DEBUG_ImageQueue__:
            print("DEBUG:main release")
        self.cond.release()


def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

'''
         -----------------------------------------> x
         |    image_col                        |
         |                                     |
         |                                     |
         |                                     |
         |                                     |
         |                                     | image_row
         |                                     |
         |                                     |
         |                                     |
         |                                     |
         |                                     |
         |      -------------------            |
         |      |  (lt)           |            |
         |      |   .---------.   |            |
         |      |   |         |   |            |
         |      |   |         |...|            |
         |      |   |         |ext|            |
         |      |   `---------`   |            |
         |      |            (rb) |            |
         |      -------------------            |
         |                                     |
         |                                     |
         |                                     |
         ---------------------------------------
         |
         |
        \|/
        y
'''

#return LeftTop,RightBot of crop ROI
#to extend center point but but roi, lt == rb
def calcCropROI( lt, rb, ext, image_row, image_col):
    left, top = lt
    right, bot = rb
    ltPntx=(left - ext > 0) and (left - ext) or 0
    ltPnty=(top - ext > 0) and (top - ext) or 0
    rbPntx=(right + ext > image_col-1) and (image_col-1) or (right + ext)
    rbPnty=(bot + ext > image_row-1) and (image_row-1) or (bot + ext)
    return (ltPntx, ltPnty, rbPntx, rbPnty)


#------------------------------------------------------------------------
##NOTES: function used to dup output log from stdout/stderr. NOT threadsafe.
##DONOT touch the implementation unless you know clearly the effects.

class DupOutput(object):
    def __init__(self):
        self.stdout_save = None
        self.stderr_save = None
        self.debugout    = None
        self.outMap={}
        self.mutex=threading.Lock()
        self.cache={}
        self.pat=re.compile(r'^\s*([a-zA-Z]+)\s*:')
        self.ShowModuleInfo = False #NOTES: enable it for module name and line no. but it will result low perf.

    def dup(self, dupMap) :
        print("stdout/stderr dup!")
        self.outMap = {}
        for key,val in dupMap.items(): #record lowercase
            self.outMap[key]=[i.lower() for i in val]
        self.debugout    = sys.stdout #reserved for debug purpose
        self.stdout_save = sys.stdout
        self.stderr_save = sys.stderr
        sys.stdout       = self
        #NOTES:comment below for debug
        sys.stderr       = self
        self.write("dup output table: %s" % self.outMap)

    def flush(self):
        for obj in self.outMap:
            obj.flush()

    def write(self, s):
        threadidstr=str(threading.currentThread())
        if not self.cache.has_key(threadidstr):
            self.cache[threadidstr]=""
        self.cache[threadidstr] += s

        '''
        #debug:
        self.debugout.write("%s.%s" %(threadidstr,repr(s)))
        '''

        if s == '\n':
            self.mutex.acquire()
            outputline=self.cache[threadidstr]
            self.cache[threadidstr]=""
            self.mutex.release()

            if self.ShowModuleInfo:
                stackline=inspect.stack()
                if len(stackline) >= 2:
                    _, modulepath, lineno, _, _, _ = stackline[1]
                else:
                    _, modulepath, lineno, _, _, _ = stackline[-1]

            logtags=re.match(self.pat, outputline)

            tag="notset" #use level info by default if not explicitly specified
            if logtags:
                prefix=logtags.group(1).lower() #convert to lower before comparison
                for key,val in self.outMap.items():
                    if prefix in val:
                        tag = prefix
                        break
                '''
                #debug
                self.debugout.write("%s\n" % tag)
                '''

            for obj,val in self.outMap.items():
                '''
                #debug
                self.debugout.write("%s\n" %tag)
                '''
                if tag in val:
                    '''
                    #debug
                    self.debugout.write(outputline)
                    '''
                    self.mutex.acquire()
                    if self.ShowModuleInfo:
                        obj.write("%s.%d: %s" % (os.path.split(modulepath)[-1], lineno, outputline))
                    else:
                        obj.write("%s" % outputline)
                    self.mutex.release()

    def recover(self):
        if self.stdout_save and self.stderr_save:
            self.flush()
            self.cache={}
            self.outMap=[]
            sys.stdout = self.stdout_save
            sys.stderr = self.stderr_save
            self.stdout_save=None
            self.stderr_save=None
        print("stdout/stderr recovered!")

'''
usage:
    fd=open('log.txt', 'w+')
    output=DupOutput()
    output.dup({sys.stdout:['critical','error','warning','info'],\
        fd:['critical','error','warning','info','debug','notset']})
    print "test"
    output.recover()
'''

#------------------------------------------------------------------------

#*************************************************************************#
# 
#    Y
#    ^
#    |                                     A'(m,n)
#  t |.....................................
#    |            Y'                    . . .              X'
#    |              \                 .   .   .          /
#    |                \             .     .     .      /
#    |                  \         .       .       .  /
#    |                    \     .         .        / m
#    |                      \ .           .      /
#    |                     n  \           .    /.
#    |                          \         .  /    .
#    |                            \        /       . theta (angle, >0 for counter-clockwise rotation)
#    |                              \    /.        .
#    |                                \/. ...................
#    |                          (p, q)    .
#    |                                    .
#    |                                    .
#    |                                    .
#    |                                    .
#    |                                    .
#    |                                    .
#    ------------------------------------------------------------>X
#                                         s
#
#   X'OY', derived from XOY by offset (p,q), with counter-clockwise angle theta > 0.
#   for point A'(m,n) in X'OY' , its new position in XOY is
#       (p + m*np.cos(theta) - n*np.sin(theta),q + m*np.sin(theta) + n*np.cos(theta))
#
#

def coordinateTransform(offset, theta, pos):
    p, q = offset
    m, n = pos
    s = p + m*np.cos(theta) - n*np.sin(theta)
    t = q + m*np.sin(theta) + n*np.cos(theta)
    return (s,t)

#
#    (0,0)
#    .----------------------.
#    |         .            |
#    |         . pos[1]     |
#    |         .            |
#    |..........            |
#    |   pos[0]             |
#    |                      |
#    |                      | height
#    |                      |
#    |                      |
#    |                      |
#    |                      |
#    `----------------------
#           width
#
#

def pic2screen(cfg, picPos):
    pic_w, pic_h = cfg.get('pic_width'), cfg.get('pic_height')
    scr_w, scr_h = cfg.get("screen_width"), cfg.get("screen_height")

    if picPos[0] > pic_w or picPos[1] > pic_h or pic_w <=0 or pic_h <= 0 or scr_w <= 0 or scr_h <= 0:
        print("ERROR:Invalid parameters. check screen/pic size config or pic position. \
                pic pos %s, pic size %s, screen size %s" % ((picPos),(pic_w, pic_h),(scr_w, scr_h)))
        raise Exception()
    screen_pos = (scr_w*picPos[0]/pic_w, scr_h*picPos[1]/pic_h)
    print("DEBUG:screen size=%s, pic size=%s, pic pos=%s  =>  screen pos=%s" % ((scr_w,scr_h),(pic_w,pic_h),picPos, screeen_pos))
    return screen_pos

def screen2pic(cfg, scrPos):
    pic_w, pic_h = cfg.get('pic_width'), cfg.get('pic_height')
    scr_w, scr_h = cfg.get("screen_width"), cfg.get("screen_height")
    if scrPos[0] > scr_w or scrPos[1] > scr_h or pic_w <=0 or pic_h <= 0 or scr_w <= 0 or scr_h <= 0:
        print("ERROR:Invalid parameters. check screen/pic size config or screen position. \
                screen pos %s, screen size %s, pic size %s" % ((scrPos),(scr_w, scr_h), (pic_w, pic_h)))
        raise Exception()
    pic_pos = (pic_w*scrPos[0]/scr_w, pic_h*scrPos[1]/scr_h)
    print("DEBUG:screen size=%s, pic size=%s, screen pos=%s  =>  pic pos=%s" % ((scr_w,scr_h),(pic_w,pic_h), scrPos, pic_pos))
    return pic_pos

#===========================================================================

gestureTypeList=["onDown", "onUp", "onScroll", "onFling", "onSingleTapUp", "onSingleTapConfirmed"]
gesturePat=re.compile('^\s*\d+\s+([a-zA-Z_]+)\s+')
timePat=re.compile('^\s*(\d+)\s+')
pointPat=re.compile('\(\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*\)')

'''
def parseLine(line):
    event=Container()
    event.type=None
    event.time=-1
    event.pos=[]
    for gestureType in gestureTypeList:
        if re.search(gestureType,line) is not None:
            event.type=gestureType

    gestureTime=re.search(timePat, line)
    if gestureTime is not None:
        event.time = int(gestureTime.group())

    gesturePos=re.findall(pointPat, line)
    #print gesturePos
    for pos in gesturePos:
        event.pos.append(eval(pos))

    print(str(event.type) + ":" + str(event.time) + ":" + str(event.pos))
    return event
'''

def parseLine(line):
    event=Container()
    event.time=-1
    event.type=None
    event.pos=[]

    gestureType=re.match(gesturePat, line)
    if gestureType is not None:
        event.type=str(gestureType.group(1))

    gestureTime=re.match(timePat, line)
    if gestureTime is not None:
        event.time = int(gestureTime.group(1))

    gesturePos=re.findall(pointPat, line)
    #print gesturePos
    for pos in gesturePos:
        event.pos.append(eval(pos))

    print(str(event.type) + ":" + str(event.time) + ":" + str(event.pos))
    return event

def gestureParse(f):
    if not os.path.isfile(f):
        raise Exception("Invalid gesture file %s" % f)

    eventSeq=[]
    for line in file(f):
        event = parseLine(line)
        if event.time != -1 and event.type is not None:
            eventSeq.append(event)
    return eventSeq

def gestureKeyDown(f):
    eventSeq=gestureParse(f)
    if len(eventSeq) != 0:
        event = eventSeq[0]
        if event.type == "onDown":
            return list(event.pos[0])
    return []

def gestureIsSingleClick(f, touch_start):
    eventSeq=gestureParse(f)
    if len(eventSeq) == 3:
        if eventSeq[0].type == "onDown" and eventSeq[1].type == "onSingleTapUp" and eventSeq[2].type == "onSingleTapConfirmed":
            if touch_start not in ['onDown', 'onSingleTapUp', 'onSingleTapConfirmed']:
                raise Exception, "Invalid touch start point specified: " + str(touch_start)
                #touch_start='onSingleTapUp'
            if touch_start == 'onDown':
                touch_start_time = eventSeq[0].time
            elif touch_start == 'onSingleTapUp':
                touch_start_time = eventSeq[1].time
            elif touch_start == 'onSingleTapConfirmed':
                touch_start_time = eventSeq[2].time
            clickDuration = touch_start_time - eventSeq[0].time
            if clickDuration >= 0:
                return True, clickDuration
    return False, 0

def getTouchSeq(gesture_file):
    eventSeq=gestureParse(gesture_file)
    pntList = []
    for event in eventSeq:
        if event.pos:
            pntList.append(event.pos[-1])
    return pntList

#if __name__ == "__main__":
#    f = "Gesture.txt"
#    eventSeq=gestureParse(f)


def compressImg(srcdir, dstfile, cfg):
    gUseColorSpace = cfg.get("COLOR_SPACE")
    if gUseColorSpace in ["BGR","YUV","HLS"]:
        loadtype = 1
    elif gUseColorSpace in ["GRAY"]:
        loadtype = 0
    loadtype=0 #always use GRAY for video compression by default
    if os.path.isfile(dstfile):
        os.remove(dstfile)

    videowrite=None
    generator = PicGenerator(srcdir, ["gif", "bmp", "jpg"])
    for aFile,img in ImageQueue(generator, loadtype):
        f=aFile.split(os.path.sep).pop()
        if videowrite is None:
            framesize=(img.shape[1], img.shape[0])
            #cv.CV_FOURCC('D','I','V','X')
            #cv.CV_FOURCC(*'DIB ')
            videowrite = cv2.VideoWriter(dstfile, cv.CV_CAP_PROP_FOURCC, 10, framesize) #hardcode 10fps, which doesn't matter
            if not videowrite.isOpened():
                print("ERROR: open videowriter failed!")
                return False

        #put image index info
        if False:
            text=f
            ret, textsize = cv2.getTextSize(text=f, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.0, thickness=1)
            x, y = 5, (2*textsize)
            cv2.putText(img, text, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness = 1, lineType=cv2.CV_AA)

        s=videowrite.write(img)
    videowrite=None
    return True

def cur_file_dir():
    path = sys.path[0]
    if os.path.isdir(path):
        return path
    elif os.path.isfile(path):
        return os.path.dirname(path)

def compressImgWithFFMPEG(srcdir, dstfile, cfg):
    if os.path.isfile(dstfile):
        os.remove(dstfile)
    videowrite=None

    generator = PicGenerator(srcdir, ["gif", "bmp", "jpg"])
    ret, aFile = generator.readFilePath(0)
    if not ret:
        print("ERROR: fail to read images from %s" % srcdir)
        return False
    fileName=aFile.split(os.path.sep).pop()
    pat = re.search('(.*?)(\d+)(\..*)$', fileName)
    if pat is None or len(pat.groups()[1])  == 0:
        print("ERROR: invalid file name sequence %s" % aFile)
        return False
    prefix, str_start, postfix = pat.groups()
    print("INFO:Compress images with prefix: %s, postfix: %s, from index: %s" % (prefix, postfix, str_start))

    import platform
    sysarc = platform.system()
    if sysarc == "Linux":
        binary  = os.path.join('ffmpeg','bin_linux','ffmpeg')
    else:
        binary  = os.path.join('ffmpeg','bin','ffmpeg.exe')

    try:
        BIXI_TOOL   = os.environ["BIXI_TOOL"]
        ffmpegexe    = os.path.join(BIXI_TOOL,'OfflineCollect','ImageProcess', binary)
    except Exception, e:
        ffmpegexe   = os.path.join(cur_file_dir(), binary)

    ## cmd=r'ffmpeg\bin\ffmpeg.exe -r 00 -i srcdir\frame_%06d.bmp -y -b:v 9000k dstfile '
    cmd= ffmpegexe + r' -r 10 -i '+ os.path.join(srcdir, prefix+r'%0'+str(len(str_start))+r'd' + postfix) + \
        r' -y -b:v 9000k ' + dstfile

    print("INFO: run cmd: %s" % cmd)
    s=subprocess.Popen(cmd, shell=True)
    s.wait()

    if not os.path.isfile(dstfile):
        print("ERROR: compress file failed. %s" % dstfile)
        return False

    print("INFO: compress image sequence in dir %s done" % srcdir)
    return True

def compressImgWithJPEG(srcdir, dstdir, cfg):
    if os.path.isdir(dstdir):
        __import__('shutil').rmtree(dstdir)
    os.mkdir(dstdir)

    generator = PicGenerator(srcdir, ["bmp"])
    for aFile,img in ImageQueue(generator, loadtype=0):
        f=os.path.basename(aFile)
        new_f = f.split('.')[0]+'.jpg'
        nFile = os.path.join(dstdir, new_f)
        try:
            cv2.imwrite(nFile, img, [int(cv.CV_IMWRITE_JPEG_QUALITY), 80])
        except Exception, e:
            print "ERROR: fail to compress %s" % aFile, e
            return False

    print("INFO: compress image sequence in dir %s done" % srcdir)
    return True
