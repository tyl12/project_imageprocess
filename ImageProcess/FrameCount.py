
'''
read framcount info from file 'result_frame.txt', which has the format like below:
    ----------------------------------------------------
    @FRAMECOUNT
    FILE, DECIMAL_INCR, DECIMAL, WEIGHT
     00000.bmp,          0,          0,         99
     00001.bmp,          0,          0,         99
     00002.bmp,          0,          0,         99
     ...
     00007.bmp,          0,          0,         99
     00008.bmp,          1,          1,         99
     00009.bmp,          1,          1,         99
     ...
     00088.bmp,          15,         15,        99
     00089.bmp,          15,         15,        99
     00090.bmp,          16,         0,         99
     00091.bmp,          16,         0,         99
     ...
     00120.bmp,          19,         3,         99
     00121.bmp,          19,         3,         99
     ...
    ----------------------------------------------------

the output file 'longframe.txt' would be like below. note that if the start/end part of the image sequence
are LF, it will *NOT* be presented.
    ----------------------------------------------------
    #framecount based long frame result
    @FRAMECOUNT
              INDEX,      FRAMECOUNT,          LENGTH,           BEGIN,             END
              ##   16,               0,             781,       00016.bmp,       00796.bmp
                903,               6,              38,       00903.bmp,       00940.bmp
               1005,              10,             164,       01005.bmp,       01168.bmp
               1185,              11,              54,       01185.bmp,       01238.bmp
               1289,              14,              20,       01289.bmp,       01308.bmp
               1623,              32,             138,       01623.bmp,       01760.bmp
               1794,              34,              24,       01794.bmp,       01817.bmp
               2093,              50,             813,       02093.bmp,       02905.bmp
              ## 3180,              66,             820,       03180.bmp,       03999.bmp
    @END
    ----------------------------------------------------
'''
import os
import sys
import math
from localUtils import getContent
from localUtils import getMap


class FrameCountInfo(object):
    def __init__(self):
        self.index=[] #index in origin frame/file sequence
        self.framecount=[]
        self.fList=[]

def getFrameCountInfo(inFile):
    '''
    convert the contents of inFile to FrameCountInfo.
    '''
    fcInfo=FrameCountInfo()
    if not os.path.isfile(inFile):
        print "ERROR:Invalid framecount file %s" % inFile
        return fcInfo

    tag = '@FRAMECOUNT'
    cont = getContent(inFile, tag)
    if not cont:
        print "ERROR:tag %s not found" % tag
        return fcInfo

    KEY_TUPLE=("FILE", "DECIMAL_INCR")
    cont_map = getMap(cont)

    fcInfo.index=range(len(cont_map['FILE']))
    fcInfo.framecount=map(int, cont_map['DECIMAL_INCR'])
    fcInfo.fList=cont_map['FILE']
    return fcInfo

class FrameCountSwitchInfo(object):
    def __init__(self):
        self.switchIndex=[]
        #self.framecount=[] #which could be got from FrameCountInfo
        self.length=[]

def getFrameCountSwitchInfo(fcInfo):
    '''
    get the framecount switch point.
                              ________
                  ___________|
    _____________|
                  ^
                  |
            switchIndex
                  ^          ^
                  |----------|
                    length
    '''
    fcSwitchInfo=FrameCountSwitchInfo()

    index=fcInfo.index
    fc=fcInfo.framecount
    if not index:
        print "Too less framecount sample available"
        return fcSwitchInfo

    diffFCList = map(lambda x,y:y-x, fc[0:-1], fc[1:])
    nonZeroIndex = [idx for idx in range(len(diffFCList)) if diffFCList[idx]!=0]
    switchIndex_begin=[i+1 for i in nonZeroIndex]
    switchIndex_begin.insert(0,0)
    switchIndex_end=nonZeroIndex[:] #copyout instead of reference
    switchIndex_end.append(index[-1])

    fcSwitchInfo.switchIndex=switchIndex_begin #the first point included as 0
    fcSwitchInfo.length=map(lambda x,y:y-x+1, switchIndex_begin, switchIndex_end) #the first point is also included

    return fcSwitchInfo

class LongFrameInfo(object):
    def __init__(self):
        self.LFIndex=[]
        self.length=[] #which will be different from FrameCountSwitchInfo

#TODO: some hardcoding parameters
threshold_single = 1.5
lcd_fps = 60 #TODO
camera_fps=1000
vsync_samplingcnt = camera_fps*1.0/lcd_fps
threshold = int(vsync_samplingcnt*threshold_single)

def getLongFrameInfo(fcInfo, fcSwitchInfo, bStrip):
    '''
    get the LFs index and length based on input framecount switch point info
    bStrip: whether strip the LF at start/stop of target sequence
    '''
    lfInfo=LongFrameInfo()

    switchIndex=fcSwitchInfo.switchIndex
    length=fcSwitchInfo.length
    lfIndex=[]
    lfLength=[]
    for i in range(len(switchIndex)):
        if length[i] > threshold:
            lfIndex.append(switchIndex[i])
            lfLength.append(length[i])

    #
    phase_start_index = 0
    phase_stop_index = fcInfo.index[-1]

    #remove the first/last LF if it's at begin/end point
    if not lfIndex:
        print "No LF found"
        return lfInfo, phase_start_index, phase_stop_index

    start_bool = (lfIndex[0]==fcInfo.index[0]) #whether LF start from begin of whole sequence
    end_bool = ((lfIndex[-1]+lfLength[-1]-1)==fcInfo.index[-1])
    if len(lfIndex)==1 and start_bool and end_bool:
        print "ERROR: no change in whole image sequence or no framecount display??"
        return lfInfo, -1, -1

    if bStrip:
        print "Notes: the start/stop LF will be striped from LF list"
        #NOTES: the first/last LF will NOT be presented if it starts from sequence begin/end.
        if start_bool:
            phase_start_index = lfIndex[0] + lfLength[0]
            del lfIndex[0]
            del lfLength[0]
        if end_bool:
            phase_stop_index = lfIndex[-1]
            del lfIndex[-1]
            del lfLength[-1]

    print 'framecount based phase start index: %s, phase stop index: %s' % (phase_start_index, phase_stop_index)

    #do 1 vsync shift for framecount internal strategy
    shift = int(math.ceil(vsync_samplingcnt))
    lfIndex = [idx+shift for idx in lfIndex] #shift to real LF start point
    lfLength = [dur-shift for dur in lfLength]

    lfInfo.LFIndex=lfIndex
    lfInfo.length=lfLength
    return lfInfo, phase_start_index, phase_stop_index

def appendLFResult(outFile, fcInfo, fcSwitchInfo, lfInfo, idxOffset):
    with open(outFile, 'a') as f:
        f.write('#framecount based long frame result\n')
        f.write('@FRAMECOUNT\n')
        lfIndex=lfInfo.LFIndex
        lfLength=lfInfo.length
        lfFrameCount=[fcInfo.framecount[i] for i in lfIndex]
        lfFile_begin=[fcInfo.fList[i] for i in lfIndex]

        lfIndex_end=map(lambda x,y:x+y-1, lfIndex, lfLength)
        lfFile_end=[fcInfo.fList[i] for i in lfIndex_end]

        taglist=('INDEX', 'LENGTH', 'FRAMECOUNT', 'BEGIN', 'END')
        f.write('%15s, %15s, %15s, %25s, %25s\n' % taglist)
        for i in range(len(lfIndex)):
            f.write('%15s, %15s, %15s, %25s, %25s\n' % (lfIndex[i]+idxOffset, lfLength[i], lfFrameCount[i], lfFile_begin[i], lfFile_end[i]))
        f.write('@END\n\n')

def getFPSFromFile(summaryFile):
    fps = -1
    if not os.path.isfile(summaryFile):
        print "Invalid image based fps file %s" % summaryFile
    else:
        prefix='equivalent fps during phase transition'
        import re
        rep = re.compile(prefix+'[\s]*=[\s]*([\d.]+)[\s]*fps')
        fps=None
        for line in file(summaryFile):
            val = re.search(rep, line.strip())
            if val:
                fps=val.group(1)
                break

    if not fps:
        print "Fail to get fps from file %s" % summaryFile
        fps = -1

    print ">>>>>Image based fps from file %s is %s" %(summaryFile, fps)
    return fps


def getImageLFResult(outFile, imgFile, bStrip, idxOffset):
    '''
    bStrip: whether strip the LF at start/stop of target sequence
    '''
    if not os.path.isfile(imgFile):
        print "No image based LF file available", imgFile
        return None, -1, -1

    ##------------------------------------------------
    tag = '@LONGFRAME'
    cont = getContent(imgFile, tag)
    if not cont:
        print "ERROR:tag %s not found" % tag
        return None, -1, -1

    KEY_TUPLE=("INDEX", "FILE", "FRAMECOUNT", "LF_PNTCNT_THRESHOLD")
    cont_map = getMap(cont)

    LF_index     = map(int, cont_map['INDEX'])
    LF_length    = map(int, cont_map['FRAMECOUNT'])
    LF_fList     = cont_map['FILE']
    LF_threshold = map(int, cont_map['LF_PNTCNT_THRESHOLD'])


    ##------------------------------------------------
    tag = '@DURATION'
    cont = getContent(imgFile, tag)
    KEY_TUPLE = ("BEGIN_IDX", "BEGIN_FILE", "END_IDX", "END_FILE", "PHASE_START_IDX", "PHASE_START_FILE", "PHASE_STOP_IDX", "PHASE_STOP_FILE", "FRAMES_CNT", "DURATION")
    cont_map = getMap(cont)

    begin_idx        = int(cont_map['BEGIN_IDX'][0])
    begin_file       = cont_map['BEGIN_FILE'][0]
    end_idx          = int(cont_map['END_IDX'][0])
    end_file         = cont_map['END_FILE'][0]

    phase_start_index= int(cont_map['PHASE_START_IDX'][0])
    phase_stop_index = int(cont_map['PHASE_STOP_IDX'][0])
    phase_start_file = cont_map['PHASE_START_FILE'][0]
    phase_stop_file  = cont_map['PHASE_STOP_FILE'][0]

    if not bStrip:
        ##------------------------------------------------
        tag = '@SWITCH'
        cont = getContent(imgFile, tag)

        KEY_TUPLE = ("INDEX", "FILE", "DIRECTION", "PREV_CNT", "NEXT_CNT")
        cont_map = getMap(cont)

        switch_index    = map(int, cont_map['INDEX'])
        switch_file     = cont_map['FILE']
        switch_direct   = cont_map['DIRECTION']
        switch_pre      = map(int, cont_map['PREV_CNT'])
        switch_next     = map(int, cont_map['NEXT_CNT'])

        if switch_direct[0] == 'up':
            start_dur = switch_pre[0]
        else:
            start_dur = 0;

        if switch_direct[-1] == 'down':
            stop_dur = switch_next[-1]
        else:
            stop_dur = 0

        if start_dur >= vsync_samplingcnt:
            LF_index.insert(0,0)
            LF_length.insert(0,start_dur)
            LF_fList.insert(0,begin_file)
            LF_threshold.insert(0,vsync_samplingcnt)
        if stop_dur >= vsync_samplingcnt:
            LF_index.append(switch_index[-1])
            LF_length.append(stop_dur)
            LF_fList.append(end_file)
            LF_threshold.append(vsync_samplingcnt)

        phase_start_index = begin_idx
        phase_stop_index  = end_idx

    with open(outFile, 'a') as f:
        f.write('\n')
        tag='@IMAGE'
        f.write('%s\n' % tag)
        f.write('%15s, %15s, %25s, %25s\n' % ('INDEX', 'LENGTH', 'FILE', 'LF_PNTCNT_THRESHOLD'))
        for i in range(len(LF_index)):
            f.write('%15s, %15s, %25s, %25s\n' % (LF_index[i]+idxOffset, LF_length[i], LF_fList[i], LF_threshold[i]))
        f.write('@END\n')

    print 'image diff based phase start index: %s, phase stop index: %s' % (phase_start_index, phase_stop_index)

    lfInfo = LongFrameInfo()
    lfInfo.LFIndex = [int(i) for i in LF_index]
    lfInfo.length = [int(i) for i in LF_length]
    return lfInfo, phase_start_index, phase_stop_index

#=================================================================================================

def calcFPS(lfInfo, phase_start_index,phase_stop_index):
    '''
    lfInfo should be object of LongFrameInfo, which contains longframe start index and length.
    '''
    index=lfInfo.LFIndex
    length=lfInfo.length
    if phase_start_index == phase_stop_index:
        print "No phase info available"
        fps = 0
    elif phase_start_index != phase_stop_index and not index:
        print "No LF info available"
        fps = 60
    else:
        LFduration=sum(length)
        fps = lcd_fps*(1-1.0*LFduration/(phase_stop_index-phase_start_index+1))

    print ">>>>>fps is %s" %(fps)
    return fps

def appendFPS(outFile, fps, tag):
    with open(outFile,'a') as f:
        f.write("\n%s\n" % tag)
        f.write("FPS\n%s\n" % fps)
        f.write("@END\n")


#=================================================================================================

def parseLF(workingdir):
    bStrip_Start_End_LF = False

    # process framecount based results
    fcFile       = os.path.join(workingdir, 'result_frame.txt')
    idxFile      = os.path.join(workingdir, 'src', 'summary_manual.txt')

    idxOffset = 0
    if os.path.isfile(idxFile):
        import re
        firstLine = file(idxFile).readline()
        idxs = re.search(r'\s*index-(\d+):.*',firstLine)
        if idxs:
            idxOffset = int(idxs.group(1))
    print "LF image start index Offset:", idxOffset

    fcInfo       = getFrameCountInfo(fcFile)
    fcSwitchInfo = getFrameCountSwitchInfo(fcInfo)
    lfInfo,phase_start_index,phase_stop_index = getLongFrameInfo(fcInfo, fcSwitchInfo, bStrip_Start_End_LF)

    outFile=os.path.join(workingdir, 'longframe.txt')
    if os.path.isfile(outFile):
        os.remove(outFile)

    appendLFResult(outFile, fcInfo, fcSwitchInfo, lfInfo, idxOffset)
    fps = calcFPS(lfInfo, phase_start_index,phase_stop_index)
    appendFPS(outFile, fps, '@FRAMECOUNT_FPS')
    ##------------------------------------------------------------------------------------------

    # additional operations, process image based result
    imgFile      = os.path.join(workingdir, 'phase_file.txt')
    lfInfo, phase_start_index, phase_stop_index = getImageLFResult(outFile, imgFile, bStrip_Start_End_LF, idxOffset)

    fps = calcFPS(lfInfo, phase_start_index,phase_stop_index)
    appendFPS(outFile, fps, '@IMAGE_FPS')
    ##------------------------------------------------------------------------------------------


if __name__ == '__main__':
    if len(sys.argv) == 1:
        parseLF('.')
    else:
        workingdir = sys.argv[1]
        parseLF(workingdir)

