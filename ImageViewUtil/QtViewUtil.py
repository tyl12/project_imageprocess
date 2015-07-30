
from PyQt4 import QtGui
from PyQt4 import QtCore
import shutil
import subprocess
import datetime
import cv2
from cv2 import cv
import numpy as np
import sys
import os

def centerWindow(winobj):
    screen = QtGui.QDesktopWidget().screenGeometry()
    size = winobj.geometry()
    winobj.move((screen.width()-size.width())/2, (screen.height()-size.height())/2)

class VideoPanel(QtGui.QWidget):
    def __init__(self, parent, videoname):
        super(VideoPanel, self).__init__()
        self.parent=parent
        self.prepareVideo(str(videoname))
        self.initUI()

    def initUI(self):
        self.slider=QtGui.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.totalFrameNumber)
        self.layout = QtGui.QVBoxLayout()

        self.imagelb=QtGui.QLabel()
        self.imagelb.setToolTip("doubleClickRightMouse to pin location")
        self.layout.addWidget(self.imagelb)
        self.layout.addWidget(self.slider)

        self.pntInfo=[]
        self.showPntGroup=QtGui.QButtonGroup(self)
        self.showPntList=[]
        for i in [0,1]:
            button = QtGui.QPushButton('')
            button.setStyleSheet("background-color: rgb(218, 133, 250)");
            self.showPntList.append(button)
            self.showPntGroup.addButton(button, i)
        self.connect(self.showPntGroup, QtCore.SIGNAL('buttonClicked(int)'), self.click_buttonGroup)

        self.delta=QtGui.QLabel('Delta Index:')
        self.delta.setStyleSheet("background-color: rgb(218, 133, 250)");

        self.vlay = QtGui.QVBoxLayout()
        self.vlay.addWidget(self.showPntList[0])
        self.vlay.addWidget(self.showPntList[1])
        self.vlay.addWidget(self.delta)

        self.layout.addLayout(self.vlay)

        self.setLayout(self.layout)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        centerWindow(self)
        self.connect(self.slider, QtCore.SIGNAL('valueChanged(int)'), self.display)
        self.display(0)

    def prepareVideo(self,videoname):
        print "Decode stream file %s" %videoname
        self.videoname=videoname
        cap=cv2.VideoCapture(str(videoname))
        if not cap.isOpened():
            msg="ERROR:Failed to open video %s" % videoname
            print msg
            QtGui.QMessageBox.about(None,'Error',msg)
            return False
        self.cap=cap
        self.totalFrameNumber = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))
        print("INFO:Total frame count=%d" % self.totalFrameNumber)

        self.width = cap.get(cv.CV_CAP_PROP_FRAME_WIDTH)
        self.height = cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT)
        self.fps = cap.get(cv.CV_CAP_PROP_FPS)
        self.codec = cap.get(cv.CV_CAP_PROP_FOURCC)
        self.docvt = cap.get(cv.CV_CAP_PROP_CONVERT_RGB)
        print("INFO:frame size=%dx%d, fps=%d, codec=%s, docvt=%d" %(self.width, self.height, self.fps, self.codec, self.docvt))
        return True

    def display(self, index):
        ##FIXME:here re-open stream once again as cv2 CV_CAP_PROP_POS_FRAMES operation may skip some frames,
        ##and could not handle backward seek correctly. maybe one bug.

        self.cap=cv2.VideoCapture(str(self.videoname))

        if not self.cap.isOpened():
            msg="ERROR:Failed to open video %s" % self.videoname
            print msg
            QtGui.QMessageBox.about(None,'Error',msg)
            return False

        ret, img = self.decodeFile(self.cap, index, self.docvt)
        if not ret or img is None:
            print("ERROR: Fail to get frame %d" %i)
            return
        print "display index: %s" % index

        ##for debug only
        #cv2.imshow("track", img)
        #cv2.waitKey(1)
        img=cv2.cvtColor(img, cv.CV_BGR2RGB) #convert color, otherwise, dstcvimg.copy() & QtGui.QImage.rgbSwapped() is required
        stride=img.strides[0]
        h,w=img.shape[0:2]
        data=img.data
        image = QtGui.QImage(data, w, h, stride,  QtGui.QImage.Format_RGB888) #image.rgbSwapped()

        screen = QtGui.QDesktopWidget().screenGeometry()
        image_scaled=image.scaled(screen.width()/2, screen.height()/2, QtCore.Qt.KeepAspectRatio)
        self.imagelb.setPixmap(QtGui.QPixmap.fromImage(image_scaled))
        self.setWindowTitle(os.path.basename(self.videoname) + ":"+ str(index))

    def decodeFile(self, cap, idx, docvt):
        if not cap.isOpened():
            print("ERROR:Failed to open video %s" % source)
            return False
        ret = True
        frame = None
        cap.set(cv.CV_CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            print("ERROR:fail to read frame %d" % idx)
        return ret, frame

    def keyPressEvent(self, event):
        print "keyPressEvent, %s" % event.key()
        i=self.slider.value()
        maxlen = self.totalFrameNumber
        if event.key() in [QtCore.Qt.Key_B, QtCore.Qt.Key_Left]:
            if i>0:
                i-=1
                self.slider.setValue(i)
        elif event.key() in [QtCore.Qt.Key_F, QtCore.Qt.Key_Right]:
            if i<maxlen:
                i+=1
                self.slider.setValue(i)
        event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() != QtCore.Qt.RightButton:
            event.accept()
            return
        print "mouseDoubleClickEvent"

        index=self.slider.value()
        imgfile = ""
        print "time tag for file: %s, index: %d" % (imgfile, index)
        if len(self.pntInfo) == 2:
            self.pntInfo=[]
        self.pntInfo.append((imgfile, int(index)))

        for i in range(2):
            if i <= len(self.pntInfo)-1:
                self.showPntList[i].setText('%s:%s'%(self.pntInfo[i][0], self.pntInfo[i][1]))
            else:
                self.showPntList[i].setText('')
        if len(self.pntInfo) == 2:
            self.delta.setText('Delta Index:%d' % (self.pntInfo[1][1]-self.pntInfo[0][1]))
        else:
            self.delta.setText('Delta Index:')

        event.accept()
        return

    def click_buttonGroup(self, i):
        if i not in [0,1]:
            return
        info = self.showPntList[i].text()
        index, f = self.getClosestImgFromTimeTag(info.split(':')[1])
        self.slider.setValue(index)


class PicPanel(QtGui.QWidget):
    def __init__(self, parent, folder):
        super(PicPanel, self).__init__()
        self.parent=parent
        self.folder=folder

        fList=sorted([pic for pic in os.listdir(self.folder) if pic.split('.')[-1].lower() in ['gif', 'bmp', 'jpg', 'png']])
        if fList is None or len(fList) == 0:
            print "ERROR: No images found in %s" % folder
            QtGui.QMessageBox.about(None,'Error','No images found in %s' % folder)
            return

        fList.sort()
        self.fList=fList
        self.initUI()

    def initUI(self):
        self.slider=QtGui.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.fList)-1)
        self.layout = QtGui.QVBoxLayout()

        self.imagelb=QtGui.QLabel()
        self.imagelb.setAlignment(QtCore.Qt.AlignLeft)
        self.imagelb.setToolTip("doubleClickRightMouse to pin location")
        self.layout.addWidget(self.imagelb)
        self.layout.addWidget(self.slider)

        self.pntInfo=[]
        #manual split button
        self.showPntGroup=QtGui.QButtonGroup(self)
        self.showPntList=[]
        for i in [0,1]:
            button = QtGui.QPushButton('')
            button.setStyleSheet("background-color: rgb(218, 133, 250)");
            self.showPntList.append(button)
            self.showPntGroup.addButton(button, i)
        self.connect(self.showPntGroup, QtCore.SIGNAL('buttonClicked(int)'), self.click_buttonGroup)

        self.delta=QtGui.QLabel('Delta Index:')
        self.delta.setStyleSheet("background-color: rgb(218, 133, 250)");

        #summary_manual.txt
        #initialize the manual split rows with contents in summary_manual.txt
        outfile = os.path.join(self.folder, 'summary_manual.txt')
        if os.path.isfile(outfile):
            with open(outfile, 'r') as f:
                lines = [line.strip() for line in f.readlines()]
                print "Load manual split result from file: %s, contents: %s" % (outfile, lines)
                if len(lines) > 3:
                    print "ERROR: invalid manual split result file: %s, %s" % (outfile, lines)
                else:
                    try:
                        for i in range(len(self.showPntList)):
                            pat=lines[i]
                            self.showPntList[i].setText(pat)

                        pat=lines[2]
                        self.delta.setText(pat)
                    except Exception,e:
                        pass

        self.save=QtGui.QPushButton('Save')
        self.save.setStyleSheet("background-color: rgb(218, 133, 250)");
        self.connect(self.save, QtCore.SIGNAL('clicked()'), self.click_save)

        self.vlay = QtGui.QVBoxLayout()

        self.commandhistory=QtGui.QLabel('Command List:')
        self.vlay.addWidget(self.commandhistory)

        #cmdlist.txt column
        #the cmdlist.txt file format should be:
        #   start:monthday-hour-min-sec-millisecond:eventtype
        #   stop:monthday-hour-min-sec-millisecond:eventtype
        #   ...
        cmdfile_list = [os.path.join(self.folder, 'cmdlist.txt'), os.path.join(self.folder, '..', 'cmdlist.txt')]
        cmdfile = None
        for f in cmdfile_list:
            if os.path.isfile(f):
                cmdfile = f
                print "Found cmdlist file: %s" % cmdfile
                break
        if cmdfile:
            with open(cmdfile) as f:
                lines = [line.split()[0] for line in f.readlines()]
                print "Load cmdlist from file: %s, contents: %s" % (cmdfile, lines)

            self.comboBox = QtGui.QComboBox()
            self.comboBox.addItems(lines)
            self.vlay.addWidget(self.comboBox)

            self.connect(self.comboBox, QtCore.SIGNAL('activated(const QString&)'), self.comboChange)
        else:
            print "No cmdlist file found in: %s" % cmdfile

        self.pinlocation=QtGui.QLabel('User Specified Location:')

        self.vlay.addWidget(self.pinlocation)
        for i in range(len(self.showPntList)):
            self.vlay.addWidget(self.showPntList[i])

        self.hlay_line = QtGui.QHBoxLayout()
        self.hlay_line.addWidget(self.delta)
        self.hlay_line.addWidget(self.save)
        self.hlay_line.setStretch(0,5)
        self.hlay_line.setStretch(1,1)
        self.vlay.addLayout(self.hlay_line)

        self.layout.addLayout(self.vlay)

        self.setLayout(self.layout)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        centerWindow(self)
        self.connect(self.slider, QtCore.SIGNAL('valueChanged(int)'), self.display)
        self.display(0)

    def getClosestImgFromTimeTag(self, timetag):
        if not self.fList:
            return -1, None
        for index,f in enumerate(self.fList):
            print timetag 
            if f.split('.')[0] >= timetag:
                break
        return index, f

    def comboChange(self, input_qstring):
        line = str(input_qstring)
        print "comboChange to: %s" % line
        pin_tag=line.split(':')[1]
        fsample=self.fList[0].split('.')[0]
        sep_list=['-','_']
        for sep in sep_list:
            if len(pin_tag.split(sep)) != len(fsample.split(sep)):
                print "time pattern in cmdlist file: %s is different with file name: %s" % (pin_tag, fsample)
                return
        index, f = self.getClosestImgFromTimeTag(pin_tag)

        print "Jump to nearest frame: %s, index: %s" %(f,index)
        self.slider.setValue(index)
        return

    def display(self, index):
        image = QtGui.QImage()
        imgfile = self.fList[index]
        print "display file: %s, index: %s" % (imgfile,index)

        screen = QtGui.QDesktopWidget().screenGeometry()
        image.load(os.path.join(self.folder, imgfile))
        image_scaled=image.scaled(screen.width()/2, screen.height()/2, QtCore.Qt.KeepAspectRatio)
        self.imagelb.setPixmap(QtGui.QPixmap.fromImage(image_scaled))
        self.setWindowTitle(imgfile+":"+str(index))

    def keyPressEvent(self, event):
        print "keyPressEvent, %s" % event.key()
        i=self.slider.value()
        maxlen = len(self.fList)-1
        if event.key() in [QtCore.Qt.Key_B, QtCore.Qt.Key_Left]:
            if i>0:
                i-=1
                self.slider.setValue(i)
        elif event.key() in [QtCore.Qt.Key_F, QtCore.Qt.Key_Right]:
            if i<maxlen:
                i+=1
                self.slider.setValue(i)
        event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() != QtCore.Qt.RightButton:
            event.accept()
            return
        print "mouseDoubleClickEvent"

        index=self.slider.value()
        imgfile = self.fList[index]
        print "time tag for file: %s, index: %d" % (imgfile, index)

        if len(self.pntInfo) == 2:
            self.pntInfo=[]
        self.pntInfo.append((imgfile, int(index)))

        for i in range(2):
            if i <= len(self.pntInfo)-1:
                imgfile,index = self.pntInfo[i][0], self.pntInfo[i][1]
                normStr=self.getNormStr(imgfile, index)
                self.showPntList[i].setText(normStr)
            else:
                self.showPntList[i].setText('')
        if len(self.pntInfo) == 2:
            self.delta.setText('Delta Index:%d' % (self.pntInfo[1][1]-self.pntInfo[0][1]))
        else:
            self.delta.setText('Delta Index:')

        event.accept()
        return

    def click_buttonGroup(self, i):
        if i not in [0,1]:
            return
        info = self.showPntList[i].text()
        index, f = self.getClosestImgFromTimeTag(info.split(':')[1])
        self.slider.setValue(index)

    def click_save(self):
        outfile = os.path.join(self.folder, 'summary_manual.txt')
        if os.path.isfile(outfile):
            os.remove(outfile)
        with open(outfile, 'a') as f:
            cnt=0
            for button in self.showPntList:
                normStr=button.text()
                if normStr != '':
                    f.write( '%s\n' % normStr)
                    cnt+=1
            if cnt == 2:
                f.write(self.delta.text()+'\n')
        print "Save manual split result to: %s" % outfile
        QtGui.QMessageBox.about(self,'Info','Manual split result saved to:\n%s.' % outfile)

    def getNormStr(self, imagefile, index):
        ret = "index-%d:%s:ManualSplit" % (index, imagefile.split('.')[0])
        return ret

class DataInspector(QtGui.QWidget):
    def __init__(self, parent = None):
        super(DataInspector, self).__init__()
        self.initUI()

    def initUI(self):
        self.picButton=QtGui.QPushButton('PicViewer')
        self.videoButton=QtGui.QPushButton('VideoViewer')
        self.textButton=QtGui.QPushButton('TextViewer')

        self.connect(self.picButton, QtCore.SIGNAL('clicked()'), self.click_viewPic)
        self.connect(self.videoButton, QtCore.SIGNAL('clicked()'), self.click_viewVideo)
        self.connect(self.textButton, QtCore.SIGNAL('clicked()'), self.click_viewText)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.picButton)
        layout.addWidget(self.videoButton)
        layout.addWidget(self.textButton)
        self.setLayout(layout)

        self.setWindowTitle('DataInspector')
        self.setWindowModality(QtCore.Qt.ApplicationModal) #should before show
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        centerWindow(self)

    def click_viewPic(self):
        #specify the folder contain images to display
        dirpath = str(QtGui.QFileDialog.getExistingDirectory(
            self,
            self.tr("Open Directory"),
            QtCore.QDir.currentPath(),
            QtGui.QFileDialog.ShowDirsOnly | QtGui.QFileDialog.DontResolveSymlinks)
            )
        if os.path.isdir(dirpath):
            print "viewPic from folder: %s" % dirpath

            self.picPan = PicPanel(self, dirpath)
            self.picPan.setWindowModality(QtCore.Qt.ApplicationModal)
            self.picPan.show()

    def click_viewVideo(self):
        filename = QtGui.QFileDialog.getOpenFileName(
                self,
                self.tr("Open Document"),
                QtCore.QDir.currentPath(),
                "Document files (*.avi *.rmvb *.mpeg);;All files(*.*)"
                )
        if os.path.isfile(filename):
            print "viewVideo file: %s" % filename

            self.videoPan = VideoPanel(self, filename)
            self.videoPan.setWindowModality(QtCore.Qt.ApplicationModal)
            self.videoPan.show()

    def click_viewText(self, filename=None):
        if filename is None:
            filename = QtGui.QFileDialog.getOpenFileName(
                    self,
                    self.tr("Open Document"),
                    QtCore.QDir.currentPath(),
                    "Document files (*.txt);;All files(*.*)"
                    )
            print "viewText file: %s" % str(filename)
        filename=str(filename)
        if not os.path.isfile(filename):
            QtGui.QMessageBox.about(self,'Error','Invalid file specified, %s!' % filename)
            return

        with open(filename) as f:
            fcontents = f.read()

        self.browser = QtGui.QTextBrowser()
        self.browser.setPlainText(fcontents)
        self.browser.append('\n\r')
        self.browser.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self.browser.setWindowTitle('Text Contents')
        screen = QtGui.QDesktopWidget().screenGeometry()
        self.browser.resize(screen.width()/2, screen.height()/2)
        self.browser.setWindowModality(QtCore.Qt.ApplicationModal)
        self.browser.show()


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    inspect=DataInspector()
    inspect.show()
    sys.exit(app.exec_())

