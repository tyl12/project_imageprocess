import sys
import os
import subprocess
#runfile_list = ['home-gesture', 'pre-gesture', 'gesture']
runfile_list = ['pre-run.sh', 'run.sh', 'post-run.sh']

def disp_description(casepath):
    print("------------------------------------------")
    desc_file = os.path.join(casepath, r"case_description.txt")
    if os.path.isfile(desc_file):
        with open(desc_file, 'r') as f:
            desc = f.read()
        try:
            print(desc)
        except:
            pass


def push_casescript(casepath, phonepath):
    ##
    cmdList = [\
            'adb shell rm -rf %s' % phonepath,\
            'adb shell mkdir -p %s' % phonepath,\
            'adb shell chmod 777 %s' % phonepath,\
            ]

    filelist = os.listdir(casepath)
    for f in filelist:
        fPath = os.path.join(casepath, f)
        cmd = r'adb push %s %s' %(fPath, phonepath)
        cmdList.append(cmd)
    cmdList.append(r'adb shell chmod 777 %s/*' % phonepath)

    for cmd in cmdList:
        print("run cmd: -------> {cmd}".format(cmd=cmd))
        subprocess.call(cmd)


def run_case(casepath, phonepath):
    global runfile_list
    for cmdScript in runfile_list:
        src_file = os.path.join(casepath, cmdScript)##
        scriptPath = phonepath + r"/" + cmdScript
        if os.path.isfile(src_file):
            cmd = r'adb shell sh %s %s' % (scriptPath, phonepath)
            print("------------------------------------------")
            print("begin to run cmd: %s" % cmd)
            print("------------------------------------------")
            raw_input("press any key to continue...")
            subprocess.call(cmd)
            subprocess.call("sleep 2")


#check input parameters before call in
def loopdir(productdir, casedir):
    ##
    phonepath = r'/data/local/tmp/testfolder/testcase'
    if casedir is not None:
        caselist = [casedir]
    else:
        caselist = [case for case in os.listdir(productdir) if \
                ( os.path.isdir(os.path.join(productdir, case)) and not case.lower().startswith('notuse') )]
    print("list all testcases to run:\n %s" % caselist)
    for case in caselist:
        print("==========================================")
        print("Start to run case: {case}".format(case=case))
        raw_input("press any key to continue...")
        casepath = os.path.join(productdir, case)

        disp_description(casepath)
        push_casescript(casepath, phonepath)
        run_case(casepath, phonepath)

        print("End to run case: {case}".format(case=case))
        raw_input("press any key to continue...")
        print("==========================================")


if __name__ == "__main__":
    if len(sys.argv) not in [2, 3]:
        print("Usage: python " + sys.argv[0] + " <abspath to product dir of testcase>" + " <casedir>")
        print("Notes: push arm/x86 events file to /data/local/tmp/ beforehand")
        exit(0)
    productdir = sys.argv[1]
    try:
        casedir = sys.argv[2]
    except:
        casedir = None

    print('testcase:{productdir}'.format(productdir=productdir))
    print('testcase:{casedir}'.format(casedir=casedir))

    if not os.path.isdir(productdir):
        print('ERROR: invalid product dir')
        exit(-1)
    if casedir is not None:
        if not os.path.isdir(os.path.join(productdir, casedir)):
            print('ERROR: invalid productdir & casedir')
            exit(-1)
    loopdir(productdir, casedir)


