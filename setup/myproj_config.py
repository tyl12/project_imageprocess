import os
import sys
import platform

#*******************************************************************************************#
sysarc = platform.system()
if sysarc == "Linux":
    print("Running on Linux system")
else:
    print("Running on Windows system")

def cur_file_dir():
    path = sys.path[0]
    if os.path.isdir(path):
        return path
    elif os.path.isfile(path):
        return os.path.dirname(path)

#RO, donot touch
ORIG_PATH       = os.path.abspath(os.getcwd())
SCRIPT_PATH     = os.path.abspath(os.path.join(cur_file_dir()))
MYPROJ_ROOT       = os.path.abspath(os.path.join(SCRIPT_PATH, r".."))
MYPROJ_TOOL       = os.path.abspath(os.path.join(MYPROJ_ROOT, r"MYPROJTool")) #tool path

#could be adjusted as required
MYPROJ_TESTCASE   = os.path.abspath(os.path.join(MYPROJ_ROOT, r"MYPROJTestCase")) #which contains all the input gesture
#MYPROJ_OUTPUT     = os.path.abspath(os.path.join(MYPROJ_ROOT, r"MYPROJOutput"))
##FIXME: for tomcat version
MYPROJ_OUTPUT     = os.path.abspath(os.path.join(MYPROJ_ROOT, "MYPROJBench", "webapps", "MYPROJOutput")) #which contains the output files
MYPROJ_OUT_DATA   = os.path.abspath(os.path.join(MYPROJ_OUTPUT, r"MYPROJData"))
MYPROJ_OUT_REPORT = os.path.abspath(os.path.join(MYPROJ_OUTPUT, r"MYPROJReport"))

environ_list=[  \
    'ORIG_PATH'       , \
    'SCRIPT_PATH'     , \
    'MYPROJ_ROOT'       , \
    'MYPROJ_TOOL'       , \
    'MYPROJ_OUTPUT'     , \
    'MYPROJ_OUT_DATA'   , \
    'MYPROJ_OUT_REPORT' , \
    'MYPROJ_TESTCASE'   , \
    ]
for env in environ_list:
    print "export environ: %s: %s" % (env, eval(env))
    os.environ[env]=eval(env)

