
from optparse import OptionParser
usage = "usage: %prog [-option]"
parser = OptionParser(usage)
parser.add_option("-d", "--directory", type="string", dest = "testdir", action = "store",
        help = r"use this option to specify the testcase directory of one product to run, \
                eg. python Setup\MYPROJ.py -d PR4")
parser.add_option("-c", "--case", type="string", dest="case", action = "store",
        help = r"use this option to specify the testcase to run, \
                eg. python Setup\MYPROJ.py -c PR4__case10_launch_maps")
parser.add_option("-w", "--web", dest="web", action = "store_true",
        help = r"use this option to launch the webserver, \
                eg. python Setup\MYPROJ.py -w")

(options, args) = parser.parse_args()

from myproj_config import *
import subprocess

if options.testdir:
    cmd=["python", os.path.abspath(os.path.join(MYPROJ_TOOL,"MYPROJ_pipeline.py")), "-d", options.testdir]
    print cmd
    ret = subprocess.call(cmd)
    exit(ret)

if options.case:
    cmd=["python", os.path.abspath(os.path.join(MYPROJ_TOOL,"MYPROJ_pipeline.py")), "-c", options.case]
    print cmd
    ret = subprocess.call(cmd)
    exit(ret)

if options.web:
    cmd=["python", os.path.abspath(os.path.join(MYPROJ_ROOT, "MYPROJBench", "mysite", "manage.py")), "runserver"]
    print cmd
    ret = subprocess.call(cmd)
    exit(ret)

