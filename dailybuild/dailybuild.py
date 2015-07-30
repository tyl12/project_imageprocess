#!/usr/bin/python

import time,os,shutil
import re
import os
import sys
import platform
import subprocess
import filecmp

#=====================================================================================#
##self upgrade hook
def cur_file_dir():
    path = sys.path[0]
    if os.path.isdir(path):
        return path
    elif os.path.isfile(path):
        return os.path.dirname(path)

def upgrade(cur_script, new_script):
    try:
        if not os.path.isfile(new_script):
            return False
        if filecmp.cmp(cur_script, new_script, shallow=False):
            print("scripts same, skip upgrade")
            return False
        else:
            print("scripts different, do upgrade")
            shutil.copy(new_script, cur_script)
    except Exception, e:
        print("script not updated")
        return False
    else:
        print("script updated")
        return True

#=====================================================================================#
#mail list:
mail_addr=[\
        r'tyl12@sjtu.org', \
        r'yiliang.teng@intel.com', \
        ]

#=====================================================================================#
print sys.argv

dry_run         = False
enable_upgrade  = True

if len(sys.argv)>1:
    if '-d' in sys.argv:
        dry_run=True
        print("in dry run mode")
    if '-r' in sys.argv:
        enable_upgrade=False

sysarc = platform.system()
if sysarc == "Linux":
    print("Running on Linux system")
    BUILDROOT_DIR  = r"/home/mysamba/MYPROJ_dailybuild_script/git"
    RELEASE_TOPDIR = r"/home/mysamba/MYPROJ_dailybuild"
else: #NOTES: windows version is no longer maintained.
    print("Running on Windows system")
    BUILDROOT_DIR  = r"C:\\localtest\\git"
    RELEASE_TOPDIR = r"C:\\Daily Release"

tag_str     = time.strftime('%Y_%m_%d',time.localtime(time.time()))
MYPROJ_DIR    = os.path.join(BUILDROOT_DIR,"MYPROJ")
RELEASE_DIR = os.path.join(RELEASE_TOPDIR,tag_str)
logFile     = os.path.join(BUILDROOT_DIR,"build.log")

if not os.path.isdir(BUILDROOT_DIR):
    os.makedirs(BUILDROOT_DIR)
if not os.path.isdir(RELEASE_TOPDIR):
    os.makedirs(RELEASE_TOPDIR)

#prepare log output file
stdout_save=sys.stdout
stderr_save=sys.stderr

logFd=open(logFile, r'w+')
sys.stdout=logFd
sys.stderr=logFd

#log time stamp
logStamp=time.strftime("%Y-%m-%d %H:%M:%S ")
print(40*'*')
print("Start to build MYPROJ pkg. Build timestamp:%s" %logStamp)

if enable_upgrade:
    #create git folder and clone source
    '''
    #FIXME
    if os.path.isdir(MYPROJ_DIR):
        try:
            __import__('shutil').rmtree(MYPROJ_DIR)
        except Exception,e:
            print("ERROR:Fail to remove old dir: %s:%s. Exit." % (MYPROJ_DIR,e))
            raise e
    gitcmd="git clone git@xxx.git:MYPROJ.git " + MYPROJ_DIR
    ret=os.system(gitcmd)
    '''
    if os.path.isdir(MYPROJ_DIR):
        os.chdir(MYPROJ_DIR)
        gitcmd=r"git reset --hard HEAD && git clean -f -d && git fetch origin && git checkout origin/master"
    else:
        gitcmd=r"git clone git@xxx..com:MYPROJ.git " + MYPROJ_DIR
    ret=os.system(gitcmd)

    if ret:
        print("ERROR:Fail to run cmd: %s" % gitcmd)
        exit(-1)
    else:
        print("Success to run cmd: %s" % gitcmd)

    #self upgrade
    scriptname='dailybuild.py'
    cur_script = os.path.join(os.path.abspath(os.path.join(cur_file_dir())), scriptname)
    new_script = os.path.join(MYPROJ_DIR, 'Setup', scriptname)
    ret = upgrade(cur_script, new_script)
    if ret:
        print("Re-launch upgraded script")
        if dry_run:
            s=subprocess.Popen(['python', cur_script, '-r', '-d'])
        else:
            s=subprocess.Popen(['python', cur_script, '-r'])
        s.wait()
        exit(0)
else:
    print("trigger after upgrade, skip git clone")

os.chdir(MYPROJ_DIR)

#last git tag
cmd='git tag'
pipe=subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
ret = pipe.wait()
if ret:
    print "ERROR: fail to execute %s" % cmd
    exit(-1)
tag_list=pipe.stdout.readlines()
tag_list=[i.strip() for i in tag_list]
print "available tag list: %s" % tag_list

pat=re.compile('([\d]{4}_[\d]{2}_[\d]{2})')
tag_match=[]
for tag in tag_list:
    tag_sea = re.search(pat, tag)
    if tag_sea and len(tag_sea.group()) >= 1:
        tag_match.append(tag_sea.group())

last_tag=None
tag_sorted=sorted(tag_match)
for i in tag_sorted[-1::-1]:
    if i < tag_str: #NOTES: if tag_str already exists, it will be ignored here as well, so nothing wrong.
        last_tag = i
        break
    else:
        print "Warning: skip invalid tag :%s, current tag :%s" % (i, tag_str)
if last_tag:
    print("Last git tag: %s" % last_tag)
    cmd="git log %s...%s" % (last_tag, 'HEAD') #NOTES: here tag_str should not be used, which had not be created!
    proc = subprocess.Popen(cmd,stdout=subprocess.PIPE,cwd=os.path.join(BUILDROOT_DIR, "MYPROJ"), shell=True, stderr=sys.stderr)
else:
    print("No available tag found")
    cmd="git log"
    proc = subprocess.Popen(cmd,stdout=subprocess.PIPE,cwd=os.path.join(BUILDROOT_DIR, "MYPROJ"),shell=True, stderr=sys.stderr)

sys.stderr.flush()

ret = proc.wait()
if ret:
    print "ERROR: fail to execute %s" % cmd
    exit(-1)
cont = proc.stdout.readlines()
print "Result of cmd: %s" % cmd
print cont

bNoChange=False
if not cont:
    print "NO CHANGES SINCE LAST BUILD: %s" % last_tag
    bNoChange=True
else:
    #create git tag
    #NOTES: should create before using
    print("Current git tag: %s" % tag_str)
    os.system("git tag --force "+tag_str)
    if dry_run == True:
        os.system("git push --tags --dry-run")
    else:
        os.system("git push --tags -f")

    #create release folder
    if os.path.isdir(RELEASE_DIR):
        shutil.rmtree(RELEASE_DIR)
        os.makedirs(RELEASE_DIR)

    changelog=os.path.join(RELEASE_DIR,r"changelog.txt")
    with open(changelog, 'w+') as f:
        for i in cont:
            f.write(i)

    #shutil.copytree('MYPROJ','C:\Temp\MYPROJtmp',ignore=shutil.ignore_patterns('.git','.gitignore'))
    sys.stdout.flush()

    os.chdir(RELEASE_DIR)

    #start to setup rls pkg
    if sysarc =="Linux":
        cmd_setup_pkg=r'bash '+ os.path.join(MYPROJ_DIR,"Setup","makeReleasePkg.sh")
    else:
        os.system("C:\\Gow\\bin\\unix2dos.exe \"" + changelog)
        cmd_setup_pkg=os.path.join(MYPROJ_DIR,"Setup","makeReleasePkg.bat")

    print("Start to launch cmd: %s\n"%cmd_setup_pkg)
    sys.stdout.flush()

    #os.system(cmd_setup_pkg)
    s=subprocess.Popen(cmd_setup_pkg, shell=True, stdout=sys.stdout, stderr=sys.stderr)
    ret = s.wait()

    sys.stdout.flush()

    if ret:
        print "ERROR: fail to execute %s" % cmd_setup_pkg
        exit(-1)

if sysarc =="Linux":
    #=============================================================
    #based on postfix & sendmail on ubuntu.
    #update additional mail addr list if any
    mail_file       = 'mail_additional.txt'
    mail_additional = os.path.join(os.path.abspath(os.path.join(cur_file_dir())), mail_file)
    if os.path.isfile(mail_additional):
        print("append additional mail addr from file: %s" % mail_additional)
        for i in file(mail_additional):
            s=i.split()[0]
            if s not in mail_addr:
                print("append additional mail addr: %s" % s)
                mail_addr.append(s)

    to_list=''
    for i in mail_addr:
        to_list = to_list+i
        if i != mail_addr[-1]:
            to_list = to_list + ','
    to_tag='To: '+to_list

    other_tags=[
        r'From: myproj-dailybuild@myprojserver.xxx.com',
        r'Subject: MYPROJ Daily Build ' + time.strftime('%B %d'),
        r'Content-Type: text/html',
        r'MIME-Version: 1.0'
        ]

    proc=subprocess.Popen(['ifconfig', 'eth0'], stdout=subprocess.PIPE, stderr=sys.stderr)
    proc.wait()
    links=proc.stdout.readlines()
    ip=None
    for i in links:
        ipstr=re.search('inet addr:([\d]+.[\d]+.[\d]+.[\d]+)', i)
        if ipstr:
            ip=ipstr.group(1)
            break

    if ip is not None:
        ip = ip.split()[0]
    else:
        ip = 'Invalid_ip_addr'

    if bNoChange:
        link=r'\\%s\%s\%s' % (ip, os.path.basename(RELEASE_TOPDIR), last_tag)

        print "get local link addr from hostname: %s" % link

        sys.stdout.flush()

        body1 = "<html><body><strong>"+ time.strftime('%B %d') + "<sup>th</sup>, 2013</strong> -<font>MYPROJ Daily Build</font> \
        <br><br><br><strong><em><i>No change since last build %s.</i></em></strong><br><br>" % last_tag
        body2 = "<br><br><i><b>Latest image available in</b></i>&nbsp;<a href=\""+ link +"\">here</a>"

        sys.stdout.flush()

        with open("mailcontents.txt",'w') as f:
            f.write(to_tag+'\n')
            for s in other_tags:
                f.write(s+'\n')

            f.write(body1+'\n')
            f.write('\n')

            f.write(body2+'\n')

    else:
        link=r'\\%s\%s\%s' % (ip, os.path.basename(RELEASE_TOPDIR), tag_str)

        print "get local link addr from hostname: %s" % link

        sys.stdout.flush()

        body1 = "<html><body><strong>"+ time.strftime('%B %d') + "<sup>th</sup>, 2013</strong> -<font>MYPROJ Daily Build</font> \
        <br><br><br><strong><em><i>What's new?</i></em></strong><br><br>"
        body2 = "<br><br><i><b>Images available in:</b></i>&nbsp;<a href=\""+ link +"\">here</a>\
        <br><br><b><i>The release note is posted:</i></b>&nbsp;<a href=\""+ link +"\">here</a></body></html>"

        if last_tag:
            cmd="git log "+ last_tag + ".." + tag_str + " --format=\"&bull; Author: %an <br>&nbsp;&nbsp; %s<br>\""
            proc = subprocess.Popen(cmd,stdout=subprocess.PIPE,cwd=MYPROJ_DIR, shell=True, stderr=sys.stderr)
        else:
            cmd="git log --format=\"&bull; Author: %an <br>&nbsp;&nbsp; %s<br>\""
            proc = subprocess.Popen(cmd,stdout=subprocess.PIPE,cwd=MYPROJ_DIR, shell=True, stderr=sys.stderr)

        proc.wait()

        sys.stdout.flush()

        res_proc=proc.stdout.read()
        print "Result of cmd: %s" % cmd
        print res_proc
        with open("mailcontents.txt",'w') as f:
            f.write(to_tag+'\n')
            for s in other_tags:
                f.write(s+'\n')

            f.write(body1+'\n')
            f.write('\n')

            f.write(res_proc)
            f.write('\n')
            f.write(body2+'\n')

    s=subprocess.Popen("cat mailcontents.txt |sendmail -t", shell=True, stderr=sys.stderr, stdout=sys.stdout)
    ret = s.wait()
    if not ret:
        print(logStamp + ": send success !\n")
    else:
        print(logStamp + ": send failed !\n")
    #=====================================================================================#

else:
    #=====================================================================================#
    #mail build results
    import win32com.client as win32

    #update additional mail addr list if any
    mail_file       = 'mail_additional.txt'
    mail_additional = os.path.join(os.path.abspath(os.path.join(cur_file_dir())), mail_file)
    if os.path.isfile(mail_additional):
        print("append additional mail addr from file: %s" % mail_additional)
        for i in file(mail_additional):
            if i not in mail_addr:
                print("append additional mail addr: %s" % i)
                mail_addr.append(i)

    #outlook = win32.Dispatch("Outlook.Application")
    #outlook2 = win32.Dispatch("Outlook.Application").GetNamespace("MAPI")
    outlook = win32.gencache.EnsureDispatch('Outlook.Application')

    link="http://windows-smifbc0:8080/" + tag_str
    mail = outlook.CreateItem(win32.constants.olMailItem)

    for addr in mail_addr:
        print("add mail recipient: %s" % addr)
        recip = mail.Recipients.Add(addr)
    subj = mail.Subject = 'MYPROJ DailyBuild '+ time.strftime('%B %d')

    if last_tag:
        proc = subprocess.Popen(["git", "log", last_tag + ".." + tag_str, "--format=&bull; Author: %an <br>&nbsp;&nbsp; %s<br>"],stdout=subprocess.PIPE,cwd="c:\\localtest\\git\\MYPROJ", shell=True, stderr=sys.stderr)
    else:
        proc = subprocess.Popen(["git", "log", "--format=&bull; Author: %an <br>&nbsp;&nbsp; %s<br>"],stdout=subprocess.PIPE,cwd="c:\\localtest\\git\\MYPROJ")

    body1 = "<html><body><strong>"+ time.strftime('%B %d') + "<sup>th</sup>, 2013</strong> -<font>MYPROJ Daily Build</font> \
    <br><br><br><strong><em><i>What's new?</i></em></strong><br><br>"
    body2 = "<br><br><i><b>Images available in:</b></i>&nbsp;<a href=\""+ link +"\">here</a>\
    <br><br><b><i>The release note is posted:</i></b>&nbsp;<a href=\""+ link +"\">here</a></body></html>"

    proc.wait()
    mail.HTMLBody = body1 + proc.stdout.read() + body2
    mail.Send()

    print(logStamp + ": send success !\r\n")
    #=====================================================================================#

print(40*'*')
sys.stdout=stdout_save
sys.stderr=stderr_save
logFd.close()

#move the build logfile to publish folder
shutil.move(logFile, RELEASE_DIR)

