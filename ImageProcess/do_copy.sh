srcdir="./release"
dstdir="../../pub_BIXI/BIXITool/OfflineCollect/ImageProcess"

echo "srcdir=$srcdir"
echo "dstdir=$dstdir"
echo "check-in src&dst done before copying release?! y/n? "
read ans
if [ "$ans" == "y" ]; then
    cp -r $srcdir/*  $dstdir/
    srcinfo="$(git log HEAD^...HEAD)"
    echo "$srcinfo" > $dstdir/version.info
    echo "done!"
else
    echo "quit!"
fi
