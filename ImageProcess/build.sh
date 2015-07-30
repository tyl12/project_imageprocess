#!/bin/bash

#python -c "import compileall; compileall.compile_dir('.')"
EXT='c'
for i in *.py; do
    echo "pycompile $i"
    rm -f ${i}o ${i}c
    python pyobfuscate.py -a -r ${i} > ${i}.obf
    python -c "import py_compile; py_compile.compile(\"${i}.obf\",\"${i}${EXT}\")"
done

OUTPUT_DIR=release
rm -rf ./$OUTPUT_DIR
mkdir -p ./$OUTPUT_DIR
ls ./$OUTPUT_DIR/

rls_list=( \
    "__init__.py" \
    "decodeROI.py${EXT}" \
    "detectPhase.py${EXT}" \
    "detectROI.py${EXT}" \
    "preProcessing.py${EXT}" \
    "localUtils.py${EXT}" \
    "imageprocess.py" \
    "FrameCount.py${EXT}" \
    "README" \
    "default_config_file.txt" \
    "input_config_file.txt" \
    "ffmpeg" \
    )

install_len=${#rls_list[*]}
for ((i=0; i<${install_len}; i++)); do
    pkg=${rls_list[${i}]};
    echo "copy ${pkg} to ${OUTPUT_DIR}..."
    cp -r -L ${pkg} ${OUTPUT_DIR}/
done

version_info=$(git log HEAD^...HEAD)
echo $version_info > ${OUTPUT_DIR}/version.info
