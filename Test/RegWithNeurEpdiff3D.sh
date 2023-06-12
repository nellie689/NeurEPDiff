#!/bin/bash
execPath="../"
maxIter=100
stepSizeGD=5.0e-2
truncX=16
truncY=16
truncZ=16
numStep=10


imageSize=192
# imageSize=128
# imageSize=64

gamma=1.0
lpower=3
sigma=0.01
alpha=3.0


# idx=1
# sourcePath="/newdisk/wn/DataSet/NP/Brain${imageSize}/source/src${idx}.mhd"
# targetPath="/newdisk/wn/DataSet/NP/Brain${imageSize}/target/tar${idx}.mhd"
# outPrefix="/newdisk/wn/DataSet/NP/Brain${imageSize}/temp"
sourcePath="SOURCE_PATH"
targetPath="TARGET_PATH"
outPrefix="DIRECTORY_OF_RESULTS"
${execPath}/Optimization ${idx} ${sourcePath} ${targetPath} ${outPrefix} ${truncX} ${truncY} ${truncZ} ${numStep} ${maxIter} ${stepSizeGD} ${alpha} ${gamma} ${lpower} ${sigma}