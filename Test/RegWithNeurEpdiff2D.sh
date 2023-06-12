#!/bin/bash
execPath="../"
maxIter=100
stepSizeGD=5.0e-3  #5.0e-2
truncX=16
truncY=16
truncZ=1
numStep=10


imageSize=192
gamma=1.0
sigma=0.005

alpha=6.0
lpower=6
lpower=12

alpha=3.0
# lpower=9
lpower=12



imageSize=128
lpower=6
sigma=0.01
alpha=3.0



imageSize=64
lpower=3
sigma=0.03
alpha=3.0


# idx=1007
# idx=1009
# idx=1008
# idx=1010
# idx=1003
# idx=1015
# idx=1001
# idx=1002
# idx=1005
# idx=1006
# sourcePath="/newdisk/wn/DataSet/NP/EyeBigDiff${imageSize}/source/src${idx}.mhd"
# targetPath="/newdisk/wn/DataSet/NP/EyeBigDiff${imageSize}/target/tar${idx}.mhd"
# outPrefix="/newdisk/wn/DataSet/NP/EyeBigDiff$/{imageSize}/temp"


sourcePath="SOURCE_PATH"
targetPath="TARGET_PATH"
outPrefix="DIRECTORY_OF_RESULTS"
${execPath}/Optimization2d ${idx} ${sourcePath} ${targetPath} ${outPrefix} ${truncX} ${truncY} ${truncZ} ${numStep} ${maxIter} ${stepSizeGD} ${alpha} ${gamma} ${lpower} ${sigma}

