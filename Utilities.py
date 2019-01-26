import numpy as np



def createVizPosMap(srcPosMap):

    dstColorMap = np.zeros((*(srcPosMap.shape[0:2]),3))
    for r in range(srcPosMap.shape[0]):
        for c in range(srcPosMap.shape[1]):
            dstColorMap[r, c][0] = int(float(srcPosMap[r, c][1]) / float(srcPosMap.shape[1]) * float(255))
            dstColorMap[r, c][1] = int(float(srcPosMap[r, c][0]) / float(srcPosMap.shape[1]) * float(255))
            dstColorMap[r, c][2] = 255
    return dstColorMap

def createMask(srcColor, maskColor, maskVal=0, nonMaskVal=255):
    dstMask = np.zeros(srcColor.shape)


    for r in range(srcColor.shape[0]):
        for c in range(srcColor.shape[1]):
            color = srcColor[r, c]  # cv::Vec3b color = srcColor(r, c);
            if color[0] == maskColor[0] and color[1] == maskColor[1] and color[2] == maskColor[2]:
                dstMask[r, c] = maskVal
            else:
                dstMask[r, c] = nonMaskVal
    return srcColor, maskColor, dstMask
