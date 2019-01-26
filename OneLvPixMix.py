import cv2
import matplotlib.pyplot as plt
import numpy as np
import Utilities as Util
import sys

np.random.seed(3545)
class OneLvPixMix:
    def __init__(self, color, mask):
        self.color = color
        self.mask = mask
        np.savetxt("Mask.csv", mask, delimiter=",")
        self.mMask = np.array([0, 1])
        self.borderSize = 2
        self.borderSizePosMap = 1
        self.windowSize = 5
        self.toLeft = np.array([0, -1])
        self.toRight = np.array([0, 1])
        self.toUp = np.array([-1, 0])
        self.toDown = np.array([1, 0])
        self.mPosMap = {}
        self.WO_BORDER, self.W_BORDER = 0, 1
        self.vSptAdj = [np.array([-1, -1]), np.array([-1, 0]), np.array([-1, 1]), np.array([0, -1]), np.array([0, 1]),
                        np.array([1, -1]), np.array([1, 0]), np.array([1, 1])]

        self.mColor = np.zeros(self.color.shape)
        self.mColor = {0: None, 1: self.mColor}
        self.mColor[self.W_BORDER] = cv2.copyMakeBorder(self.color, self.borderSize, self.borderSize, self.borderSize,
                                                        self.borderSize, cv2.BORDER_REFLECT)

        self.mMask = np.pad(np.zeros(self.mask.shape),
                            ((self.borderSize, self.borderSize), (self.borderSize, self.borderSize)), 'edge')
        self.mMask = {0: self.mMask, 1: self.mMask}
        self.mMask[self.W_BORDER] = cv2.copyMakeBorder(self.mask, self.borderSize, self.borderSize, self.borderSize,
                                                       self.borderSize, cv2.BORDER_REFLECT)
        self.mColor[self.WO_BORDER] = self.mColor[self.W_BORDER][self.borderSize:self.borderSize + color.shape[0],
                                      self.borderSize:self.borderSize + color.shape[1]]

        self.mMask[self.WO_BORDER] = np.array(self.mMask[
                                                  self.W_BORDER][self.borderSize:self.borderSize + mask.shape[0],
                                              self.borderSize:self.borderSize + mask.shape[1]])
        self.mPosMap[self.WO_BORDER] = np.zeros((*self.mColor[self.WO_BORDER].shape[0:2], 2))

        for r in range(self.mPosMap[self.WO_BORDER].shape[0]):
            for c in range(self.mPosMap[self.WO_BORDER].shape[1]):
                if self.mMask[self.WO_BORDER][r, c] == 0:
                    self.mPosMap[self.WO_BORDER][r, c] = self.getValidRandPos()
                else:
                    self.mPosMap[self.WO_BORDER][r, c] = [r, c]

        # self.mPosMap[self.W_BORDER] = np.pad(np.zeros(self.mPosMap[self.WO_BORDER].shape), ((self.borderSizePosMap, self.borderSizePosMap), (self.borderSizePosMap, self.borderSizePosMap), (0, 0),(0,0)), 'edge')
        self.mPosMap[self.W_BORDER] = cv2.copyMakeBorder(self.mPosMap[self.WO_BORDER], self.borderSizePosMap,
                                                         self.borderSizePosMap, self.borderSizePosMap,
                                                         self.borderSizePosMap, cv2.BORDER_REFLECT)

        # self.mPosMap[self.W_BORDER] = np.pad(self.mPosMap[self.WO_BORDER], ((self.borderSizePosMap, self.borderSizePosMap), (self.borderSizePosMap, self.borderSizePosMap), (0, 0),(0,0)), 'edge')

        self.mPosMap[self.WO_BORDER] = np.array(self.mPosMap[
                                                    self.W_BORDER][1:self.color.shape[0] + 1, 1:self.color.shape[
                                                                                                    1] + 1])

    def execute(self, scAlpha, maxItr, maxRandSearchItr, threshDist):

        acAlpha = 1.0 - scAlpha
        thDist = pow(max(self.mColor[self.WO_BORDER].shape[0], self.mColor[self.WO_BORDER].shape[1]) * threshDist, 2.0)
        for itr in range(maxItr):

            self.vizPosMap = Util.createVizPosMap(self.mPosMap[self.WO_BORDER])
            self.vizPosMap = cv2.resize(self.vizPosMap, (640, 480), interpolation=cv2.INTER_NEAREST)
            # plt.imshow(self.vizPosMap)
            # plt.show()
            fig, ax = plt.subplots(1, 2, figsize=(10, 10), sharex=True, sharey=True,
                                   subplot_kw={'adjustable': 'box-forced'})
            self.vizColor = cv2.resize(self.mColor[self.WO_BORDER], (640, 480), cv2.INTER_NEAREST)
            ax[0].imshow(self.vizColor)
            # plt.show()
         
            if itr % 2 == 0:
                self.fwdUpdate(scAlpha, acAlpha, thDist, maxRandSearchItr)

            else:
                self.bwdUpdate(scAlpha, acAlpha, thDist, maxRandSearchItr)


            self.vizPosMap = Util.createVizPosMap(self.mPosMap[self.WO_BORDER])
            self.vizPosMap = cv2.resize(self.vizPosMap, (640, 480), cv2.INTER_NEAREST)
            # plt.imshow(self.vizPosMap)
            # plt.show()
            cv2.imwrite("./PixMix-Inpainting-master/data/mcolor.tif", self.mColor[0])
            vizColor = cv2.resize(self.mColor[self.WO_BORDER], (640, 480), cv2.INTER_NEAREST)
            ax[1].imshow(vizColor)
            plt.show()

            self.inpaint()

    def inpaint(self):
        for r in range(self.mColor[self.WO_BORDER].shape[0]):

            for c in range(self.mColor[self.WO_BORDER].shape[1]):
                self.mColor[self.WO_BORDER][r, c] = self.mColor[self.WO_BORDER][
                    int(self.mPosMap[self.WO_BORDER][r, c, 0]), int(self.mPosMap[self.WO_BORDER][r, c, 1])]


    def calcSptCost(self, target, ref, maxDist, w=0.125):

        normFactor = maxDist * float(2.0)
        sc = 0.0

        for v in self.vSptAdj:
            temp = target + (self.borderSizePosMap, self.borderSizePosMap) + v
            diff = (ref + v) - self.mPosMap[self.W_BORDER][temp[0], temp[1], :]
            sc += min(np.dot(diff, diff), maxDist)
            # sc += min(diff.dot(diff), maxDist)

        return sc * w / normFactor

    def calcAppCost(self, target, ref, w=.04):

        normFctor = 255.0 * 255.0 * 3.0
        ac = 0.0
        for r in range(self.windowSize):
            for c in range(self.windowSize):

                if self.mMask[self.W_BORDER][int(r + ref[0]), int(c + ref[1])] == 0:
                    ac += sys.float_info.max / 25.0

                else:

                    diff = (self.mColor[self.W_BORDER][int(r + target[0]), int(c + target[1])]).astype(float) - (
                        self.mColor[self.W_BORDER][int(r + ref[0]), int(c + ref[1])]).astype(float)
                    ac += diff[0]**2 + diff[1]**2 + diff[2]**2
                    #                 ac += diff.dot(diff)

        return ac * w / normFctor

    def fwdUpdate(self, scAlpha, acAlpha, thDist, maxRandSearchItr):
        # pragma omp parallel for // NOTE: This is not thread-safe

        for r in range(self.mColor[self.WO_BORDER].shape[0]):

            for c in range(self.mColor[self.WO_BORDER].shape[1]):

                if self.mMask[self.WO_BORDER][r, c] == 0:

                    target = np.array([r, c])
                    ref = self.mPosMap[self.WO_BORDER][r, target[1]]
                    top = target + self.toUp
                    left = target + self.toLeft
                    if top[0] < 0:
                        top[0] = 0
                    if left[1] < 0:
                        left[1] = 0
                    topRef = self.mPosMap[self.WO_BORDER][top[0], top[1]] + self.toDown
                    leftRef = self.mPosMap[self.WO_BORDER][left[0], left[1]] + self.toRight
                    if topRef[0] >= self.mColor[self.WO_BORDER].shape[0]:
                        topRef[0] = self.mPosMap[self.WO_BORDER][top[0], top[1]][0]
                    if leftRef[1] >= self.mColor[self.WO_BORDER].shape[1]:
                        leftRef[1] = self.mPosMap[self.WO_BORDER][left[0], left[1]][1]

                    # propagate
                    x = self.calcSptCost(target, ref, thDist)
                    a = self.calcAppCost(target, ref)
                    cost = scAlpha * x + acAlpha * a
                    costTop = sys.float_info.max
                    costLeft = sys.float_info.max

                    if self.mMask[self.WO_BORDER][top[0], top[1]] == 0 and self.mMask[self.WO_BORDER][
                        int(topRef[0]), int(topRef[1])] != 0:
                        a = self.calcAppCost(target, topRef)
                        x = self.calcSptCost(target, topRef, thDist)
                        costTop = scAlpha * x + acAlpha * a

                    if self.mMask[self.WO_BORDER][left[0], left[1]] == 0 and self.mMask[self.WO_BORDER][
                        int(leftRef[0]), int(leftRef[1])] != 0:
                        a = self.calcAppCost(target, leftRef)
                        x = self.calcSptCost(target, leftRef, thDist)
                        costLeft = scAlpha * x + acAlpha * a

                    if costTop < cost and costTop < costLeft:

                        cost = costTop

                        self.mPosMap[self.WO_BORDER][r, target[1]] = topRef


                    elif (costLeft < cost):

                        cost = costLeft
                        self.mPosMap[self.WO_BORDER][r, target[1]] = leftRef

                    # random search
                    itrNum = 0
                    costRand = sys.float_info.max

                    for s in [1]:
                        refRand = self.getValidRandPos()
                        a = self.calcAppCost(target, refRand)
                        x = self.calcSptCost(target, refRand, thDist)
                        costRand = scAlpha * x + acAlpha * a

                        while costRand >= cost and itrNum < maxRandSearchItr:
                            itrNum += 1
                            refRand = self.getValidRandPos()
                            a = self.calcAppCost(target, refRand)
                            x = self.calcSptCost(target, refRand, thDist)
                            costRand = scAlpha * x + acAlpha * a

                    if costRand < cost:
                        self.mPosMap[self.WO_BORDER][r, target[1]] = refRand

    def bwdUpdate(self, scAlpha, acAlpha, thDist, maxRandSearchItr):
        # pragma omp parallel for // NOTE: This is not thread-safe
        for r in range(self.mColor[self.WO_BORDER].shape[0] - 1, -1, -1):
            for c in range(self.mColor[self.WO_BORDER].shape[1] - 1, -1, -1):
                if self.mMask[self.WO_BORDER][r, c] == 0:

                    target = np.array([r, c])
                    ref = self.mPosMap[self.WO_BORDER][r, target[1]]
                    bottom = target + self.toDown
                    right = target + self.toRight
                    if bottom[0] >= self.mColor[self.WO_BORDER].shape[0]:
                        bottom[0] = target[0]
                    if right[1] >= self.mColor[self.WO_BORDER].shape[1]:
                        right[1] = target[1]
                    bottomRef = self.mPosMap[self.WO_BORDER][bottom[0], bottom[1]] + self.toUp
                    rightRef = self.mPosMap[self.WO_BORDER][right[0], right[1]] + self.toLeft
                    if bottomRef[0] < 0:
                        bottomRef[0] = 0
                    if rightRef[1] < 0:
                        rightRef[1] = 0
                    # propagate
                    a = self.calcAppCost(target, ref)
                    x = self.calcSptCost(target, ref, thDist)
                    cost = scAlpha * x + acAlpha * a
                    costTop = sys.float_info.max
                    costLeft = sys.float_info.max

                    if self.mMask[self.WO_BORDER][bottom[0], bottom[1]] == 0 and self.mMask[self.WO_BORDER][
                        int(bottomRef[0]), int(bottomRef[1])] != 0:
                        a = self.calcAppCost(target, bottomRef)
                        x = self.calcSptCost(target, bottomRef, thDist)
                        costTop = scAlpha * x + acAlpha * a

                    if self.mMask[self.WO_BORDER][right[0], right[1]] == 0 and self.mMask[self.WO_BORDER][
                        int(rightRef[0]), int(rightRef[1])] != 0:
                        a = self.calcAppCost(target, rightRef)
                        x = self.calcSptCost(target, rightRef, thDist)
                        costLeft = scAlpha * x + acAlpha * a

                    if costTop < cost and costTop < costLeft:
                        cost = costTop

                        self.mPosMap[self.WO_BORDER][r, target[1]] = bottomRef

                    elif costLeft < cost:
                        cost = costLeft

                        self.mPosMap[self.WO_BORDER][r, target[1]] = rightRef
                    # random search
                    itrNum = 0
                    costRand = sys.float_info.max
                    for s in [1]:
                        refRand = self.getValidRandPos()
                        a = self.calcAppCost(target, refRand)
                        x = self.calcSptCost(target, refRand, thDist)
                        costRand = scAlpha * x + acAlpha * a
                        while costRand >= cost and itrNum < maxRandSearchItr:
                            itrNum += 1
                            refRand = self.getValidRandPos()
                            a = self.calcAppCost(target, refRand)
                            x = self.calcSptCost(target, refRand, thDist)
                            costRand = scAlpha * x + acAlpha * a

                    if costRand < cost:
                        self.mPosMap[self.WO_BORDER][r, target[1]] = refRand

        tmp = np.concatenate((self.mPosMap[self.WO_BORDER], np.zeros((*self.mPosMap[self.WO_BORDER].shape[0:2], 1))),
                             axis=2)
        # plt.imshow(np.uint8(255 * tmp / np.max(tmp)))
        # plt.show()

    def getColorPtr(self):
        return self.mColor[self.WO_BORDER]

    def getMaskPtr(self):
        return self.mMask[self.WO_BORDER]

    def getPosMapPtr(self):
        return self.mPosMap[self.WO_BORDER]

    def getValidRandPos(self):

        p = (np.random.randint(0, self.color.shape[0] - 1), np.random.randint(0, self.color.shape[1] - 1))
        while self.mMask[self.WO_BORDER][p] != 255:
            p = (np.random.randint(0, self.color.shape[0] - 1), np.random.randint(0, self.color.shape[1] - 1))
        return p
