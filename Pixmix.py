import numpy as np
from OneLvPixMix import OneLvPixMix
import cv2
import matplotlib.pyplot as plt


class PixMix:
    def __init__(self, color, mask, blurSize=5):
        self.color = color
        self.mask = mask

        assert (color.shape[0:2] == mask.shape[0:2]), "image and mask size doesn't match!"

        self.pm = np.empty([self.calcPyrmLv(self.color.shape[0], self.color.shape[1])], dtype=type(OneLvPixMix))
        self.pm[0] = OneLvPixMix(color, mask)

        for lv in range(1, len(self.pm)):
            lvSize = (int(self.pm[lv - 1].getColorPtr().shape[1] / 2), int(
                self.pm[lv - 1].getColorPtr().shape[0] / 2))

            tmpColor = cv2.resize(self.pm[lv - 1].getColorPtr(),  dsize=lvSize, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
            tmpMask = cv2.resize(self.pm[lv - 1].getMaskPtr(),  dsize=lvSize, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
            for r in range(tmpMask.shape[0]):
                ptrMask = np.zeros(tmpMask.shape)
                for c in range(tmpMask.shape[1]):
                    if tmpMask[r, c] < 255:
                        tmpMask[r, c] = 0
                    else:
                        ptrMask[r, c] = 255
            self.pm[lv] = OneLvPixMix(tmpColor, tmpMask)

        # for the final composite
        self.mColor = color.copy()  # color.clone()
        self.mAlpha = cv2.blur(mask, (blurSize, blurSize))

    def execute(self, dst, alpha):
        for lv in range(len(self.pm)- 1, -1, -1):
            self.pm[lv].execute(alpha, 2, 1, 0.5)
            if lv > 0:
                self.fillInLowerLv(self.pm[lv], self.pm[lv - 1])
        dst = self.blendBorder(dst)
        return dst

    def calcPyrmLv(self, width, height):
        size = min(width, height)
        pyrmLv = 1
        size /= 2

        while size >= 5:
            pyrmLv += 1
            size /= 2
        return min(pyrmLv, 6)

    def fillInLowerLv(self, pmUpper, pmLower):

        mColorUpsampled = cv2.resize(pmUpper.getColorPtr(), (pmLower.getColorPtr().shape[1],pmLower.getColorPtr().shape[0]), fx=0.0, fy=0.0,
                                     interpolation=cv2.INTER_LINEAR)

        mPosMapUpsampled = cv2.resize(pmUpper.getPosMapPtr(), (pmLower.getPosMapPtr().shape[1],pmLower.getPosMapPtr().shape[0]), fx=0.0, fy=0.0,
                                      interpolation=cv2.INTER_NEAREST)


        for r in range(mPosMapUpsampled.shape[0]):
            for c in range(mPosMapUpsampled.shape[1]):
                mPosMapUpsampled[r, c] = mPosMapUpsampled[r, c] * 2 + np.array([r % 2, c % 2])

        mColorLw = pmLower.getColorPtr()
        mMaskLw = pmLower.getMaskPtr()
        mPosMapLw = pmLower.getPosMapPtr()

        wLw = mColorUpsampled.shape[1]
        hLw = mColorUpsampled.shape[0]

        for r in range(hLw):
            for c in range(wLw):
                if mMaskLw[r, c] == 0:
                    mColorLw[r, c] = mColorUpsampled[r, c]
                    mPosMapLw[r, c] = mPosMapUpsampled[r, c]
        return pmUpper, pmLower

    def blendBorder(self, dst):
        mColorF = self.mColor
        cv2.convertScaleAbs(self.mColor, mColorF, 1.0, 1.0 / 255.0)
        mPMColorF = self.pm[0].getColorPtr()
        mDstF = self.pm[0].getColorPtr()  # mDstF(pm[0].getColorPtr()->size())

        cv2.convertScaleAbs(self.pm[0].getColorPtr(), mPMColorF, 1.0,
                            1.0 / 255.0)  # pm[0].getColorPtr()->convertTo(mPMColorF, CV_32FC3, 1.0 / 255.0);
        mAlphaF = np.zeros(self.mAlpha.shape)
        cv2.convertScaleAbs(self.mAlpha, mAlphaF, 1.0, 1.0 / 255.0)

        for r in range(self.mColor.shape[0]):
            for c in range(self.mColor.shape[1]):
                mDstF[r, c] = mAlphaF[r, c] * mColorF[r, c] + (1.0 + mAlphaF[r, c]) * mPMColorF[r, c]

        dst = cv2.convertScaleAbs(mDstF)
        return dst
