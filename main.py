import numpy as np
import matplotlib.pyplot as plt
import cv2
from Pixmix import PixMix
import Utilities as Util

pathToSrcColor = "./PixMix-Inpainting-master/data/birds.png"
pathToDstColor = "./PixMix-Inpainting-master/data/birds_res.png"
pathToMask = "./PixMix-Inpainting-master/data/birds_mask.png"

case = 'magenta'

if case == 'magenta':
    src = cv2.imread(pathToSrcColor)
    # src = src[:, :, ::-1]
    dst = np.zeros(src.shape)
    mask = cv2.imread(pathToMask, 0)

    fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})

    ax[0].imshow(src)
    ax[0].set_title("Input color image")
    ax[1].imshow(mask)
    ax[1].set_title("Input mask image")
    plt.show()

    pm = PixMix(src, mask)
    dst = pm.execute(dst, 0.05)
    cv2.imwrite(pathToDstColor, dst)
    plt.imshow(dst)
    plt.show()

else:
    pathToSrcColor = "../PixMix-Inpainting/data/birds_magenta.png"
    pathToDstColor = "../PixMix-Inpainting/data/birds_magenta_res.png"
    src = cv2.imread(pathToSrcColor)
    dst = np.array([src.size(), src.type()])

    maskColor = np.array([255, 0, 255])
    src, maskColor, mask = Util.createMask(src, maskColor)

    fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})

    ax[0, 0].imshow(src)
    ax[0, 0].set_title("Input color image")
    ax[0, 1].imshow(mask)
    ax[0, 1].set_title("Input mask image")
    plt.show()


    pm = PixMix(src, mask)

    dst = pm.execute(dst, 0.05)
    plt.imshow(dst)
    plt.show()
