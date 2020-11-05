import cv2 as opencv
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import math
from PyQt5.QtCore import QThread, pyqtSignal

import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("log.txt",mode = "w")
handler.setLevel(logging.INFO)
logger.addHandler(handler)
logger.info("-----------------------------------------")
f = open('./log.txt','w')
print("-----  output my log  -----",file=f)

class LocalLaplaceImageConverter(QThread):
    sendStartInfoFromConverter = pyqtSignal(int)
    sendInfoFromConverter = pyqtSignal(int)

    def __init__(self, kAlphaIn: float, kBetaIn: float, kSigmaRIn: float, inputFileName: str, skala: int):
        QThread.__init__(self)
        self.kAlpha = kAlphaIn
        self.kBeta = kBetaIn
        self.kSigmaR = kSigmaRIn
        self.inputFileName = inputFileName
        self.rows = 0
        self.cols = 0
        self.result = 0
        self.color = ''
        self.gaussian_pyramid = []
        self.laplacian_pyramid = []
        self.output_write = []
        self.reconstructed_image = None
        self.width = 0
        self.height = 0
        self.dim = (self.width, self.height)
        if self.inputFileName:
            self.input = opencv.imread(self.inputFileName, opencv.IMREAD_UNCHANGED)
            self.input = opencv.cvtColor(self.input,opencv.COLOR_BGR2GRAY)
            if len(self.input.shape) == 2:
                self.color = 'lum'
            else:
                self.color = 'rgb'
            self.scale_percent = skala  # percent of original size
            self.width = int(math.ceil(self.input.shape[1] * self.scale_percent / 100))
            self.height = int(math.ceil(self.input.shape[0] * self.scale_percent / 100))
            self.dim = (self.width, self.height)
            if self.input is not None:
                self.img_norm2 = self.input.astype(float)/255.0
                self.img_resized = opencv.resize(self.img_norm2, self.dim, interpolation=opencv.INTER_AREA)
                self.result = 1
            else:
                self.result = 0
            self.num_levels = self.get_num_levels(self.img_resized)
        else:
            self.result = 0

    def __del__(self):
        self.wait()

    def run(self):
        dummy = 0

    # number of pyramid levels, as many as possible up 1x1
    def get_num_levels(self, image):
        self.rows, self.cols = image.shape[:2]
        min_d = min(self.rows, self.cols)
        nlev = 1
        while min_d > 1:
            nlev = nlev + 1
            min_d = int((min_d+1)/2)
        return nlev

    # 2D for building Gaussian and Laplacian pyramids
    def filter(self):
        a = [.05, .25, .4, .25, .05]
        kernel_1 = np.array(a, np.float64)
        kernel_2 = np.array(kernel_1)[np.newaxis]
        kernel = kernel_2.T
        f = np.multiply(kernel, kernel_1, None)
        return f

    # smooth step edge
    def smooth_step(self, xmin, xmax, x):
        y = (x - xmin) / (xmax - xmin)
        y = np.minimum(y, 1)
        y = np.maximum(y, 0)
        y_1 = np.multiply(y, y - 2)
        y_2 = np.square(y_1)
        return y_2

    # detail remapping function
    def fd(self, d):
        noise_level = float(0.01)
        out = d ** self.kAlpha
        if self.kAlpha < 1.0:
            tau = self.smooth_step(noise_level, 2*noise_level, d*self.kSigmaR)
            out = tau * out + (1-tau) * d
        return out

    # edge remapping function
    def fe(self, a):
        out = self.kBeta * a
        return out

    # color remapping function
    def r_color(self, i, g0, sigma_r):
        g0 = np.tile(g0, [i.shape[0], i.shape[1], 1])
        dnrm = np.sqrt(np.sum((i - g0) ** 2, axis=2))
        eps_dnrm = np.spacing(1) + dnrm
        unit = (i - g0) / np.tile(eps_dnrm[..., None], [1, 1, 3])
        rd = g0 + unit * np.tile(sigma_r * self.fd(dnrm / sigma_r)[..., None], [1, 1, 3])
        re = g0 + unit * np.tile(sigma_r + self.fe(dnrm - sigma_r)[..., None], [1, 1, 3])
        isedge = np.tile((dnrm > sigma_r)[..., None], [1, 1, 3])
        return np.logical_not(isedge) * rd + isedge * re

    # grayscale remapping function
    def r_gray(self, i, g0, sigma_r):
        dnrm = abs(i-g0)
        dsgn = np.sign(i-g0)

        rd = g0 + dsgn*sigma_r*self.fd(dnrm/sigma_r)
        re = g0 + dsgn*(self.fe(dnrm - sigma_r) + sigma_r)

        isedge = dnrm > sigma_r
        return np.logical_not(isedge) * rd + isedge * re

    def remapping(self, image, gauss, remapping_type):
        if remapping_type == 'rgb':
            return self.r_color(image, gauss, self.kSigmaR)
        if remapping_type == 'lum':
            return self.r_gray(image, gauss, self.kSigmaR)
        return None

    def child_window(self, parent):
        child = parent.copy()

        child[0] = math.ceil((float(child[0]) + 1.0) / 2.0)
        child[2] = math.ceil((float(child[2]) + 1.0) / 2.0)
        child[1] = math.floor((float(child[1]) + 1.0) / 2.0)
        child[3] = math.floor((float(child[3]) + 1.0) / 2.0)
        return child

    def upsample(self, image, subwindow, color):
        R = []
        Z = []
        r = int(subwindow[1] - subwindow[0] + 1)
        c = int(subwindow[3] - subwindow[2] + 1)
        if color == 'rgb':
            k = int(image.shape[2])
        reven = ((subwindow[0]) % 2) == 0
        ceven = ((subwindow[2]) % 2) == 0
        border_mode = 'reweighted'

        if color == 'lum':
            R = np.zeros((int(r), int(c)))
            Z = np.zeros((int(r), int(c)))
        if color == 'rgb':
            R = np.zeros((int(r), int(c), k))
            Z = np.zeros((int(r), int(c), k))
        kernel = self.filter()

        if border_mode == 'reweighted':
            if color == 'lum':
                if R[reven: r: 2, ceven: c: 2].shape != image.shape:
                    rslice = slice(reven, r, 2).indices(r)
                    cslise = slice(ceven, c, 2).indices(c)
                    np.put(R, [rslice, cslise], image.copy())
                else:
                    R[reven: r: 2, ceven: c: 2] = image.copy()

                R = opencv.filter2D(R.astype(np.float64), -1, kernel)
                Z[reven: r:2, ceven: c: 2] = 1.0
                Z = opencv.filter2D(Z.astype(np.float64), -1, kernel)
                R = np.divide(R, Z)

            if color == 'rgb':
                if R[reven: r: 2, ceven: c: 2, :].shape != image.shape:
                    rslice = slice(reven, r, 2).indices(r)
                    cslise = slice(ceven, c, 2).indices(c)
                    np.put(R, [rslice, cslise], image.copy())
                else:
                    R[reven: r: 2, ceven: c: 2, :] = image.copy()
                R = opencv.filter2D(R.astype(np.float64), -1, kernel)
                Z[reven: r:2, ceven: c: 2, :] = 1.0
                Z = opencv.filter2D(Z.astype(np.float64), -1, kernel)
                R = np.divide(R, Z)

            return R
    def imshow(self,src,dst):
        plt.subplot(121)
        plt.imshow(src)
        plt.subplot(122)
        plt.imshow(dst)
        plt.show()
    def downsample(self, image, subwindow, color):
        r = image.shape[0]
        c = image.shape[1]
        if not subwindow:
            subwindow = np.arange(r*c).reshape(r, c)
        print("par window:",file=f)
        print(subwindow,file=f)
        subwindow_child = self.child_window(subwindow)
        print("child window:",file=f)
        print(subwindow_child,file=f)
        border_mode = 'reweighted'
        R = None
        kernel = self.filter()

        if border_mode == 'reweighted':
            R = opencv.filter2D(image.astype(np.float64), -1, kernel)
            if color == 'rgb':
                Z = numpy.ones([r, c, 3], dtype=np.float64)
                Z = opencv.filter2D(Z, -1, kernel)
                R = np.divide(R, Z)
            if color == 'lum':
                Z = numpy.ones([r, c], dtype=np.float64)
                Z = opencv.filter2D(Z, -1, kernel)
                R = np.divide(R, Z)
        else:
            R = opencv.filter2D(R.astype(np.float64), -1, kernel, borderType=opencv.BORDER_REPLICATE)
        reven = np.remainder(subwindow[0], 2) == 0
        ceven = np.remainder(subwindow[2], 2) == 0

        if color == 'rgb':
            R = R[reven: r: 2, ceven: c: 2, :]
        if color == 'lum':
            R = R[reven: r: 2, ceven: c: 2]
        return R, subwindow_child

    def reconstruct_laplacian_pyramid(self, subwindow=None):
        nlev = self.num_levels
        subwindow_all = np.zeros((nlev, 4))
        if not subwindow:
            subwindow_all[1, :] = [1, self.height, 1, self.cols]
        else:
            subwindow_all[1, :] = subwindow
        for lev in range(2, nlev):
            subwindow_all[lev, :] = self.child_window(subwindow_all[lev-1,:])
        R = self.laplacian_pyramid[nlev-1].copy()
        for lev in range(nlev-1, 0, -1):
            upsampled = self.upsample(R, subwindow_all[lev, :], self.color)
            R = np.add(self.laplacian_pyramid[lev-1], upsampled)
        return R

    def gauss_pyramid(self, image, nlev, subwindow):
        r = image.shape[0]
        c = image.shape[1]
        if not subwindow:
            subwindow = [1, r, 1, c]
        if not nlev:
            nlev = self.get_num_levels(image)
        pyr = [None] * nlev
        pyr[0] = image.copy()

        for level in range(1, nlev):
            print("subwindow:",file = f)
            print(subwindow,file=f)
            print("-----start downsample-----",file=f)
            image_para = image.copy()
            image, subwindow_child = self.downsample(image, subwindow, self.color)
            # self.imshow(image_para,image)
            print("nlev:",nlev,"par:",image_para.shape,"child:",image.shape,file=f)
            print("-----end downsample-----",file =f)
            print("subwindow_child:",file=f)
            print(subwindow_child,file=f)
            pyr[level] = image.copy()
        return pyr

    def laplace_pyramid(self, image, nlev, subwindow):
        r = image.shape[0]
        c = image.shape[1]
        j_image = []
        if not subwindow:
            subwindow = [1, r, 1, c]
        if not nlev:
            nlev = self.get_num_levels(image)
        pyr = [None] * nlev
        for level in range(0, nlev-1):
            j_image = image.copy()
            image, subwindow_child = self.downsample(j_image, subwindow, self.color)
            upsampled = self.upsample(image, subwindow, self.color)
            a = np.subtract(j_image, upsampled)
            pyr[level] = a

            subwindow = subwindow_child.copy()

        pyr[nlev-1] = j_image
        return pyr

    def LocalLaplacianFilter(self, input, color):
        gauss = 0
        self.sendStartInfoFromConverter.emit(self.num_levels)
        print("---------------------------gaussian_pyramid----------------------------------",file=f)
        print("start gaussian pyramid",file=f)
        self.gaussian_pyramid = self.gauss_pyramid(input, None, None)
        print("end gaussian pyramid",file=f)
        print("---------------------------laplacian_pyramid----------------------------------",file=f)
        self.laplacian_pyramid = self.gaussian_pyramid.copy()

        for level in range(1, self.num_levels):
            hw = 3 * 2**level - 2
            print("----------nleve-----------",file=f)
            print("hw:",hw,file=f)
            for y in range(1, self.gaussian_pyramid[level-1].shape[0]+1):
                for x in range(1, self.gaussian_pyramid[level - 1].shape[1]+1):
                    print("gauss x,y",x,y,file=f)
                    yf = (y - 1) * 2**(level - 1) + 1
                    xf = (x - 1) * 2**(level - 1) + 1
                    yrng = [max(1, yf - hw), min(self.dim[1], yf + hw)]
                    xrng = [max(1, xf - hw), min(self.dim[0], xf + hw)]
                    print("isub xis:",yrng[0] - 1,yrng[1], xrng[0] - 1,xrng[1],file=f)
                    isub = input[yrng[0] - 1:yrng[1], xrng[0] - 1: xrng[1]]
                    print("isub:",isub,file=f)
                    if color == 'lum':
                        gauss = self.gaussian_pyramid[level - 1][y - 1, x - 1]
                    if color == 'rgb':
                        gauss = self.gaussian_pyramid[level - 1][y - 1, x - 1, :]
                    print("gauss pyramid:",gauss.shape,gauss,file=f)

                    img_remapped = self.remapping(isub, gauss, "lum")
                    print("remap:",img_remapped.shape,file=f)
                    l_remap = self.laplace_pyramid(img_remapped, level + 1, [yrng[0], yrng[1], xrng[0], xrng[1]])
                    # print("---------remap")
                    # print("remap",l_remap)
                    yfc = yf - yrng[0] + 1
                    xfc = xf - xrng[0] + 1

                    yfclev0 = math.floor((yfc-1)/2**(level-1)) + 1
                    xfclev0 = math.floor((xfc-1)/2**(level-1)) + 1

                    if color == 'rgb':
                        self.laplacian_pyramid[level - 1][y - 1, x - 1, :] = l_remap[level - 1][yfclev0 - 1,
                                                                             xfclev0 - 1, :]
                    if color == 'lum':
                        self.laplacian_pyramid[level - 1][y - 1, x - 1] = l_remap[level - 1][yfclev0 - 1, xfclev0 - 1]
                    # print("kojrjonr",self.laplacian_pyramid[level - 1][y - 1, x - 1])
            self.sendInfoFromConverter.emit(level)
        out = self.reconstruct_laplacian_pyramid()
        return out

    def LocalLaplaceImageProcessor(self):
        output = self.LocalLaplacianFilter(self.img_resized, self.color)

        output_w = (output * 255.0).copy()
        self.output_write = opencv.imwrite("output.png", output_w)
        return

if __name__ == '__main__':
    path = r"E:\local_laplacian_filter\my_laplacian\data\inputdata\flower.png"
    LocalLaplaceImageConverter(0.1,0.1,0.1,path,100).LocalLaplaceImageProcessor()