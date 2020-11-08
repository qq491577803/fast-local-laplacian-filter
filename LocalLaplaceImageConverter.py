import cv2 as opencv
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import math
import os
from PyQt5.QtCore import QThread, pyqtSignal
from downUpSample import Sample
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
            # self.input = opencv.resize(self.input,dsize=(128,128))
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

    def downSampleKernel(self):
        a = [.05, .25, .4, .25, .05]
        kernel_1 = np.array(a, np.float64)
        kernel_2 = np.array(kernel_1)[np.newaxis]
        kernel = kernel_2.T
        f = np.multiply(kernel, kernel_1, None)
        return f
    def upSampleKernel(self):
        kernel =[
                [1/64, 4/64,  6/64,  4/64,   1/64],
                [4/64, 16/64, 24/64, 16/64,  4/64],
                [6/64, 24/64, 36/64, 24/64,  6/64],
                [4/64, 16/64, 24/64, 16/64,  4/64],
                [1/64, 4/64,  6/64,  4/64,   1/64],
        ]
        return np.array(kernel)
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


    def imshow(self,src,dst):
        plt.subplot(121)
        plt.imshow(src)
        plt.subplot(122)
        plt.imshow(dst)
        plt.show()


    def downSample(self,inImg, cols, rows):
        outImg = np.zeros(shape=(cols, rows), dtype=np.float64)
        outImg = cv2.filter2D(outImg, -1, self.downSampleKernel())
        x = 0
        for col in range(cols):
            y = 0
            for row in range(rows):
                y = y if y < inImg.shape[1] else inImg.shape[1] - 1
                x = x if x < inImg.shape[0] else inImg.shape[0] - 1
                outImg[col][row] = inImg[x][y]
                y += 2
            x += 2
        return outImg


    def upSample(self,inImg,cols,rows):
        upSample = np.zeros(shape=(cols,rows),dtype=np.float32)
        x = 0
        for col in range(0,cols,2):
            y = 0
            for row in range(0,rows,2):
                upSample[col][row] = inImg[x][y]
                y += 1
            x += 1
        upSample = cv2.filter2D(upSample, -1, kernel=self.upSampleKernel())
        return upSample

    def rebuidLaplacian_pyramid(self,laplacainPyraimd):
        leves = len(laplacainPyraimd)
        out = laplacainPyraimd[leves - 1]
        for i in range(leves-1,0,-1):
            upSample = self.upSample(out,laplacainPyraimd[i-1].shape[0],laplacainPyraimd[i-1].shape[1])
            out = np.add(upSample,laplacainPyraimd[i-1])
        savePath = r"C:\Users\Administrator\Desktop"
        cv2.imwrite(os.path.join(savePath,"res.png"),out.astype(np.uint8))
        return out

    def gaussPyramid(self,img,leves):
        guassLayers = [None] * leves

        guassLayers[0] = img.copy()
        for leve in range(1,leves):
            tmpImge = guassLayers[leve - 1]
            rows = int((tmpImge.shape[0] + 1) / 2)
            cols = int((tmpImge.shape[1] + 1)/ 2)
            downSampleImg = self.downSample(tmpImge,rows,cols)
            guassLayers[leve] = downSampleImg

        return guassLayers

    def laplacianPyramid(self,img,nlev):
        # nlev = 4
        j_image = []
        pyr = [None] * nlev
        for level in range(0, nlev - 1):
            j_image = img.copy()
            rows = int((j_image.shape[0] + 1)/2)
            cols = int((j_image.shape[1] + 1) / 2)
            image = self.downSample(j_image,rows,cols )
            upsampled = self.upSample(image, j_image.shape[0], j_image.shape[1])
            a = np.subtract(j_image, upsampled)
            pyr[level] = a
        pyr[nlev - 1] = j_image
        return pyr



    def LocalLaplacianFilter(self, input, color):
        gauss = 0
        self.sendStartInfoFromConverter.emit(self.num_levels)
        # self.gaussian_pyramid = self.gauss_pyramid(input, None, None)
        self.gaussian_pyramid = Sample().gaussPyramid(input,self.num_levels)

        self.laplacian_pyramid = self.gaussian_pyramid.copy()





        for level in range(1, self.num_levels):
            hw = 3 * 2**level - 2
            print("----------nleve-----------",file=f)
            print("hw:",hw,file=f)
            for y in range(1, self.gaussian_pyramid[level-1].shape[0]+1):
                for x in range(1, self.gaussian_pyramid[level - 1].shape[1]+1):
                    print("gauss x,y",x,y)
                    yf = (y - 1) * 2**(level - 1) + 1
                    xf = (x - 1) * 2**(level - 1) + 1
                    yrng = [max(1, yf - hw), min(self.dim[1], yf + hw)]
                    xrng = [max(1, xf - hw), min(self.dim[0], xf + hw)]
                    isub = input[yrng[0] - 1:yrng[1], xrng[0] - 1: xrng[1]]
                    if color == 'lum':
                        gauss = self.gaussian_pyramid[level - 1][y - 1, x - 1]
                    if color == 'rgb':
                        gauss = self.gaussian_pyramid[level - 1][y - 1, x - 1, :]

                    img_remapped = self.remapping(isub, gauss, "lum")
                    l_remap = self.laplacianPyramid(img_remapped, level + 1)
                    # print(l_remap)
                    yfc = yf - yrng[0] + 1
                    xfc = xf - xrng[0] + 1

                    yfclev0 = math.floor((yfc-1)/2**(level-1)) + 1
                    xfclev0 = math.floor((xfc-1)/2**(level-1)) + 1

                    if color == 'rgb':
                        self.laplacian_pyramid[level - 1][y - 1, x - 1, :] = l_remap[level - 1][yfclev0 - 1,
                                                                             xfclev0 - 1, :]
                    if color == 'lum':
                        self.laplacian_pyramid[level - 1][y - 1, x - 1] = l_remap[level - 1][yfclev0 - 1, xfclev0 - 1]


        out = self.rebuidLaplacian_pyramid(self.laplacian_pyramid)
        cv2.imshow("out",(out*255).astype(np.uint8) )
        cv2.waitKey(0)
        return out

    def LocalLaplaceImageProcessor(self):
        output = self.LocalLaplacianFilter(self.img_resized, self.color)

        output_w = (output * 255.0).copy()
        self.output_write = opencv.imwrite("output.png", output_w)
        return

if __name__ == '__main__':
    path = r"E:\local_laplacian_filter\my_laplacian\data\inputdata\flower.png"
    lp = LocalLaplaceImageConverter(0.1,0.1,0.1,path,100).LocalLaplaceImageProcessor()
