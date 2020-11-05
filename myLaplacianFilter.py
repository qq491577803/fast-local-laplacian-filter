import cv2
import numpy as np
import matplotlib.pyplot as plt
from downUpSample import Sample
import pandas as pd
from LocalLaplaceImageConverter import LocalLaplaceImageConverter
f = open(r"E:\local_laplacian_filter\local_laplacian_filter\log.txt",'w')
class MylocalLaplacianFilter():
    def __init__(self,path,alpha,beta,sigma,noise_leve):
        self.input = cv2.imread(path,cv2.IMREAD_UNCHANGED)
        self.input = cv2.cvtColor(self.input,cv2.COLOR_RGB2GRAY)
        self.input = cv2.resize(self.input,dsize=(64,64))
        self.sigma = sigma
        self.beta = beta
        self.alpha = alpha
        self.noise_leve = noise_leve
        self.leves = 4
        self.width = self.input.shape[1]
        self.hight = self.input.shape[0]
        self.dim = (self.width,self.hight)
        self.img_norm = self.input.astype(np.float) / 255
        self.gaussian_pyramid = [None] * self.leves
        self.laplacian_pyramic = [None] * self.leves
    def get_kernel(self):
        kernel = [[0.0025, 0.0125, 0.02,   0.0125, 0.0025],
                  [0.0125, 0.0625, 0.1,    0.0625, 0.0125],
                  [0.02,   0.1,    0.16,   0.1,    0.02  ],
                  [0.0125, 0.0625, 0.1,    0.0625, 0.0125],
                  [0.0025, 0.0125, 0.02,   0.0125, 0.0025],
                 ]
        return np.array(kernel)



# """copy start"""
    # detail remapping function
    def fd(self, d):
        noise_level = float(0.01)
        out = d ** self.kAlpha
        if self.kAlpha < 1.0:
            tau = self.smooth_step(noise_level, 2 * noise_level, d * self.kSigmaR)
            out = tau * out + (1 - tau) * d
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
        dnrm = abs(i - g0)
        dsgn = np.sign(i - g0)

        rd = g0 + dsgn * sigma_r * self.fd(dnrm / sigma_r)
        re = g0 + dsgn * (self.fe(dnrm - sigma_r) + sigma_r)

        isedge = dnrm > sigma_r
        return np.logical_not(isedge) * rd + isedge * re

    def remapping(self, image, gauss, remapping_type):
        if remapping_type == 'rgb':
            return self.r_color(image, gauss, self.sigma)
        if remapping_type == 'lum':
            return self.r_gray(image, gauss, self.sigma)
        return None

# """copy end"""


    def get_leves(self,image):
        rows,cols = image.shape[0],image.shape[1]
        mid_d = min(rows,cols)
        n = 3
        # while mid_d > 1:
        #     n = n+1
        #     mid_d = int((mid_d + 1) / 2)
        return n


    def down_sample(self,input):
        gauss_kernel = self.get_kernel()
        img = cv2.filter2D(input.astype(np.float64),-1,kernel=gauss_kernel)
        # self.imshow(input,img)
        img = img[::2,::2]
        return img
    def upsample(self,image):
        upsammle_img = cv2.pyrUp(image)
        return upsammle_img





    def fe(self,x):
        return self.beta * x
    def fd(self,x):
        if self.alpha >= 1:
            return x ** self.alpha
        else:
            x1 = ((x * self.sigma - self.noise_leve) / self.noise_leve) ** 2
            x2 = ((x * self.sigma - self.noise_leve) / self.noise_leve - 2) ** 2
            return x1 * x2
    def remap(self,g0,src):
        img_rempaed = np.zeros_like(src)
        for i in range(src.shape[0]):
            for j in range(src.shape[1]):
                pixl_val = src[i][j]
                diff = abs(pixl_val - g0)
                dsign = np.sign(pixl_val - g0)
                if diff > self.sigma:#边缘增强re
                    re = g0 + dsign * (self.fe(diff - self.sigma) + self.sigma)
                    img_rempaed[i][j] = re
                else:               #细节平滑rd
                    de = g0 + dsign * self.sigma * self.fd(diff / self.sigma)
                    img_rempaed[i][j] = de
        return img_rempaed
    def gaussion_layers(self,input_img,leves):
        gauss_layers = [None] * leves
        gauss_layers[0] = input_img.copy()
        for leve in range(1,leves):
            img_dowmsample = self.down_sample(input_img)
            input_img = img_dowmsample.copy()
            gauss_layers[leve] = img_dowmsample.copy()
        return gauss_layers
    def laplacian_layers(self,input_img):
        leves = self.get_leves(input_img)
        laylacian_layers = [None] * leves
        srcImg = input_img.copy()
        us_rows,us_cols = input_img.shape[0],input_img.shape[1]

        for leve in range(leves-1):
            src_cols,src_rows = srcImg.shape[0],srcImg.shape[1]
            rows_ds, cols_ds = int((src_rows + 1) / 2), int((src_cols + 1) / 2)
            dsImg = Sample().downSample(srcImg,cols_ds,rows_ds)
            upImg = Sample().upSample(dsImg,src_cols,src_rows)
            diff = np.subtract(srcImg,upImg)
            laylacian_layers[leve] = diff.copy()
            srcImg = upImg.copy()
            laylacian_layers[leves-1] = srcImg
        return laylacian_layers

    def imageProcessor(self):
        self.gaussian_pyramid = Sample().gaussPyramid(self.img_norm,self.leves)
        self.laplacian_pyramic = self.gaussian_pyramid.copy()
        for leve in range(1,self.leves):
            hw = 3 * 2 ** (leve-1) - 2
            print("-----------------leve",leve,"-----------------")
            for  y in range(self.gaussian_pyramid[leve-1].shape[0] + 1):
                for x in range(self.gaussian_pyramid[leve-1].shape[1] + 1):
                    g0 = self.gaussian_pyramid[leve-1][y - 1,x - 1]
                    x_src = x * 2 ** (leve-1) + 1
                    y_src = y * 2 ** (leve-1) + 1
                    sample_with = int(2 ** (leve))
                    x_start = max(x_src - sample_with,1) - 1
                    x_end = min(x_src + sample_with,self.dim[0])
                    y_start = max(y_src - sample_with,1) - 1
                    y_end = min(y_src + sample_with,self.dim[1])
                    src = self.img_norm[y_start:y_end,x_start:x_end]
                    # print("src",src)
                    # print("g0",g0)
                    # g0_remap = self.remapping(src,g0,"lum")
                    # print(g0_remap)
                    g0_remap = LocalLaplaceImageConverter(0.1, 0.1, 0.1, './data/inputdata/flower.png', 100).remapping(src,g0,"lum")
                    _,l_remap = Sample().laplacian_pyramid(g0_remap, leve)

                    self.laplacian_pyramic[leve -1][y-1][x-1] = l_remap[leve-1][0][0]
            # cv2.imshow("lap",np.array((self.laplacian_pyramic[leve-1] * 255)).astype(np.uint8))
            # cv2.waitKey(0)
            fn = os.path.join(r"C:\Users\Administrator\Desktop",str(leve) + ".png")
            cv2.imwrite(fn,np.array((self.laplacian_pyramic[leve-1] * 255)).astype(np.uint8))
        self.laplacian_pyramic[self.leves-1] = self.gaussian_pyramid[self.leves-1]



    def imshow(self,src,dst):
        plt.subplot(121)
        plt.imshow(src)
        plt.subplot(122)
        plt.imshow(dst)
        plt.show()
if __name__ == '__main__':
    path = './data/inputdata/flower.png'
    MylocalLaplacianFilter(path,alpha = 0.1,beta = 0.1,sigma =0.1,noise_leve=0.01).imageProcessor()



