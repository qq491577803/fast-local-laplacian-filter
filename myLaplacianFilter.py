import cv2
import os
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
        self.input = cv2.resize(self.input,dsize=(128,128))
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
        n = self.leves
        # while mid_d > 1:
        #     n = n+1
        #     mid_d = int((mid_d + 1) / 2)
        return n


    def upSampleKernel(self):
        kernel =[
                [1/64, 4/64,  6/64,  4/64,   1/64],
                [4/64, 16/64, 24/64, 16/64,  4/64],
                [6/64, 24/64, 36/64, 24/64,  6/64],
                [4/64, 16/64, 24/64, 16/64,  4/64],
                [1/64, 4/64,  6/64,  4/64,   1/64],
        ]
        return np.array(kernel)
    def downSampleKernel(self):
        kernel=[
               [1/256,  4/256,  6/256,  4/256, 1/256],
               [4/256, 16/256, 24/256, 16/256, 4/256],
               [6/256, 24/256, 36/256, 24/256, 6/256],
               [4/256, 16/256, 24/256, 16/256, 4/256],
               [1/256,  4/256,  6/256,  4/256, 1/256]
        ]
        return np.array(kernel)
    def downSample(self,inImg,cols,rows):
        outImg = np.zeros(shape=(cols,rows),dtype=np.float32)
        outImg = cv2.filter2D(outImg,-1,self.downSampleKernel())
        x = 0
        for col in range(cols):
            y = 0
            for row in range(rows):
                y = y if y < inImg.shape[1] else inImg.shape[1] - 1
                x = x if x < inImg.shape[0] else inImg.shape[0] - 1
                outImg[col][row] = inImg[x][y]
                y += 2
            x += 2
        outImg = cv2.filter2D(outImg,-1,kernel=self.downSampleKernel())
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
    def rebuidLaplacian_pyramid(self,laplacainPyraimd):
        leves = len(laplacainPyraimd)
        out = laplacainPyraimd[leves - 1]
        for i in range(leves-1,0,-1):
            upSample = self.upSample(out,laplacainPyraimd[i-1].shape[0],laplacainPyraimd[i-1].shape[1])
            out = np.add(upSample,laplacainPyraimd[i-1])
        savePath = r"C:\Users\Administrator\Desktop"
        cv2.imwrite(os.path.join(savePath,"res.png"),(out * 255).astype(np.uint8))
    def imageProcessor(self):
        self.gaussian_pyramid = Sample().gaussPyramid(self.img_norm,self.leves)
        self.laplacian_pyramic = self.gaussian_pyramid.copy()
        for leve in range(1,self.leves):
            hw = 3 * 2 ** (leve-1) - 2
            print("-----------------leve",leve,"-----------------")
            for  y in range(1,self.gaussian_pyramid[leve-1].shape[0] + 1):
                for x in range(1,self.gaussian_pyramid[leve-1].shape[1] + 1):
                    print(x,y)
                    g0 = self.gaussian_pyramid[leve-1][y - 1,x - 1]
                    x_src = x * 2 ** (leve-1) + 1
                    y_src = y * 2 ** (leve-1) + 1
                    sample_with = int(2 ** (leve))
                    x_start = max(x_src - sample_with,1) - 1
                    x_end = min(x_src + sample_with,self.dim[0])
                    y_start = max(y_src - sample_with,1) - 1
                    y_end = min(y_src + sample_with,self.dim[1])
                    src = self.img_norm[y_start:y_end,x_start:x_end]
                    g0_remap = LocalLaplaceImageConverter(0.1, 0.1, 0.1, './data/inputdata/flower.png', 100).remapping(src,g0,"lum")
                    _,l_remap = Sample().laplacian_pyramid(g0_remap, leve)

                    x = x_end


                    self.laplacian_pyramic[leve -1][y-1][x-1] = l_remap[leve-1][0][0]
            fn = os.path.join(r"C:\Users\Administrator\Desktop",str(leve) + ".png")
            cv2.imwrite(fn,np.array((self.laplacian_pyramic[leve-1] * 255)).astype(np.uint8))
        self.laplacian_pyramic[self.leves-1] = self.gaussian_pyramid[self.leves-1]
        self.rebuidLaplacian_pyramid(self.laplacian_pyramic)


    def imshow(self,src,dst):
        plt.subplot(121)
        plt.imshow(src)
        plt.subplot(122)
        plt.imshow(dst)
        plt.show()
if __name__ == '__main__':
    import time
    stime = time.time()
    path = r"C:\Users\Administrator\Desktop\IMG_2405.JPG"
    MylocalLaplacianFilter(path,alpha = 0.1,beta = 0.1,sigma =0.1,noise_leve=0.01).imageProcessor()
    etime = time.time()
    print("total time:",etime- stime)


