import cv2
import numpy as np
import matplotlib.pyplot as plt
class myLaplacianFliter():

    def __init__(self,path,alpha,beta,sigma):
        self.fp = path
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.gaussian_pyramid = []
        self.laplacian_pyramid = []
        self.color = 'rgb'
        self.input = [] #store origin image
        self.input = cv2.imread(self.fp,cv2.IMREAD_UNCHANGED)
        self.input =  cv2.cvtColor(self.input,cv2.COLOR_RGB2GRAY)
        self.width = self.input.shape[1]
        self.hight = self.input.shape[0]
        self.dim = (self.width,self.hight)#533 800
        self.img_norm = self.input.astype(float) / 255.0 + 0.5
        self.img_resized = cv2.resize(self.img_norm,self.dim,interpolation=cv2.INTER_AREA)
        self.nlevels = 6
    def downsample(self,image,subwindow):
        r = image.shape[0]
        c = image.shape[1]
        subwindow





    def gauss_pyramid(self,image):
        r = image.shape[0]
        c = image.shape[1]
        subwindow = [1,r,1,c]
        nlev = self.nlevels
        pyr=[None] * nlev
        pyr[0] = image.copy()
        for lev in range(1,nlev):
            image,subwindow = self.



        pass
    def laplacian_filter(self,input):
        self.gaussian_pyramid = self.gauss_pyramid(input)
        self.laplacian_pyramid = self.gaussian_pyramid.copy()
        for leve in range(1,self.nlevels):
            hw = 3 * 2 ** leve - 2



    def imageProcessor(self):
        self.laplacian_filter(self.img_resized)
    def imshow(self,src,dst):
        plt.subplot(121)
        plt.imshow(src)
        plt.subplot(122)
        plt.imshow(dst)
        plt.show()
if __name__ == '__main__':
    path = './data/inputdata/flower.png'
    myLaplacianFliter(path,alpha = 0.1,beta = 0.1,sigma =0.1).imageProcessor()
    ll = cv2.imread(path)
