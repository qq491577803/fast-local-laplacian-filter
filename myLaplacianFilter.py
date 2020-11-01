import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
f = open(r"E:\local_laplacian_filter\local_laplacian_filter\log.txt",'w')
class MylocalLaplacianFilter():
    def __init__(self,path,alpha,beta,sigma,noise_leve):
        self.input = cv2.imread(path,cv2.IMREAD_UNCHANGED)
        self.input = cv2.cvtColor(self.input,cv2.COLOR_RGB2GRAY)
        self.input = cv2.resize(self.input,dsize=(32,32))
        self.sigma = sigma
        self.beta = beta
        self.alpha = alpha
        self.noise_leve = noise_leve
        self.leves = 6
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
    def get_leves(self,image):
        rows,cols = image.shape[0],image.shape[1]
        mid_d = min(rows,cols)
        n = 1
        while mid_d > 1:
            n = n+1
            mid_d = int((mid_d + 1) / 2)
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
        img_tmp = input_img.copy()
        for leve in range(leves):
            downsample_layer = self.down_sample(img_tmp)
            downsample_layer = cv2.pyrDown(downsample_layer)
            upsample_layer = self.upsample(downsample_layer)
            print("src",img_tmp.shape,"down",downsample_layer.shape,"upsample",upsample_layer.shape)
            diff =np.subtract(img_tmp,upsample_layer)
            img_tmp = downsample_layer
            laylacian_layers[leve] = diff


    def imageProcessor(self):
        self.gaussian_pyramid = self.gaussion_layers(self.img_norm,self.leves)
        self.laplacian_pyramic[0] = self.gaussian_pyramid[0].copy()
        for n in range(self.leves):
            print("leve:",str(n),file=f)
            print(self.gaussian_pyramid[n],file=f)
            # df = pd.DataFrame(np.array(self.gaussian_pyramid)[n])
            # df.to_csv(r"E:\local_laplacian_filter\local_laplacian_filter\log.csv",mode='a+')
        for leve in range(1,self.leves):
            hw = 3 * 2 ** (leve-1) - 2
            print("-----------------leve",leve,"-----------------",file=f)
            print("leve:",leve,"hw:",hw,file=f)
            for  y in range(self.gaussian_pyramid[leve-1].shape[0] + 1):
                for x in range(self.gaussian_pyramid[leve-1].shape[1] + 1):
                    '''
                                        # print("leve:", leve, "hw:", hw, file=f)
                    # print("leve:",leve,"y,x",y,x,file=f)
                    # yf = (y-1) * 2** (leve - 1) + 1
                    # xf = (x-1) * 2** (leve - 1) + 1
                    # print("leve:",leve,"yf,xf",yf,xf,file=f)
                    # yrng = [max(1,yf - hw),min(self.dim[1],yf + hw)]
                    # xrng = [max(1,xf - hw),min(self.dim[0],xf + hw)]
                    # print("leve:",leve,"yrng,xrng",xrng,yrng,file=f)
                    # isub = self.input[yrng[0] - 1: yrng[1],xrng[0] - 1 : xrng[1]]
                    # # print("leve:",leve,"isub:",isub / 255.0,file=f)
                    '''
                    g0 = self.gaussian_pyramid[leve-1][y - 1,x - 1]
                    x_src = x * 2 ** (leve-1) + 1
                    y_src = y * 2 ** (leve-1) + 1
                    sample_with = int(2 ** (leve))
                    x_start = max(x_src - sample_with,1) - 1
                    x_end = min(x_src + sample_with,self.dim[0])
                    print(sample_with,y_src)
                    y_start = max(y_src - sample_with,1) - 1
                    y_end = min(y_src + sample_with,self.dim[1])
                    # print(self.img_norm[y_start:y_end][x_start:x_end])
                    print("leve:",leve,"hw:",sample_with,"xy:",x,y,"xse",x_start,x_end,"yse",y_start,y_end)
                    print(y_start,y_end,x_start,x_end)
                    print("norma shape:",self.img_norm.shape)
                    src = self.img_norm[y_start:y_end,x_start:x_end]
                    print("src shape:",src.shape)
                    print("---------------------------------------")
                    g0_remap = self.remap(g0,src)
                    self.laplacian_layers(g0_remap)

                    # print()



    def imshow(self,src,dst):
        plt.subplot(121)
        plt.imshow(src)
        plt.subplot(122)
        plt.imshow(dst)
        plt.show()
if __name__ == '__main__':
    path = './data/inputdata/flower.png'
    MylocalLaplacianFilter(path,alpha = 0.25,beta = 1,sigma =0.4,noise_leve=0.01).imageProcessor()
    array = np.array([
        [1,2,3,3],
        [4,5,6,6],
        [7,8,9,7],
    ])
    print(array[0:4,0:2])
    # print(array.shape)
    # res = np.zeros((array.shape[0] * 2,array.shape[1] * 2),dtype=np.float64)
    # res = cv2.pyrUp(array)
    # print(res)


