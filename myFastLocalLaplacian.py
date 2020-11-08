#！usr/bin/env python
#encoding:utf-8
'''
__Author__:Lsz
Function:Image detail enhance
Author Loca:BeiJing China Haidian
Email:www.491577803@qq.com
Berif:Use Python base Lang
Todo:Use Numpy to optim process speed or c++
'''
import os
import cv2
import time
import math
import numpy as np
import matplotlib.pyplot as plt
class FastLaplacianFilter():
    def __init__(self,ipath:str,opath:str,sigma:float,alpha:float,beta:float,noise:float,gamma:float):
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
        self.noise = noise
        self.gamma = gamma
        self.maxPyramidNum = 64
        self.opath = opath
        self.imgRGB = cv2.imread(ipath,cv2.IMREAD_UNCHANGED)
        # self.imgRGB = cv2.resize(self.imgRGB,dsize=(128,128))
        # rgb to gray [b,g,r]  or use cv2.cvtColor
        self.imgGray = self.rgb2gray(self.imgRGB)
        self.maxLeves = math.floor(math.log2(min(self.imgGray.shape[0],self.imgGray.shape[1])))
        self.maxLeves = 6
        self.imgGray = self.imgGray.astype(np.float64)
        # Recode Time
        self.sTime = time.time()
        self.eTime = -1

    def rgb2gray(self,image):
        rgb2yCoeff = [0.0722,0.7152,0.2126]
        imgGray = np.zeros(shape=(image.shape[0], image.shape[1]),dtype=np.float64)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                imgGray[i][j] = image[i][j][0] * rgb2yCoeff[0] + image[i][j][1] * rgb2yCoeff[1] + \
                                     image[i][j][2] * rgb2yCoeff[2]
        return imgGray

    def upSampleKernel(self):
        kernel =[
                [1/64, 4/64,  6/64,  4/64,   1/64],
                [4/64, 16/64, 24/64, 16/64,  4/64],
                [6/64, 24/64, 36/64, 24/64,  6/64],
                [4/64, 16/64, 24/64, 16/64,  4/64],
                [1/64, 4/64,  6/64,  4/64,   1/64],
        ]
        return np.array(kernel,dtype=np.float64)
    def downSampleKernel(self):
        kernel=[
                [1/256,  4/256,  6/256,  4/256, 1/256],
                [4/256, 16/256, 24/256, 16/256, 4/256],
                [6/256, 24/256, 36/256, 24/256, 6/256],
                [4/256, 16/256, 24/256, 16/256, 4/256],
                [1/256,  4/256,  6/256,  4/256, 1/256]
         ]
        return np.array(kernel,dtype=np.float64)
    def fiter2D(self,inImg,sampleType):
        if sampleType == "downSample":
            kernel = self.downSampleKernel()
        if sampleType == "upSample":
            kernel = self.upSampleKernel()
        kernel = np.array(kernel,dtype=np.float64)
        outImg = np.zeros_like(inImg,dtype=np.float64)
        height = inImg.shape[0]
        width = inImg.shape[1]
        width2 =  (width - 1) << 1
        height2 = (height- 1) << 1
        ksizeY = 5 >> 1
        ksizeX = 5 >> 1
        idx = 0
        for y in range(height):
            for x in range(width):
                sum = 0.0
                kIdx = 0
                for yy in range(y-ksizeY,y+ksizeY + 1):
                    yyC = yy
                    if yyC >= height:
                        yyC = height2 - yyC
                    elif yyC < 0:
                        yyC = - yyC
                    for xx in range(x - ksizeX,x+ksizeX + 1):
                        xxC = xx
                        if xxC >= width:
                            xxC = width2 - xxC
                        elif xxC < 0:
                            xxC = -xxC
                        sum += inImg[yyC][xxC] * kernel[kIdx]
                kIdx += 1
                outImg[y][x] = sum
        return  outImg

    #get imageGray mean val for fast Lapcian
    def getMeans(self,imageGray,maxPyramidNum):
        minVal = np.min(imageGray).astype(np.float64)
        maxVal = np.max(imageGray).astype(np.float64)
        means = np.linspace(minVal,maxVal,maxPyramidNum).astype(np.float64)
        print("mean:",means)
        return means
    #remap lum fucntion
    def  remapLum(self,imgGray,mean,alpha,beta,sigma):
        out = np.zeros_like(imgGray,dtype=np.float64)
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                diff = imgGray[i][j] - mean
                diffAbs = abs(diff)
                sign = 1 if diff >= 0 else -1
                if(diffAbs > sigma):
                    out[i][j] = mean + sign * ((diffAbs - sigma) * beta + sigma)
                else:
                    out[i][j] = mean + sign * math.pow(diffAbs / sigma,alpha) * sigma
        return out

    #down sample image
    def downSample(self,inImg, rows, cols):
        outImg = np.zeros(shape=(rows, cols), dtype=np.float64)
        outImg = cv2.filter2D(outImg, -1, self.downSampleKernel())
        # outImg = self.fiter2D(outImg,"downSample")
        xRow = 0
        for row in range(rows):
            yCol = 0
            for col in range(cols):
                yCol = yCol if yCol < inImg.shape[1] else inImg.shape[1] - 1
                xRow = xRow if xRow < inImg.shape[0] else inImg.shape[0] - 1
                outImg[row][col] = inImg[xRow][yCol]
                yCol += 2
            xRow += 2
        return outImg
    #up sample image
    def upSample(self,inImg,rows,cols):
        upSample = np.zeros(shape=(rows,cols),dtype=np.float64)
        xRow = 0
        for row in range(0,rows,2):
            yCol = 0
            for col in range(0,cols,2):
                upSample[row][col] = inImg[xRow][yCol]
                yCol += 1
            xRow += 1
        upSample = cv2.filter2D(upSample, -1, kernel=self.upSampleKernel())
        # upSample = self.fiter2D(upSample,"upSample")
        return upSample

    #build laplacian Pyramid
    def buildLaplacianPyramid(self,imgRmaped,guassionPyramid,laplacianPyramid,maxLeves):
        gaussionPyramid = [None] * maxLeves
        laplacianPyramid = [None] * maxLeves
        gaussionPyramid[0] = imgRmaped
        for leve in range(1,maxLeves):
            leveImg = gaussionPyramid[leve - 1]
            rows,cols = int((leveImg.shape[0] + 1) / 2),int((leveImg.shape[1] + 1) /2)
            downSampleImg = self.downSample(leveImg,rows,cols)
            upSampleImg = self.upSample(downSampleImg,leveImg.shape[0],leveImg.shape[1])
            gaussionPyramid[leve] = downSampleImg
            laplacianPyramid[leve - 1] = np.subtract(leveImg,upSampleImg)
        laplacianPyramid[maxLeves - 1] = gaussionPyramid[maxLeves - 1]
        return gaussionPyramid,laplacianPyramid

    def remapLaplacianPyramid(self,imgGray,means,leves):
        maxPyramidNum = len(means)
        laplacianLayers = [None] * maxPyramidNum # 50 * 6 layers
        for i in range(maxPyramidNum):
            # self.imshow(self.imgGray)
            tmp = self.remapLum(self.imgGray,means[i],self.alpha,self.beta,self.sigma)
            # self.imshow(tmp)
            #build gaussion and laplacian pyramid
            gaussionPyramid, laplacianPyramid = [],[]
            gaussionPyramid, laplacianPyramid = self.buildLaplacianPyramid(tmp,gaussionPyramid,laplacianPyramid,leves)
            laplacianLayers[i] = laplacianPyramid
        return laplacianLayers

    def rebuildLaplacianPyramid(self,lapLacianPyramid):
        maxLeve = len(lapLacianPyramid)
        outImg = lapLacianPyramid[maxLeve - 1]
        for leve in range(maxLeve - 1,0,-1):
            rows,cols = lapLacianPyramid[leve -1].shape[0],lapLacianPyramid[leve-1].shape[1]
            upSampleImg = self.upSample(outImg,rows,cols)
            outImg = np.add(upSampleImg,lapLacianPyramid[leve -1])
        return outImg

    def clipBit(self,inPut,min,max):
        if min > max :
            return "error"
        if inPut > max:
            outPut = max
        elif inPut < min:
            outPut = min
        else:
            outPut = inPut
        return outPut

    def fastLoalConstrast(self,imgGray,means,laplacaianPyramids):
        maxPyramidNum = len(laplacaianPyramids)
        maxLeve = len(laplacaianPyramids[0])
        print("maxPyramid num = {0},maxLeve = {1},means len = {2}".format(maxPyramidNum,maxLeve,len(means)))
        gaussianPyramid,laplacianPyramid = [],[]
        gaussianPyramid, laplacianPyramid = self.buildLaplacianPyramid(imgGray,gaussianPyramid,laplacianPyramid,maxLeve)
        print("gaussPyrShape={0} lapLacPyrShape = {1}".format(len(gaussianPyramid),len(laplacianPyramid)))
        # laplacianPyramid = gaussianPyramid.copy()
        wtKernel = np.array([1/9,2/9,3/9,2/9,1/9]).astype(np.float64)
        for leve in range(maxLeve - 1):
            g = gaussianPyramid[leve]
            for row in range(g.shape[0]):
                for col in range(g.shape[1]):
                    gaussVal = g[row][col]
                    minDistance = 9999999
                    layerIndex = -1
                    for x in range(maxPyramidNum):
                        distance = abs(gaussVal - means[x])
                        if distance < minDistance:
                            minDistance = distance
                            layerIndex = x
                    #Get nerast layer index
                    rLayerBefSecIndex = self.clipBit(layerIndex - 2,0,maxPyramidNum - 1)
                    rLayerBefFstIndex = self.clipBit(layerIndex - 1,0,maxPyramidNum - 1)
                    rLayerCurrenIndex = self.clipBit(layerIndex - 0,0,maxPyramidNum - 1)
                    rLayerAftFstIndex = self.clipBit(layerIndex + 1,0,maxPyramidNum - 1)
                    rLayerAftSecIndex = self.clipBit(layerIndex + 2,0,maxPyramidNum - 1)
                    #Blending nerest layer
                    Blended =   laplacaianPyramids[rLayerBefSecIndex][leve][row][col] * wtKernel[0] +\
                                laplacaianPyramids[rLayerBefFstIndex][leve][row][col] * wtKernel[1] +\
                                laplacaianPyramids[rLayerCurrenIndex][leve][row][col] * wtKernel[2] +\
                                laplacaianPyramids[rLayerAftFstIndex][leve][row][col] * wtKernel[3] +\
                                laplacaianPyramids[rLayerAftSecIndex][leve][row][col] * wtKernel[4]
                    laplacianPyramid[leve][row][col] = Blended
        print("Finally lapLacianPyramid length = {0}".format(len(laplacianPyramid)))
        outImg = self.rebuildLaplacianPyramid(laplacianPyramid)
        return outImg

    #recovery img from log domin to rgb
    def recoverRgbImage(self,inImage):
        imgGray = inImage.copy()
        outImage = np.zeros_like(self.imgRGB,dtype=np.float64)
        minVal = np.min(imgGray)
        maxVal = np.max(imgGray)
        interV = maxVal - minVal
        for row in range(self.imgRGB.shape[0]):
            for col in range(self.imgRGB.shape[1]):
                BGR = [0.00,0.00,0.00]
                for c in range(self.imgRGB.shape[2]):
                    BGR[c] = imgGray[row][col] - minVal
                    BGR[c] = BGR[c] + self.gamma * (self.imgRGB[row][col][c] - self.imgGray[row][col])
                    BGR[c] = BGR[c] / interV
                outImage[row][col][0],outImage[row][col][1],outImage[row][col][2] = BGR[0],BGR[1],BGR[2]
        return outImage

    def normalizeImgRgb(self,inImgRgb):
        norImgRgb = np.zeros_like(inImgRgb,dtype=np.float64)
        imgGray = inImgRgb.copy()
        imgGray = self.rgb2gray(imgGray)
        minVal = np.min(imgGray)
        maxVal = np.max(imgGray)
        interV = 255 / (maxVal - minVal)
        for row in range(inImgRgb.shape[0]):
            for col in range(inImgRgb.shape[1]):
                BGR = [inImgRgb[row][col][0],inImgRgb[row][col][1],inImgRgb[row][col][2]]
                for c in range(inImgRgb.shape[2]):
                    BGR[c] = BGR[c] - minVal
                    BGR[c] = BGR[c] * interV
                norImgRgb[row][col][0], norImgRgb[row][col][1], norImgRgb[row][col][2] = BGR[0], BGR[1], BGR[2]
        return norImgRgb

    def grayImgNorm(self,inImgGray):
        minVal = np.min(inImgGray)
        maxVal = np.max(inImgGray)
        interV = 255 / maxVal - minVal
        imgNormed = np.zeros_like(inImgGray,dtype=np.float64)
        for row in range(inImgGray.shape[0]):
            for col in range(inImgGray.shape[1]):
                imgNormed[row][col] = math.exp(inImgGray[row][col])
        self.imshow(imgNormed)

    def clipImage(self,inImg):
        outImg = np.zeros_like(inImg)
        for row in range(inImg.shape[0]):
            for col in range(inImg.shape[1]):
                for c in range(inImg.shape[2]):
                    tmp = self.clipBit(inImg[row][col][c],0,255)
                    outImg[row][col][c] = tmp
        self.eTime = time.time()
        return outImg
    def writeResImage(self,resImgRgb):
        cv2.imwrite(os.path.join(self.opath,"./srcRgb_gamma_{0}_alpha_{1}_beta_{2}_sigma_{3}_num_{4}_leve_{5}.png"\
                    .format(self.gamma,self.alpha,self.beta,self.sigma,self.maxPyramidNum,self.maxLeves)),\
                    self.imgRGB.astype(np.uint8))
        cv2.imwrite(os.path.join(self.opath,"./srcRgb_gamma_{0}_alpha_{1}_beta_{2}_sigma_{3}_num_{4}_leve_{5}_costT_{6}.png"\
                    .format(self.gamma,self.alpha,self.beta,self.sigma,self.maxPyramidNum,self.maxLeves,int(self.eTime - self.sTime))), \
                    resImgRgb.astype(np.uint8))
        cv2.imshow("srcRgb",self.imgRGB.astype(np.uint8))
        cv2.imshow("resImgRgb",resImgRgb.astype(np.uint8))
        cv2.waitKey(0)


        return 0

    def imageProcesser(self):
        means = self.getMeans(self.imgGray,self.maxPyramidNum)
        # build remap laplayers 50 * 6
        laplacianPyramids = self.remapLaplacianPyramid(self.imgGray,means,self.maxLeves)
        outImgGray = self.fastLoalConstrast(self.imgGray,means,laplacianPyramids)

        #show res Laplacian filter resoult
        # cv2.imshow("resImg",outImgGray.astype(np.uint8))
        # cv2.imshow("srcGray", self.imgGray.astype(np.uint8))
        # cv2.waitKey(0)
        #trans from gray2 rgb and apply gamma
        imgRecorv = self.recoverRgbImage(outImgGray)
        # cv2.imshow("srcRGBrecorv", (imgRecorv * 255).astype(np.uint8))
        # cv2.waitKey(0)
        #normalize 1bit to 8bit
        norImgRgb = self.normalizeImgRgb(imgRecorv)
        # cv2.imshow("norImgRgb",norImgRgb.astype(np.uint8))
        # cv2.waitKey(0)

        #clip img to 0 ~  255
        resImgRgb = self.clipImage(norImgRgb)

        #write resImage
        self.writeResImage(resImgRgb)

    def testLaplacian(self):
        path = r"E:\local_laplacian_filter\my_laplacian\data\inputdata\flower.png"
        gray = cv2.imread(path,cv2.IMREAD_UNCHANGED)
        gray = cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)
        guassionPyramid, laplacianPyramid = [],[]
        gaussPyramid,laplacianPyramid = self.buildLaplacianPyramid(gray, guassionPyramid, laplacianPyramid, 7)
        outImg = self.rebuildLaplacianPyramid(laplacianPyramid)
        self.imshow(outImg)
    def imshow(self,image):
        cv2.imshow("image",image.astype(np.uint8))
        cv2.waitKey(0)

if __name__ == '__main__':
    sTime = time.time()
    iPath = r".\data\20201108213401.jpg"
    oPath = r".\data"
    fastLapcian = FastLaplacianFilter(ipath=iPath,opath=oPath,sigma= 80,alpha= 1.5,beta=1,noise=0.01,gamma=1.3).imageProcesser()
    eTime = time.time()
    print("Total cost time :{0}".format(eTime - sTime))
    # fastLapcian = FastLaplacianFilter(path=path, sigma=0.9, alpha=1.0, beta=0.0, noise=0.01, gamma=1.2).testLaplacian()



