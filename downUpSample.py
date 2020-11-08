import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


class Sample():
    def __init__(self):
        pass
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
        a = [.05, .25, .4, .25, .05]
        kernel_1 = np.array(a, np.float64)
        kernel_2 = np.array(kernel_1)[np.newaxis]
        kernel = kernel_2.T
        f = np.multiply(kernel, kernel_1, None)
        return f

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
            print("dddd")
            upSample = self.upSample(out,laplacainPyraimd[i-1].shape[0],laplacainPyraimd[i-1].shape[1])
            out = np.add(upSample,laplacainPyraimd[i-1])
        savePath = r"C:\Users\Administrator\Desktop"
        cv2.imwrite(os.path.join(savePath,"res.png"),out.astype(np.uint8))
        return out




    def laplacian_pyramid(self,img,leves):
        laplacainLayers = [None] * leves
        guassLayers = [None] * leves

        guassLayers[0] = img.copy()
        for leve in range(1,leves):
            tmpImge = guassLayers[leve - 1]
            rows = int((tmpImge.shape[0] + 1) / 2)
            cols = int((tmpImge.shape[1] + 1)/ 2)
            downSampleImg = self.downSample(tmpImge,rows,cols)
            upSampleImg = self.upSample(downSampleImg,tmpImge.shape[0],tmpImge.shape[1])
            guassLayers[leve] = downSampleImg
            laplacainLayers[leve-1] = np.subtract(tmpImge,upSampleImg)
        laplacainLayers[leves-1] = guassLayers[leves-1]
        return laplacainLayers
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

    def processer(self,path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,dsize=(999,999))
        #dowsample
        cols = int((img.shape[0] + 1) / 2)
        rows = int((img.shape[1] + 1) / 2)
        imgDs = self.downSample(img,cols,rows)
        #upsample
        cols = img.shape[0]
        rows = img.shape[1]
        imgUs = self.upSample(imgDs,cols,rows)
        # imgUs = cv2.filter2D(imgUs,-1,kernel=self.upSampleKernel())
        print("src shape",img.shape,"ds shape",imgDs.shape,"us shape",imgUs.shape)
        diff = np.subtract(img,imgUs)

        cv2.imshow("srcImage",img)
        cv2.imshow("Ds",imgDs.astype(np.uint8))
        cv2.imshow("Us",imgUs.astype(np.uint8))
        cv2.imshow("diff",diff.astype(np.uint8))
        cv2.waitKey(0)

    def downsample11(self,image, subwindow):
        r = image.shape[0]
        c = image.shape[1]
        subwindow = np.arange(r*c).reshape(r, c)
        subwindow_child = child_window(subwindow)
        border_mode = 'reweighted'
        R = None
        kernel = self.filter()
        R = cv2.filter2D(image.astype(np.float64), -1, kernel)
        Z = numpy.ones([r, c], dtype=np.float64)
        Z = cv2.filter2D(Z, -1, kernel)
        R = np.divide(R, Z)
        reven = np.remainder(subwindow[0], 2) == 0
        ceven = np.remainder(subwindow[2], 2) == 0
        R = R[reven: r: 2, ceven: c: 2]
        print(R)
        cv2.imshow("ds111",R)
        cv2.waitKey(0)
        return R
if __name__ == '__main__':
    path = r"E:\local_laplacian_filter\my_laplacian\data\inputdata\flower.png"
    # Sample().processer(path)
    # img = cv2.imread(path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = img.astype(float) / 255.
    # Sample = Sample()
    # lap = Sample.laplacian_pyramid(img,4)
    # out = Sample.rebuidLaplacian_pyramid(lap)
    # print(out.shape)
    # cv2.imshow("out",out)
    # cv2.waitKey(0)

    # subwindow = [1, r, 1, c]

    Sample = Sample()
    inImg = np.zeros(shape=(267,400))
    rows = 800
    cols = 533
    Sample.upSample(inImg,rows,cols)
