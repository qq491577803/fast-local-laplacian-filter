import numpy as np
import cv2
import matplotlib.pyplot as plt



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
        return upSample


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
        imgUs = cv2.filter2D(imgUs,-1,kernel=self.upSampleKernel())
        print("src shape",img.shape,"ds shape",imgDs.shape,"us shape",imgUs.shape)
        cv2.imshow("srcImage",img)
        cv2.imshow("Ds",imgDs.astype(np.uint8))
        cv2.imshow("Us",imgUs.astype(np.uint8))
        cv2.waitKey(0)
if __name__ == '__main__':
    path = r"C:\Users\Administrator\Desktop\IMG_2405.JPG"
    Sample().processer(path)






