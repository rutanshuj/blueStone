from PIL import Image,ImageFilter
from scipy.misc import toimage
from operator import itemgetter
from skimage import measure
import numpy as np
import copy
import heapq
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage.filters import median_filter


#----------------------------------------------------------------

class preprocessing:
    def pre_proc_image(self,img):
        #img_removed_noise=self.apply_median_filter(img)
        img_removed_noise=self.remove_noise(img)
        p1,p2,LL=self.get_line_position(img_removed_noise)
        img=self.remove_line(p1,p2,LL,img_removed_noise)
        img=median_filter(np.asarray(img),1)
        return img

    def remove_noise(self,img):
        img_gray=img.convert('L')
        w,h=img_gray.size
        max_color=np.asarray(img_gray).max()
        pix_access_img=img_gray.load()
        row_img=list(map(lambda x:255 if x in range(max_color-15,max_color+1) else 0,np.asarray(img_gray.getdata())))
        img=np.reshape(row_img,[h,w])
        return img

    def apply_median_filter(self,img):
        img_gray=img.convert('L')
        img_gray=cv2.medianBlur(np.asarray(img_gray),3)
        img_bw=(img_gray>np.mean(img_gray))*255
        return img_bw

    def eliminate_zeros(self,vector):
        return [(dex,v) for (dex,v) in enumerate(vector) if v!=0 ]

    def get_line_position(self,img):
        sumx=img.sum(axis=0)
        list_without_zeros=self.eliminate_zeros(sumx)
        min1,min2=heapq.nsmallest(2,list_without_zeros,key=itemgetter(1))
        l=[dex for [dex,val] in enumerate(sumx) if val==min1[1] or val==min2[1]]
        mindex=[l[0],l[len(l)-1]]
        cols=img[:,mindex[:]]
        col1=cols[:,0]
        col2=cols[:,1]
        col1_without_0=self.eliminate_zeros(col1)
        col2_without_0=self.eliminate_zeros(col2)
        line_length=len(col1_without_0)
        dex1=col1_without_0[round(len(col1_without_0)/2)][0]
        dex2=col2_without_0[round(len(col2_without_0)/2)][0]
        p1=[dex1,mindex[0]]
        p2=[dex2,mindex[1]]
        return p1,p2,line_length

    def remove_line(self,p1,p2,LL,img):
        m=(p2[0]-p1[0])/(p2[1]-p1[1]) if p2[1]!=p1[1] else np.inf
        w,h=len(img),len(img[0])
        x=list(range(h))
        y=list(map(lambda z : int(np.round(p1[0]+m*(z-p1[1]))),x))
        img_removed_line=list(img)
        for dex in range(h):
            i,j=y[dex],x[dex]
            i=int(i)
            j=int(j)
            rlist=[]
            while True:
                f1=i
                if img_removed_line[i][j]==0 and img_removed_line[i-1][j]==0:
                    break
                rlist.append(i)
                i=i-1

            i,j=y[dex],x[dex]
            i=int(i)
            j=int(j)
            while True:
                f2=i
                if img_removed_line[i][j]==0 and img_removed_line[i+1][j]==0:
                    break
                rlist.append(i)
                i=i+1
            if np.abs(f2-f1) in [LL+1,LL,LL-1]:
                rlist=list(set(rlist))
                for k in rlist:
                    img_removed_line[k][j]=0

        return img_removed_line

if __name__ == '__main__':
    img = cv2.imread("captcha.png")
    p = preprocessing()
    imgNew = p.pre_proc_image(img)
    cv2.imshow("Image", imgNew)
    cv2.waitKey(0)
