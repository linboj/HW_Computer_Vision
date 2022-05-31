#CV HW6
#R08945043
import cv2
import numpy as np

def yokoi(img):
    #add padding
    img_tmp=np.zeros((img.shape[0]+2,img.shape[1]+2))
    img_tmp[1:img_tmp.shape[0]-1,1:img_tmp.shape[1]-1]=img
    img_con=np.full(img.shape,None)
    #define 4 h function
    def h1(img,r,c):
        s1,s2,s3,s4=img[r,c],img[r,c+1],img[r-1,c+1],img[r-1,c]
        if s1==s2 and (s1!=s3 or s1!=s4):
            return 'q'
        elif s1==s2 and (s1==s3 and s1==s4):
            return 'r'
        elif s1!=s2:
            return 's'
    def h2(img,r,c):
        s1,s2,s3,s4=img[r,c],img[r-1,c],img[r-1,c-1],img[r,c-1]
        if s1==s2 and (s1!=s3 or s1!=s4):
            return 'q'
        elif s1==s2 and (s1==s3 and s1==s4):
            return 'r'
        elif s1!=s2:
            return 's'
    def h3(img,r,c):
        s1,s2,s3,s4=img[r,c],img[r,c-1],img[r+1,c-1],img[r+1,c]
        if s1==s2 and (s1!=s3 or s1!=s4):
            return 'q'
        elif s1==s2 and (s1==s3 and s1==s4):
            return 'r'
        elif s1!=s2:
            return 's'
    def h4(img,r,c):
        s1,s2,s3,s4=img[r,c],img[r+1,c],img[r+1,c+1],img[r,c+1]
        if s1==s2 and (s1!=s3 or s1!=s4):
            return 'q'
        elif s1==s2 and (s1==s3 and s1==s4):
            return 'r'
        elif s1!=s2:
            return 's'
    for i in range(1,1+img.shape[0]):
        for j in range(1,1+img.shape[1]):
            if img_tmp[i,j]!=0:
                res=h1(img_tmp,i,j),h2(img_tmp,i,j),h3(img_tmp,i,j),h4(img_tmp,i,j)
                if res.count('r')==4:
                    img_con[i-1,j-1]=5
                else:
                    img_con[i-1,j-1]=res.count('q')
    return img_con
        

#read file and img binary 
filepath='Absolute path of lena.bmp'
img_ori=cv2.imread(filepath,flags=cv2.IMREAD_GRAYSCALE)
#downsampling and binarization
img_downsample=np.zeros((64,64),np.uint8)
for i in range(img_ori.shape[0]//8):
    for j in range(img_ori.shape[1]//8):
        img_downsample[i,j]=img_ori[0+8*i,0+8*j]
img_downsample[np.where(img_downsample<128)]=0
img_downsample[np.where(img_downsample>=128)]=255
#Yokoi connectivity number
img_yokoi=yokoi(img_downsample)
#align
img_yokoi[np.where(img_yokoi==None)]=' '
img_yokoi[np.where(img_yokoi==0)]=' '
#save as txt
np.savetxt('result.txt',img_yokoi,fmt='%s',delimiter=" ")

