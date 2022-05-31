#CV HW8
#R08945043
import cv2
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)

def gaussian_noise(img,amp):
    p=np.random.normal(size=img.shape)
    img_tmp=img+(np.multiply(p,amp)).astype(np.uint8)
    return img_tmp

def salt_pepper_noise(img,threshold):
    img_tmp=img.copy()
    p=np.random.uniform(size=img.shape)
    img_tmp[np.where(p<threshold)]=0
    img_tmp[np.where(p>(1-threshold))]=255
    return img_tmp

def dilation (img,kernel):
    row_ker,col_ker=(kernel.shape[0]-1)//2,(kernel.shape[0]-1)//2
    kernel_value=np.zeros((kernel.shape),dtype=np.uint8)
    img_padding=np.zeros((img.shape[0]+2*row_ker,img.shape[1]+2*col_ker),dtype=np.uint8)
    img_tmp=np.zeros((img.shape[0]+2*row_ker,img.shape[1]+2*col_ker),dtype=np.uint8)
    img_padding[row_ker:img.shape[0]+row_ker,col_ker:img.shape[1]+col_ker]=img[:,:].copy()
    for i in range(row_ker,img_padding.shape[0]-row_ker):
        for j in range(col_ker,img_padding.shape[1]-col_ker):
            tmp=img_padding[i-row_ker:i+row_ker+1,j-col_ker:j+col_ker+1]+kernel_value
            max_value=max(tmp[kernel==1])
            img_tmp[i,j]=max_value
    return img_tmp

def ersion (img,kernel):
    row_ker,col_ker=(kernel.shape[0]-1)//2,(kernel.shape[0]-1)//2
    kernel_value=np.zeros((kernel.shape),dtype=np.uint8)
    img_padding=np.full((img.shape[0]+2*row_ker,img.shape[1]+2*col_ker),255,dtype=np.uint8)
    img_tmp=np.zeros((img.shape[0]+2*row_ker,img.shape[1]+2*col_ker),dtype=np.uint8)
    img_padding[row_ker:img.shape[0]+row_ker,col_ker:img.shape[1]+col_ker]=img[:,:].copy()
    for i in range(row_ker,img_padding.shape[0]-row_ker):
        for j in range(col_ker,img_padding.shape[1]-col_ker):
            tmp=img_padding[i-row_ker:i+row_ker+1,j-col_ker:j+col_ker+1]+kernel_value
            min_value=min(tmp[kernel==1])
            img_tmp[i,j]=min_value
    return img_tmp

def morphology_open (img,kernel=np.array([[0,1,1,1,0],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[0,1,1,1,0]])):
    row_ker,col_ker=(kernel.shape[0]-1)//2,(kernel.shape[0]-1)//2
    img_tmp=ersion(img,kernel)
    img_tmp=img_tmp[row_ker:img.shape[0]+row_ker,col_ker:img.shape[1]+col_ker]
    img_tmp=dilation(img_tmp,kernel)
    return img_tmp[row_ker:img.shape[0]+row_ker,col_ker:img.shape[1]+col_ker]

def morphology_close (img,kernel=np.array([[0,1,1,1,0],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[0,1,1,1,0]])):
    row_ker,col_ker=(kernel.shape[0]-1)//2,(kernel.shape[0]-1)//2
    img_tmp=dilation(img,kernel)
    img_tmp=img_tmp[row_ker:img.shape[0]+row_ker,col_ker:img.shape[1]+col_ker]
    img_tmp=ersion(img_tmp,kernel)
    return img_tmp[row_ker:img.shape[0]+row_ker,col_ker:img.shape[1]+col_ker]

def snr(img_ori,img_proc):
    img_ori_tmp=img_ori.astype(float)/255.0
    img_proc_tmp=img_proc.astype(float)/255.0
    u=np.sum(img_ori_tmp)/img_ori_tmp.shape[0]/img_ori_tmp.shape[1]
    vs=np.sum(np.power((img_ori_tmp-u),2))/img_ori_tmp.shape[0]/img_ori_tmp.shape[1]
    un=np.sum(img_proc_tmp-img_ori_tmp)/img_ori_tmp.shape[0]/img_ori_tmp.shape[1]
    vn=np.sum(np.power((img_proc_tmp-img_ori_tmp-un),2))/img_ori_tmp.shape[0]/img_ori_tmp.shape[1]
    return 20*np.log10((vs**0.5)/(vn**0.5))

def boxfilter(img,size):
    size_ker=size//2
    img_padding=cv2.copyMakeBorder(img,size_ker,size_ker,size_ker,size_ker,cv2.BORDER_REFLECT)
    img_tmp=np.zeros((img.shape[0]+2*size_ker,img.shape[1]+2*size_ker),dtype=np.uint8)
    for i in range(size_ker,img_padding.shape[0]-size_ker):
        for j in range(size_ker,img_padding.shape[1]-size_ker):
            tmp=img_padding[i-size_ker:i+size_ker+1,j-size_ker:j+size_ker+1]
            tmp=np.sum(tmp)//(size**2)
            img_tmp[i,j]=tmp
    return img_tmp[size_ker:img.shape[0]+size_ker,size_ker:img.shape[1]+size_ker]

def medianfilter(img,size):
    size_ker=size//2
    img_padding=cv2.copyMakeBorder(img,size_ker,size_ker,size_ker,size_ker,cv2.BORDER_REFLECT)
    img_tmp=np.zeros((img.shape[0]+2*size_ker,img.shape[1]+2*size_ker),dtype=np.uint8)
    for i in range(size_ker,img_padding.shape[0]-size_ker):
        for j in range(size_ker,img_padding.shape[1]-size_ker):
            tmp=img_padding[i-size_ker:i+size_ker+1,j-size_ker:j+size_ker+1]
            tmp=np.median(tmp)
            img_tmp[i,j]=tmp
    return img_tmp[size_ker:img.shape[0]+size_ker,size_ker:img.shape[1]+size_ker]


#read file and img binary 
#filepath='Absolute path of lena.bmp'
filepath='D:\\report\\course_data\\109-1\\CV\\homework\\lena.bmp'
img_ori=cv2.imread(filepath,flags=cv2.IMREAD_GRAYSCALE)


#gaussian noise 10
'''
img_gn10=gaussian_noise(img_ori,10)
img_gn10_bf3=boxfilter(img_gn10,3)
img_gn10_bf5=boxfilter(img_gn10,5)
img_gn10_mf3=medianfilter(img_gn10,3)
img_gn10_mf5=medianfilter(img_gn10,5)
img_gn10_oc=morphology_close(morphology_open(img_gn10))
img_gn10_co=morphology_open(morphology_close(img_gn10))
cv2.imshow("noise",img_gn10)
cv2.imshow("box3",img_gn10_bf3)
cv2.imshow("box5",img_gn10_bf5)
cv2.imshow("median3",img_gn10_mf3)
cv2.imshow("median5",img_gn10_mf5)
cv2.imshow("oc",img_gn10_oc)
cv2.imshow("co",img_gn10_co)
cv2.waitKey()
cv2.destroyAllWindows()
'''
#gaussian noise 30
'''
img_gn30=gaussian_noise(img_ori,30)
img_gn30_bf3=boxfilter(img_gn30,3)
img_gn30_bf5=boxfilter(img_gn30,5)
img_gn30_mf3=medianfilter(img_gn30,3)
img_gn30_mf5=medianfilter(img_gn30,5)
img_gn30_oc=morphology_close(morphology_open(img_gn30))
img_gn30_co=morphology_open(morphology_close(img_gn30))
cv2.imshow("noise",img_gn30)
cv2.imshow("box3",img_gn30_bf3)
cv2.imshow("box5",img_gn30_bf5)
cv2.imshow("median3",img_gn30_mf3)
cv2.imshow("median5",img_gn30_mf5)
cv2.imshow("oc",img_gn30_oc)
cv2.imshow("co",img_gn30_co)
cv2.waitKey()
cv2.destroyAllWindows()
'''
#salt-and-pepper noise 0.05
'''
img_spn5=salt_pepper_noise(img_ori,0.05)
img_spn5_bf3=boxfilter(img_spn5,3)
img_spn5_bf5=boxfilter(img_spn5,5)
img_spn5_mf3=medianfilter(img_spn5,3)
img_spn5_mf5=medianfilter(img_spn5,5)
img_spn5_oc=morphology_close(morphology_open(img_spn5))
img_spn5_co=morphology_open(morphology_close(img_spn5))
cv2.imshow("noise",img_spn5)
cv2.imshow("box3",img_spn5_bf3)
cv2.imshow("box5",img_spn5_bf5)
cv2.imshow("median3",img_spn5_mf3)
cv2.imshow("median5",img_spn5_mf5)
cv2.imshow("oc",img_spn5_oc)
cv2.imshow("co",img_spn5_co)
cv2.waitKey()
cv2.destroyAllWindows()
'''
#salt-and-pepper noise 0.1
'''
img_spn10=salt_pepper_noise(img_ori,0.1)
img_spn10_bf3=boxfilter(img_spn10,3)
img_spn10_bf5=boxfilter(img_spn10,5)
img_spn10_mf3=medianfilter(img_spn10,3)
img_spn10_mf5=medianfilter(img_spn10,5)
img_spn10_oc=morphology_close(morphology_open(img_spn10))
img_spn10_co=morphology_open(morphology_close(img_spn10))
cv2.imshow("noise",img_spn10)
cv2.imshow("box3",img_spn10_bf3)
cv2.imshow("box5",img_spn10_bf5)
cv2.imshow("median3",img_spn10_mf3)
cv2.imshow("median5",img_spn10_mf5)
cv2.imshow("oc",img_spn10_oc)
cv2.imshow("co",img_spn10_co)
cv2.waitKey()
cv2.destroyAllWindows()
'''

path='D:\\report\\course_data\\109-1\\CV\\homework\\hw8\\'
#name='GN10'
#img_noise=gaussian_noise(img_ori,10)
#name='GN30'
#img_noise=gaussian_noise(img_ori,30)
#name='SPN5'
#img_noise=salt_pepper_noise(img_ori,0.05)
name='SPN10'
img_noise=salt_pepper_noise(img_ori,0.1)
print(round(snr(img_ori,img_noise),3))
cv2.imwrite(path+name+'_noise.png',img_noise)
img_bf3=boxfilter(img_noise,3)
print(round(snr(img_ori,img_bf3),3))
cv2.imwrite(path+name+'_bf3.png',img_bf3)
img_bf5=boxfilter(img_noise,5)
print(round(snr(img_ori,img_bf5),3))
cv2.imwrite(path+name+'_bf5.png',img_bf5)
img_mf3=medianfilter(img_noise,3)
print(round(snr(img_ori,img_mf3),3))
cv2.imwrite(path+name+'_mf3.png',img_mf3)
img_mf5=medianfilter(img_noise,5)
print(round(snr(img_ori,img_mf5),3))
cv2.imwrite(path+name+'_mf5.png',img_mf5)
img_oc=morphology_close(morphology_open(img_noise))
print(round(snr(img_ori,img_oc),3))
cv2.imwrite(path+name+'_oc.png',img_oc)
img_co=morphology_open(morphology_close(img_noise))
print(round(snr(img_ori,img_co),3))
cv2.imwrite(path+name+'_co.png',img_co)