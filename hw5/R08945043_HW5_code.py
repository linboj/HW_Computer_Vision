#CV HW5
#R08945043
import cv2
import numpy as np

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

def morphology_open (img,kernel):
    row_ker,col_ker=(kernel.shape[0]-1)//2,(kernel.shape[0]-1)//2
    img_tmp=ersion(img,kernel)
    img_tmp=img_tmp[row_ker:img.shape[0]+row_ker,col_ker:img.shape[1]+col_ker]
    img_tmp=dilation(img_tmp,kernel)
    return img_tmp

def morphology_close (img,kernel):
    row_ker,col_ker=(kernel.shape[0]-1)//2,(kernel.shape[0]-1)//2
    img_tmp=dilation(img,kernel)
    img_tmp=img_tmp[row_ker:img.shape[0]+row_ker,col_ker:img.shape[1]+col_ker]
    img_tmp=ersion(img_tmp,kernel)
    return img_tmp


#read file and img binary 
filepath='Absolute path of lena.bmp'
img_ori=cv2.imread(filepath,flags=cv2.IMREAD_GRAYSCALE)
#kernel
kernel_oct=np.array([[0,1,1,1,0],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[0,1,1,1,0]])
#dilation
img_dilation=dilation(img_ori,kernel_oct)
#ersion
img_ersion=ersion(img_ori,kernel_oct)
#open (ersion then dilation)
img_open=morphology_open(img_ori,kernel_oct)
#close (dilation and ersion)
img_close=morphology_close(img_ori,kernel_oct)


#show image
cv2.imshow("ori",img_ori)
cv2.imshow("dilation",img_dilation)
cv2.imshow("ersion",img_ersion)
cv2.imshow("open",img_open)
cv2.imshow("close",img_close)
cv2.waitKey()
cv2.destroyAllWindows()

#image output
#outpath='outpath'
#cv2.imwrite(outpath+'dilation.png',img_dilation)
#cv2.imwrite(outpath+'ersion.png',img_ersion)
#cv2.imwrite(outpath+'close.png',img_close)
#cv2.imwrite(outpath+'open.png',img_open)
