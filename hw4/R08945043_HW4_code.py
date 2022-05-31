#CV HW4
#R08945043
import cv2
import numpy as np

def dilation (img,kernel):
    row_ker,col_ker=(kernel.shape[0]-1)//2,(kernel.shape[0]-1)//2
    img_padding=np.zeros((img.shape[0]+2*row_ker,img.shape[1]+2*col_ker))
    img_tmp=np.zeros((img.shape[0]+2*row_ker,img.shape[1]+2*col_ker))
    img_padding[row_ker:img.shape[0]+row_ker,col_ker:img.shape[1]+col_ker]=img[:,:].copy()
    for i in range(row_ker,img_padding.shape[0]-row_ker):
        for j in range(col_ker,img_padding.shape[1]-col_ker):
            if np.logical_and(img_padding[i-row_ker:i+row_ker+1,j-col_ker:j+col_ker+1],kernel).any():
                img_tmp[i,j]=255
    return img_tmp

def ersion (img,kernel):
    row_ker,col_ker=(kernel.shape[0]-1)//2,(kernel.shape[0]-1)//2
    img_padding=np.zeros((img.shape[0]+2*row_ker,img.shape[1]+2*col_ker))
    img_tmp=np.zeros((img.shape[0]+2*row_ker,img.shape[1]+2*col_ker))
    img_padding[row_ker:img.shape[0]+row_ker,col_ker:img.shape[1]+col_ker]=img[:,:].copy()
    for i in range(row_ker,img_padding.shape[0]-row_ker):
        for j in range(col_ker,img_padding.shape[1]-col_ker):
            if (np.logical_and(img_padding[i-row_ker:i+row_ker+1,j-col_ker:j+col_ker+1],kernel)==kernel).all():
                img_tmp[i,j]=255
    return img_tmp

def hit_and_miss (img,kernel_j,kernel_k):
    img_rev=np.zeros(img.shape[:])
    img_rev[np.where(img==0)],img_rev[np.where(img==255)]=255,0
    img_ers=ersion(img,kernel_j)
    img_rev=ersion(img_rev,kernel_k)
    img_output=np.multiply(img_ers,img_rev)
    return img_output

#read file and img binary 
filepath='Absolute path of lena.bmp'
img_bin=cv2.imread(filepath,flags=cv2.IMREAD_GRAYSCALE)
img_bin[np.where(img_bin<128)],img_bin[np.where(img_bin>=128)]=0,255
#kernel
kernel_oct=np.array([[0,1,1,1,0],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[0,1,1,1,0]])
kernel_j=np.array([[0,0,0,0,0],[0,0,0,0,0],[1,1,0,0,0],[0,1,0,0,0],[0,0,0,0,0]])
kernel_k=np.array([[0,0,0,0,0],[0,1,1,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]])
#dilation
img_dilation=dilation(img_bin,kernel_oct)
#ersion
img_ersion=ersion(img_bin,kernel_oct)
#open (ersion then dilation)
img_open=dilation(img_ersion,kernel_oct)
#close (dilation and ersion)
img_close=ersion(img_dilation,kernel_oct)
#hit and miss
img_hitmiss=hit_and_miss(img_bin,kernel_j,kernel_k)


#show image
cv2.imshow("ori",img_bin)
cv2.imshow("dilation",img_dilation)
cv2.imshow("ersion",img_ersion)
cv2.imshow("open",img_open)
cv2.imshow("close",img_close)
cv2.imshow("hit&miss",img_hitmiss)
cv2.waitKey()
cv2.destroyAllWindows()

#image output
#outpath='outpath'
#cv2.imwrite(outpath+'dilation.png',img_dilation)
#cv2.imwrite(outpath+'ersion.png',img_ersion)
#cv2.imwrite(outpath+'close.png',img_close)
#cv2.imwrite(outpath+'open.png',img_open)
#cv2.imwrite(outpath+'hitandmiss.png',img_hitmiss)