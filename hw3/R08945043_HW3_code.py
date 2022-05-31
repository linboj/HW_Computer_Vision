#CV HW3 
#R08945043
import cv2
import numpy as np
import matplotlib.pyplot as plt 

#convert image to intensity-number data
def intensity_count(img):
    stat_data=np.zeros((2,256))
    stat_data[0,:]=[v for v in range(256)]
    h,w=img.shape[0:2]
    for i in range(h):
        for j in range(w):
            stat_data[1,img[i,j]]=stat_data[1,img[i,j]]+1
    return stat_data

#filepath='file path of lena.bmp'
filepath='The absolute path of lena.bmp'
img_ori=cv2.imread(filepath,flags=cv2.IMREAD_GRAYSCALE)
h,w=img_ori.shape[0:2]

#(a)original image and its histogram
hist_ori=intensity_count(img_ori)
#(b)image with intensity divided by 3 and its histogram
img_proc=img_ori.copy()
img_proc[:,:]=np.round(img_proc[:,:]/3)
hist_proc=intensity_count(img_proc)
#(c)image after applying histogram equalization to (b) and its histogram
img_equ=img_proc.copy()
histsum=np.sum(hist_proc[1,:])
cumsumlist=np.cumsum(hist_proc[1,:])
for i in range(h):
    for j in range(w):
        img_equ[i,j]=round(255*cumsumlist[img_equ[i,j]]/histsum)
hist_equ=intensity_count(img_equ)

#output histogram
f1=plt.figure(1)
plt.bar(hist_ori[0,:],hist_ori[1,:],width=1)
plt.xlabel('Intensity')
plt.ylabel('Number')
f2=plt.figure(2)
plt.bar(hist_proc[0,:],hist_proc[1,:],width=1)
plt.xlabel('Intensity')
plt.ylabel('Number')
f3=plt.figure(3)
plt.bar(hist_equ[0,:],hist_equ[1,:],width=1)
plt.xlabel('Intensity')
plt.ylabel('Number')
plt.show()
#show image
cv2.imshow("ori",img_ori)
cv2.imshow("proc",img_proc)
cv2.imshow("equ",img_equ)
cv2.waitKey()
cv2.destroyAllWindows()

#image output
outpath='outpath'
cv2.imwrite(outpath+'proc.png',img_proc)
cv2.imwrite(outpath+'equ.png',img_equ)