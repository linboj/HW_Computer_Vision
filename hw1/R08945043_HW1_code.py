#CV HW1 
#R08945043
import cv2
import numpy as np

filepath='file path of lena.bmp'
img_ori=cv2.imread(filepath)
h,w=img_ori.shape[0:2]

#(a)upside-down
img_UD=np.zeros(img_ori.shape,np.uint8)
for i in range(h):
    img_UD[i,:,:]=img_ori[h-1-i,:,:]

#(b)right-side-left
img_RSL=np.zeros(img_ori.shape,np.uint8)
for i in range(w):
    img_RSL[:,i,:]=img_ori[:,w-1-i,:]

#(c)diagonally flip
img_DF=np.zeros(img_ori.shape,np.uint8)
for i in range(h):
    for j in range(w):
        img_DF[i,j,:]=img_ori[j,i,:]

#(d)rotate 45 degrees clockwise
M=cv2.getRotationMatrix2D((w//2,h//2),-45,1.0)
img_rot=cv2.warpAffine(img_ori,M,(w,h))

#(e) shrink in half
img_resize=cv2.resize(img_ori,(w//2,h//2))

#(f) binarize at 128 to get a binary image
_,img_binary=cv2.threshold(img_ori,128,255,cv2.THRESH_BINARY)

#show images
cv2.imshow("ori",img_ori)
cv2.imshow("a",img_UD)
cv2.imshow("b",img_RSL)
cv2.imshow("c",img_DF)
cv2.imshow("d",img_rot)
cv2.imshow("e",img_resize)
cv2.imshow("f",img_binary)
cv2.waitKey()
cv2.destroyAllWindows()

#image output
'''cv2.imwrite(outpath+'a.png',img_UD)
cv2.imwrite(outpath+'b.png',img_RSL)
cv2.imwrite(outpath+'c.png',img_DF)
cv2.imwrite(outpath+'d.png',img_rot)
cv2.imwrite(outpath+'e.png',img_resize)
cv2.imwrite(outpath+'f.png',img_binary)'''