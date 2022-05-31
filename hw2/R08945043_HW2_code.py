#CV HW2 
#R08945043
import cv2
import numpy as np
import matplotlib.pyplot as plt 

#filepath='file path of lena.bmp'
filepath='D:\\report\\course_data\\109-1\\CV\\homework\\lena.bmp'
img_ori=cv2.imread(filepath,flags=cv2.IMREAD_GRAYSCALE)
h,w=img_ori.shape[0:2]

#(a)a binary image (threshold at 128) and (b) count the number of pixel with different intensity in the image
img_binary=img_ori.copy()
stat_data=np.zeros((2,256))
stat_data[0,:]=[v for v in range(256)]
for i in range(h):
    for j in range(w):
        if img_ori[i,j]<128:
            img_binary[i,j]=0
        else:
            img_binary[i,j]=255
        stat_data[1,img_ori[i,j]]=stat_data[1,img_ori[i,j]]+1


#(c) connected components(regions with + at centroid, bounding box)
img_label_tmp=np.zeros(img_binary.shape)
num_label=0
for i in range(img_binary.shape[0]):
    for j in range(img_binary.shape[1]):
        if img_binary[i,j]==255:
            set_tmp=set([img_label_tmp[index[0],index[1]] \
                for index in ((i-1,j-1),(i-1,j),(i-1,j+1),(i,j-1)) \
                    if index[0]>-1 and index[1]>-1 and index[0]<img_binary.shape[0] and index[1]<img_binary.shape[1]  ])
            set_tmp.discard(0)
            if len(set_tmp)==0:
                num_label+=1
                img_label_tmp[i,j]=num_label
            elif len(set_tmp)==1:
                img_label_tmp[i,j]=min(set_tmp)
            else:
                min_group=min(set_tmp)
                img_label_tmp[i,j]=min_group
                set_tmp.remove(min_group)
                for group in set_tmp:
                    img_label_tmp[np.where(img_label_tmp==group)]=min_group
#draw centroid and bounding box
img_draw=cv2.cvtColor(img_binary,cv2.COLOR_GRAY2BGR)
setnum=set(img_label_tmp.flatten())
setnum.discard(0)
for label in setnum:
    pixelpos=np.where(img_label_tmp==label)
    if len(pixelpos[0])>=500:
        cen_r=int(round(sum(pixelpos[0])/len(pixelpos[0])))
        cen_c=int(round(sum(pixelpos[1])/len(pixelpos[1])))
        img_draw=cv2.circle(img_draw,(cen_c,cen_r),3,(0,0,255),-1)
        img_draw=cv2.rectangle(img_draw,(min(pixelpos[1]),min(pixelpos[0])),(max(pixelpos[1]),max(pixelpos[0])),(255,0,0),2)

#output histogram
plt.bar(stat_data[0,:],stat_data[1,:],width=1)
plt.xlabel('Intensity')
plt.ylabel('Number')
plt.show()
#show image
cv2.imshow("ori",img_ori)
cv2.imshow("binary",img_binary)
cv2.imshow("draw",img_draw)
cv2.waitKey()
cv2.destroyAllWindows()

#image output
#outpath='outpath'
#cv2.imwrite(outpath+'binary.png',img_binary)
#cv2.imwrite(outpath+'draw.png',img_draw)