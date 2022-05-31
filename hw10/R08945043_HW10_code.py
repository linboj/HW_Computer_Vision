#CV HW10
#R08945043
import cv2
import numpy as np

def Laplacian_operator (img,kernel,thr):
    img_result=np.full(img.shape[:2],0,np.int8)
    img=cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REFLECT)
    for i in range(1,img_result.shape[0]+1):
        for j in range(1,img_result.shape[1]+1):
            value=np.sum(((img[i-1:i+2,j-1:j+2]).astype(float)*kernel))
            if np.round(value) >= thr:
                img_result[i-1,j-1]=1
            elif np.round(value) <= -thr:
                img_result[i-1,j-1]=-1
    return img_result

def Minvar_Laplacian(img,thr):
    img_result=np.full(img.shape[:2],0,np.int8)
    img=cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REFLECT)
    kernel=np.array([[2, -1, 2], [-1, -4, -1], [2, -1, 2]])/3
    for i in range(1,img_result.shape[0]+1):
        for j in range(1,img_result.shape[1]+1):
            value=np.sum(((img[i-1:i+2,j-1:j+2]).astype(float)*kernel))
            if np.round(value) >= thr:
                img_result[i-1,j-1]=1
            elif np.round(value) <= -thr:
                img_result[i-1,j-1]=-1
    return img_result

def LOG(img,thr):
    img_result=np.full(img.shape[:2],0,np.int8)
    img=cv2.copyMakeBorder(img,5,5,5,5,cv2.BORDER_REFLECT)
    kernel=np.array([
        [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0],
        [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
        [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
        [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
        [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
        [-2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2],
        [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
        [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
        [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
        [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
        [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0]
        ])
    for i in range(5,img_result.shape[0]+5):
        for j in range(5,img_result.shape[1]+5):
            value=np.sum(((img[i-5:i+6,j-5:j+6]).astype(float)*kernel))
            if np.round(value) >= thr:
                img_result[i-5,j-5]=1
            elif np.round(value) <= -thr:
                img_result[i-5,j-5]=-1
    return img_result

def DOG(img,thr):
    img_result=np.full(img.shape[:2],0,np.int8)
    img=cv2.copyMakeBorder(img,5,5,5,5,cv2.BORDER_REFLECT)
    kernel=np.array([
        [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],
        [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
        [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
        [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
        [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
        [-8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8],
        [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
        [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
        [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
        [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
        [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],
        ])
    for i in range(5,img_result.shape[0]+5):
        for j in range(5,img_result.shape[1]+5):
            value=np.sum(((img[i-5:i+6,j-5:j+6]).astype(float)*kernel))
            if np.round(value) >= thr:
                img_result[i-5,j-5]=1
            elif np.round(value) <= -thr:
                img_result[i-5,j-5]=-1
    return img_result

def zero_crossing(img):
    img_result=np.full(img.shape[:2],255,np.uint8)
    img=cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REFLECT)
    for i in range(1,img_result.shape[0]+1):
        for j in range(1,img_result.shape[1]+1):
            if img[i,j] == 1:
                if np.isin(img[i-1:i+2,j-1:j+2],-1).any():
                    img_result[i-1,j-1]=0
    return img_result


#read file and img binary 
filepath='Absolute path of lena.bmp'
img_ori=cv2.imread(filepath,flags=cv2.IMREAD_GRAYSCALE)

#Laplace Mask1 (0, 1, 0, 1, -4, 1, 0, 1, 0): 15
img_laplace1=zero_crossing(Laplacian_operator(img_ori,np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),15))
#Laplace Mask2 (1, 1, 1, 1, -8, 1, 1, 1, 1): 15
img_laplace2=zero_crossing(Laplacian_operator(img_ori,np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])/3,15))
#Minimum variance Laplacian: 20
img_mvl=zero_crossing(Minvar_Laplacian(img_ori,20))
#Laplace of Gaussian: 3000
img_log=zero_crossing(LOG(img_ori,3000))
#Difference of Gaussian: 1
img_dog=zero_crossing(DOG(img_ori,1))

'''
cv2.waitKey()
cv2.destroyAllWindows()
'''