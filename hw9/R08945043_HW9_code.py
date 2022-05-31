#CV HW9
#R08945043
import cv2
import numpy as np

def Robert_Operator(img,thr):
    img_result=np.full(img.shape[:2],255,np.uint8)
    img=cv2.copyMakeBorder(img,0,1,0,1,cv2.BORDER_REFLECT)
    for i in range(img_result.shape[0]):
        for j in range(img_result.shape[1]):
            r1=np.round(np.float(img[i+1,j+1])-np.float(img[i,j]))
            r2=np.round(np.float(img[i+1,j])-np.float(img[i,j+1]))
            value=(r1**2+r2**2)**0.5
            if value >= thr:
                img_result[i,j]=0
    return img_result

def Prewitt_Edge_Detector(img,thr):
    img_result=np.full(img.shape[:2],255,np.uint8)
    img=cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REFLECT)
    for i in range(1,img_result.shape[0]+1):
        for j in range(1,img_result.shape[1]+1):
            p1=np.round(np.sum((img[i+1,j-1:j+2]).astype(float)-(img[i-1,j-1:j+2]).astype(float)))
            p2=np.round(np.sum((img[i-1:i+2,j+1]).astype(float)-(img[i-1:i+2,j-1]).astype(float)))
            value=(p1**2+p2**2)**0.5
            if value >= thr:
                img_result[i-1,j-1]=0
    return img_result

def Sobel_Edge_Detector(img,thr):
    img_result=np.full(img.shape[:2],255,np.uint8)
    img=cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REFLECT)
    k=np.array([1,2,1])
    for i in range(1,img_result.shape[0]+1):
        for j in range(1,img_result.shape[1]+1):
            s1=np.round(np.sum(((img[i+1,j-1:j+2]).astype(float)-(img[i-1,j-1:j+2]).astype(float))*k))
            s2=np.round(np.sum(((img[i-1:i+2,j+1]).astype(float)-(img[i-1:i+2,j-1]).astype(float))*k))
            value=(s1**2+s2**2)**0.5
            if value >= thr:
                img_result[i-1,j-1]=0
    return img_result

def Frei_Chen_Gradient_Operator(img,thr):
    img_result=np.full(img.shape[:2],255,np.uint8)
    img=cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REFLECT)
    k=np.array([1,2**0.5,1])
    for i in range(1,img_result.shape[0]+1):
        for j in range(1,img_result.shape[1]+1):
            f1=np.round(np.sum(((img[i+1,j-1:j+2]).astype(float)-(img[i-1,j-1:j+2]).astype(float))*k))
            f2=np.round(np.sum(((img[i-1:i+2,j+1]).astype(float)-(img[i-1:i+2,j-1]).astype(float))*k))
            value=(f1**2+f2**2)**0.5
            if value >= thr:
                img_result[i-1,j-1]=0
    return img_result

def Kirsch_Compass_Operator(img,thr):
    img_result=np.full(img.shape[:2],255,np.uint8)
    img=cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REFLECT)
    k0=np.array([[-3,-3,5],[-3,0,5],[-3,-3,5]])
    k1=np.array([[-3,5,5],[-3,0,5],[-3,-3,-3]])
    k2=np.array([[5,5,5],[-3,0,-3],[-3,-3,-3]])
    k3=np.array([[5,5,-3],[5,0,-3],[-3,-3,-3]])
    k4=np.array([[5,-3,-3],[5,0,-3],[5,-3,-3]])
    k5=np.array([[-3,-3,-3],[5,0,-3],[5,5,-3]])
    k6=np.array([[-3,-3,-3],[-3,0,-3],[5,5,5]])
    k7=np.array([[-3,-3,-3],[-3,0,5],[-3,5,5]])
    for i in range(1,img_result.shape[0]+1):
        for j in range(1,img_result.shape[1]+1):
            m0=np.sum(((img[i-1:i+2,j-1:j+2]).astype(float)*k0))
            m1=np.sum(((img[i-1:i+2,j-1:j+2]).astype(float)*k1))
            m2=np.sum(((img[i-1:i+2,j-1:j+2]).astype(float)*k2))
            m3=np.sum(((img[i-1:i+2,j-1:j+2]).astype(float)*k3))
            m4=np.sum(((img[i-1:i+2,j-1:j+2]).astype(float)*k4))
            m5=np.sum(((img[i-1:i+2,j-1:j+2]).astype(float)*k5))
            m6=np.sum(((img[i-1:i+2,j-1:j+2]).astype(float)*k6))
            m7=np.sum(((img[i-1:i+2,j-1:j+2]).astype(float)*k7))
            value=max([m0,m1,m2,m3,m4,m5,m6,m7])
            if np.round(value) >= thr:
                img_result[i-1,j-1]=0
    return img_result

def Robinson_Compass_Operator(img,thr):
    img_result=np.full(img.shape[:2],255,np.uint8)
    img=cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REFLECT)
    k0=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    k1=np.array([[0,1,2],[-1,0,1],[-2,-1,0]])
    k2=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    k3=np.array([[2,1,0],[1,0,-1],[0,-1,-2]])
    for i in range(1,img_result.shape[0]+1):
        for j in range(1,img_result.shape[1]+1):
            r0=np.sum(((img[i-1:i+2,j-1:j+2]).astype(float)*k0))
            r1=np.sum(((img[i-1:i+2,j-1:j+2]).astype(float)*k1))
            r2=np.sum(((img[i-1:i+2,j-1:j+2]).astype(float)*k2))
            r3=np.sum(((img[i-1:i+2,j-1:j+2]).astype(float)*k3))
            value=max([r0,r1,r2,r3,-r0,-r1,-r2,-r3])
            if np.round(value) >= thr:
                img_result[i-1,j-1]=0
    return img_result

def Nevatia_Babu_Operator(img,thr):
    img_result=np.full(img.shape[:2],255,np.uint8)
    img=cv2.copyMakeBorder(img,2,2,2,2,cv2.BORDER_REFLECT)
    k0=np.array([[100]*5,[100]*5,[0]*5,[-100]*5,[-100]*5])
    k1=np.array([[100]*5,[100,100,100,78,-32],[100,92,0,-92,-100],[32,-78,-100,-100,-100],[-100]*5])
    k2=np.array([[100,100,100,32,-100],[100,100,92,-78,-100],[100,100,0,-100,-100],[100,78,-92,-100,-100],[100,-32,-100,-100,-100]])
    k3=np.array([[-100,-100,0,100,100]]*5)
    k4=np.array([[-100,32,100,100,100],[-100,-78,92,100,100],[-100,-100,0,100,100],[-100,-100,-92,78,100],[-100,-100,-100,-32,100]])
    k5=np.array([[100]*5,[-32,78,100,100,100],[-100,-92,0,92,100],[-100,-100,-100,-78,-32],[-100]*5])
    for i in range(2,img_result.shape[0]+2):
        for j in range(2,img_result.shape[1]+2):
            n0=np.sum(((img[i-2:i+3,j-2:j+3]).astype(float)*k0))
            n1=np.sum(((img[i-2:i+3,j-2:j+3]).astype(float)*k1))
            n2=np.sum(((img[i-2:i+3,j-2:j+3]).astype(float)*k2))
            n3=np.sum(((img[i-2:i+3,j-2:j+3]).astype(float)*k3))
            n4=np.sum(((img[i-2:i+3,j-2:j+3]).astype(float)*k4))
            n5=np.sum(((img[i-2:i+3,j-2:j+3]).astype(float)*k5))
            value=max([n0,n1,n2,n3,n4,n5])
            if np.round(value) >= thr:
                img_result[i-2,j-2]=0
    return img_result

#read file and img binary 
filepath='Absolute path of lena.bmp'
img_ori=cv2.imread(filepath,flags=cv2.IMREAD_GRAYSCALE)

#(a) Robert's Operator: 12
img_a=Robert_Operator(img_ori,12)
#(b) Prewitt's Edge Detector: 24
img_b=Prewitt_Edge_Detector(img_ori,24)
#(c) Sobel's Edge Detector: 38
img_c=Sobel_Edge_Detector(img_ori,38)
#(d) Frei and Chen's Gradient Operator: 30
img_d=Frei_Chen_Gradient_Operator(img_ori,30)
#(e) Kirsch's Compass Operator: 135
img_e=Kirsch_Compass_Operator(img_ori,135)
#(f) Robinson's Compass Operator: 43
img_f=Robinson_Compass_Operator(img_ori,43)
#(g) Nevatia-Babu 5x5 Operator: 12500
img_g=Nevatia_Babu_Operator(img_ori,12500)

'''
cv2.waitKey()
cv2.destroyAllWindows()
'''

