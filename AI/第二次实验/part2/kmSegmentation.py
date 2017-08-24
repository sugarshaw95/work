# coding=utf-8
import numpy as np
from scipy.cluster.vq import *
from PIL import Image
import random
import math
def rand_cent(dataset,k):
    return random.sample(dataset,k)
#随机选择中心

def vec_ou_distance(v1,v2):
    return np.linalg.norm(v1-v2,axis=1)
#向量化计算距离,v1是二维array,每个元素分别计算与v2的距离

#这里的depth是设置了一个迭代深度,超过深度还没收敛就停止
def mykmeans(features,k=5,depth=50):
    features=np.array(features)
    centers=rand_cent(features,k)
    changed=True
    count=0#记录一下迭代次数
    while changed:
        count=count+1
        if(count>depth):
            print("reach depth!")
            break
        changed=False

        diss=[[] for i in range(k)]
        #距离,diss[i]表示和centers[i],即第i个中心点的距离
        for i in range(k):
            diss[i]=vec_ou_distance(features,centers[i])
        #向量化计算距离
        poss=np.argmin(diss,axis=0)
        #poss取得各个点距离最近的中心的index
        for i in xrange(k):
            new_c=np.mean(features[np.nonzero(poss[:]==i)[0]],axis=0)
            #重新计算中心
            if((new_c==centers[i]).all()):
                pass
            else:
                centers[i]=new_c
                changed=True
            #如果没变pass,否则说明还没收敛,更新
    return centers,poss



im=Image.open("Sea.jpg")
pix=im.load()
features=[]
m,n=im.size
for x in xrange(m):
    for y in xrange(n):
        features.append([float(i) for i in pix[x,y]])
features=np.array(features)

k=32
'''centers=kmeans(features,k)[0]
labels=vq(features,centers)[0]
int_centers=[]
for i in centers:
    int_centers.append([int(j) for j in i])
new_f=[]
for i in xrange(len(features)):
    new_f.append(int_centers[labels[i]])
for x in xrange(m):
    for y in xrange(n):
        im.putpixel([x,y],tuple(new_f[x*n+y]))
im.show()
im.save("standard_sea_%d.jpg" % k,"JPEG")'''

centers,labels=mykmeans(features,k)
int_centers=[]
for i in centers:
    for j in i:
        if(math.isnan(j)):
            print(i)
    int_centers.append([int(j) for j in i])
new_f=[]
for i in xrange(len(features)):
    #print(int_centers[labels[i]])
    new_f.append(int_centers[labels[i]])
for x in xrange(m):
    for y in xrange(n):
        im.putpixel([x,y],tuple(new_f[x*n+y]))
im.show()
im.save("my_sea%d.jpg" % k,"JPEG")