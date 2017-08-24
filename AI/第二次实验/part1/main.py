# -*- coding: utf-8 -*-
import numpy as np
from sklearn import *
from scipy import sparse
###最基本的无优化特征提取,comment是从data文件读出的一条评论,返回稀疏矩阵存储的bag of words向量
def getFeature(comment,voc_num):
    row=np.zeros(len(comment))
    col=[]
    data=[]
    for w in comment:
        w_index,freq=w.split(':')
        w_index=int(w_index)
        freq=int(freq)
        col.append(w_index)
        data.append(freq)
    col=np.array(col)
    data=np.array(data)
    feature=sparse.coo_matrix((data,(row,col)),shape=(1,voc_num))
    return feature

##与上同理,但是对comments组成的list提取
def getFeatures(comments,voc_num):
    row=[]
    col=[]
    data=[]
    count=1
    for i in range(len(comments)):
        #print(count)
        count+=1
        for w in comments[i]:
            w_index,freq=w.split(':')
            row.append(comments.index(comments[i]))
            col.append(int(w_index))
            data.append(int(freq))
    features=sparse.coo_matrix((data,(row,col)),shape=(len(comments),voc_num))
    return features

#朴素贝叶斯分类器,采用多项式模型,分类是-1和1
def nBayesClassifier(traindata,trainlabel,testdata,testlabel,threshold=0.5):

    clf = naive_bayes.MultinomialNB()
    clf.fit(traindata,trainlabel)
    total=(len(testlabel))
    right=0.0
    proba=clf.predict_proba(testdata)
    ypred=np.ones(total)
    ypred=-ypred#初始化均为-1
    ypred[np.where(proba[:,1]>threshold)]=1 #概率大于阈值的改成1
    for i in xrange(total):
        if(testlabel[i]==ypred[i]):
            right+=1
    accuary=right/total
    ypred=tuple(ypred)
    return ypred,accuary

#最小二乘分类器,采用sklearn的RidgeClassifier
def lsClassifier(traindata,trainlabel,testdata,testlabel,Lambda=0.01):
    clf = linear_model.RidgeClassifier(alpha=Lambda)
    clf.fit(traindata,trainlabel)
    ypred=clf.predict(testdata)
    total=(len(testlabel))
    right=0.0
    for i in xrange(len(testlabel)):
        if(testlabel[i]==ypred[i]):
            right+=1
    accuary=right/total
    ypred=tuple(ypred)
    return ypred,accuary


#SVM,同样使用sklearn库
def softsvm(traindata,trainlabel,testdata,testlabel,sigma,C):
    if sigma==0:
        my_svm=svm.LinearSVC(C=C)
    else:
        my_svm=svm.SVC(C=C,kernel='rbf',gamma=(1.0/sigma**2) ) #gamma与sigma的换算关系
    my_svm.fit(traindata,trainlabel)
    ypred=my_svm.predict(testdata)
    total=len(testlabel)
    right=0.0
    for i in xrange(len(testlabel)):
        if(testlabel[i]==ypred[i]):
            right+=1
    accuary=right/total
    ypred=tuple(ypred)
    return  ypred,accuary

#5-fold交叉验证,利用sklearn库
def five_fold_cv(o_datas,datas,labels,bparms,lparms,Cs):
    labels=np.array(labels)
    o_datas=np.array(o_datas)
    kf=model_selection.KFold(n_splits=5,shuffle=True)
    bres=[]
    lres=[]
    sres=[]
    sum=0.0
    n=len(datas)
    for i in range(n):
        tmp=(datas-datas[i])**2
        sum+=np.sum(tmp)
    d=sum/(n**2) #计算SVM的d参数

    sigmas=[0.01*d,0.1*d,1.0*d,10.0*d,100.0*d]

    #注意:对于naive bayes,因为其多项式模型不能使用于小数,故直接将原始的bag of words数据输入,o_datas即原始数据
    for train_index,test_index in kf.split(o_datas):
        train_data,train_label=o_datas[train_index],labels[train_index]
        test_data,test_label=o_datas[test_index],labels[test_index]
        r=[]
        #在当前fold下对所有参数各计算一遍
        for s in bparms:
            acc=nBayesClassifier(train_data,train_label,test_data,test_label,threshold=s)[1]
            r.append(acc)
        bres.append(r)

    #对最小二乘和svm采用datas来计算
    for train_index,test_index in kf.split(datas):
        train_data,train_label=datas[train_index],labels[train_index]
        test_data,test_label=datas[test_index],labels[test_index]
        r=[]
        for l in lparms:
            acc=lsClassifier(train_data,train_label,test_data,test_label,Lambda=l)[1]
            r.append(acc)
        lres.append(r)
        r=[]
        for s in sigmas:
            for c in Cs:
                acc = softsvm(train_data, train_label, test_data, test_label, sigma=s, C=c)[1]
                r.append(acc)
        sres.append(r)
    bres=np.array(bres)
    lres=np.array(lres)
    sres=np.array(sres)

    return bres.T,lres.T,sres.T #因格式要求,最后返回转置

#从文件中获取特征和label
def load_data(fname,vocasize):
    f = open(fname)
    comments = []
    labels = []
    for l in f.readlines():
        line = l.strip().split()
        comment = line[1:]
        comments.append(comment)
        if (int(line[0]) >= 7):
            labels.append(1)
        else:
            labels.append(-1)
    features = getFeatures(comments, vocasize)
    return features,labels


dict={}
f=open('vocabulary')
vocasize=0
for l in f.readlines():
    dict[vocasize]=l.strip()
    vocasize+=1

datas,labels=load_data('dataset',vocasize)
o_datas=datas
print('start')



pca=decomposition.TruncatedSVD(n_components=100)
pca.fit(datas)
datas=pca.transform(datas)
#利用sklearn的降维模块的SVD方法进行降维,这里设定为降到100维

bayes_parms=[0.5,0.6,0.7,0.75,0.8,0.85,0.9]
ls_parms=[1e-4,0.01,0.1,0.5,1,5,10,100,1000,5000,10000]
Cs=[1,10,100,1000]
ratio=[0.01,0.1,1,10,100]
bres,lres,sres=five_fold_cv(o_datas.toarray(),datas,labels,bayes_parms,ls_parms,Cs)#进行交叉验证

bmeans=np.mean(bres,axis=1)
lmeans=np.mean(lres,axis=1)
smeans=np.mean(sres,axis=1)
#分别计算每组参数在5个fold下的平均正确率

b_best_index=np.argmax(bmeans)
l_best_index=np.argmax(lmeans)
s_best_index=np.argmax(smeans)
#取得最好参数对应的index

#输出结果
print "bayes res:",bres
print "bayes avg:",bmeans
print "ls res:",lres
print "ls avg:",lmeans
print "svm res:",sres
print "svm avg:",smeans
print "bayes best parm index:",b_best_index
print "bayes best parm:",bayes_parms[b_best_index]
print "ls best parm index",l_best_index
print "ls best parm:",ls_parms[l_best_index]

print "svm best parm index",s_best_index/len(Cs),s_best_index%len(Cs)
print "svm best parm:","%f*d" % ratio[s_best_index/len(Cs)],Cs[s_best_index%len(Cs)]







