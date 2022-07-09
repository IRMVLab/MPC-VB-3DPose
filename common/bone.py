import torch
import numpy as np

def getbonejs(seq, boneindex):  #计算Joint Shift
    bs = seq.size(0) 
    ss = seq.size(1)
    seq = seq.view(-1,seq.size(2),seq.size(3))  #把前2维展开成1维
    bone = []
    for i in range(17):
        #for j in range(i+1,17) is also OK
        for j in range(i,17):                   #j>=i
            if not ([i,j] in boneindex or [j,i] in boneindex):  #不相邻的关节
                bone.append(seq[:,j] - seq[:,i])
    bone = torch.stack(bone,1)
    bone = bone.view(bs,ss, bone.size(1),3)
    return bone


def getbonelength(seq, boneindex):  #计算骨长
    bs = seq.size(0)
    ss = seq.size(1)
    seq = seq.view(-1,seq.size(2),seq.size(3))
    bone = []
    for index in boneindex:
        bone.append(seq[:,index[0]] - seq[:,index[1]])
    bone = torch.stack(bone,1)
    bone = torch.pow(torch.pow(bone,2).sum(2),0.5)
    bone = bone.view(bs,ss, bone.size(1))
    return bone


def getbonedirect(seq, boneindex): #提取骨骼方向
    bs = seq.size(0)
    ss = seq.size(1)
    seq = seq.view(-1,seq.size(2),seq.size(3))
    bone = []
    for index in boneindex:
        bone.append(seq[:,index[0]] - seq[:,index[1]])  #计算骨骼向量
    bonedirect = torch.stack(bone,1)
    bonesum = torch.pow(torch.pow(bonedirect,2).sum(2), 0.5).unsqueeze(2)   #计算骨骼长度
    bonedirect = bonedirect/bonesum #单位向量表示骨骼方向
    bonedirect = bonedirect.view(bs,-1,3)
    return bonedirect
