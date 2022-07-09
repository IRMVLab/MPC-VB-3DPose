import torch
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from common.ske_route import ske_route as ske

def mpjpe(predicted, target, masks = torch.ones(1).cuda()):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    if (masks == 1).all:
        return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))
    else:
        result = predicted - target
        result = result * masks
        return torch.mean(torch.norm(result, dim=len(target.shape)-1))  #用最后一维，计算3D空间范数，取平均值

def pck(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    dis = torch.norm(predicted - target, dim=len(target.shape)-1)
    #print(dis.size())
    t = torch.Tensor([0.15]).cuda()  # threshold
    out = (dis < t).float() * 1
    return out.sum()/14.0   #距离小于0.15的关节总数除以14

# dis = torch.randn(10).cuda()
# t = torch.Tensor([0.15]).cuda()  # threshold
# out = (dis < t).float() * 1
# print(out.cpu())


def auc(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    dis = torch.norm(predicted - target, dim=len(target.shape)-1)
    outall = 0
    #print(dis.size())
    for i in range(150):
        t = torch.Tensor([float(i)/1000]).cuda()  # threshold
        out = (dis < t).float() * 1
        outall+=out.sum()/14.0
    outall = outall/150
    return outall

    
def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))  #对关节偏差求加权平均

def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY        #平移

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY     #缩放

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)     #这个函数是奇异值分解，看上去好像是在求旋转近似
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    
    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1))
    
def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape
    
    norm_predicted = torch.mean(torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return mpjpe(scale * predicted, target)

def mean_velocity_error(predicted, target): #速度平均误差
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    
    velocity_predicted = np.diff(predicted, axis=0)
    velocity_target = np.diff(target, axis=0)
    
    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape)-1))


#16,15,  15,14,  13,12,  12,11,  10,9,  9,8,  11,8,  14,8,  8,7,  7,0,  3,2,  2,1,  6,5,  5,4,  1,0,  4,0

def project_mean_error(inputs_2d, boneindex, cam, projection_func, roots, **x):
    pred_3d = None
    if "x_rand" in x:
        input_2d = inputs_2d
        pred_3d = x["x_rand"]
    else:
        input_2d = inputs_2d[:,int((inputs_2d.size(1)-1)/2):int((inputs_2d.size(1)-1)/2+1),:,:2]
        roots = roots[:,int((inputs_2d.size(1)-1)/2):int((inputs_2d.size(1)-1)/2+1),:,:]
        bonelength = x["bonelength"]
        bonedirect = x["bonedirect"]    #bs bone 3
        bone_3d = bonedirect * bonelength.unsqueeze(2) #batchsize * (1) * bones * 3  #改成广播 ske : joint * bone * (1)
        """pred_3d = torch.zeros(bone_3d.size(0),bone_3d.size(1)+1,3).cuda()
        for joint in range(16,0,-1):                                                            #改成矩阵运算
            parent = joint
            for i, idx in zip(range(len(boneindex)), boneindex):
                if idx[0] == parent:
                    parent = idx[1]
                    pred_3d[:,joint] += bone_3d[:,i]
                    if parent == 0:
                        break
                else:
                    continue"""
        pred_3d = bone_3d[:,:16,:].unsqueeze(1) * ske.cuda().unsqueeze(2)
        pred_3d = pred_3d.sum(2)
        pred_3d = pred_3d.unsqueeze(1)  #bs ss=1 joint 3
    pred_3d += roots    #bs ss joint 3   bs ss 1 3
    #assert torch.sum(pred_3d == 0) == 0
    #pred_3d[pred_3d==0] = 0.001     #计算损失时去掉0值



    masks = pred_3d[:,:,:, 2] <= 0
    masks = 1 - torch.sum(masks, 2)
    masks = masks.unsqueeze(2).unsqueeze(3)
    pred_2d = projection_func(pred_3d, cam)             #batchsize,ss,bones,2
    #err_1 = mpjpe(get_bone2D(pred_2d, boneindex), get_bone2D(input_2d, boneindex), True)
    err_1 = mpjpe(pred_2d, input_2d, masks)

    err = pred_2d - input_2d

    err_2 = None
    if "x_rand" in x:
        pred_2d = pred_2d[:,1:]-pred_2d[:,:-1]
        input_2d = input_2d[:,1:]-input_2d[:,:-1]       #bs,ss-1,bones,2
        err_2 =mpjpe(pred_2d, input_2d, masks)
    else:
        err_2 = []
        i = 0
        for camera_1 in cam:    #cam bs 
            j = 0
            for camera_2 in cam[i+1:]:
                j = j + 1
                if (camera_1==camera_2).all:
                    err_temp = mpjpe(pred_2d[i] - pred_2d[i+j], input_2d[i] - input_2d[i+j], masks)
                    err_2.append( torch.mean(err_temp) )
                    break
                else:
                    continue
            i = i + 1
        err_2 = torch.tensor(err_2).cuda()
        err_2 = torch.mean(err_2)
    # err_2 = 0

    return err_1, err_2

def get_bone2D(inputs_2d,boneindex):
    bs = inputs_2d.size(0)
    ss = inputs_2d.size(1)
    inputs_2d = inputs_2d.view(-1,inputs_2d.size(2),inputs_2d.size(3))
    bone = []
    for index in boneindex:
        bone.append(inputs_2d[:,index[0]] - inputs_2d[:,index[1]])
    bone = torch.stack(bone,1)      #bs*ss  bone  2D
    bone = bone.view(bs,ss,bone.size(1),2)
    return bone
