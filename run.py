import numpy as np

from common.arguments import parse_args
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno

from common.camera import *
from common.model import *
from common.loss import *
from common.bone import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from time import time
from common.utils import deterministic_random
import random
import os


args = parse_args()
print(args)



try:            #创建目录
    # Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

print('Loading dataset...')
dataset_path = 'data/data_3d_' + args.dataset + '.npz'  #读3d数据
# temporally only supports human3.6m dataset
if args.dataset == 'h36m':                          #准备数据集对象 dataset dataset[subject][action][position/camera] 
    from common.h36m_dataset import Human36mDataset
    dataset = Human36mDataset(dataset_path)
else:                                               #报错
    raise KeyError('Invalid dataset')

print('Preparing data...')
for subject in dataset.subjects():                  #迭代subject
    for action in dataset[subject].keys():          #迭代该subject的action
        anim = dataset[subject][action]             #引用
        positions_3d = []                           #
        for cam in anim['cameras']:                 #对于每个相机
            pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])     #根据数据中的相机参数和全局3d位置，计算相机坐标系下的3d位置
            pos_3d[:, 1:] -= pos_3d[:, :1] # Remove global offset, but keep trajectory in first position#从第2列开始，每一列减去第一列，消除全局偏置            ???
            positions_3d.append(pos_3d)             #连进去
        anim['positions_3d'] = positions_3d         #在每个action字典里附加position_3d词条，实际上是一个列表，按位置与相机列表对应

print('Loading 2D detections...')                       #读2d数据
keypoints = np.load('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz',allow_pickle=True)
keypoints_metadata = keypoints['metadata'].item()       #元数据     接下来是把2d data中除position以外的信息用别的东西装起来
keypoints_symmetry = keypoints['metadata'].item()['keypoints_symmetry'] #对称
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])  #把对称的信息赋给左右两列关节
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right()) #把左右关节的序号赋值给jl，jr
keypoints = keypoints['positions_2d'].item()        #keypoints只保留positions_2d

for subject in dataset.subjects():          #检查2d与3d是否对应
    assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)     #检查3d的subject在2d中有没有，否则报错
    for action in dataset[subject].keys():
        assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)    #检查action存在
        for cam_idx in range(len(keypoints[subject][action])):  #对相机编号迭代
            
            # We check for >= instead of == because some videos in H3.6M contain extra frames
            mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]   #这个镜头下有这么多帧
            assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length         #2d的帧数不比3d集少
            
            if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                # Shorten sequence
                keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]    #多余帧剪掉

        assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d']) #检查一下


print('Loading 2D keypoint visibility score...')        #加载可见性评分
#the visibility scores are predicted by Alphapose
with open('data/score.pkl', 'rb') as f:
    score = pickle.load(f)

#n joints, (n-1) bones, 2(n-1) indexs
print('Loading bone index...')
# for i, keys in zip(range(len(score.keys())+1), score.keys()):
#     if i%4 == 1:
#         print(keys)
# assert 2<1, 'stop'
boneindextemp = args.boneindex.split(',')
boneindex = []                          #准备骨骼与关节的对应表
for i in range(0,len(boneindextemp),2):
    boneindex.append([int(boneindextemp[i]), int(boneindextemp[i+1])])
for i in range(1,17):
    if i == 1 or i == 4 or i == 7:
        continue
    boneindex.append([0,i])
joints = [3,6,10,13,16]
for i in range(len(joints)):
    for j in range(i+1,len(joints)):
        boneindex.append([joints[i],joints[j]])
        
for subject in keypoints.keys():    #对2d关键点的subject迭代，归一2d位置相片的宽度，把可见性评分连接在2dpose的最后一维
    for action in keypoints[subject]:   #对该subj的action迭代
        for cam_idx, kps in enumerate(keypoints[subject][action]):  #对相机列表迭代，camidx是相机号，kps是2d位置
            # Normalize camera frame
            cam = dataset.cameras()[subject][cam_idx]   #读出相机参数
            kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])       #把相片宽度归一，保留长宽比
            kps = kps[:,:,:2]       #只保留前2列
            name = subject+'_'+action+'.'+str(cam_idx)
            # score.pkl includes the 2D keypoint visibility scores for S1, S5, S6, S7, S8, S9, S11 of H3.6M
            if subject in ["S1","S5","S6","S7","S8","S9","S11"]:                #提取可见性评分，连接在kps后面（最后一维）
              s = score[name][:len(kps)]
              # occationally the visibility score loses one frame of a video, just pad it
              if len(score[name])< len(kps):    #如果长度不够
                s2 = [s]
                for i in range(len(kps)-len(score[name])):
                      s2.append(s[-1:])
                s = np.concatenate(s2,0)        #补0
              s = np.reshape(s, (len(s),np.shape(kps)[1],1))
              # concatenate the 2D keypoints and visibility scores, N*K*3
              kps = np.concatenate((kps,s),2)
            keypoints[subject][action][cam_idx] = kps



            
subjects_train = args.subjects_train.split(',')
subjects_test = args.subjects_test.split(',')

            
def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []      #准备要输出的列表
    for subject in subjects:    #对subject迭代
        for action in keypoints[subject].keys():    #对action迭代
            if action_filter is not None:   #该列表可以给出想选择的action，忽略其他action
                found = False
                for a in action_filter:     #找以a开头的action，如果现在这个action以a开头则找到，否则换一个a，直到找完所有a
                    if action.startswith(a):
                        found = True
                        break
                if not found:               #没找到就换一个action
                    continue
                
            poses_2d = keypoints[subject][action]   #读出2d数据集中对应的2d位置
            for i in range(len(poses_2d)): # Iterate across cameras
                out_poses_2d.append(poses_2d[i])    #把2d数据集里的数据挨个拷进out里，其中每个元素是一个相机对应的一串位置+可见性
                
            if subject in dataset.cameras():        #如果有这个subject对应的相机信息
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])  #把相机的intrinsic参数放进输出里
                
            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:   #把选择的3d位置送进输出里去
                poses_3d = dataset[subject][action]['positions_3d'] #读3d位置
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)): # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])
    
    if len(out_camera_params) == 0:     #如果是空的就置none
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None
    
    stride = args.downsample            #下采样
    if subset < 1:  #随机选择占比subset的子集，
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i])//stride * subset)*stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start+n_frames:stride]  #等时间间隔采样部分帧
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start+n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]     #隔stride个取一帧
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]
    

    return out_camera_params, out_poses_3d, out_poses_2d


action_filter = None if args.actions == '*' else args.actions.split(',')    #读设定中的action_filter
if action_filter is not None:
    print('Selected actions:', action_filter)
    
cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, action_filter)    #根据action_filter取样

filter_widths = [int(x) for x in args.architecture.split(',')]  #各层的卷积核大小
if not args.disable_optimizations and not args.dense and args.stride == 1:
    # Use optimized model for single-frame predictions
    model_pos_train = TemporalModelOptimized1f(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1]-1, poses_valid[0].shape[-2], boneindex, args.temperature,
                                filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels)
else:
    # When incompatible settings are detected (stride > 1, dense filters, or disabled optimization) fall back to normal model
    model_pos_train = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1]-1, poses_valid[0].shape[-2], boneindex, args.temperature, args.randnumtest,
                                filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels,
                                dense=args.dense)
    
model_pos = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1]-1, poses_valid[0].shape[-2], boneindex, args.temperature, args.randnumtest,
                            filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels,
                            dense=args.dense)

if len(filter_widths) < 3:
    model_pos_train=nn.DataParallel(model_pos_train,device_ids=[0]) # multi-GPU #改动处
    model_pos=nn.DataParallel(model_pos,device_ids=[0]) # multi-GPU
else:
    model_pos_train=nn.DataParallel(model_pos_train,device_ids=[0,1,2]) # multi-GPU #改动处
    model_pos=nn.DataParallel(model_pos,device_ids=[0,1,2]) # multi-GPU


receptive_field = model_pos.module.receptive_field()
print('INFO: Receptive field: {} frames'.format(receptive_field))
pad = (receptive_field - 1) // 2 # Padding on each side
if args.causal:
    print('INFO: Using causal convolutions')
    causal_shift = pad
else:
    causal_shift = 0

model_params = 0
for parameter in model_pos.parameters():
    model_params += parameter.numel()
print('INFO: Trainable parameter count:', model_params)

if torch.cuda.is_available():
    model_pos = model_pos.cuda()
    model_pos_train = model_pos_train.cuda()
    
if args.resume or args.evaluate:
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))
    model_pos_train.load_state_dict(checkpoint['model_pos'])
    model_pos.load_state_dict(checkpoint['model_pos'])

test_generator = UnchunkedGenerator(cameras_valid, poses_valid, poses_valid_2d,
                                    pad=pad, causal_shift=causal_shift, augment=False,
                                    kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
print('INFO: Testing on {} frames'.format(test_generator.num_frames()))


def evaluate(test_generator, action=None, return_predictions=False):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0
    with torch.no_grad():
        model_pos.eval()
        N = 0
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()
            
            # Positional model
            if batch is not None:
              inputs_3d = torch.from_numpy(batch.astype('float32'))
              if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
              inputs_3d[:, :, 0] = 0    

            inputs_2d = inputs_2d.view(inputs_2d.size(0), inputs_2d.size(1), -1, 3)
            predicted_3d_pos = model_pos(inputs_2d)
            
            if test_generator.augment_enabled():
              if batch is not None:
                inputs_3d = inputs_3d[:1]
            if test_generator.augment_enabled():
                # Undo flipping and take average with non-flipped version
                predicted_3d_pos[1, :, :, 0] *= -1
                predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)
                
            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()
            pad = int((inputs_3d.size(1) - predicted_3d_pos.size(1))/2)
            inputs_3d = inputs_3d[:,pad:inputs_3d.size(1)-pad]
            error = mpjpe(predicted_3d_pos, inputs_3d)

            epoch_loss_3d_pos_scale += inputs_3d.shape[0]*inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos, inputs_3d).item()

            epoch_loss_3d_pos += inputs_3d.shape[0]*inputs_3d.shape[1] * error.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]
            
            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

            epoch_loss_3d_pos_procrustes += inputs_3d.shape[0]*inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)

            # Compute velocity error
            epoch_loss_3d_vel += inputs_3d.shape[0]*inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)
            
    if action is None:
        print('----------')
    else:
        print('----'+action+'----')
    e1 = (epoch_loss_3d_pos / N)*1000
    e2 = (epoch_loss_3d_pos_procrustes / N)*1000
    e3 = (epoch_loss_3d_pos_scale / N)*1000
    ev = (epoch_loss_3d_vel / N)*1000
    print('Test time augmentation:', test_generator.augment_enabled())
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
    print('Velocity Error (MPJVE):', ev, 'mm')
    print('----------')

    return e1, e2, e3, ev

err_t_save = np.inf

if not args.evaluate:
    cameras_train, poses_train, poses_train_2d = fetch(subjects_train, action_filter, subset=args.subset)

    lr = args.learning_rate
    
    optimizer = optim.Adam(model_pos_train.parameters(), lr=lr, amsgrad=True)
        
    lr_decay = args.lr_decay

    losses_3d_train = []
    losses_3d_train_proj = []   
    losses_3d_train_len = []
    losses_3d_train_rand = []
    losses_3d_train_dire = []   
    losses_3d_train_eval = []
    losses_3d_valid = []

    epoch = 0
    initial_momentum = 0.1
    final_momentum = 0.001
    
    
    train_generator = ChunkedGenerator(args.batch_size//args.stride, cameras_train, poses_train, poses_train_2d, args.randnum, boneindex, args.augdegree, args.stride,
                                       pad=pad, causal_shift=causal_shift, shuffle=True, augment=args.data_augmentation,
                                       kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    train_generator_eval = UnchunkedGenerator(cameras_train, poses_train, poses_train_2d,
                                              pad=pad, causal_shift=causal_shift, augment=False)
    print('INFO: Training on {} frames'.format(train_generator_eval.num_frames()))

    if args.resume:
        epoch = checkpoint['epoch']
        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            train_generator.set_random_state(checkpoint['random_state'])
        else:
            print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
        
        lr = checkpoint['lr']
            
    print('** Note: reported losses are averaged over all frames and test-time augmentation is not used here.')
    print('** The final evaluation will be carried out after the last training epoch.')
    
    # Pos model only
    while epoch < args.epochs:
        start_time = time()
        epoch_loss_3d_train = 0
        epoch_loss_3d_train_proj = 0   
        epoch_loss_3d_train_len = 0
        epoch_loss_3d_train_rand = 0
        epoch_loss_3d_train_dire = 0
        batch = 0                      
        N = 0
        model_pos_train.train()
        #stoppoint = 0
        for cam, batch_3d, batch_2d, batch_3d_rand, batch_2d_rand, batch_3d_randaugtraj, bonelennew, batch_3d_randauggt, in train_generator.next_epoch():
            #assert (batch_3d[:,:,:,2]>0).all
            cam = torch.from_numpy(cam.astype('float32'))
            inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()
                cam = cam.cuda()
            inputs_3d_root = inputs_3d[:, :, :1].clone()
            inputs_3d[:, :, 0] = 0
            

            inputs_3d_rand = torch.from_numpy(batch_3d_rand.astype('float32'))
            inputs_2d_rand = torch.from_numpy(batch_2d_rand.astype('float32'))
            if torch.cuda.is_available():
                inputs_3d_rand = inputs_3d_rand.cuda()
                inputs_2d_rand = inputs_2d_rand.cuda()
            inputs_3d_rand_root = inputs_3d_rand[:, :, :1].clone()
            inputs_3d_rand[:, :, 0] = 0     #根节点=0

            inputs_3d_randaugtraj = torch.from_numpy(batch_3d_randaugtraj.astype('float32'))
            inputs_3d_lengthnew = torch.from_numpy(bonelennew.astype('float32'))
            inputs_3d_randauggt = torch.from_numpy(batch_3d_randauggt.astype('float32'))
            if torch.cuda.is_available():
                inputs_3d_randaugtraj = inputs_3d_randaugtraj.cuda()
                inputs_3d_lengthnew = inputs_3d_lengthnew.cuda()
                inputs_3d_randauggt = inputs_3d_randauggt.cuda()
                
            optimizer.zero_grad()
                
            projection_func = project_to_2d_linear if args.linear_projection else project_to_2d
            #reconstruct the augmented 2D input 
            inputs_2d_randaug = projection_func(inputs_3d_randaugtraj, cam)
            #bone length prediction network doesn't need the visibility score feature as input
            inputs_2d_rand = inputs_2d_rand[:,:,:,:2].contiguous()
            #get the grounth-truth bone length (b * nb), directly average the frames since bone length is consistent across frame
            inputs_3d_length = getbonelength(inputs_3d, boneindex[:16]).mean(1)
            inputs_vir_length = getbonelength(inputs_3d, boneindex[16:])[:,int((inputs_3d.size(1)-1)/2):int((inputs_3d.size(1)-1)/2)+1]
            predicted_3d_pos, bonelength, predicted_3d_rand, bonelengthaug, predicted_3d_randaug, bonedirect_2, bonedirect_1, predicted_js_2, predicted_js_1, predicted_3d_virtual = model_pos_train(inputs_2d, inputs_2d_rand, inputs_2d_randaug)        #送进去训练

            bonedirect_2 = bonedirect_2.view(bonedirect_2.size(0),-1,3)
            bonedirect_1 = bonedirect_1.view(bonedirect_1.size(0),-1,3)
            #get the gt 3D joint locations of the current frame 
            inputs_3d = inputs_3d[:,int((inputs_3d.size(1)-1)/2):int((inputs_3d.size(1)-1)/2)+1]
            #compute the bone direction loss of each sub-network (totally 2) of the bone direction prediction network
            inputs_3d_direct = getbonedirect(inputs_3d, boneindex)
            loss_direct = args.wd*torch.pow(inputs_3d_direct - bonedirect_2,2).sum(2).mean() + args.wd*args.snd*torch.pow(inputs_3d_direct - bonedirect_1,2).sum(2).mean()
            #compute the mpjpe loss of the final 3D joint location prediction
            loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
            #compute the relative joint shifts loss
            inputs_3d_js = getbonejs(inputs_3d, boneindex[:16])
            loss_js = args.wjs*mpjpe(predicted_js_2, inputs_3d_js) + args.wjs*args.snd*mpjpe(predicted_js_1, inputs_3d_js)
                
            #randomly sample one frame to compute the mpjpe loss of the bone length prediction network
            randnum = random.randint(0,predicted_3d_rand.size(1)-1)
            #compute the mpjpe loss of the bone length prediction network (original + augmented, you can also remove the mpjpe loss of the augmented data as described in the paper)
            # 只用一帧进行计算损失
            loss_3d_pos_rand = mpjpe(predicted_3d_rand[:,randnum:randnum+1], inputs_3d_rand[:,randnum:randnum+1])
            loss_3d_pos_randaug = mpjpe(predicted_3d_randaug[:,randnum:randnum+1], inputs_3d_randauggt[:,randnum:randnum+1])
            loss_3d_pos_vir = mpjpe(predicted_3d_virtual, inputs_3d)
            #compute bone length loss (original + augmented)
            loss_length = args.wl*torch.pow(inputs_3d_length - bonelength[:,:16],2).mean()
            loss_lengthaug = args.wl*torch.pow(inputs_3d_lengthnew[:,:16] - bonelengthaug,2).mean()
            loss_lengthvir = args.wl*torch.pow(inputs_vir_length - bonelength[:,16:],2).mean()
            #total loss of the bone length prediction network
            loss_len = loss_3d_pos_rand + loss_3d_pos_randaug + loss_3d_pos_vir + loss_length + loss_lengthaug + loss_lengthvir
            loss_proj_length_1, loss_proj_length_2 = project_mean_error(inputs_2d_rand, boneindex, cam, project_to_2d, inputs_3d_rand_root, x_rand = predicted_3d_rand)
            loss_proj_lengthaug_1, loss_proj_lengthaug_2 = project_mean_error(inputs_2d_randaug, boneindex, cam, project_to_2d, inputs_3d_randaugtraj[:,:,:1,:], x_rand = predicted_3d_randaug)
            loss_proj_dire1_1, loss_proj_dire1_2 = project_mean_error(inputs_2d, boneindex, cam, project_to_2d, inputs_3d_root, bonelength = bonelength.detach(), bonedirect = bonedirect_1)
            loss_proj_dire2_1, loss_proj_dire2_2 = project_mean_error(inputs_2d, boneindex, cam, project_to_2d, inputs_3d_root, bonelength = bonelength.detach(), bonedirect = bonedirect_2)
            loss_proj  = loss_proj_dire2_1 * args.w_projd1  + loss_proj_dire1_1 * args.snd * args.w_projd1
            loss_proj += loss_proj_dire2_2 * args.w_projd2  + loss_proj_dire1_2 * args.snd * args.w_projd2
            loss_proj += loss_proj_length_2 * args.w_projl2
            loss_proj += loss_proj_length_1 * args.w_projl1
            loss_proj += loss_proj_lengthaug_2 * args.w_projl2
            loss_proj += loss_proj_lengthaug_1 * args.w_projl1

            epoch_loss_3d_train += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_3d_pos.item()
            epoch_loss_3d_train_proj += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_proj.item()   
            epoch_loss_3d_train_len += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_len.item()
            epoch_loss_3d_train_rand += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_3d_pos_rand.item()
            epoch_loss_3d_train_dire += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_direct.item() 
            N += inputs_3d.shape[0]*inputs_3d.shape[1]

            assert epoch_loss_3d_train != None
            assert epoch_loss_3d_train_proj != None
            assert epoch_loss_3d_train_len != None
            assert epoch_loss_3d_train_dire != None

            #total loss of the model
            loss_total = loss_3d_pos + loss_len + loss_direct + loss_js + loss_proj
            loss_total.backward()

            optimizer.step()
        
        losses_3d_train.append(epoch_loss_3d_train / N)
        losses_3d_train_proj.append(epoch_loss_3d_train_proj / N)
        losses_3d_train_len.append(epoch_loss_3d_train_len / N)
        losses_3d_train_rand.append(epoch_loss_3d_train_rand / N)
        losses_3d_train_dire.append(epoch_loss_3d_train_dire / N)

        # End-of-epoch evaluation
        if (epoch+1) % args.eva_frequency == 0:
            with torch.no_grad():
                model_pos.load_state_dict(model_pos_train.state_dict())
                model_pos.eval()

                epoch_loss_3d_valid = 0
                N = 0
                if not args.no_eval:
                    # Evaluate on test set
                    for cam, batch, batch_2d in test_generator.next_epoch():
                        inputs_3d = torch.from_numpy(batch.astype('float32'))
                        inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                        if torch.cuda.is_available():
                            inputs_3d = inputs_3d.cuda()
                            inputs_2d = inputs_2d.cuda()
                        inputs_traj = inputs_3d[:, :, :1].clone()
                        inputs_3d[:, :, 0] = 0
                        # Predict 3D poses
                        predicted_3d_pos = model_pos(inputs_2d)
                        pad = int((inputs_3d.size(1) - predicted_3d_pos.size(1))/2)
                        inputs_3d = inputs_3d[:,pad:inputs_3d.size(1)-pad]
                        loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                        epoch_loss_3d_valid += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_3d_pos.item()
                        N += inputs_3d.shape[0]*inputs_3d.shape[1]

                    losses_3d_valid.append(epoch_loss_3d_valid / N)

        elapsed = (time() - start_time)/60

        if args.no_eval or (epoch+1) % args.eva_frequency != 0:
            print('[%d] time %.2f lr %f 3d_train %f proj_loss %f len_loss %f rand_loss %f dire_loss %f ' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    losses_3d_train[-1] * 1000,
                    losses_3d_train_proj[-1] * 1000,
                    losses_3d_train_len[-1] * 1000,
                    losses_3d_train_rand[-1] * 1000,
                    losses_3d_train_dire[-1] * 1000))
        else:
            print('[%d] time %.2f lr %f 3d_train %f 3d_valid %f proj_loss %f len_loss %f rand_loss %f dire_loss %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    losses_3d_train[-1] * 1000,
                    losses_3d_valid[-1]  *1000,
                    losses_3d_train_proj[-1] * 1000,
                    losses_3d_train_len[-1] * 1000,
                    losses_3d_train_rand[-1] * 1000,
                    losses_3d_train_dire[-1] * 1000))
            torch.cuda.empty_cache()
        
        # Decay learning rate exponentially     
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
        epoch += 1
        
        # Decay BatchNorm momentum
        momentum = initial_momentum * np.exp(-epoch/args.epochs * np.log(initial_momentum/final_momentum))
        model_pos_train.module.set_bn_momentum(momentum)
            
        # Save checkpoint if necessary
        if epoch % 5 == 0 and epoch >= 20 :
            if args.render:
                print('Rendering...')
                input_keypoints = keypoints[args.viz_subject][args.viz_action][args.viz_camera].copy()
                ground_truth = None
                if args.viz_subject in dataset.subjects() and args.viz_action in dataset[args.viz_subject]:
                    if 'positions_3d' in dataset[args.viz_subject][args.viz_action]:
                        ground_truth = dataset[args.viz_subject][args.viz_action]['positions_3d'][args.viz_camera].copy()
                if ground_truth is None:
                    print('INFO: this action is unlabeled. Ground truth will not be rendered.')
                    
                gen = UnchunkedGenerator(None, None, [input_keypoints],
                                        pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                                        kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
                prediction = evaluate(gen, return_predictions=True)        
                if ground_truth is not None:
                    # Reapply trajectory
                    trajectory = ground_truth[:, :1]
                    ground_truth[:, 1:] += trajectory
                    prediction += trajectory
                
                # Invert camera transformation
                cam = dataset.cameras()[args.viz_subject][args.viz_camera]
                if ground_truth is not None:
                    prediction = camera_to_world(prediction, R=cam['orientation'], t=cam['translation'])
                    ground_truth = camera_to_world(ground_truth, R=cam['orientation'], t=cam['translation'])
                else:
                    # If the ground truth is not available, take the camera extrinsic params from a random subject.
                    # They are almost the same, and anyway, we only need this for visualization purposes.
                    for subject in dataset.cameras():
                        if 'orientation' in dataset.cameras()[subject][args.viz_camera]:
                            rot = dataset.cameras()[subject][args.viz_camera]['orientation']
                            break
                    prediction = camera_to_world(prediction, R=rot, t=0)
                    # We don't have the trajectory, but at least we can rebase the height
                    prediction[:, :, 2] -= np.min(prediction[:, :, 2])
                
                anim_output = {'Reconstruction': prediction}
                if ground_truth is not None and not args.viz_no_ground_truth:
                    anim_output['Ground truth'] = ground_truth
                
                input_keypoints = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])
                
                from common.visualization import render_animation
                render_animation(input_keypoints, keypoints_metadata, anim_output, 
                                dataset.skeleton(), dataset.fps(), args.viz_bitrate, cam['azimuth'], args.viz_output,
                                limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                                input_video_path=args.viz_video, viewport=(cam['res_w'], cam['res_h']),
                                input_video_skip=args.viz_skip)
            else:
                print('Evaluating...')
                if not args.evaluate:
                    model_pos.load_state_dict(model_pos_train.state_dict())
                all_actions = {}
                all_actions_by_subject = {}
                for subject in subjects_test:
                    if subject not in all_actions_by_subject:
                        all_actions_by_subject[subject] = {}

                    for action in dataset[subject].keys():
                        action_name = action.split(' ')[0]
                        if action_name not in all_actions:
                            all_actions[action_name] = []
                        if action_name not in all_actions_by_subject[subject]:
                            all_actions_by_subject[subject][action_name] = []
                        all_actions[action_name].append((subject, action))
                        all_actions_by_subject[subject][action_name].append((subject, action))

                def fetch_actions(actions):
                    out_poses_3d = []
                    out_poses_2d = []

                    for subject, action in actions:
                        poses_2d = keypoints[subject][action]
                        for i in range(len(poses_2d)): # Iterate across cameras
                            out_poses_2d.append(poses_2d[i])

                        poses_3d = dataset[subject][action]['positions_3d']
                        assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                        for i in range(len(poses_3d)): # Iterate across cameras
                            out_poses_3d.append(poses_3d[i])

                    stride = args.downsample
                    if stride > 1:
                        # Downsample as requested
                        for i in range(len(out_poses_2d)):
                            out_poses_2d[i] = out_poses_2d[i][::stride]
                            if out_poses_3d is not None:
                                out_poses_3d[i] = out_poses_3d[i][::stride]
                    
                    return out_poses_3d, out_poses_2d

                def run_evaluation(actions, action_filter=None):
                    errors_p1 = []
                    errors_p2 = []
                    errors_p3 = []
                    errors_vel = []

                    for action_key in actions.keys():
                        if action_filter is not None:
                            found = False
                            for a in action_filter:
                                if action_key.startswith(a):
                                    found = True
                                    break
                            if not found:
                                continue

                        poses_act, poses_2d_act = fetch_actions(actions[action_key])
                        gen = UnchunkedGenerator(None, poses_act, poses_2d_act,
                                                pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                                                kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
                        e1, e2, e3, ev = evaluate(gen, action_key)
                        errors_p1.append(e1)
                        errors_p2.append(e2)
                        errors_p3.append(e3)
                        errors_vel.append(ev)

                    print('Protocol #1   (MPJPE) action-wise average:', round(np.mean(errors_p1), 1), 'mm')
                    print('Protocol #2 (P-MPJPE) action-wise average:', round(np.mean(errors_p2), 1), 'mm')
                    print('Protocol #3 (N-MPJPE) action-wise average:', round(np.mean(errors_p3), 1), 'mm')
                    print('Velocity      (MPJVE) action-wise average:', round(np.mean(errors_vel), 2), 'mm')
                    return np.mean(errors_p1)



                if not args.by_subject:
                    # run_evaluation(all_actions, ['Purchases'])
                    err_t = run_evaluation(all_actions)
                else:
                    for subject in all_actions_by_subject.keys():
                        print('Evaluating on subject', subject)
                        err_t = run_evaluation(all_actions_by_subject[subject], action_filter)
                

            if err_t < err_t_save:
                err_t_save = err_t
                chk_path = os.path.join(args.checkpoint, 'epochfinal_%s_%s_d%d_l%d.bin'%(args.keypoints,args.architecture,args.w_projd2,args.w_projl2))
                print('Saving checkpoint to', chk_path)
                
                torch.save({
                    'epoch': epoch,
                    'lr': lr,
                    'random_state': train_generator.random_state(),
                    'optimizer': optimizer.state_dict(),
                    'model_pos': model_pos_train.state_dict(),
                }, chk_path)
