# Dataset and Visibility Score

The experiments in our paper is trained/evaluated on the [Human3.6M](http://vision.imar.ro/human3.6m) dataset. The dataset used in our code is introduced by [Martinez et al.](https://github.com/una-dinosauria/3d-pose-baseline) and [Pavllo et al.](https://github.com/facebookresearch/VideoPose3D). Please refer to [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) to set up the dataset. The CPN and the ground-truth 2D keypoint data and the corresponding 3D ground truth and visibility score are required and need to be placed in (`./data` directory).

The 2D keypoint visibility scores of the Human3.6M dataset is predicted by [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose).