# MF2N: Multiview feature fusion network for pancreatic cancer segmentation.
We present a novel deep network, MF2N network, equipped with three feature fusion modules in different views for automated segmentation of the variable pancreas and tumors, which is challenging due to 1) large variations occurring within/crossing the narrow and irregular pancreas, 2) a small proportion of voxels on pancreatic tumors in the abdomen and 3) ambiguous lesion boundaries caused by low contrast between target surroundings. We first propose a novel adaptive morphological feature fusion (AMF2) module to dynamically learn and fuse morphological features of the pancreas and tumors from the skeleton to boundaries, aiming to mitigate undersegmentation of targets. Then, bidirectional semantic feature fusion (BSF2) module is proposed to optimize mutual information between prediction and manual delineation as well as discard redundant information between input and attentive features to capture more consistent feature expressions and alleviate noise interference caused by the large fraction of background in the abdomen. Furthermore, we develop a local-global dependency feature fusion (LGDF2) module module to fuse local features from CNNs and global information provided by shallow features through a lightweight transformer to enhance MF2N’s capability to grasp more boundary and content features. 

建立并激活虚拟环境
```
cd /root/LL
source ./bin/activate
cd /root/nnUNet
pip install nnunet
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```
进入程序，并修改设置文件（按照自己的路径进行修改），后激活
```
cd /root/LL/nnUNet/nnunet
source /root/LL/nnUNet/nnunet/.bashrc
```
利用设备进行数据处理、训练
```
export CUDA_VISIBLE_DEVICES=0
```
数据处理，生成网络可识别的形式。
```
nnUNet_plan_and_preprocess -t 007
```
网络训练，进行五折交叉验证实验（0,1,2,3,4），最终将训练过的模型存储于RESULTS_FOLDER/nnUNet文件夹。
```
nnUNet_train 3d_lowres nnUNetTrainerV2BraTSRegions_DA3 7 0 
```
训练完成后测试
```
nnUNet_predict -i /public/home/../ -o /public/home/../predict -tr nnUNetTrainerV2BraTSRegions_DA3 -m 3d_lowres -p nnUNetPlansv2.1 -t Task007_Pancreas
