
# Installation
nnU-Net在Linux上运行
需要GPU!为了进行inference，GPU应该有4GB的VRAM。为了训练nnU-Net模型，GPU应该至少有10GB。
利用pytorch(see [here](https://github.com/pytorch/pytorch#from-source)),cuDNN 8.0.2 or newer. 
训练:CPU要足够配合GPU。建议CPU核数至少为6核(12线程)。对于像brat这样使用4种图像模式的数据集，它们的分辨率更高。

建议在虚拟环境中安装nnU-Net。
[Here is a quick how-to for Ubuntu.](https://linoxide.com/linux-how-to/setup-python-virtual-environment-ubuntu/)
设置环境变量OMP_NUM_THREADS=1(在bashrc中使用' export OMP_NUM_THREADS=1 ')。
请确保您使用的是python3。
步骤：
1) 安装 [PyTorch](https://pytorch.org/get-started/locally/). 版本至少 1.6以上
2) 安装 nnU-Net:
    1) For use as **standardized baseline**, **out-of-the-box segmentation algorithm** or for running **inference with pretrained models**:
      
        ```pip install nnunet```
    
    2) For use as integrative **framework** (this will create a copy of the nnU-Net code on your computer so that you can modify it as needed):
          ```bash
          git clone https://github.com/MIC-DKFZ/nnUNet.git
          cd nnUNet
          pip install -e .
          ```
3) nnU-Net需要知道你打算在哪里保存原始数据、预处理数据和训练过的模型。为此，需要设置一些环境变量。Please follow the instructions [here](documentation/setting_up_paths.md).
安装nnU-Net完成后将为您的终端添加几个新命令。这些命令用于运行整个nnU-Net。为了便于识别，所有nnU-Net命令都有“nnUNet_”前缀。
如果在虚拟环境中安装了nnU-Net，那么在执行这些命令时必须激活该环境。

nnU-Net命令有一个' -h '选项，它提供了如何使用它们的信息。
# Usage
### Dataset conversion
nnU-Net要求数据集采用结构化格式。这种格式紧密(但不是完全)遵循了 [Medical Segmentation Decthlon](http://medicaldecathlon.com/)的数据结构.
详细信息见[this](documentation/dataset_conversion.md)，为此，此文件夹中包含了两个数据集的结构化格式修改的代码。

### Experiment planning and preprocessing
第一步，nnU-Net提取数据集指纹(一组特定于数据集的属性，如图像大小、体素间隔、强度信息等)。
该信息用于创建三个U-Net配置:一个2D U-Net，一个在全分辨率图像上运行的3D U-Net，以及3D U-Net级联，其中第一个U-Net在下采样图像中创建一个粗分割map，然后由第二个U-Net进行细化。
原始数据集的文件夹 (`nnUNet_raw_data_base/nnUNet_raw_data/TaskXXX_MYTASK`, 
步骤:
```bash
nnUNet_plan_and_preprocess -t XXX --verify_dataset_integrity
```
`XXX` 是任务名 `TaskXXX_MYTASK`.
这里我们利用`nnUNet_plan_and_preprocess -t 007`处理胰腺数据集，随后产生preprocessed data in nnUNet_preprocessed/TaskXXX_MYTASK.
这些文件包含生成的配置，将被训练器读取。 注意预处理的数据文件夹只包含the training cases. The test images are not preprocessed. 
如果在预处理过程中内存耗尽，需要使用`-tl` and `-tf` 选项来调整进程的数量。
`nnUNet_plan_and_preprocess`之后,配置已经创建，文件位于nnUNet_preprocessed/TaskXXX_MYTASK.

### Model training
nnUNet可以自动划分数据，并进行5折交叉验证。当然我们也可以自己划分训练和测试数据，进而进行交叉验证实验。
自动划分数据进行实验：
```bash
nnUNet_train CONFIGURATION TRAINER_CLASS_NAME TASK_NAME_OR_ID FOLD 
```
“CONFIGURATION”是字符串，用于标识请求的U-Net配置。TRAINER_CLASS_NAME是模型训练器的名称。
TASK_NAME_OR_ID指定应该对哪些数据集进行训练，FOLD指定训练5折交叉验证中的哪一折。
如果您需要继续训练，只需在命令中添加一个`-c` 。
#### 2D U-Net
For FOLD in [0, 1, 2, 3, 4], run:
```bash
nnUNet_train 2d nnUNetTrainerV2 TaskXXX_MYTASK FOLD 
```
#### 3D full resolution U-Net
For FOLD in [0, 1, 2, 3, 4], run:
```bash
nnUNet_train 3d_fullres nnUNetTrainerV2 TaskXXX_MYTASK FOLD
```
#### 3D U-Net cascade
##### 3D low resolution U-Net
For FOLD in [0, 1, 2, 3, 4], run:
```bash
nnUNet_train 3d_lowres nnUNetTrainerV2 TaskXXX_MYTASK FOLD
```
##### 3D full resolution U-Net
For FOLD in [0, 1, 2, 3, 4], run:
```bash
nnUNet_train 3d_cascade_fullres nnUNetTrainerV2CascadeFullRes TaskXXX_MYTASK FOLD --npz
```
级联需要提前完成低分辨率U-Net的5折。

我们使用的训练命令为：使用自动划分数据的交叉验证：`nnUNet_train 3d_lowres nnUNetTrainerV2BraTSRegions_DA3 7 FOLD`  or `nnUNet_train 3d_lowres nnUNetTrainerV2BraTSRegions_DA3 7 FOLD -c` 
不使用自动划分数据的交叉验证，即自己划分训练和测试集：`nnUNet_train 3d_lowres nnUNetTrainerV2BraTSRegions_DA3 7 all`  or`nnUNet_train 3d_lowres nnUNetTrainerV2BraTSRegions_DA3 7 all -c` 
最终将训练过的模型写入RESULTS_FOLDER/nnUNet文件夹
nnUNet_preprocessed/CONFIGURATION/TaskXXX_MYTASKNAME/TRAINER_CLASS_NAME__PLANS_FILE_NAME/FOLD
For Task007_Pancreas:

    RESULTS_FOLDER/nnUNet/
    ├──  3d_lowres
    │   └── Task07_Pancreas
    │       └── nnUNetTrainerV2__nnUNetPlansv2.1#我们改变了训练器
    │           ├── fold_0
    │           │   ├── debug.json
    │           │   ├── model_final_checkpoint.model
    │           │   ├── model_final_checkpoint.model.pkl
    │           │   ├── progress.png
    │           ├── fold_1
    │           ├── fold_2
    │           ├── fold_3
    │           └── fold_4
    └── 

- debug.json: 包含用于训练该模型的blueprint和inferred parameters。
- model_final_checkpoint.model / model_final_checkpoint.model.pkl: 最终模型的checkpoint文件(训练结束后)。这就是测试和推理所使用的。
- progress.png: 训练期间的训练损失图(蓝色)。

### Run inference
测试：
```
nnUNet_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -t TASK_NAME_OR_ID -m CONFIGURATION 
```
# 快速运行-MF2N: Multiview feature fusion network for pancreatic cancer segmentation.
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
```