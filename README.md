# OCFER
预训练模型的数据增强和超参数设置是参考的是这个仓库：https://github.com/LetheSec/Fer2013-Facial-Emotion-Recognition-Pytorch 在此非常感谢
---
这个仓库所包含的内容是One-Class Classification for Facial Expression Recognition
## data
**FER2013**：https://www.kaggle.com/datasets/deadskull7/fer2013
**AffectNet**：https://www.kaggle.com/datasets/arafatshovon/affectnet
下载这两个数据集放在- [data]下

## 预训练模型
**Fer2013**:- [一级目录](#一级目录)
  - [二级目录](#二级目录)
    - [三级目录](#三级目录)

**AffectNet**:- [checkpoint]
              - [AffectNet]
## train
注意训练单类模型请修改<span style="background-color: #ddd; color: #fff; padding: 3px 5px;">paramers.py</span>文件中的target_class
python train_fer2013.py
