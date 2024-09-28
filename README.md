# GNSS-Spoofing-Detection-Using-MM-KD

本文使用一个具有多模态输入的知识蒸馏模型来检测GNSS欺骗
TEXBAT数据集下载地址：https://radionavlab.ae.utexas.edu/texbat/
下载数据集后需要使用GNSS软件接收机对场景数据进行处理以获得捕获、跟踪和解算阶段的GNSS观测值，软件接收机下载地址：https://github.com/nlsfi/FGI-GSRx
从观测值中提取可见卫星的载噪比、多普勒频移、伪距等特征按照卫星编号和时间构造特征矩阵，并构造特征矩阵的热图，分别作为第一和第二模态输入以训练知识蒸馏模型。
