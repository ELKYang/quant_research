# Simpler VAE

模型使用pytorch-lightning包裹，weight&basis训练过程可视化

`config-defaults.yaml`为模型的配置文件，包括数据集配置，模型配置等。

`dataset.py`和 `dataset_rand.py`为数据准备dataset，分别为按天feed股票数据和随机feed数据

`model.py`为模型的实现，`model-factorvae.py`为FactorVAE的实现（仅供参考）

`train.py`为训练代码

`utils.py`为工具类代码
