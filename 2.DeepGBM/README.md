# DeepGBM

### 参考代码

[DeepGBM](https://github.com/motefly/DeepGBM)

主函数入口在 `main.py`， 包含GBDT训练，DeepGBM训练，滚动预测等。

DeepGBM训练流程在 `train_models.py`中，主要包含了三部分

1. 训练LightGBM获得树；
2. 训练树的叶子节点embedding；
3. 训练gbdt2NN拟合刚才训练好的embedding。

`model_components.py`是对训练好的树的结构的解析
