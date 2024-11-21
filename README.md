# Print-ML

#### 安装: 
```
git clone https://github.com/EDiRobotics/printml
cd printml
pip install -e .
```
#### 配置
```
accelerate config
```
选择GPU卡数, 混合精度等(建议bf16)。然后在`Train.py`中配置训练参数。

#### 运行
1. 生成模拟数据集
```
cd Datasets
python GenerateFakeDataset.py
```
2. 开始训练
```
cd Scripts
accelerate launch TrainDeformPred.py # 变形检测
accelerate launch TrainSegment.py # 区域分割
```

#### 损失曲线
[变形检测](https://wandb.ai/lizh9785/PrintML/runs/7145xw6q)

[区域分割](https://wandb.ai/lizh9785/PrintML/runs/jbimm1vy)

