# ML_pipeline

## 项目简介

本项目为机器学习流水线（ML Pipeline），实现了数据加载、模型训练、预测和评估等功能，适用于回归与分类等任务。项目结构清晰，便于扩展和复用。

## 文件结构

```
ML_pipeline/
├── config.txt              # 配置文件
├── pipeline.py             # 流水线主程序
├── train.py                # 训练脚本
├── predict.py              # 预测脚本
├── requirements.txt        # Python依赖包列表
├── models/                 # 各类模型实现
│   ├── LearningNetworks.py
│   ├── NLearningNetworks.py
│   └── RidgeRegression.py
├── utils/                  # 工具函数
│   ├── data_utils.py
│   └── evaluate.py
└── README.md               # 项目说明文档
```

## 安装依赖

建议使用Python 3.8及以上版本。

在命令行中运行以下命令安装依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

```bash
python train.py --config config.txt
```

### 2. 进行预测

```bash
python predict.py --config config.txt --input <测试集路径> --output <预测结果路径>
```

### 3. 配置文件说明

`config.txt` 用于设置模型参数、数据路径等。请根据实际需求修改。

## 示例

训练示例：

```bash
python train.py --config config.txt
```

预测示例：

```bash
python predict.py --config config.txt --input data/nyc_test.csv --output result.csv
```

## 贡献

欢迎提交Issue或Pull Request改进本项目。

## License

本项目遵循 MIT License。
