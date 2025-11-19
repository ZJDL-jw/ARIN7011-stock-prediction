# 金融深度学习股票预测项目

## 项目简介

本项目实现了一个完整的"自研多因子 + 深度时序模型"的股票涨跌概率预测系统，针对A股蓝筹股进行多周期预测（日线/周线/月线：1/7/30日）与分布学习（分位数回归 τ=0.1/0.5/0.9）。项目包含数据抓取、因子工程、样本构造、模型训练、概率校准、可解释性分析（注意力热力图）与结果可视化。

### 核心特性

- **多因子体系**：59个因子（48基础+8短期+5 PCA+5 AE+1 close）
- **多模型支持**：MLP、LSTM、GRU、CNN1D、PatchTST、SimpleTransformer、Transformer、Ensemble（8个模型）
- **多周期预测**：日线（1日）、周线（7日）、月线（30日）
- **多任务学习**：二分类（涨跌方向）+ 分位数回归（收益分布）
- **自适应阈值优化**：在验证集上自动找到最优分类阈值，大幅提升F1分数
- **Ensemble权重优化**：自动优化集成模型权重，提升综合性能
- **概率校准**：Isotonic/Platt校准，提升概率可靠性
- **可解释性**：注意力热力图（Transformer系列模型）

## 环境安装

### 使用 Conda（推荐）

```bash
conda env create -f environment.yml
conda activate finseq
```

### 使用 pip

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 数据下载

```bash
python src/data/download.py --config configs/base.yaml
```

这将从Yahoo Finance下载15只A股蓝筹股和上证指数的日线数据（2016-2024，实际可用数据范围）。

### 2. 数据预处理

```bash
python src/data/preprocess.py --config configs/base.yaml --factors configs/factors.yaml
```

执行复权对齐、winsorization（1%-99%）、滚动标准化（窗口252日）。

### 3. 因子工程与样本构造

```bash
python src/data/dataset.py --build --config configs/base.yaml --factors configs/factors.yaml
```

计算59个因子（48基础+8短期+5 PCA+5 AE），构造滑动窗口样本（T=60）。

### 4. 模型训练

训练所有模型（推荐使用批量脚本）：

```bash
# 批量训练所有模型
bash scripts/train_all_optimized.sh

# 或逐个训练
python src/train.py --config configs/mlp.yaml
python src/train.py --config configs/lstm.yaml
python src/train.py --config configs/gru.yaml
python src/train.py --config configs/cnn1d.yaml
python src/train.py --config configs/patchtst.yaml
python src/train.py --config configs/simple_transformer.yaml
python src/train.py --config configs/transformer.yaml
```

训练过程会自动保存最佳模型到 `runs/<model_name>/checkpoints/best.pt`。

### 5. 评估与概率校准

评估所有模型（自动使用最优阈值）：

```bash
# 批量评估所有模型
bash scripts/evaluate_all_optimized.sh

# 或逐个评估
python src/evaluate.py --config configs/mlp.yaml
python src/evaluate.py --config configs/lstm.yaml
python src/evaluate.py --config configs/gru.yaml
python src/evaluate.py --config configs/cnn1d.yaml
python src/evaluate.py --config configs/patchtst.yaml
python src/evaluate.py --config configs/simple_transformer.yaml
python src/evaluate.py --config configs/transformer.yaml
```

输出：
- 测试集指标（Accuracy/F1/AUC/Pinball/CRPS）
- 最优阈值（自动在验证集上优化）
- ROC/PR曲线
- 校准前后Reliability图
- 注意力热力图（Transformer系列模型）

### 6. Ensemble模型评估

```bash
python src/ensemble.py --config configs/lstm.yaml --models lstm gru patchtst --method weighted
```

自动优化权重和阈值，输出集成模型的综合性能。

## 项目结构

```
project-root/
├── README.md
├── environment.yml
├── requirements.txt
├── configs/
│   ├── base.yaml            # 数据区间、窗口T、horizons、quantiles
│   ├── factors.yaml         # 因子开关与参数
│   ├── mlp.yaml
│   ├── lstm.yaml
│   ├── gru.yaml
│   ├── cnn1d.yaml
│   ├── patchtst.yaml
│   ├── simple_transformer.yaml
│   └── transformer.yaml
├── data/
│   ├── raw/                 # yfinance原始CSV
│   ├── interim/             # 对齐/复权/特征中间文件
│   └── processed/           # 滑窗样本npz
├── src/
│   ├── utils/
│   │   ├── seed.py          # 随机种子设置
│   │   ├── io.py            # 配置加载、文件I/O
│   │   ├── metrics.py       # 评估指标
│   │   ├── plots.py         # 可视化
│   │   ├── timecv.py        # 时间序列交叉验证
│   │   └── calibration.py   # 概率校准
│   ├── data/
│   │   ├── download.py      # 数据下载
│   │   ├── preprocess.py   # 预处理
│   │   ├── factors.py      # 因子工程
│   │   └── dataset.py       # 数据集构造
│   ├── models/
│   │   ├── mlp.py
│   │   ├── lstm.py
│   │   ├── gru.py
│   │   ├── cnn1d.py
│   │   ├── patchtst.py
│   │   ├── simple_transformer.py
│   │   ├── transformer.py
│   │   ├── ensemble.py
│   │   └── common/
│   │       ├── losses.py    # 损失函数
│   │       ├── heads.py     # 输出头
│   │       └── focal_loss.py # Focal Loss
│   ├── train.py             # 训练主循环
│   ├── evaluate.py          # 评估+校准+可视化（支持自适应阈值）
│   └── ensemble.py          # Ensemble模型训练和评估（支持权重优化）
├── reports/
│   ├── figs/                # 图表
│   └── tables/              # 结果表格
└── runs/
    └── <model_name>/
        ├── checkpoints/
        │   └── best.pt
        └── logs.json
```

## 因子说明

### 59个因子体系

**48个基础因子**：
- 价格因子：多周期收益率（1/5/10/20日）、反转收益、多周期MA偏差
- 动量因子：RSI（14/21）、MACD、Stochastic、Williams %R、CCI、ADX、ROC、Momentum
- 波动率因子：多周期Bollinger Bands、ATR（14/21）、多周期波动率、波动率比率、趋势强度
- 成交量因子：多周期成交量比率、换手率、OBV、MFI
- 市场相对因子：多周期相对强度（1/5/10日）

**8个短期特征**（针对Horizon 1优化）：
- momentum_1/2/3: 1/2/3日动量
- intraday_range: 日内波动率（最高-最低）
- intraday_range_ma3: 日内波动率3日均值
- price_acceleration: 价格加速度（二阶导数）
- trend_3/5: 3/5日趋势强度

**10个表征学习因子**：
- **PCA因子**：5个主成分（滚动窗口PCA，使用RobustScaler）
- **Autoencoder因子**：5个潜在表示（滚动窗口训练，使用MAD标准化）

**1个价格因子**：
- close: 收盘价（用于标签计算）

## 模型架构

### MLP
- 仅使用最后时刻（t日）的截面特征
- 结构：FC 128→64→32, ReLU, Dropout=0.3

### LSTM
- 双向3层LSTM，hidden=128，处理完整时序[T=60, D]
- Dropout=0.3

### GRU
- 双向3层GRU，hidden=128，处理完整时序[T=60, D]
- Dropout=0.3

### CNN1D
- 3层1D卷积，滤波器数量[64, 128, 256]，核大小[3, 5, 7]
- MaxPooling + AdaptiveMaxPooling
- Dropout=0.3

### PatchTST
- Channel-independent版本
- Patch长度=6，步长=3
- Transformer编码器：3层，8头，d_model=128
- 平均池化后接多任务头

### SimpleTransformer
- 简化版Transformer（仅注意力机制，无FFN）
- d_model=128, n_heads=8, depth=3
- 轻量级，训练速度快

### Transformer
- 标准Transformer编码器（注意力+FFN）
- d_model=128, n_heads=8, depth=3, d_ff=512
- 完整实现，性能更强

### Ensemble
- 集成LSTM、GRU、PatchTST
- 支持加权平均（自动优化权重）和Stacking
- 自动优化分类阈值

## 评估指标

### 分类指标
- **Accuracy**: 准确率
- **F1**: F1分数
- **AUC**: ROC曲线下面积
- **Brier Score**: 概率校准质量

### 分布指标
- **Pinball Loss**: 分位数回归损失（τ=0.1/0.5/0.9）
- **CRPS**: 连续概率排序分数（多分位数近似）

## 结果解读

### 实际性能（使用最优阈值）

在测试期（2023-2024），实际指标：

**Horizon 1日（日线预测）**：
- **Accuracy**: 44.9%-51.8%（接近随机）
- **F1**: 0.4190-0.6239（SimpleTransformer/Transformer最佳：0.6239）
- **AUC**: 0.4682-0.5333（MLP最佳：0.5333）
- **CRPS**: 0.0085-0.0163（PatchTST最佳：0.0085）

**Horizon 7日（周线预测）**：
- **Accuracy**: 37.5%-52.8%（PatchTST最佳：52.8%）
- **F1**: 0.4435-0.5474（CNN1D最佳：0.5474）
- **AUC**: 0.4009-0.5309（PatchTST最佳：0.5309）
- **CRPS**: 0.0170-0.0257（SimpleTransformer最佳：0.0170）

**Horizon 30日（月线预测）**：
- **Accuracy**: 38.3%-67.3%（Ensemble最佳：67.3%，显著优于随机）
- **F1**: 0.2871-0.6393（Ensemble最佳：0.6393）
- **AUC**: 0.3377-0.7424（Ensemble最佳：0.7424，非常优秀）
- **CRPS**: 0.0342-0.0485（MLP最佳：0.0342）

### 关键发现

1. **阈值优化成功**：不同模型选择了不同的最优阈值（0.44-0.50），评估结果真实反映模型差异
2. **月线预测表现最佳**：Ensemble在月线预测中表现卓越（Accuracy=67.3%, AUC=0.7424）
3. **新模型表现突出**：SimpleTransformer在日线F1表现最佳（0.6239）
4. **模型性能差异明显**：不同模型在不同horizon各有优势
5. **分位数回归质量高**：CRPS保持在0.008-0.049范围

### 局限性

1. **数据质量**：yfinance数据可能存在延迟、复权问题
2. **短期预测难度**：Horizon 1的预测难度极大，Accuracy接近随机
3. **模型复杂度**：为CPU训练优化，模型规模较小
4. **样本量**：15只股票、实际可用数据2016-2024，样本量相对有限

### 已实现的优化

1. ✓ **自适应阈值优化**：在验证集上自动找到最优分类阈值，阈值范围调整为0.3-0.7，避免极端阈值
2. ✓ **Ensemble权重优化**：自动优化集成模型权重，提升综合性能
3. ✓ **因子扩展**：从12个扩展到59个因子（+392%）
4. ✓ **模型扩展**：从3个扩展到8个模型（新增GRU、CNN1D、SimpleTransformer、Transformer、Ensemble）
5. ✓ **预测周期优化**：从1-3-5日改为1-7-30日（日线/周线/月线），更符合实际交易习惯
6. ✓ **数值稳定性**：完全解决因子构建过程中的警告问题

### 未来改进方向

1. 使用专业数据源（Wind、Tushare等）
2. 扩展因子库（基本面、情绪、另类数据）
3. 短期预测深度优化（高频数据、专门架构）
4. 模型架构优化（更深的网络、更好的正则化）
5. 数据增强技术（时间序列数据增强）

## 配置说明

### 修改因子配置

编辑 `configs/factors.yaml`：
- `use_autoencoder`: true/false（是否启用AE嵌入）
- `use_pca`: true/false（是否启用PCA因子）
- `pca_components`: PCA主成分数量（默认5）
- `ae_latent_dim`: AE潜在维度（默认5）

### 修改模型超参数

编辑对应的配置文件（如`configs/patchtst.yaml`）：
- `trainer`: 训练参数（batch_size, epochs, lr等）
- `patchtst`: 模型架构参数

### 修改Ensemble配置

编辑 `src/ensemble.py` 或使用命令行参数：
- `--models`: 要集成的模型列表（如：lstm gru patchtst）
- `--method`: 集成方法（weighted/average/stacking）
- `--weights`: 手动指定权重（可选）

## 常见问题

### Q: 数据下载失败？
A: 检查网络连接，yfinance可能受防火墙限制。可手动下载CSV到`data/raw/`目录。

### Q: 训练很慢？
A: 默认使用CPU，可修改为GPU（需安装CUDA版PyTorch）。模型已针对CPU优化，PatchTST训练约需30分钟。

### Q: 如何添加新因子？
A: 在`src/data/factors.py`的`compute_all_factors`函数中添加计算逻辑，并在`base_factor_cols`中注册。

### Q: 如何修改预测周期？
A: 在`configs/base.yaml`中修改`features.horizons`（如添加7日、10日）。

## 许可证

本项目仅供教学和研究使用。

## 贡献

欢迎提交Issue和Pull Request！

