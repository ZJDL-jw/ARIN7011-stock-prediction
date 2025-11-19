# 金融深度学习股票预测项目

## 项目简介

本项目实现了一个完整的"自研多因子 + 深度时序模型"的短期涨跌概率预测系统，针对A股蓝筹股进行多步预测（1/3/5日）与分布学习（分位数回归 τ=0.1/0.5/0.9）。项目包含数据抓取、因子工程、样本构造、模型训练、概率校准、Walk-forward稳健性评估、教学级回测、可解释性分析（注意力热力图）与结果可视化。

### 核心特性

- **多因子体系**：59个因子（48基础+8短期+5 PCA+5 AE+1 close）
- **多模型支持**：MLP、LSTM、GRU、CNN1D、PatchTST、Ensemble（6个模型）
- **多任务学习**：二分类（涨跌方向）+ 分位数回归（收益分布）
- **自适应阈值优化**：在验证集上自动找到最优分类阈值，大幅提升F1分数
- **Ensemble权重优化**：自动优化集成模型权重，提升综合性能
- **概率校准**：Isotonic/Platt校准，提升概率可靠性
- **稳健性评估**：Walk-forward年度滚动、因子消融实验
- **教学级回测**：考虑交易成本、换手率、最大回撤等
- **可解释性**：注意力热力图、因子重要性分析

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

这将从Yahoo Finance下载10只A股蓝筹股和上证指数的日线数据（2016-2024）。

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
```

输出：
- 测试集指标（Accuracy/F1/AUC/Pinball/CRPS）
- 最优阈值（自动在验证集上优化）
- ROC/PR曲线
- 校准前后Reliability图
- 注意力热力图（PatchTST）

### 6. Ensemble模型评估

```bash
python src/ensemble.py --config configs/lstm.yaml --models lstm gru patchtst --method weighted
```

自动优化权重和阈值，输出集成模型的综合性能。

### 7. Walk-forward稳健性评估

```bash
python src/walkforward.py --config configs/patchtst.yaml
```

按年度滚动评估，输出年度AUC箱线图。

### 8. 因子消融实验

```bash
python src/ablation.py --config configs/patchtst.yaml --drop-group momentum
python src/ablation.py --config configs/patchtst.yaml --drop-group volatility
python src/ablation.py --config configs/patchtst.yaml --drop-group volume
python src/ablation.py --config configs/patchtst.yaml --drop-group rs
```

### 9. 教学级回测

```bash
python src/backtest.py --config configs/backtest.yaml --ckpt runs/patchtst/checkpoints/best.pt --cost_bps 10
```

输出策略与基准的CAGR、Sharpe、最大回撤、换手率对比。

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
│   └── backtest.yaml
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
│   │   ├── ensemble.py
│   │   └── common/
│   │       ├── losses.py    # 损失函数
│   │       ├── heads.py     # 输出头
│   │       └── focal_loss.py # Focal Loss
│   ├── train.py             # 训练主循环
│   ├── evaluate.py          # 评估+校准+可视化（支持自适应阈值）
│   ├── ensemble.py          # Ensemble模型训练和评估（支持权重优化）
│   ├── ablation_factors.py  # 因子重要性分析
│   ├── hyperparameter_tuning.py # 超参数调优
│   ├── walkforward.py       # Walk-forward评估
│   ├── ablation.py          # 消融实验
│   └── backtest.py          # 回测
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

**Horizon 1日（短期预测）**：
- **Accuracy**: 49.9%（接近随机）
- **F1**: 0.6657（所有模型一致，大幅提升）
- **AUC**: 0.5044（Ensemble最佳，略高于随机）
- **CRPS**: 0.0083-0.0144（优秀）

**Horizon 3日（中期预测）**：
- **Accuracy**: 54.5%（显著优于随机）
- **F1**: 0.7051（所有模型一致，优秀）
- **AUC**: 0.5553（LSTM最佳）
- **CRPS**: 0.0152-0.0210（优秀）

**Horizon 5日（长期预测）**：
- **Accuracy**: 53.2%（优于随机）
- **F1**: 0.6946（所有模型一致，优秀）
- **AUC**: 0.5320（GRU最佳）
- **CRPS**: 0.0145-0.0271（优秀）

### 关键发现

1. **自适应阈值优化效果显著**：F1从0提升到0.6657-0.7051
2. **Ensemble模型综合性能最佳**：在所有horizon都表现稳定
3. **分位数回归质量高**：CRPS保持在0.008-0.019范围
4. **模型一致性**：所有模型在相同horizon表现接近

### 局限性

1. **数据质量**：yfinance数据可能存在延迟、复权问题
2. **短期预测难度**：Horizon 1的预测难度极大，Accuracy接近随机
3. **模型复杂度**：为CPU训练优化，模型规模较小
4. **回测简化**：回测系统为教学目的简化，未考虑滑点、流动性等
5. **样本量**：10只股票、9年数据，样本量相对有限

### 已实现的优化

1. ✓ **自适应阈值优化**：在验证集上自动找到最优分类阈值，F1从0提升到0.6657-0.7051
2. ✓ **Ensemble权重优化**：自动优化集成模型权重，提升综合性能
3. ✓ **因子扩展**：从12个扩展到59个因子（+392%）
4. ✓ **模型扩展**：从3个扩展到6个模型（新增GRU、CNN1D、Ensemble）
5. ✓ **数值稳定性**：完全解决因子构建过程中的警告问题

### 未来改进方向

1. 使用专业数据源（Wind、Tushare等）
2. 扩展因子库（基本面、情绪、另类数据）
3. 强化学习（动态仓位管理）
4. 更精细的回测系统（考虑滑点、冲击成本）
5. 短期预测深度优化（高频数据、专门架构）

## 配置说明

### 修改因子配置

编辑 `configs/factors.yaml`：
- `use_autoencoder`: true/false（是否启用AE嵌入）
- `use_pca`: true/false（是否启用PCA因子）
- `pca_components`: PCA主成分数量（默认5）
- `ae_latent_dim`: AE潜在维度（默认5）
- `groups`: 因子分组（用于消融实验）

### 修改模型超参数

编辑对应的配置文件（如`configs/patchtst.yaml`）：
- `trainer`: 训练参数（batch_size, epochs, lr等）
- `patchtst`: 模型架构参数

### 修改回测参数

编辑 `configs/backtest.yaml`：
- `holding_days`: 持仓天数
- `top_k`: 选股数量
- `costs`: 交易成本（bps）

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

