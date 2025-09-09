# Snake Game

本项目实现了一个基于遗传算法（Genetic Algorithm, GA）和神经网络（Neural Network, NN）的贪吃蛇agent。

- 神经网络负责预测贪吃蛇的移动方向。
- 遗传算法用于优化神经网络的参数（即权重，作为基因）。

---

## 目录结构说明

- `train.py`：主训练脚本，包含遗传算法主流程。
- `ai.py`：AI自动游玩与可视化，支持单体和多体对战。
- `human.py`：人类玩家游玩界面。
- `model.py`：神经网络模型定义。
- `inits.py`：全局参数与常量设置。
- `genes/`：存储父代和最优个体的基因参数。
- `seed/`：存储每个最优个体的随机种子。
- `generation_all.txt`：记录每一代的训练数据。

---

## 遗传算法流程

### 1. 初始化种群
- 随机生成 `P_SIZE`（如100）个父代，每个个体的基因为一组神经网络权重。
- 每个个体通过 `Individual` 类封装。

### 2. 适应度评估
- 每个个体用其基因（神经网络权重）游玩一局游戏，获得分数（score）、步数（steps）等。
- 适应度函数综合考虑分数和步数，具体见 `Individual.get_fitness()`。

### 3. 选择父代
- 采用精英选择（elitism selection），每代保留适应度最高的 `P_SIZE` 个体作为父代。
- 也可选用轮盘赌（roulette wheel）或锦标赛（tournament）等策略。

### 4. 交叉（Crossover）
- 两点交叉（Two-point crossover）：随机选两个切点，交换区间基因片段。
- 交叉概率 `pc`（如0.8），未命中则直接复制。

### 5. 变异（Mutation）
- 高斯变异（Gaussian mutation）：以概率 `MUTATE_RATE`（如0.1）对每个基因加高斯噪声。
- 变异幅度可调节。

### 6. 子代生成
- 每次从父代中选两人交配，产生2个子代，直到生成 `C_SIZE`（如400）个子代。
- 子代和父代合并为新一代种群。

### 7. 记录与保存
- 每代保存最优个体的基因和种子到 `genes/best/` 和 `seed/`。
- 每20代保存当前父代基因到 `genes/parents/`，便于断点续训。
- 每代训练数据（代数、历史最高分、本代最高分、均分）写入 `generation_all.txt`。

---

## 神经网络结构

- 输入层：32维（包括头/尾方向、8方向视野等）
- 隐藏层1：24维
- 隐藏层2：12维
- 输出层：4维（上、下、左、右）
- 激活函数：ReLU + Sigmoid
- 权重参数总数：`GENES_LEN = N_INPUT * N_HIDDEN1 + N_HIDDEN1 * N_HIDDEN2 + N_HIDDEN2 * N_OUTPUT + N_HIDDEN1 + N_HIDDEN2 + N_OUTPUT`

---

## 训练与测试

### 训练命令
- `python train.py`：从头开始训练。
- `python train.py -i`：继承 `genes/parents/` 目录下的父代基因继续训练。

### 可视化与测试
- `python ai.py`：调用 `play_best(score)` 或 `play_all(n)`，可视化最优个体或全部父代的表现。
- `python human.py`：人类手动游玩。

---

## 参数说明（inits.py）
- `P_SIZE`：父代数量（如100）
- `C_SIZE`：子代数量（如400）
- `MUTATE_RATE`：变异概率（如0.1）
- `N_INPUT`、`N_HIDDEN1`、`N_HIDDEN2`、`N_OUTPUT`：神经网络结构
- `DIRECTIONS`：动作空间

---

## 文件与数据说明
- `genes/parents/`：每20代保存一次父代基因，便于断点续训。
- `genes/best/`：每次出现新高分时保存最优个体基因。
- `seed/`：与 `genes/best/` 配套，记录最优个体的随机种子。
- `generation_all.txt`：每20代追加写入训练数据，格式为：`generation record best_score avg_score`

---

## 常见问题与改进建议

- **模型容易“自尽”**：当前适应度函数和奖励机制较为简单，建议引入更复杂的奖励（如距离食物远近、探索性奖励等）。
- **收敛速度慢**：可尝试增加种群规模、调整变异率、引入多样性保持机制。
- **神经网络结构简单**：可尝试更深/更宽的网络，或引入卷积等结构。
- **断点续训**：每20代自动保存父代基因，支持 `-i` 参数断点续训。
- **可视化**：`ai.py` 支持多体对战和最优个体回放，便于观察训练效果。

---

## 参考文献

- https://github.com/Chrispresso/SnakeAI.git
- Bell, Okezue. "Applications of Gaussian Mutation for Self Adaptation in Evolutionary Genetic Algorithms." *arXiv preprint arXiv:2201.00285* (2022).
- Almalki, Ali Jaber, and Pawel Wocjan. "Exploration of reinforcement learning to play snake game." *2019 International Conference on Computational Science and Computational Intelligence (CSCI)*. IEEE, 2019.
