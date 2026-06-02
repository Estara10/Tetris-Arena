# Tetris Python — AI 对战平台

基于 Pygame 的俄罗斯方块 AI 对战项目，支持人机对战、同盘竞技，以及强化学习训练。

▶️ **[观看演示视频](./演示视频.mp4)**

## 功能概览

### 游戏机制

- 经典 7 种方块（I/J/L/O/S/T/Z），7-bag 随机发牌
- 方块移动、旋转、硬降
- 碰撞检测、锁定、消行、计分
- 预测落点（Ghost Piece）、下一块预览
- 垃圾行攻击与抵消系统
- 陷阱能量机制（消行充能 → 按 C 释放，限制对手洞位分布）
- Combo 连击与 Back-to-Back (B2B) 奖励

### 视觉效果（新增）

- **粒子爆发**：消行时彩色粒子向四周飞溅，带重力加速度和透明度衰减
- **屏幕震动**：消行时屏幕轻微抖动，强度与消行数成正比
- **浮动文字**：消行、Combo、B2B、Tetris 时弹出飞字提示
- **锁定闪光**：方块落定瞬间在落点产生短暂白色闪光
- **菜单过渡动画**：菜单切换时 500ms 三次方缓出淡入效果
- **计时器脉冲**：Arena 模式最后 30 秒计时器红色脉冲闪烁
- **陷阱能量条**：Unicode 进度条直观显示陷阱能量状态

## 游戏模式

| 按键 | 模式 | 说明 |
|------|------|------|
| 1 | Classic 经典对战 | 左右双棋盘，同步 7-bag 对战 |
| 2 | Arena 同盘竞技 | 玩家与 AI 在 33×25 同一棋盘推挤对抗 |

### Arena 同盘竞技规则

- 玩家与 AI 在同一棋盘同时下落，活动方块之间不可穿透
- 固定出生位：玩家 x=9，AI1 x=17，AI2 x=25（共 3 个实体）
- 双方重力相位错开，避免持续互挤
- 紧挨碰撞：同向不变向，异向传导（对方已靠墙则不传导）
- 得分：最后一格落地锁定者 +1 分
- 规定时间结束后得分高者获胜

## 控制说明

### 玩家操作

| 按键 | 功能 |
|------|------|
| A | 左移 |
| D | 右移 |
| S | 直接落底 |
| L | 变换（旋转） |
| W | Classic 中暂停，Arena 中硬降 |
| C | 释放陷阱（Classic 模式，需满能量） |

> **Arena 模式按键差异**：A / D 左右移动（支持长按），S 软降，W 硬降。

### 全局操作

| 按键 | 功能 |
|------|------|
| P | 暂停 / 恢复 |
| R | 重开当前对局 |
| M | 返回菜单 |
| Q / Esc | 退出 |

## 运行方式

### ① 玩游戏（无需 torch）

只需 Python 3.10+ 和 pygame：

```bash
# 1. 安装依赖
pip install pygame

# 2. 运行游戏
python3 main.py
```

默认使用启发式 AI，不依赖任何模型文件。

### ② 启用 AI 模型推理（可选）

用预训练 DQN 模型替代启发式 AI，需额外安装 torch：

```bash
# 1. 安装 torch（CPU 版即可）
pip install torch

# 2. 启用模型模式运行
export TETRIS_AI_MODE=model
python3 main.py

# 也可指定其他模型路径
export TETRIS_AI_MODEL_PATH=models/next_state_v3/best.pt
```

> 若 torch 未安装，自动回退到启发式 AI，不影响游戏运行。

---

## AI 训练：训练自己的模型

仓库中的预训练模型由 `scripts/train_nextstate.py` 训练得到。以下是完整的训练流程说明。

### 训练环境要求

| 依赖 | 用途 | 安装命令 |
|------|------|---------|
| Python 3.10+ | 运行时 | — |
| torch | 神经网络训练 | `pip install torch` |
| pygame | 游戏环境仿真 | `pip install pygame` |
| tensorboard（可选） | 训练曲线可视化 | `pip install tensorboard` |
| matplotlib（可选） | 生成训练曲线图 | `pip install matplotlib` |

> **GPU 训练**：有 NVIDIA 显卡时安装 CUDA 版 torch 可大幅加速；无 GPU 时 CPU 也能训，只是慢一些。

### 训练方法：Next-State 落点评分

不同于传统 DQN（输出"左移/右移/旋转/硬降"等 5 个离散动作），本项目使用 **Next-State Value** 方法：

```
当前方块 → 枚举所有可能落点（旋转×平移，约 40~60 个候选）
         → 每个候选提取 51 维棋盘特征
         → MLP 神经网络对每个候选打分
         → 选最高分的落点执行
```

训练算法：**Double DQN + Prioritized Experience Replay + 教师蒸馏**

### 三步课程学习

训练分三个阶段，逐步增加难度：

| 阶段 | 回合范围 | 内容 |
|------|---------|------|
| **1. Solo 预训练** | 1 ~ 1200 | 对手不攻击（垃圾行=0），AI 专心学会消行和保持棋盘整洁 |
| **2. 过渡期** | 1201 ~ 1800 | 垃圾行强度从 0% 逐渐升至 100%，对手从弱到强 |
| **3. 完全对抗** | 1801 ~ 结束 | 正常对战，对面会攻击、会用陷阱、会加速 |

同时，**教师（启发式 AI）** 在早期指导模型：训练初期 85% 概率直接采用教师的落点选择，随训练逐渐降至 5%，让模型从模仿过渡到自主决策。

### 训练命令

```bash
# ===== 从头训练 =====
python scripts/train_nextstate.py \
    --episodes 5000 \
    --save_dir models/next_state_v4 \
    --device cuda

# ===== 仅训 300 轮（快速验证）=====
python scripts/train_nextstate.py \
    --episodes 300 \
    --save_dir models/next_state_test \
    --device cuda

# ===== 无 GPU 时用 CPU =====
python scripts/train_nextstate.py \
    --episodes 1000 \
    --save_dir models/next_state_cpu \
    --device cpu
```

### 恢复训练

默认 `--resume`（自动恢复），中断后用**相同命令**即可接着训：

```bash
# 中断后，相同命令自动检测 checkpoint 并恢复
python scripts/train_nextstate.py \
    --save_dir models/next_state_v3 \
    --device cuda

# 强制从头开始（忽略已有 checkpoint）
python scripts/train_nextstate.py \
    --save_dir models/next_state_v3 \
    --device cuda \
    --no-resume
```

### 关键训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--episodes` | 5000 | 训练总回合数 |
| `--save_dir` | `models/next_state` | 模型和日志输出目录 |
| `--device` | 自动检测 | `cuda` 或 `cpu` |
| `--batch_size` | 256 | 每次梯度更新的样本数 |
| `--lr` | 1e-4 | 学习率 |
| `--gamma` | 0.99 | 折扣因子 |
| `--replay_capacity` | 150000 | 经验回放池大小 |
| `--curriculum_episodes` | 1200 | 课程学习回合数 |
| `--solo_pretrain_episodes` | 1200 | Solo 预训练回合数 |
| `--solo_transition_episodes` | 600 | 过渡期回合数 |
| `--teacher_prob_start` / `--teacher_prob_end` | 0.85 / 0.05 | 教师指导概率起止值 |
| `--eval_interval` | 100 | 每隔多少回合评估一次 |
| `--seed` | 20260404 | 随机种子 |

### 训练产物

训练过程中 `--save_dir` 指定目录下会生成：

```
models/next_state_v3/
├── best.pt                  # 评估指标最优的模型
├── last.pt                  # 最新模型
├── latest_checkpoint.pth    # 完整 checkpoint（含优化器状态，用于 resume）
├── history.json             # 每轮详细训练记录
├── training_summary.json    # 最新评估摘要
├── training_curve.png       # 训练曲线图
└── tensorboard/             # TensorBoard 日志
```

查看训练曲线：

```bash
tensorboard --logdir models/next_state_v3/tensorboard
```

### 关键评估指标

| 指标 | 含义 |
|------|------|
| **Eval/AvgLines** | AI 平均每局消行数（核心指标，越高越好） |
| **Eval/ClearRate** | 至少消过行对局比例 |
| **Eval/AvgSteps** | 平均存活步数 |
| **Train/Loss** | 网络训练损失（应持续下降） |

## 项目结构

```
Tetris_Python/
├── main.py                      # 程序入口
├── settings.py                  # 全局配置中心（GameConfig）
├── requirements.txt             # 运行时依赖
├── tetris.spec                  # PyInstaller 打包配置
├── background.png               # 背景图资源
├── 演示视频.mp4                  # 项目演示视频
│
├── src/
│   ├── game/                    # 核心游戏逻辑
│   │   ├── tetromino.py         #   方块定义与旋转
│   │   ├── piece_sequence.py    #   7-bag 发牌序列
│   │   ├── game_core.py         #   对战核心逻辑
│   │   ├── shared_game_core.py  #   Arena 同盘逻辑
│   │   ├── player_input.py      #   玩家输入处理
│   │   └── game_modes.py        #   模式/难度定义
│   │
│   ├── render/                  # 渲染 & 视觉效果
│   │   ├── render.py            #   棋盘渲染引擎
│   │   ├── background.py        #   动态背景
│   │   ├── menu_screens.py      #   菜单界面
│   │   ├── ui_fonts.py          #   字体管理
│   │   ├── ui_primitives.py     #   共享绘制原语
│   │   └── effects.py           #   视觉特效
│   │
│   ├── ai/                      # AI & 强化学习
│   │   ├── ai_controller.py     #   AI 控制器（启发式 + 模型）
│   │   ├── dqn_model.py         #   DQN 模型加载
│   │   ├── deep_q_network.py    #   神经网络定义
│   │   ├── next_state_features.py  # 特征提取
│   │   ├── replay_memory.py     #   经验回放
│   │   ├── rl_trainer.py        #   RL 训练器
│   │   ├── tetris_env.py        #   RL 环境（对战）
│   │   ├── shared_tetris_env.py #   RL 环境（Arena）
│   │   ├── shared_arena_model.py   # Arena AI 模型
│   │   └── model_paths.py       #   模型路径工具
│   │
│   └── app/                     # 应用层
│       ├── app_controller.py    #   主控制器
│       ├── versus_match.py      #   人机对战会话
│       └── shared_arena_match.py   # Arena 对战会话
│
├── scripts/                     # 独立脚本
│   ├── train.py                 #   离散动作 DQN 训练
│   ├── train_nextstate.py       #   Next-state 落点评分训练（主用）
│   ├── evaluate_model.py        #   模型评估
│   ├── test_load.py             #   模型加载测试
│   └── test_shared_load.py      #   Arena 导入测试
│
└── models/                      # 预训练模型（DQN 权重）
```

## 配置

所有可调参数集中在 `settings.py` 的 `GameConfig` 中，支持环境变量覆盖：

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `TETRIS_AI_MODE` | AI 模式 (heuristic/model) | heuristic |
| `TETRIS_AI_MODEL_PATH` | 模型路径 | models/next_state_v3/best.pt |
| `TETRIS_AI_DEVICE` | 推理设备 | cpu |
| `TETRIS_ARENA_FALL_MS` | Arena 下落速度 | 500 |
| `TETRIS_AREA_SCALE` | 单元格缩放 | 1.6 |
| `TETRIS_FPS` | 帧率 | 60 |
| `TETRIS_RL_ENABLED` | 启用 RL 训练 | false |
| `TETRIS_RL_EPISODES` | 训练回合数 | 300 |

对战参数（攻击表、Combo 奖励、陷阱消耗/冷却等）参见 `settings.py` 中的 `GameConfig`。

## 对战规则

1. **同步公平**：玩家与 AI 使用同步 7-bag 序列，相同基础重力
2. **攻击与抵消**：消行攻击先抵消待接收垃圾行，余量发送给对手
3. **攻击表**：1 行→0 / 2 行→1 / 3 行→2 / 4 行→4（可配置）
4. **陷阱机制**：消行充能 → 按 C 释放 → 限制对手垃圾行洞位分布

## License

仅用于学习与练习，可按需扩展。
