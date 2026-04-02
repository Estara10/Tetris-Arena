# Tetris Python AI Versus

基于 Pygame 的俄罗斯方块对战项目，当前版本已支持：
- 同步 7-bag 发牌的人机对战
- 垃圾行攻击与抵消
- 陷阱能量与洞位限制机制
- AI 双模式（启发式基线 / 模型推理）
- 为强化学习训练准备的参数集中管理

## 当前功能

- 经典 7 种方块（I/J/L/O/S/T/Z）
- 方块移动、旋转、软降、硬降
- 碰撞检测、锁定、消行、计分
- 预测落点（Ghost Piece）
- 下一块预览
- 左右双棋盘对战（玩家 vs AI）
- Classic / Challenge 模式
- Arena 同盘竞技模式（玩家和 AI 在同一棋盘）
- AI 启发式一层前瞻
- 可选模型推理模式（如果检测到模型和 torch）

## 对战规则（本版）

说明：若按统一规则进行对战，请使用 Arena 模式（按 3 进入），该模式采用 25x25 与统一按键映射，并支持高频可调节节奏。

### 1) 同步公平

- 玩家和 AI 使用同步 7-bag 序列
- 双方同一基础重力规则

### 2) 攻击与抵消

- 消行会按规则转换成攻击行
- 攻击先抵消自己待接收的来袭垃圾，再把剩余攻击发送给对手
- 未消行锁定时会结算来袭垃圾（每次有上限）

默认攻击表在配置中：
- 1 行: 0
- 2 行: 1
- 3 行: 2
- 4 行: 4

### 3) 陷阱机制

- 通过消行累积陷阱能量
- 玩家按 C 主动释放陷阱（有能量消耗与冷却）
- 陷阱会限制后续攻击垃圾的洞位分布（左/中/右段）
- 双方有预警提示文本

### 4) 同盘竞技（Arena）

- 玩家与 AI 在同一棋盘同时下落
- 活动方块之间会发生碰撞阻挡（不允许互相穿透）
- 棋盘固定为 25x25
- 高频推进，默认约 0.18 秒下落 1 格，控制周期约 0.016 秒
- AI 与玩家采用重力相位错开，减少同高同拍导致的持续互挤
- 出生位固定：玩家 x=9，AI x=17
- 计分规则：谁完成“最后一格落地锁定”谁得 1 分
- 规定时间结束后，得分高者获胜
- 紧挨碰撞规则：同向时双方不变向；异向时左右传导给对方（若对方已靠墙则不传导）

## 控制说明

### 玩家操作（统一）

- A: 左移
- D: 右移
- S: 直接到底
- L: 变换
- W: 暂停

### 全局操作

- P: 暂停 / 恢复（兼容）
- R: 重开当前对局
- M: 返回菜单
- Q / Esc: 退出

## 运行方式

安装依赖：

```bash
pip install pygame
```

运行：

```bash
python3 main.py
```

## AI 双模式

### 启发式模式（默认）

- 不依赖 torch
- 使用落点评分 + next_piece 一层前瞻

### 模型推理模式

通过环境变量启用：

```bash
export TETRIS_AI_MODE=model
export TETRIS_AI_MODEL_PATH=models/tetris_dqn.pt
python3 main.py
```

说明：
- 若未安装 torch，或模型文件不存在/加载失败，会自动回退到启发式模式。

## 模式选择

- 按 1 进入 Classic
- 按 2 进入 Challenge
- 按 3 进入 Arena（同盘竞技）

## 配置中心

所有主要参数集中在 settings.py 的 GameConfig：

- 对战参数：
  - versus_attack_mapping
  - versus_combo_bonus
  - versus_b2b_bonus
  - versus_garbage_cap_per_lock
  - versus_garbage_apply_on_nonclear
  - versus_trap_energy_cost
  - versus_trap_cooldown_ms
  - versus_trap_forced_lines
- AI 参数：
  - ai_controller_mode
  - ai_model_path
  - ai_model_device
  - ai_model_action_interval_ms
  - ai_model_reaction_delay_ms
- RL 训练参数：
  - rl_enabled
  - rl_gamma
  - rl_batch_size
  - rl_replay_capacity
  - rl_target_sync_interval
  - rl_learning_rate
  - rl_epsilon_start / rl_epsilon_end / rl_epsilon_decay_steps

## 强化学习接入说明（当前阶段）

本仓库已先完成规则侧改造与参数收口，训练侧建议按下列模块推进：

- tetris_env.py: reset/step 封装（已提供）
- dqn_model.py: Q 网络定义（已提供）
- replay_memory.py: 经验回放（已提供）
- rl_trainer.py: 训练与评估循环（已提供）

当前环境接口示例：

```python
from tetris_env import TetrisEnv

env = TetrisEnv(step_dt_ms=90)
state = env.reset(seed=42)

done = False
while not done:
  action = env.sample_action()  # 后续可替换为模型动作
  outcome = env.step(action)
  state = outcome.state
  done = outcome.done
```

训练命令示例：

```bash
pip install torch
export TETRIS_RL_WARMUP=200
export TETRIS_RL_EVAL_EPISODES=20
export TETRIS_RL_CURRIC_EPISODES=160
python3 rl_trainer.py --episodes 300 --seed 42 --opponent-mode heuristic --device cuda
```

训练产物默认写入 models 目录：
- last.pt
- best.pt
- training_history.json

建议评估指标：
- 平均存活步数
- 平均消行数
- 平均承受垃圾行数
- 固定种子集上的胜率

## 目录概览

```text
Tetris_Python/
├── app_controller.py
├── main.py
├── settings.py
├── game_core.py
├── shared_game_core.py
├── versus_match.py
├── ai_controller.py
├── tetris_env.py
├── dqn_model.py
├── replay_memory.py
├── rl_trainer.py
├── piece_sequence.py
├── tetromino.py
├── render.py
├── background.py
├── menu_screens.py
├── game_modes.py
└── player_input.py
```

## License

仅用于学习与练习，可按需继续扩展。
