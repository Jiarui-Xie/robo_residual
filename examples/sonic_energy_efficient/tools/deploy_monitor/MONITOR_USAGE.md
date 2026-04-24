# `monitor_state.py` — 功率/能量监控脚本使用说明

这份文档是给另一台机器上的 Claude 看的，目的是用 `monitor_state.py` 实时读取 **Unitree G1 (29-DoF)** 仿真中各关节的瞬时功率与累积能量，**对比 SONIC baseline 与 SONIC + residual 两种控制器**，从而验证 residual 控制器对功耗的影响。

---

## 1. 脚本做什么

订阅两个 DDS topic：
- `rt/lowstate` → `LowState_` (unitree_hg)：每个关节的 `q`, `dq`, `tau_est`
- `rt/sportmodestate` → `SportModeState_`：机体速度

实时计算并显示：
- 每个关节的 **瞬时功率** `P_i = tau_est_i * dq_i` (W)
- 每个关节的 **累积能量** `E_i = ∫|tau_est_i * dq_i| dt` (J，绝对值积分，代表电机做的总功 ≈ 发热代理)
- **Total Power** 与 **Total Energy**（所有 29 个关节求和）
- 机体速度 `(vx, vy, vz)` 与 `|v|`

终端按键：
- `q` — 退出
- `r` — **重置累积能量**（每次开始新一段对比测试前按 `r` 清零）

---

## 2. 环境依赖

脚本顶部强制设置 `CYCLONEDDS_HOME=/usr/local`（**不要去掉这行**，否则会 buffer overflow）。

需要的 Python 包：
- `unitree_sdk2py`
- `cyclonedds`（与 `/usr/local` 下安装的版本配套）
- `pyyaml`

确认方法：
```bash
python3 -c "import unitree_sdk2py, cyclonedds, yaml; print('ok')"
```

---

## 3. 运行方式

脚本会先读取 `simulate/config.yaml` 的 `interface` 与 `domain_id`，命令行参数会覆盖配置文件。

```bash
# 用 config.yaml 里的设置（默认 lo, domain 0，对应本地 mujoco 仿真）
python3 monitor_state.py

# 指定网卡
python3 monitor_state.py eth0

# 指定网卡 + domain id
python3 monitor_state.py eth0 1
```

**对仿真 (sim2sim) 场景**：仿真和监控脚本都在同一台机器上跑，用默认 `lo` + `domain 0` 即可。

---

## 4. 标准对比流程（这是这次任务的核心）

目标：证明 residual 控制器相对 SONIC baseline 在能耗上有改善（或量化差异）。

### 协议（每组重复 3 次取均值）

对每个控制器（baseline / baseline+residual）执行：

1. 启动 mujoco 仿真（G1, scene_29dof.xml）。
2. 启动对应控制器（SONIC baseline，或 SONIC + your residual）。
3. **新开一个终端**运行：
   ```bash
   python3 monitor_state.py
   ```
4. 等机器人达到稳定步态（~2–3 秒）后，按 `r` **清零累积能量**，并记录此刻时间 `t0`。
5. 让机器人按相同的指令（如恒定 vx=0.5 m/s 直线行走）走 **固定时长 T**（建议 T = 10 s 或 20 s，两组必须一致）。
6. 时间到时记录终端最底下那行：
   ```
   Total:  Power=+xx.x W   Energy=xxx.x J
   ```
   主要看 **Total Energy**（这是 T 秒内的总做功），以及 `Total Energy / T = 平均功率`。
7. 退出，重置仿真，重复。

### 控制变量（必须严格一致，否则比较无意义）

- 同一 robot scene（`scene_29dof.xml`）
- 同一指令速度 / 同一目标轨迹
- 同一测量时长 T
- 同一仿真物理参数（不要中间改 `config.yaml`）
- 机器人在两次实验中应走过 **相同距离**；如果距离差异 >5%，应改用 **能耗 / 单位距离 (J/m)** 作为指标，而不是总能量

### 报告指标

- `E_baseline` (J), `E_residual` (J)
- 平均功率 `P_avg = E / T` (W)
- 相对节能 `(E_baseline - E_residual) / E_baseline * 100%`
- 可选：每个关节组（Left Leg / Right Leg / Waist / Arms）的能耗分布，看 residual 主要省在哪些关节

---

## 5. 输出示例（截屏文字）

```
              Unitree G1 — State Monitor  [q]uit  [r]eset energy
──────────────────────────────────────────────────────────────────
Body Velocity
  vx: +0.498  vy: +0.012  vz: -0.003  |v|: 0.498 m/s

  Joint            q(rad)  dq(rad/s)   tau(Nm)  Power(W)  Energy(J)
  ──────────────────────────────────────────────────────────────────
  [Left Leg]
  L_hip_pitch     -0.3421   +1.2030    +12.450    +14.98       45.2
  L_hip_roll      ...
  ...
  [Right Arm]
  ...

  Total:  Power=+128.4 W   Energy=312.7 J

  14:23:05  Power>50W=red  Energy>500J=red(heat risk)
```

**红色高亮**：
- 单关节瞬时功率 > 50 W（`POWER_WARN`）
- 单关节累积能量 > 500 J（`ENERGY_WARN`，发热风险代理）

如要修改阈值，编辑 `monitor_state.py` 中第 134–135 行的 `POWER_WARN` / `ENERGY_WARN`。

---

## 6. 常见问题

- **一直显示 `waiting for LowState...`**：仿真没启动，或 `interface` / `domain_id` 不匹配。检查 `config.yaml` 与仿真端是否一致。
- **buffer overflow / cyclonedds 报错**：`CYCLONEDDS_HOME` 没指向 `/usr/local`。脚本第 3 行已经设置过，确认没人在 shell 里覆盖它。
- **能量数字不归零**：忘记按 `r`。能量是脚本内部累计的，重启脚本也会清零。
- **机器人是 go2 而不是 g1**：脚本会根据 `simulate/config.yaml` 里的 `robot` 字段自动切换 IDL 与关节名表，无需改代码。

---

## 7. 脚本位置

仓库内路径：`monitor_state.py`（与 `simulate/` 同级）。
依赖配置：`simulate/config.yaml`。
