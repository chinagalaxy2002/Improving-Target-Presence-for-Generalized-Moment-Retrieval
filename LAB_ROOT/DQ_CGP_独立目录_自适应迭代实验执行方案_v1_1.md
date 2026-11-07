# DQ-CGPv3 → SI-Bind：自适应、事件驱动的实验与验证方案（Isolated Lab / Agent Execution Protocol v1.1）

> **定位**：可直接交给代码/实验代理执行的研究实施协议。本文定义新模块设计、代码改造、单 seed 从头训练、两卡调度、即时反馈、动态迭代、严格验证、证据归档与最终论文产出。本文不是已完成实验的结果，不代表任何新结构已经验证有效。  
> **研究基线**：只读参考仓库 `chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval` 的 `dia` 分支。**禁止在原项目中创建分支、工作树、文件或任何实验产物**。在全新 `LAB_ROOT` 内克隆/复制代码、实现新模块并完成所有训练、验证、测试、诊断和论文制图；历史 checkpoint / 数据 / 特征仅作为只读输入。  
> **默认 GPU**：2 × NVIDIA RTX 3090 24GB；每卡**最多**并行 3 个训练进程，允许因显存/稳定性下降为 2 或 1 个。最大总并发为 6，不代表无条件占满。  
> **最高优先级：单根目录写入隔离**：只允许写入 `LAB_ROOT` 及其真实子目录；严禁修改原仓库、原有 `configs/`、`src/`、`diagnosed/`、历史 checkpoints、数据/特征、系统环境和用户主目录缓存。任何潜在越界写入立即中止、记录并报告。  
> **固定原则**：**一个 seed（2023），不做多 seed；所有新结构从头训练 GMR；已有 Baseline 与 DQ-CGPv3 检查点仅作参考和诊断；不修改原生 Self-Attention，不直接复制 Query Dropout，不改原生 Hungarian one-to-one；测试集不参加循环调参。**

---

## 0. 交给实验代理的执行指令

代理接收本文件后，应按以下顺序行动：

1. **读取真实仓库并验证条件**：确认 `dia` HEAD commit、`docs/checkpoints_and_paths.md`、`src/soccergmr/{config,cli,train,test,checkpoint,predict}.py`、`modules/dq_cgp.py`、八骨干 adapter、`diagnosed/`。以真实代码为准；本文提到的新文件、新命令均为**待实现**，不可声称已经存在。
2. **全新单根目录隔离**：先执行 §0A：创建此前不存在的 `LAB_ROOT`；在它里面独立克隆 `dia` 源码、建立版本快照和运行目录。**不得对原仓库执行 git checkout/branch/worktree/commit、不得原地编辑任意现有代码**。新实验采用隔离副本 Git SHA + 配置指纹 + 资产指纹 + checkpoint SHA 的完整身份追踪。
3. **只在 `LAB_ROOT/revisions/` 下做工程最小闭环**：实现精简 Binding、损失解耦、训练期 Slot 使用率统计、诊断新 checkpoint 的适配能力；在 1–2 个 batch 上做前/反向、负样本、多 GT、checkpoint 恢复与评测校验。**没有通过 Gate-0 不得启动正式训练。**
4. **全部训练、验证、测试写入 `LAB_ROOT/runs/`，立即开跑首批 6 个任务**（§7）；按各任务完成事件即刻读取 Val 指标及 Val 诊断，更新研究登记册，为腾出的槽位安排新任务；**不用等六个任务全部完成**。
5. **闭环迭代**：遵循 §8–§10 的实验分支决策，允许提出新的单因素变体；**一次只改变一个可解释机制**，每次新的结构/损失/训练协议都必须产生独立的 run ID 和新的从头训练。
6. **坚持验证边界**：Standard Train 用于拟合，Standard Val 用于结构选择与即时反馈；Standard Test 和 Full-Decodable Test 仅对**冻结入围版本**执行，绝不依据 Test 结果继续调参。日志、bug 修复、硬件 OOM 与科学结果必须分开记录。
7. **结束条件**：当候选不再改善 Val 上的有效独立覆盖/多 GT 找齐/任务主指标、预算触顶或用户终止时，冻结最终候选，输出科学结论、负面结果、消融和复现包。不得为证明某机制有效而改写统计。

**代理每次报告首先声明 `LAB_ROOT`、隔离代码版本和越界写入审计结果；此外必须说清**：刚完成哪个实验、该实验用的哪版代码/配置、真实数字、是否通过门槛、否定了/支持了哪个假设、下一轮拟执行的最少实验、GPU 槽位分配，以及可复现文件位置。

---

## 0A. 绝对写入隔离：全新实验根目录（最高优先级；覆盖其他章节）

> **强制性边界**：用户要求“全新目录中进行实验训练和测试，不得改动其他地方的任何代码”。本文中凡提及“修改/新增 `src/`、`configs/`、`diagnosed/`、`tests/`、`train.py` 或 vendor 文件”，**全部严格指 `LAB_ROOT/revisions/<revision>/` 内的独立副本**。不能把这条约束理解成仅把 `runs/` 放到新位置，却仍在原项目中修改代码。

### 0A.1 唯一写入根目录、只读输入、明令禁止的行为

建议在服务器创建一个此前不存在的目录：

```bash
export LAB_ROOT=/home/guoxiangyu/SoccerGMR_SI_Bind_IsolatedLab
# 首次启动必须确保不存在，若已存在且不属于本次实验：停止，改用另一个新名称
if [ -e "$LAB_ROOT" ]; then
  echo "LAB_ROOT 已存在，禁止覆盖或混用" >&2
  exit 1
fi
mkdir -p "$LAB_ROOT"/{source_mirror,revisions,runs,agent_runs,scripts,logs,cache,tmp,envs,audits,artifacts}
```

以上路径仅为**建议新建位置**；代理必须在创建之前使用 `realpath` 检查 `LAB_ROOT` 不等于、也不位于原仓库、历史检查点或数据/特征根目录内部，且不存在符号链接越界。用户若另行指定新根目录，以用户设置为准。**启动前不得预先清理、覆盖任何同名旧目录**。

| 目录/资源 | 权限 | 具体约束 |
|---|---|---|
| `LAB_ROOT/source_mirror` | 只在新目录内创建；创建后只读 | 从 `dia` 获取源码的独立副本；不在原仓库建工作树 |
| `LAB_ROOT/revisions/rev_xxxx` | 新目录内可修改；训练时冻结 | 每个代码版本独立完整源码，含必要 vendor 与 configs；所有代码改动只发生这里 |
| `LAB_ROOT/runs/` | 可写 | 所有 GMR 训练、Val、Test、诊断、检查点、日志、图表及显存记录 |
| `LAB_ROOT/agent_runs/` | 可写 | 队列、研究代理状态、失败分析、实验决策记录 |
| `LAB_ROOT/cache/`、`tmp/`、`envs/` | 可写 | pip/PyTorch/HF/CUDA/JIT/Matplotlib/临时文件/虚拟环境；不写 `~/.cache` 或系统 Python |
| 原项目任意目录（包含 `.git`） | **只读** | 不改源码/配置/文档/诊断结果；不在此 `git commit/checkout/worktree/clean/reset`；不写 `__pycache__` |
| `docs/checkpoints_and_paths.md` 列出的旧 checkpoint | **只读** | 只允许校验 SHA、加载和计算诊断；不保存替换、重命名、修改、重新选模 |
| Standard/Full 标签及所有原有特征目录 | **只读** | 读取原始资产；任何生成的缓存、索引、加工特征均写入 `LAB_ROOT/cache` |
| GitHub 原始远程仓库 | **只读** | 此次实验不 push，不创建 PR/分支/提交，不改远程文件；如未来需要提交，由用户另行授权 |

> 代理的**写权限白名单只有 `realpath(LAB_ROOT)` 及其子路径**。发现写目标是绝对路径、符号链接、环境默认缓存或脚本隐式导出时，都应先规范化路径再判断；如果无法证明在白名单内，**fail closed：拒绝执行**。禁止用 sudo、全局 pip install、系统配置修改或进程级修改用户其他项目来解决问题。

### 0A.2 独立获取原始源码：不使用关联原项目的 Git 工作树

必须使用**独立克隆**（`git clone` 在 `LAB_ROOT` 内产生它自己的 `.git`），或把只读源码完整复制到新目录。**不得使用** `git -C 原项目 worktree add ...`，因为它仍会修改原项目 `.git` 元数据。

```bash
# 请在已创建的 LAB_ROOT 中操作。source_mirror 必须为空。
rmdir "$LAB_ROOT/source_mirror"
git clone --branch dia --single-branch \
  https://github.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval.git \
  "$LAB_ROOT/source_mirror"

# 若服务器不能访问远程 GitHub，可替换为：
# git clone --no-hardlinks --branch dia "$ORIGINAL_REPO" "$LAB_ROOT/source_mirror"
# 此处 ORIGINAL_REPO 仅作为读取来源；切勿执行原仓库写操作。

# 在新的实验根目录内创建第一个可编辑且独立的版本副本
git clone --no-hardlinks "$LAB_ROOT/source_mirror" "$LAB_ROOT/revisions/rev_0001"
cd "$LAB_ROOT/revisions/rev_0001"
git switch dia 2>/dev/null || git switch -c dia --track origin/dia
# 此处之后的改动、提交、测试只能作用于该新克隆，不可 push 至原仓库。
```

如果 `git switch` 没有现成 `dia` 本地分支，则应先验证当前 `HEAD` 对应镜像的 `dia` commit，再在**新克隆内部**创建本地分支。原仓库的未提交修改**不会自动包含在远程 `dia`**；若确需对照，应先做原仓库只读差异审计，在 `LAB_ROOT` 内生成显式补丁，不能直接从原仓库改写。

之后的 `rev_0002` 等版本，也从 **`LAB_ROOT` 内已冻结的源码快照**完整复制/独立克隆得到。每个正在训练的 revision 永久冻结；**不能修改一个被任意训练或评测进程使用的 revision**。若代码有 bug，创建下一版修复并明确记录哪些训练失效、是否需要同配置重训。

### 0A.3 推荐的唯一目录结构

```text
/home/guoxiangyu/SoccerGMR_SI_Bind_IsolatedLab/   <-- LAB_ROOT，唯一可写根
├── source_mirror/                   # dia 原始代码镜像；不改
├── revisions/
│   ├── rev_0001/                   # 完整可运行项目（src/configs/diagnosed/tests）
│   ├── rev_0002/                   # 新机制/bug 修复的独立源码版本
│   └── ...
├── runs/
│   ├── rev_0001/moment_detr/E1/{gmr,val_diagnosis,test}/
│   ├── rev_0001/moment_detr/E2/{gmr,val_diagnosis,test}/
│   └── ...
├── agent_runs/{queue,registry,reports,status}/
├── scripts/                         # 只在此放启动器、越界校验器和队列调度
├── cache/{pycache,torch,hf,cuda,pip,matplotlib,triton}/
├── tmp/                             # 各进程子目录
├── envs/                            # 若确需新环境，只在此新建
├── logs/                            # stdout/stderr/scheduler/GPU 监控
├── audits/                          # 隔离前后源文件/旧 checkpoint 不变性清单
└── artifacts/                       # 论文图、汇总表、模型对比与最终报告
```

**禁止在 `source_mirror/` 写实验代码或输出**；它只用于证明实验起点。建议每个 revision 所有实验配置也位于该 revision 的 `configs/experiments/interlayer/`。不允许把新 `configs/*.json` 写回原项目。

### 0A.4 新配置必须去除指向原项目源码的绝对路径

现有 `configs/experiments/*.json` 的 `architecture` 字段可能指向 `/home/guoxiangyu/SoccerGMR-DQ-CGPv3/configs/backbones/*.json`。**不能直接复用这些硬编码配置开展新训练**：否则训练时可能误读旧架构选项。

每个 `rev_xxxx` 内新建实验配置，并将 `architecture` 明确写成同一 revision 内的文件路径（由配置生成器以相对路径或 `realpath` 绝对路径生成）。标签和特征的原有绝对路径可以**作为只读输入保留**。输出 `--output`、诊断 `--output-dir`、日志、checkpoint 路径一律位于 `LAB_ROOT`。

**新配置生成器要打印并断言**：`resolved_architecture_path` 属于当前 revision；所有可写输出路径属于 `LAB_ROOT`；`data.splits/features` 只读；旧 checkpoint 只读；所有新 checkpoint 指向当前 run 的 `gmr/`。不得通过原项目相同文件名的“相对路径”误写旧项目。

### 0A.5 运行时隐藏写入：缓存、Python 字节码和临时文件

代理必须为每个进程准备 `LAB_ROOT/tmp/<run_id>` 和缓存/运行主目录，**只在实验进程环境内**设置：

```bash
export RUN_ID=rev_0001_moment_detr_E2
export LAB_ROOT=/home/guoxiangyu/SoccerGMR_SI_Bind_IsolatedLab
mkdir -p "$LAB_ROOT/tmp/$RUN_ID" "$LAB_ROOT/cache/$RUN_ID" "$LAB_ROOT/envs/runtime_home/$RUN_ID"
export TMPDIR="$LAB_ROOT/tmp/$RUN_ID"
export TEMP="$TMPDIR" TMP="$TMPDIR"
export HOME="$LAB_ROOT/envs/runtime_home/$RUN_ID"
export XDG_CACHE_HOME="$LAB_ROOT/cache/$RUN_ID/xdg"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$LAB_ROOT/cache/$RUN_ID/pycache"
export TORCH_HOME="$LAB_ROOT/cache/$RUN_ID/torch"
export HF_HOME="$LAB_ROOT/cache/$RUN_ID/hf"
export PIP_CACHE_DIR="$LAB_ROOT/cache/$RUN_ID/pip"
export MPLCONFIGDIR="$LAB_ROOT/cache/$RUN_ID/matplotlib"
export CUDA_CACHE_PATH="$LAB_ROOT/cache/$RUN_ID/cuda"
export TRITON_CACHE_DIR="$LAB_ROOT/cache/$RUN_ID/triton"
export TORCH_EXTENSIONS_DIR="$LAB_ROOT/cache/$RUN_ID/torch_extensions"
export WANDB_DIR="$LAB_ROOT/logs/$RUN_ID/wandb"
export WANDB_MODE=disabled
```

每个任务使用独立 `RUN_ID`，避免 6 个作业抢写同一缓存。原有 conda/Python/CUDA 环境可以**只读使用**；如果必须安装新依赖，只能新建 `LAB_ROOT/envs/` 中的虚拟环境，不得修改系统或原项目环境。个别库可能有额外缓存变量；启动器必须扫描并重定向，不允许忽略。宿主机的 GPU 运行时内部行为未必完全受环境变量控制；若要求操作系统层面保证不写其他目录，需要在服务器使用只允许写 `LAB_ROOT` 的容器/挂载沙箱，不能将普通 shell 约定称为绝对物理隔离。

### 0A.6 运行前后审计与强制停止条件

**Gate-I（目录隔离审计）必须先于原 Gate-0：**

1. 核对 `LAB_ROOT` 是全新、非符号链接、非旧项目/旧 checkpoint/数据的子目录；写入白名单仅 `LAB_ROOT`。
2. 在 `LAB_ROOT/audits/` 保存原仓库所有 `.py/.json/.md/.sh/.yaml/.toml` 等源码与配置文件的**只读 SHA-256 清单**，加旧 checkpoint 的文件 SHA-256 清单（允许复用原已核实的完整 hash 再抽样核验）；必要时保存原有数据/特征目录文件数量与元数据清单。**清单文件只写在 LAB_ROOT 内**。
3. 对所有命令的工作目录、配置路径、日志重定向、`--output`、`--output-dir`、临时文件路径、符号链接和可疑默认缓存做规范化路径检查；一旦写路径不在白名单，拒绝启动。
4. 用新目录源码执行真实 batch 和最小 Test 流程，检查 `strace`/文件系统监控（若可用）中的写行为，或使用容器只读挂载旧目录。不得把缺少监控解释为已经验证“零越界写入”。
5. 每次创建 revision、每轮正式训练完成、最终结题时，重复只读哈希审计并比较；任何旧代码/旧 checkpoint 哈希变化都必须立即停止队列、封存日志、报告差异和恢复方案，**不得静默修复或覆盖旧文件**。

**禁止清理旧项目（即使只是 `git clean`、删除临时文件或覆盖 `best.pt`）；禁止软链接写入到旧目录；禁止从新项目用 `git push` 修改原远程。**

---

## 1. 科学问题、现有事实与不可越界的解释

### 1.1 五项诊断结论（作为出发点，不是新实验的结论）

- **Multi-GT 是真实任务**：Standard Test 160/1,036 查询为多 GT，Full-Decodable Test 567/2,970；独立事件检索不能仅追求 Top-1。
- **Top-1 固定索引集中**：Standard Test 的 Baseline，Moment-DETR 和 BM-DETR 的 Top-2 Slot 贡献 100% Top-1，UVCOM 为 98.2%；但这几个模型的 Top-2 Slot 独立命中占比分别约 39.8%、40.9%、16.3%。**这不是其他 Slot 全部失活的证据。**
- **候选冗余与全候选覆盖不足并存**：UVCOM/Sim-DETR 有较多未增加 GT 独立覆盖的冗余命中；Moment-DETR/BM-DETR 首要问题是合格候选不足。
- **候选生成/排名分离**：Moment、BM 有较大 All-K 缺失；UVCOM 具明显 Top-5 选择遗漏。不同骨干不应强行用同一失效故事解释。
- **层间退化**：Moment-DETR 标准集 Baseline 的 D1→D2，独立匹配数 H 从 321 降为 266；GT 可覆盖集合 |S| 净变化 −61；原版 DQ-CGPv3 提升至 389→416、|S| 净变化 +33。**H 的净变化（−55→+27）与 |S| 的净变化（−61→+33）不可混用。**

数据参照：`diagnosed/tables/` 下的 `table_a/b/c/d_*`、`comparison_multigt_coverage_*`、`comparison_layer_evolution_*`。原结果基于 tIoU≥0.4、全 K 候选及一对一最大匹配。若新结果与旧报告有不一致，优先排查严格加载、特征及打分；保留差异。

### 1.2 论文启发及我们的区别

参考用户提供的 *Beyond Caption-Based Queries for Video Moment Retrieval*，尤其第 6–8 页及其 Table 5–8：单独移除 Decoder Self-Attention 或初始 Query Dropout 的效果有限；联合可能增加活跃 Query；但单纯 One-to-Many 激活往往造成重复预测。**抽取思想：减少固定索引长期优势，减少不必要的 Slot 协调依赖，同时保持一对一的事件级有效分工。**

**禁止照搬**：
- 不在原生 Decoder 里直接删除 SA；
- 不直接将原生可学习 Query Embedding 以 0.25 概率置零；
- 不以让 Top-1 接近均匀分布为优化目标；
- 不改用 One-to-Many/Group Matching 作为默认方案；
- 不用 NMS 掩盖原生重复预测；
- 不因为现有数据改善就宣称“解决了 SA 抑制”的因果机制。

### 1.3 主假设与可证伪预测

| 假设 | 可被验证的预期 | 反证/否定条件 |
|---|---|---|
| H1：完整 DQ-CGPv3 含冗余组件 | 去掉 RCG/BPS/FRF 后 E2 在 Val 任务效果接近或超过 R1，参数/推理成本更低 | E2 显著下降且原因不是实现问题 |
| H2：Binding 是主要有效部分 | E1/E2 表现良好；关闭 Binding 的 E7 下降 | 关闭 Binding 无明显损失，甚至更好 |
| H3：固定索引的层间惯性损害有效覆盖 | 定向层间刷新 E5 优于强度匹配的均匀随机 E4，且独立命中/Top-5 有收益 | E5 只降低 Top-1 集中度，不增加有效覆盖 |
| H4：保留 Slot 私有时间创新有助独立分工 | E3/E6 的多 GT 找齐/独立覆盖上升，冗余相对不恶化 | 只有冗余预测增多，或 GT 找齐下降 |
| H5：不同 Decoder 层需要不同干预 | 只有在后续层退化被确认时，E8 多层共享插入能带来增益 | 相比单处插入无效或破坏深层命中 |

**不可偷换指标**：索引熵增加 ≠ Query 激活增加；Query 激活增加 ≠ 独立命中增加；冗余减少 ≠ 多事件找齐；Val 上增益 ≠ 已证明因果机制。

---

## 2. 模型的最小改造范围

### 2.1 当前实现已核验的事实

- `src/soccergmr/modules/dq_cgp.py` 目前：每 Slot 时间绑定 → RCG Router → 16 个 Basis Prompt × 6 tokens → FRF → `beta=0.05` 固定残差。
- `compute_query_cgp_loss()` 依赖 `query_cgp_temporal_attention`、`query_cgp_basis_weights`、`query_cgp_video_mask`，包含 Binding 与 Route Entropy；主权重通常 Binding 0.2 / Route 0.01。
- `dq_cgp.py` 的 `beta=0` **提前 return**，不计算注意力。因此**E1 不可直接 `set_beta(0)`**，否则 Binding Loss 无法训练。
- Moment-DETR `moment_transformer.py` 在第一个解码层之后注入 adapter；BM、BAM、UVCOM 的 adapter 有各自接入逻辑；其他骨干也要检查实际插入路径。
- `src/soccergmr/train.py` 统一从头训练，`--stage gmr` 以 Standard Val `brief.mAP` 选择 `best.pt`，默认 seed=2023、max_epochs=200、patience=20。
- `src/soccergmr/test.py` 对本仓库新格式检查点要求 `complete.json`、严格 `state_dict`、训练/测试配置和源码指纹一致；不可在源码改变后使用新源码直接测旧 run。
- `src/soccergmr/predict.py` **只导出排名前 10，且不含原 Query 索引**；不能作为全 K 诊断原始输入。
- `diagnosed/run_backbone_diagnostics.py` 目前只支持旧的 `m1/m2/m3` 检查点映射；`diagnosed/src/model_runner.py` 使用 `load_state_dict(..., strict=False)`，**新实验必须新增严格的通用 checkpoint 加载入口**，不可把旧加载流程直接指向新模型。
- `src/soccergmr/backbones/registry.py` 一个进程只能装载一个 vendor backbone，故每个训练/分析任务独立 Python 进程。

### 2.2 建议在**新目录独立源码副本**中新增/修改的文件

> 下表中的每个相对路径均以 **`LAB_ROOT/revisions/rev_xxxx/` 为根目录**，不是服务器原有的 `src/`、`configs/` 或 `diagnosed/`。执行代理若发现当前 `pwd` 不在独立 revision 下，应拒绝任何修改。所有实验产物则必须位于 `LAB_ROOT/runs/`、`agent_runs/` 或 `artifacts/`。

| 文件 | 工程任务 |
|---|---|
| `src/soccergmr/modules/interlayer_bind.py` **新增** | E1–E8 的轻量 Binding / Slot-Private Innovation / Slot Refresh，统一层间接口 |
| `src/soccergmr/modules/dq_cgp.py` | 保持旧 DQ 路径不变；抽离可复用 Binding Loss，或新增独立 `binding_loss.py` |
| `src/soccergmr/backbones/{loss,bm_adapter,bam_adapter,uvcom_adapter}.py` | 原生一对一匹配结果传入新 Binding；多实验配置驱动，不写死 E1/E2 |
| 各 vendor Decoder 中已有 `interlayer_adapter` 接口 | 接入选择器；只在确实缺少可复用 hook 时修改，并限于最小变更 |
| `src/soccergmr/backbones/registry.py` | 透传架构选项/adapter 类型，确保八骨干一致 |
| `src/soccergmr/train.py` | 训练期使用率统计/EMA、日志和必要的 hook；保持 Val 选模逻辑不变 |
| `src/soccergmr/config.py`、`configs/experiments/interlayer/` **新增配置** | 所有模型因素显式参数化、运行身份可哈希 |
| `diagnosed/run_variant_diagnostics.py` **新增** | 从任意实验 `best.pt` 严格加载，抽取所有原生 Query 和逐层输出 |
| `diagnosed/src/{core,analyzers}.py` | 优先复用精确 tIoU、匹配与原生计数；新审计项作为扩展字段 |
| `tests/test_interlayer_bind.py`、`tests/test_variant_diagnostics.py` **新增** | 形状、梯度、负样本、指标代数闭合、保存/恢复、推理关闭刷新等测试 |
| `agent_runs/{registry.jsonl,queue.yaml,reports/}` **新增目录** | 代理运行账本、待办、决策和每轮结论。与训练产物分离 |

**架构配置要防止暗中失效**：例如 BM/BAM/UVCOM 的现有 adapter 构造器中 DQ-CGP 是硬编码创建的；不能只在 JSON 中写 `use_new_adapter=true` 而不检查构造和参数注册。对每个骨干打印注入模块实际类型、插入层、调用次数、训练参数量、Binding 是否有梯度。

---

## 3. SI-Bind 轻量模块设计（可开关且适合单因素迭代）

设第 l 层输出 `h ∈ [B,K,d]`，视频 memory `v ∈ [B,T,d]`、文本语义 `s ∈ [B,d]`，视频 padding mask 限制有效帧。所有新逻辑位于原生 Decoder 层之间，不改变 SA/CA/FFN。

### 3.1 A：独立时间绑定 + 小型残差

```text
q_i = Wq LN(h_i) + Ws s
k_t = Wk LN(v_t)
a_it = masked_softmax_t( <q_i,k_t> / sqrt(d) )
c_i = Σ_t a_it Wv v_t
r_i = Wo GELU( Wr[ h_i ; c_i ; s ] )
h_i^(+) = h_i + β r_i,    β = 0.05 initially
```

Binding Loss 使用**原生 Hungarian 匹配的正 GT-Slot 对**，计算 GT 有效时间帧上的注意力总质量：

```text
L_bind = - mean_{(i,g) ∈ native matched pairs}
                 log( Σ_{t intersects g} a_it + eps )
```

- 默认 Binding 权重 `0.2`（与旧方案相同）；训练从头开始。
- **E1**：计算 `a_it` 和 `L_bind`，层间状态原样返回 `h`；测试它是否单独通过辅助监督影响上游。须验证梯度流向上游 Decoder/投影层；`r_i` 不参与训练。
- **E2**：计算同一 Binding 和小残差 `β r_i`；完全去掉 RCG、Basis Prompt、FRF、Route Entropy。
- 新模块的 `forward` 必须在 E1 时仍计算注意力，不可沿用旧 `beta=0` fast path。
- 不额外新增 Query 数量；使用视频有效长度和 padding mask；全负样本 `L_bind=0`，但原生存在性/背景损失必须仍有效。

### 3.2 B：Slot-Private Innovation（不改变原生 SA）

```text
c_bar = stop_grad(mean_i c_i)
nov_i = Wn( c_i - c_bar )
h_i^(+) = h_i + β [ r_i + λ_nov nov_i ],   λ_nov = 0.2
```

解释：保留每个 Slot 自己读取的视频时间证据相对共享背景的差异，而非增加 Query-to-Query Attention 或施加硬正交。

- **风险**：若所有 `c_i` 相似，`nov_i≈0`；若噪声占多数，也可能扩大噪声。属于待检验假说，不宣称天然破除重复预测。
- 若 E3 明显无效，**删除此分支**；不要强行叠加多种 diversity loss。必要时只能基于 Val 的 `GT hit crowding`、`H`、多 GT 四态提出一个新机制并单独消融。

### 3.3 C：训练期 Dominance-Aware Interlayer Refresh（不置零初始 Query）

训练正样本中记录原生最终排名 Top-1 的 Slot index，计算每个 epoch/step 的分布，并维护 EMA `u_i`。若某些 Slot 长期高于均匀基线 `1/K`，对其赋予更高的**训练时**刷新概率：

```text
excess_i = max(u_i - 1/K, 0)
p_i = p_max * excess_i / (max_j excess_j + eps)
z_bi ~ Bernoulli(p_i)       # per-sample, per-slot, training only
h_refresh_bi = (1 - α*z_bi)*h_bi + α*z_bi*LN(Wm [c_bi ; s_b])
h_bi^(+) = h_refresh_bi + β [r_bi + λ_nov nov_bi]
```

首轮固定：`p_max=0.25, α=0.25, warmup=5 epochs, EMA decay=0.95`。这些是预设实验值，不是已有最优超参数。`p_i` 应**由已有训练统计（上一迭代/上一 epoch）产生，不用当前 batch 的 GT 直接选择 Slot**；EMA 只使用训练正样本 Top-1 频数。刷新对本次 train batch 的全部样本作用，以免只训练正样本分支，推理时完全禁用刷新。**不得用 Val/Test Top-1 分布更新 EMA。**

- **E4 均匀对照**：每个 Slot 刷新概率等于 E5 同期的平均 `mean_i p_i`（按 epoch 预先记录/同步），保持相同总扰动量及 α、warmup。若 E5 统计为零，E4 也应零，并注明该对比不具辨识力。
- **E5 定向实验**：概率按长期 Top-1 占优频数变化。
- **E6 联合**：E5 + Slot-Private Innovation。
- 若 E5 仅降低 Top-1 集中度、却未提高独立覆盖或降低最终任务性能，不保留；不将“均匀性”写成目标函数。

### 3.4 训练与推理一致性、可复现的技术要求

- EMA 注册为 buffer 或训练状态；在 checkpoint 内保存并恢复。`model.train()` 与 `model.eval()` 下行为不同；Eval 必须**不调用任何随机数采样**。
- 使用独立于文本和 GT 的稳定 query index，数组按原顺序读写；不得重排、删除、隐藏 Query。
- 训练中每次更新 EMA 只能使用训练样本；原生分数来自每个骨干当前 `predict.export()` 同等的排序逻辑，不能在 Sim/UVCOM 等模型统一套 softmax fg 误判。
- 训练阶段负样本不能制造额外 GT；沿用原生 BCE/背景分类/存在性监督。
- 在 E4/E5 中记录 `actual_refresh_count / opportunities` 以验证等强度，不能只比较标称 p 值。
- `beta=0` 身份、E1 恒等返回、E2 架构无 basis、`strict=True` 恢复均须有测试。

---

## 4. 实验 IDs、原则性对比与动态扩展规则

| ID | 模块 | Bind | Innovation B | Refresh C | 目的 | 优先级 |
|---|---|---:|---:|---|---|---|
| R0 | 既有原生 Baseline checkpoint | 无 | 无 | 无 | 外部参照（不重训） | 固定 |
| R1 | 既有完整 DQ-CGPv3 checkpoint | 0.2 | 无 | 无 | 外部参照（不重训） | 固定 |
| E1 | Binding-only，层间完全 identity | 0.2 | 无 | 无 | 验证 Binding 本身 | 核心 |
| E2 | 轻量 Binding + residual | 0.2 | 无 | 无 | 验证“去 basis 的精简” | 核心 |
| E3 | E2+B | 0.2 | 有 | 无 | 验证独立时间创新 | 条件启动 |
| E4 | E2+均匀随机 Refresh | 0.2 | 无 | uniform | E5 的等扰动对照 | 必须在 E5 入围时启动 |
| E5 | E2+占优定向 Refresh | 0.2 | 无 | dominance | 验证索引偏置中断 | 核心 |
| E6 | E2+B+dominance | 0.2 | 有 | dominance | 验证组合是否值得 | 条件启动 |
| E7 | E6−Bind | 0 | 有 | dominance | 验证 Binding 必要性 | E6 入围后 |
| E8 | E6 或最终冠军，多层间共享插入 | 与冠军相同 | 与冠军相同 | 与冠军相同 | 定位 D2→D3 | 仅有层间证据时 |
| Xn | 动态新假设 | 单因素改变 | 显式 | 显式 | 只为解释一个新发现 | 见 §8 |

**禁止未经消融就把 E6 设为必然胜者。** 如果 E2 胜出，则最后模型是 E2；如果 E5 胜出，则可直接采用 E5；若 E3 对多 GT 有好处但负样本明显退化，应先处理负面机制而非直接全量推广。

所有新模型：随机初始化（包括原生 GMR 网络；使用相同预提取输入特征）、固定一个 seed=2023、与相同骨干其他变体相同训练数据与协议。不要从 R0/R1 载入任何预训练 GMR 参数。禁止从上一轮胜者继续 fine-tune 并把它当作“从头训练”结果。

**历史比较提醒**：R0/R1 可能有历史采样长度、原生厂商配置差异；新变体之间的结论优先做严格同代码版本、同协议比较。若需对原生历史模型做严格因果宣称，应先证明旧 checkpoint 与新评估协议的数值可比；无法证明则标注历史参考，必要时在单 seed 下补一个同协议空模块对照，不得默默假设一致。

---

## 5. 训练/评测数据和指标的统一口径

### 5.1 数据与模型选择

- Standard Train：训练（完整正负样本）；Standard Val：每 epoch 的 `brief.mAP` 选模，研究反馈；Standard Test 和 Full-Decodable Test：仅在最终入围且冻结之后运行。
- 原仓库 `src/soccergmr/config.py` 默认 `seed=2023, batch_size=8, accumulation=1, learning_rate=3e-5, weight_decay=1e-4, max_epochs=200, patience=20`。**实验内可调整 GPU 并行度，不可为了 OOM 单独降低某个候选的 batch size 导致不公平。** 若确需修改 batch size，整个同骨干公平对照组一起改，并在新版本中重训；首选降低并行数。
- 不训练 GEQ Stage 2 作为本轮主实验；`GEQ` 和 `NMS` 作为冻结最终 GMR 后的独立对照项，不参与主结构搜索。
- `tIoU≥0.4` 作为主命中阈值；0.3/0.5 只做入围模型稳健性分析，不能根据其 Test 结果二次选参。

### 5.2 五项诊断必须输出**原生整数计数**

一个正样本 video-query 有 M 个 GT、K 个**全部原生 Decoder Slot**。定义 `B_ij = 1[tIoU(p_i,g_j)≥0.4]`：

```text
T = number of candidate slots with any B_ij=1
H = max-cardinality one-to-one bipartite matched GT count
D = T - H                  # 非独立/冗余命中，不等同于严格同事件复制
U = K - T
K = H + D + U             # 必须逐样本、逐模型代数闭合

H_all    = independent matches using all K slots
H_at_k   = independent matches using native-score Top-k slots
missing  = M - H_all
omission_at_k = H_all - H_at_k
selected_at_k = H_at_k
M = missing + omission_at_k + selected_at_k
```

统计 `H@1/3/5/10/All`、`missing`、`omission@5/10`、`T/D/U`、每个 GT 的候选拥挤 `C_j = Σ_i B_ij`、各 query index 的 Top-1 次数和**独立命中次数**、Multi-GT 完整找齐/部分找齐/仅找到一个/零覆盖四态、逐层 `H` 与 `S/Gain/Loss/Keep`。

- `Top-k` 的 k 小于 GT 数时存在结构性预算限制；`Top-1` 不应单独用来断言排序失败。
- 原生排序分数按骨干原有实现，不能跨模型拿统一 0.5 激活阈值评判“死 Query”。
- 负样本 M=0 另外统计 false positive、原生预测分数、存在性门控错误；不要纳入 H/D 的“重复命中率”分母。
- 不使用 NMS、Top10 截断、GEQ 门控进行主诊断。
- 所有整数指标同时输出 Standard Val 以及最终冻结版两个 Test；Val 多 GT 为 90 条（相对小），避免根据 1–2 条查询差异无限调参。
- 表示不确定性时可用**按视频分组的 paired bootstrap**，这不属于多 seed 模型训练；不以单一 bootstrap 检验代替实际整数计数。

### 5.3 每轮自动汇总表的固定字段

```text
run_id, backbone, parent_hypothesis, git_sha, config_sha, seed,
status, gpu_physical_id, start_at, finish_at, best_epoch, best_val_mAP,
val_GT_total, val_H_all, val_H_top5, val_missing, val_omission_top5,
val_multigt_total, val_multigt_full_all, val_multigt_full_top5,
val_multigt_single_all, val_multigt_zero_all,
val_T, val_D, val_U,
val_top1_slot_counts[K], val_independent_hit_slot_counts[K],
val_top2_top1_concentration, val_top2_hit_concentration,
val_layer_H, val_layer_gain_loss_keep,
val_false_positive_count, extra_params, peak_vram_gb, latency_ms,
acceptance_status, rationale, parent_run_id, artifact_dir
```

不得把没有统计的字段填 0；用 `null` 或 `not_available`，说明原因。**缺数据不是负结果。**

---

## 6. Gate-0：正式训练前必须通过的工程验收

代理应写实际自动化测试，至少包括：

1. **结构断言**：E1、E2、E3、E4、E5、E6 的 adapter 真正注册到相应 Decoder；`h_in.shape == h_out.shape`；在指定层确实调用，而不是配置字段无效。
2. **E1 梯度**：`h_out` 与 `h_in` 数值恒等，辅助 Binding 有非零有效梯度传向上游；没有 Router/Basis 参数；负样本全空 GT 不出现 NaN。
3. **E2 梯度**：Binding 与轻量残差梯度都非零；`beta=0.05` 生效；eval 两次完全一致；padding 不进入 softmax。
4. **E4/E5**：Eval/测试不刷新，Train 只对合法 Slot 刷新；E4/E5 实际扰动强度可以匹配；EMA 仅使用 Train，不读取 Val/Test 标签。
5. **不同 Slot 数**：至少 K=10（Moment-DETR）和 K=30（UVCOM）测试；多 GT、单 GT、Null 及不同视频长度测试。
6. **一对一监督完整**：不改变 native matcher；匹配外候选仍有原生背景监督；改变过的 span loss 与 aux loss 应与原生行为对应。
7. **checkpoint**：`last.pt` 含优化器、EMA、RNG、模型、config/hash；中断在 epoch 边界恢复与连续训练一致；变体误加载不能通过 `strict=True`。
8. **真实一批数据**：分别在 Moment-DETR 和 UVCOM 上跑一次完整 forward→loss→backward→optimizer.step 和小批 Val；确认 `best.pt` 可被**同版本源代码**严格加载。
9. **诊断代数**：人工构造单 GT、多 GT、重叠 GT、无 GT、排名倒置等 toy 样本，逐样本满足 `K=H+D+U` 与 `M=missing+omission+selected`。
10. **资源预检**：每个待并发骨干在真实最大长度 batch 上测量显存峰值（含 backward），预留足够余量再按卡安排任务。

Gate-0 失败：**只允许在全新 LAB_ROOT 目录的下一版 revision 中**修复工程问题，**不计作科学实验失败**。修复任何 `.py` 文件都更新代码版本；旧运行不应继续混入新版本的因果对照中。

---

## 7. 两张 3090：六槽位、谁先完成谁补位的事件驱动调度

### 7.1 初始任务六槽排程

| 槽位 | 物理 GPU | 首个实验 | 必要输出 | 首选原因 |
|---|---:|---|---|---|
| GPU0-S1 | 0 | Moment-DETR / E1 | Binding-only Val + 诊断 | 验证监督本身 |
| GPU0-S2 | 0 | Moment-DETR / E2 | Lite-Val + 诊断 | 核心精简模型 |
| GPU0-S3 | 0 | Moment-DETR / E5 | Dominance-Val + 诊断 | 固定赢家干预 |
| GPU1-S1 | 1 | UVCOM / E1 | Binding-only Val + 诊断 | 高 K/偏排序模型 |
| GPU1-S2 | 1 | UVCOM / E2 | Lite-Val + 诊断 | 核心精简模型 |
| GPU1-S3 | 1 | UVCOM / E5 | Dominance-Val + 诊断 | Top-1 固定索引集中 |

**重要：上表是最高并发意向，不是无条件直接拉起六个作业。** 如果单作业峰值显存使同卡三作业无法在 24GB 内稳定运行，代理必须先降至每卡 2 个或 1 个，并在完成作业后继续补位。不要把 OOM 当成方法失败。

### 7.2 调度原则

- 可用物理卡为 0、1；每卡训练进程 `active_count ≤ 3`。
- 每次入队都估算同卡峰值显存总和，至少预留 **2–3GB** 容量给 CUDA 框架、验证时峰值、其他进程；允许实际预留更多。对未知骨干先单任务探测显存。
- 一卡多个任务使用**独立进程**。推荐 `CUDA_VISIBLE_DEVICES=0/1` 隔离，每个进程内部 `--device cuda:0`；恢复时维持相同的可见设备集合，符合仓库 RNG 恢复要求。
- 使用 `OMP_NUM_THREADS=1` / 适当 PyTorch 线程限制，避免 6 个作业 CPU 线程互相抢占；控制 data workers 总量。
- 优先调度**完成实验的直接判别性对照**，而不是随意填充新组合。例如 E5 完成且初步有效，优先排 E4；E2 没有通过工程验证时，不应直接排 E6。
- 每个进程独立 `run_id`、配置和输出目录，不并发写同一个 `best.pt` / `last.pt`。
- 训练中需要 `kill` 或抢占时使用优雅信号，使仓库在 epoch 边界保存 `last.pt`，然后用相同代码、配置、可见设备 `--resume`；禁止把未完成的 `last.pt` 当 `best.pt` 评测。
- 不重复启动已 `complete.json` 并哈希核验完成的任务。若该任务是相同 `model/config/seed/source/data` 的完全重复，优先重用缓存，不消耗训练槽。

### 7.3 事件轮询（代理在服务器中持续运行）

建议间隔 **2–5 分钟**扫描进程与日志（这是服务器代理的执行建议，不代表当前聊天能在后台运行），每个 epoch 结束/失败/完成时即时触发不同事件：

```text
on EPOCH_DONE(run):
  append epoch metrics, loss components, Slot usage and actual refresh counts
  if NaN / corruption / identity fail: BLOCK_AND_DEBUG
  every 5 epochs or when Val best updates: optional light Val diagnostics
  do not select by Test; do not alter active run hyperparameters

on RUN_FINISHED(run):
  verify complete.json, best.pt hash and source/config identity
  run strict-load Val ALL-K diagnostics for best.pt
  assert count identities and sample coverage
  compare against same-backbone fair candidates
  classify: PROMOTE / KEEP_AS_CONTROL / REJECT / INCONCLUSIVE
  create the next smallest discriminative experiment(s) following §8
  immediately fill available GPU slot(s)

on RUN_FAILED(run):
  classify: GPU_OOM / DATA / SOFTWARE / NaN / SCIENTIFIC_NONIMPROVEMENT
  if infrastructure: repair/retry same run identity or fair new protocol
  if scientific: do not retry same seed hoping for a better outcome

on CODE_CHANGE_REQUESTED:
  create next independent revision only under LAB_ROOT/revisions/
  never edit original repository or an active frozen revision
  finish or preserve existing runs under previous source SHA
  do not test old checkpoints through changed source
```

**不允许**：在 `LAB_ROOT` 之外写任何源代码、日志、缓存、checkpoint、测试结果；在训练进行中悄悄修改 loss 系数、刷新概率、选模指标或数据；在 Test 结果不理想后直接修改并重测。

---

## 8. 动态迭代决策树（研究重点）

### 8.1 三重判断：任务质量、事件覆盖、机制目标

在同一骨干的 Standard Val 上，与 E2 和最相关的对照比较，每轮依次判断：

**第一层：任务质量是否守住？**
- 主指标 `Val mAP` 使用代码库原本 `brief.mAP`，不改 `Selector`。默认非劣门槛 `ΔVal mAP ≥ −0.3 个百分点`（**只是工程推进筛选线，不是显著性证明**）。不同模型输入/评价尺度必须确认后使用。
- 对正/负样本分层记录定位与存在性错误；若 null false positives 明显恶化，自动标记 `SAFETY_OF_TASK_FAIL`，不因 Slot 分布更均匀而升级。

**第二层：实际多事件收益？**
- 首看 `H_all`、`H@5`、`M≥2 full@5` 的整数差，以及 `M≥2 zero@5` 是否减少；再看 `D`、`C_j` 和召回缺口。
- 以 `H_all`/`H@5` 中至少一项改善、且另一项不出现大幅退化为首要要求；多 GT 仅有 90 条 Val 查询，**1–2 条变化只能列为趋势，不可自动认定机制已证实**。对边缘结果标记 `INCONCLUSIVE`，用更直接的消融或配对案例解释，不靠换 seed。

**第三层：机制变化是否对应假设？**
- E5 应比 E4 对固定赢家更有针对性，同时保留有效 H；
- E3 需要真实分工收益，而非各 Slot 得分都更高；
- E7 说明 Binding 的必要性；
- E8 必须改善之前定位的特定层间退化，不能只看最终总分。

### 8.2 自动决策规则（可转换为机器状态机）

| 观察到的 Val 反馈 | 下一步自动动作 | 决策理由 |
|---|---|---|
| E1≈R1，且 E2≈E1 / 不明显更好 | 把 **E1 或 E2** 放入 Pareto 候选；优先考察轻量化收益，暂不开发更大 Router | 复杂结构可能不必要 |
| E2 明显弱于 R1，而 E1 较好 | 审查残差映射引入的干扰；启动一个 **E2-residual-off** 或 beta 单因素对照，禁止一次新增多模块 | 分离辅助监督与残差损害 |
| E1 与 E2 都显著落后 R1 | 验证严格评测/模块调用；若合法，再单独恢复一个 RCG 或 FRF 成分形成 Xn | 不能预设 basis 一定冗余 |
| E5 比 E2 更好且 Top-1 垄断下降、H@5/Full@5 改善 | **立刻安排 E4** 等扰动随机对照；同时可安排 E3 | 验证定向机制而非一般噪声 |
| E5 Top-1 变均匀，但 H@5 / Full@5 不升反降 | **淘汰 E5**；回到 E2；优先尝试 E3 | 反对伪去垄断 |
| E5 ≈ E4 | 不声称定向机制有独有作用；保留更简单的方式，或关闭刷新 | 避免过度机制解释 |
| E3 改善覆盖但提高 D | 保留为“覆盖—冗余权衡”，分析 C_j；严禁以 D 绝对变化直接判死 | 更多成功候选也可能增加 D |
| E3/E5 任一有明确增益 | 启动 **E6**，判断叠加是否增益而非相互干扰 | 只有有依据才组合 |
| E6 入围 | 启动 **E7**（bind=0），必要时对照 E2/no-bind | 核验核心损失贡献 |
| 多层模型只有 D1→D2 改善，D2→D3 仍严重丢失 | 启动 **E8 多层共享**，只在 BM/UVCOM 代表性模型先做 | 问题定位明确 |
| CG/QD 迁移下降 | 标记架构依赖；仅允许提出一个兼容性单因素 Xn，不得无限堆新 loss | 保留负面证据 |
| 所有新模块不如 R1 | 保留完整 DQ-CGPv3 作为当前优选；输出负结果和代码验证，不强行叙事 | 科学结论可能是“不应精简” |

**默认用于排序，不是硬性显著性判断**：同骨干 Val 上先淘汰明显违反任务质量门槛的候选；在其余候选中以 `(Val mAP, H@5, full_multigt@5, H_all, 低成本)` 保留 Pareto 前沿。**不把不同单位的数字随意加权求总分**，不让索引均匀度单独赢得选模权。

### 8.3 何时新增 Xn，何时应该停止改结构？

新增 Xn 必须满足全部条件：
1. 有**真实 Val 观测或具体梯度/输出审计**作为触发证据；
2. 有一个清晰、可反证的假设及对照；
3. 每次仅改变一个机制或超参数；
4. 预计可以解释一个已经出现的现象；
5. 不需要读取 Test 才能决定。

允许的首轮小范围自适应调节：
- `λ_nov`: 默认 0.2；只有看到更新过强/过弱且 E3 有潜力时，追加一次 0.1 **或** 0.4（只改一个值）；
- `p_max`: 默认 0.25；只在 E5 显示效果趋势且刷新计数合乎预期时追加一次 0.1 **或** 0.35；
- `α`: 默认 0.25；只在状态幅度/梯度审计显示过强时追加一次 0.1；
- `binding_loss_coef`: 默认 0.2，E7=0；若有明确证据显示过强/过弱才增加一个对照点；
- Adapter 插入位置：只在逐层诊断看到特定后层退化时做 E8。

禁止同时网格搜索以上超参数；禁止反复根据 Test 小幅差异选择设置。新参数必须记录 `why_now`、`parent_run_id`、与父任务的唯一配置差异。

### 8.4 训练早期反馈与防止过度早停

- `train.py` 已经每个 epoch Val + patience 控制，默认按其原生规则训练；**正式科学结果以完成后的 best.pt 为准**。
- 训练前 5–10 epoch 的 Slot 变化可用作调试与调度优先级，**不可因为当前 mAP 一时低而直接判定模型无效**，因为旧版本最佳 epoch 可达数十到上百。
- 可在严重 NaN、错配、持续 OOM、Grad 全零、loss 未被加入主目标等**工程错误**时即时终止；正常收敛中的性能落后，原则上完成 Val early stopping 后再科学裁决。
- 在同一运行完成前，不得取某个中间 epoch 换新参数继续训练、并将它与从头训练实验混为一谈。

---

## 9. 持续反馈的实验状态机和自动账本

每条实验任务在 `agent_runs/registry.jsonl` 中有明确状态：

```text
PLANNED → CODE_READY → PREFLIGHT_PASSED → QUEUED → RUNNING
        → TRAINED → VAL_DIAGNOSING → VAL_VERIFIED
        → PROMOTED | RETAINED_CONTROL | REJECTED | INCONCLUSIVE
        → FROZEN_FOR_TEST → TESTED → ARCHIVED

Any state → BLOCKED_ENGINEERING / BLOCKED_RESOURCES / PAUSED
```

每一个 RUN_FINISHED 必须创建 `agent_runs/reports/<run_id>.md`，格式固定为：

```markdown
# RUN DECISION: <run_id>
- Hypothesis: ...
- Git SHA / Config SHA / checkpoint SHA: ...
- Backbone / seed / best epoch / GPU peak: ...
- Control: <id> (same protocol?)
- Val mAP: A -> B (Δ ... pp)
- Val H_all / H@5 / missing / omission@5: A -> B (raw counts)
- Val Multi-GT full@5 / single / zero: A -> B (raw counts)
- Top-2 Top-1 concentration / Hit concentration: A -> B
- H/D/U and layer Gain/Loss/Keep: ...
- Null false positives: ...
- Validity checks: pass/fail, notes
- Verdict: PROMOTE | CONTROL | REJECT | INCONCLUSIVE
- Falsified/supported hypotheses: ...
- Proposed next run: <id>, parent, single changed factor
- Test used for iteration? No
```

研究登记册还须记载最初假设、所有失败尝试、程序 bug、代码修复和无效实验；不得只展示成功版本。

---

## 10. 从核心实验到跨骨干泛化：动态运行规模

### 第一阶段：即时六槽实验（最先执行）

**Moment-DETR + UVCOM × {E1,E2,E5} = 6 次从头训练。** 两种骨干分别对应“生成缺失/层间退化”和“Slot Top-1 垄断/前列选择遗漏”。每结束一个就诊断和补位。

### 第二阶段：按实际反馈追加实验

- E5 入围 → 优先 E4 等扰动对照，避免“随机扰动也有效”的替代解释；
- E2 或 E5 有价值 → E3、E6 逐项验证；
- E6 入围 → E7 Binding 消融；
- 完整 DQ-CGPv3 明显更强 → 优先分析具体损失/Router 是否关键，避免无休止组合；
- 一张卡先空出来 → 优先调度**已经具备必要对照**的下一项，不必等另一个骨干结束。

### 第三阶段：迁移与负面验证

- 第一批迁移：BM-DETR（候选生成不足、后层退化）、CG-DETR（历史 DQ-CGPv3 部分诊断退化）。分别训练 **E2 + Val 上的精简冠军**。两者若相同，补一个明确的机制对照而不是重复跑相同配置。
- 后续迁移：EaTR、QD-DETR、Sim-DETR、BAM-DETR，各从头训练冻结的最终候选一次。保留所有负面结果。
- E8 多层共享插入仅对逐层证据支持的骨干做，例如 BM-DETR D2→D3 丢失较多；优先单骨干验证，未改善即停止。

**预算可调，但默认上限**：约 22–24 次新 GMR 训练（不含条件性工程重跑；新 Xn 超预算必须在代理决策记录中说明目的和代价）。若第 6 次就已表明模型无效，可以提前终止某条假设；若 E2/E5 明确有效，可在总预算内优先扩大跨架构验证。**预算限制的是无意义的搜索，不限制合理、可证伪的即时迭代。**

---

## 11. 独立版本冻结、运行命令与 checkpoint 校验（全部位于新目录）

### 11.1 代码冻结要求与版本生命周期

已核对 `src/soccergmr/checkpoint.py::source_identity()` 会哈希当前源码目录中的所有 `.py`。`src/soccergmr/test.py` 还会拒绝训练/测试源码不一致的 checkpoint。**因此所有新实验必须从头到尾使用同一个 `LAB_ROOT/revisions/rev_xxxx/`**，直到训练、Val 和最终 Test 结束。

- 代码开发只能在 `LAB_ROOT/revisions/rev_xxxx` 中进行；在**新克隆自己的 Git 元数据**内提交，并记录该 revision 的 Git SHA，**绝不改原仓库 Git/远程 GitHub**。
- 一旦启动该 revision 的任何训练，立刻冻结 `.py`、配置、vendor、依赖锁文件；训练、诊断和最终 Test 均只读该 revision 的源码。
- 如果代理需要改代码，则从 `LAB_ROOT` 的既有源码快照再完整克隆/复制一个 `rev_xxxx+1`；旧 revision 保留，旧作业继续使用旧代码。不能直接 `rsync --delete` 覆盖正在使用的 revision。
- 训练结束的 `best.pt` 只能在匹配的旧 revision 下做严格 Test；不能通过 `strict=False` 或篡改 `source_identity` 绕过验证。
- 不允许在原有 `docs/checkpoints_and_paths.md` 指定的 checkpoint 所在目录写任何文件；R0/R1 只读，所有新 checkpoint 只在 `LAB_ROOT/runs`。
- 本协议允许代理根据 Val 反馈在**新目录里**反复改代码、训练、测试，但全部代码迭代只发生于新目录的独立复制版本，不触碰原代码。

### 11.2 独立配置与示例训练命令

以下假设新实验根目录为 `/home/guoxiangyu/SoccerGMR_SI_Bind_IsolatedLab`，且已在新 revision 内创建 `configs/experiments/interlayer/moment_detr/E2.json`，所有写入目录都在 `LAB_ROOT`。这些新配置与新诊断脚本为**待实现**产物，当前原仓库并不存在。

```bash
set -euo pipefail
export LAB_ROOT=/home/guoxiangyu/SoccerGMR_SI_Bind_IsolatedLab
export REV="$LAB_ROOT/revisions/rev_0001"
export RUN_ID=rev_0001_moment_detr_E2
export OUT="$LAB_ROOT/runs/rev_0001/moment_detr/E2"
mkdir -p "$OUT" "$LAB_ROOT/logs/$RUN_ID" "$LAB_ROOT/tmp/$RUN_ID"

# 启动前调用 LAB_ROOT/scripts/check_write_scope.py（Gate-I 实现后才允许正式运行）
# python "$LAB_ROOT/scripts/check_write_scope.py" --lab "$LAB_ROOT" --revision "$REV" --output "$OUT"

# 根据 §0A.5 使用新目录 env.sh 或任务包装器设置每项缓存变量
cd "$REV"
export PYTHONPATH="$REV/src"
export TMPDIR="$LAB_ROOT/tmp/$RUN_ID"
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$LAB_ROOT/cache/$RUN_ID/xdg"

# 每个进程单独一张卡；卡内最多 3 个独立任务，显存不足时自动降并发
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
python -m soccergmr train \
  --config "$REV/configs/experiments/interlayer/moment_detr/E2.json" \
  --stage gmr \
  --output "$OUT/gmr" \
  --device cuda:0 \
  > "$LAB_ROOT/logs/$RUN_ID/train.stdout.log" 2> "$LAB_ROOT/logs/$RUN_ID/train.stderr.log"

# 中断恢复仍使用相同 rev + config + run output，且配置与 CUDA 可见设备集合一致
# 同样执行 env.sh / Gate-I 之后，在上述训练命令最后附加 --resume

# 只用 Standard Val 决策；以下严格全 K Val 诊断接口需先在新 revision 中实现
CUDA_VISIBLE_DEVICES=0 python "$REV/diagnosed/run_variant_diagnostics.py" \
  --config "$REV/configs/experiments/interlayer/moment_detr/E2.json" \
  --checkpoint "$OUT/gmr/best.pt" \
  --split val --iou-threshold 0.4 \
  --output-dir "$OUT/val_diagnosis"

# 严格冻结入围后，使用同一 revision 测两个 Test；所有输出仍只在 LAB_ROOT
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
python -m soccergmr test \
  --config "$REV/configs/experiments/interlayer/moment_detr/E2.json" \
  --checkpoint "$OUT/gmr/best.pt" \
  --output "$OUT/test" --device cuda:0
```

**注意**：这段示例仅展示 CLI/目录关系。实际作业必须由 `LAB_ROOT/scripts/launch_job.sh` 包装，执行 **§0A.5 的全部缓存/临时目录重定向、Gate-I 白名单校验、同 revision 的配置身份检查**；不能因为示例只设置了几个变量，就跳过完整隔离。用于 Test 的输出目录也必须检查绝对规范路径；`--checkpoint` 只能读新实验的 frozen `best.pt`。

### 11.3 严格一致性与零污染审计

- 每个 run 记录镜像基础 `dia` commit、新目录 revision Git SHA、所有 `src/soccergmr/` 源码文件哈希、配置指纹、标签/特征指纹及 `best.pt` SHA。
- 在新源码环境中，新模型 `model.load_state_dict(..., strict=True)` 必须成功；训练时保存的身份和测试时读取的身份完全一致。
- 官方 Top-10 导出与全 K 诊断抽取器的前十个排序、窗口和置信度必须一致；不一致时停止基于该结果的自动迭代。
- 原始 R0/R1 检查点不通过未经验证的旧 `strict=False` 路径假装重现；若需重新诊断，使用**只读旧 checkpoint + 新目录中的专用历史兼容加载器**，并记录 load-key 完整性与已知历史评测差异。
- 训练、验证、测试、绘图、失败重跑所有文件只写入 `LAB_ROOT`；读取训练配置绝不回写其外部输入路径。
- 每轮从 `LAB_ROOT/audits/` 执行**只读原项目文件与旧 checkpoint 哈希核验**，确认原项目的代码、旧运行和模型资产未变化；只要出现修改迹象，立即停止并向用户报告。

---

## 12. 需要向用户持续展示的仪表盘与产物

建议在**全新根目录**生成固定的 `LAB_ROOT/agent_runs/STATUS.md`，**每个 run 结束立即刷新**（严禁写原仓库的同名目录）：

```text
A. GPU0: S1 [Moment-E2, epoch ...] | S2 [BM-E2, queued] | S3 [idle]
   GPU1: S1 [UVCOM-E5, epoch ...] | S2 [...] | S3 [...]
B. 完成/正在训练/排队/失败的实验数，显存峰值，最新日志位置
C. 同骨干 Val Pareto 前沿：mAP、H@All、H@5、Multi-GT Full@5、Top2 Index、params
D. 本轮被验证/否定的假设及整数证据
E. 接下来最多 3 个高价值任务及其理由
F. 固定 Test 尚未使用/已冻结的标志
```

最终最少产生以下图表/结果：

1. **结构复杂度表**：R1（原版）vs E1/E2/冠军的参数、FLOPs（可测时）、峰值显存、推理耗时。
2. **索引集中双图**：每个 Slot 的 Top-1 次数 vs 独立命中次数，不能仅画 Top-1 熵。
3. **候选分解柱图**：所有 K 的 H、D、U，以及 750 GT（Standard Test）的 missing、omission@5、selected@5（只在冻结后）。
4. **多 GT 四态图**：全候选/Top5，各骨干 160 条 Standard Test / 567 条 Full Test 的找齐、部分、只一个、零。
5. **层间演化图**：H^(l) 与 |S^(l)| 分开；Gain/Loss/Keep 的实际 GT 集合数。
6. **训练期机会再分配图**：u_i EMA、每 Slot 实际刷新频数与训练匹配频数，说明 E5 是否改变了监督流向。
7. **负例复盘**：至少 CG/QD/BAM 中不改善的真实案例；不能只挑成功视频。
8. **论文主表**：Baseline R0、DQ R1、新冠军、关键消融、单 seed、双数据集、官方指标和原生命中整数，不混淆 mAP 与本论文的多 GT 检索指标。

---

## 13. 额外反事实实验（无需再训练、仅用于机制辨析）

为了验证“其他 Slot 被相互协调所抑制”是否真实存在，而不仅是排名集中：

- 在冻结 checkpoint 和 Val 上，暂时衰减当前占优 Slot 在 D1→D2 处的状态强度（不替换历史 checkpoint、不改变 GT、不调优阈值），观察其他 Slot 的最终分数、预测区间和独立命中。需要设置**等强度随机 Slot 扰动**作为对照。
- 记录 `H_all`、`H@5`、`D` 和 Slot Top1 变化。若其他 Slot 的真实覆盖恢复，支持“存在可被释放的竞争效应”；若只是排名换了赢家，不能归因于 SA 抑制。
- 必须以反事实诊断身份输出，与正式模型训练/官方结果分开。

这项反事实**不删除原生 SA**，也不是论文 QD 复现。

---

## 14. 何时冻结、何时运行 Test、何时结束整个研究循环

### 14.1 冻结标准

当某个候选满足下列条件时，可进入 `FROZEN_FOR_TEST`：

1. 训练结束，`best.pt`/`complete.json`/源码/配置严格一致；
2. 和对应同骨干 E2 / R1 或适当控制对照相比，Val 主任务没有明显牺牲；
3. 至少一个**真实**的独立 GT 覆盖或低预算多 GT 找齐指标得到有意义的改善，或证明显著的模型精简且主任务近似保持；
4. 机制必要的消融已完成或给出明确不完成的理由；
5. 不根据任何 Test 结果选择该候选。

### 14.2 Test 规则

- **仅冻结入围模型**运行 Standard Test 和 Full-Decodable Test；同一冻结模型可运行完整官方与五项诊断，两者必须一致。
- 同一模型 Test 可因程序 bug 严格重跑，但必须解释 bug、修正前后 checksum 和影响范围；不允许以改模型参数作为“重跑”。
- 若 Test 显示方法无效，诚实报告推广失败；若要继续研究，则开启**另一个明确标记的研究阶段**，不能把已读 Test 变成原本阶段的未见验证集。

### 14.3 整体停止条件

达到以下任一条件即可停止新增结构：
- 核心假设均有清楚的支持或反证，Pareto 前沿连续数轮无实质进展；
- 发现 E1/E2 精简已经足够，新增 B/C 没有独立收益；
- 新结构无法在较优的至少两类骨干上稳定提升 Val 的有效指标；
- 到达预设运行预算或用户要求停止；
- 数据/实现身份无法严格核验，需要先修复实验基础设施。

**成功可能只是“精简 DQ-CGPv3”，也可能是“定向层间刷新有效”，甚至可能是“现有 DQ-CGPv3 仍最优”。任何一种都应成为真实、可复现的结论。**

---

## 15. Agent 启动提示词（可直接复制给编程/实验模型）

> **你是 SoccerGMR 层间模块自主研究代理。第一优先级不是启动训练，而是严格隔离写入。** 请先完整阅读本方案，尤其 §0A、§6、§7、§8 和 §11。**用户的硬性要求：所有实验代码修改、从头训练、Val、Test、诊断、日志、缓存、checkpoint、绘图和队列记录必须全部位于一个全新实验根目录 `LAB_ROOT` 内；不得修改原仓库、原代码、原配置、原 `diagnosed/`、历史检查点及任何其他位置的代码。** 不允许在原项目创建 Git 分支/工作树/提交，不允许 push/PR；必须从只读 `dia` 分支独立克隆到 `LAB_ROOT/source_mirror`，再在 `LAB_ROOT/revisions/rev_xxxx` 修改独立代码副本。原训练数据、特征和历史 Baseline/DQ-CGPv3 checkpoint 只读复用；不复制覆盖或重新训练历史 checkpoint。首次创建新目录前先确认它不存在、不是旧项目子目录且不是符号链接；配置、临时目录、Python 字节码、HF/PyTorch/CUDA/PIP 缓存、HOME 等可能产生的写入都重定向到 `LAB_ROOT`。先实现 `check_write_scope.py` 与只读原项目 SHA 基线审计；Gate-I 未通过不得执行训练、测试或改代码。**不得为方便开发改动其他地方任何代码**。
>
> 隔离完成后，以 `dia` 只读代码为基线，在新 revision 中实现 E1/E2/E5：轻量层间 Binding、Binding Loss 与 Router/Basis 解耦、Slot 使用率 EMA/定向轻度刷新、严格的全原生 Query 诊断。先通过 Gate-0 的形状、梯度、负样本、多 GT、checkpoint 恢复、原生一对一匹配、Val 诊断与显存实测，再开展新模型**从头训练**。固定 seed=2023，不做多 seed；不移除原生 SA、不直接照抄论文 Query Dropout、不修改原生 Hungarian One-to-One、不默认 NMS。2 × RTX 3090 最高每卡并发 3 个任务，显存不足自动降级。首批安排 Moment-DETR 和 UVCOM 的 E1/E2/E5；任何任务完成就立即读取 Standard Val mAP、H/D/U、All/Top5、Multi-GT 四态、Slot Top-1/独立命中、逐层变化，并按预注册决策树即时补位。只依据 Standard Val 迭代；Standard Test 与 Full-Decodable Test 仅针对冻结入围方案，在其训练源码对应 revision 中执行。每次修改机制新建一个 `LAB_ROOT/revisions/` 下的**独立代码版本**，不能修改正在运行的源码。持续更新 `LAB_ROOT/agent_runs/registry.jsonl`、STATUS、日志、失败分析与结果报告，报告所有负面结果。每轮先提供 `LAB_ROOT`、版本 SHA、越界写入审计状态，再展示真实指标与下一步行动。发现任何潜在越界写入、身份不符或检查点异常，立即停止并报告，绝不静默修复旧项目。你的目标是发现最小有效机制而不是证明某一个设计一定成功。

---

## 16. 代理的首次工作清单（按优先级）

- [ ] 先按 §0A 检查 **`LAB_ROOT` 为此前不存在的全新独立目录**，且不是原项目/数据/checkpoint 的子目录或符号链接；创建完整目录结构。
- [ ] 只读检查原 `dia`，独立 clone 至 `LAB_ROOT/source_mirror`，再 clone 至 `LAB_ROOT/revisions/rev_0001`；绝不在原项目创建分支/工作树或修改 `.git`。
- [ ] 读取并核对旧 checkpoint 清单，写入 `LAB_ROOT/audits/`；建立**唯一写入白名单**、缓存/临时目录重定向、源码/检查点只读哈希基线。通过 Gate-I 才允许继续。
- [ ] 在 `LAB_ROOT/revisions/rev_0001` 的新代码副本中实现，不在原项目的任何 `src/`、`configs/`、`diagnosed/` 中工作。
- [ ] 仅在新 revision 完成新的配置 schema/adapter 工厂，保存每个骨干的真实插入层、K、loss weights、checkpoint 来源。
- [ ] 拆分 Binding Loss，使无 basis 的 E1/E2 真正产生监督；验证 E1 返回恒等且有梯度。
- [ ] 实现 E2 小残差以及训练期 E5 Top-1 EMA/定向刷新；确保 Eval 不随机。
- [ ] 实现并验证严格的通用 Val 全 K 诊断（不可仅导出 Top10），包含所有五项指标与 Null 统计。
- [ ] 在新 revision 的 Moment/UVCOM 真数据上跑 Gate-0；真实输入只读；统计每训练进程峰值显存，全部日志落在 `LAB_ROOT/logs/`。
- [ ] 按可容纳显存安排最多 6 任务；每项都有 `run_id`、日志、hash、GPU、队列状态。
- [ ] 运行中即时监控 Val；任务完成即时诊断、按决策树补位。
- [ ] 根据 E1/E2/E5 反馈确定是否启动 E4、E3、E6、E7；保存负面结果和新假设。
- [ ] 将最小有效模型推广到 BM/CG，再检查 EaTR/QD/Sim/BAM；必要时做 E8。
- [ ] 冻结候选，**随后仅用同一新 revision** 做正式双 Test、完整原生五项诊断、官方指标和论文图表；所有 Test 产物必须落在 `LAB_ROOT/runs/`。
- [ ] 交付完整复现包：新克隆源码 revision Git SHA、配置、检查点、日志、分析 JSON、表格、图、负面案例、研究迭代轨迹；附**原项目及历史 checkpoint 不变性审计**，证明未改动其他位置。

---

### 文件与原始依据索引

- 仓库：`https://github.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval/tree/dia`
- 检查点：`https://github.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval/blob/dia/docs/checkpoints_and_paths.md`
- 原版模块：`https://github.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval/blob/dia/src/soccergmr/modules/dq_cgp.py`
- 现有五项诊断：`https://github.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval/tree/dia/diagnosed`
- 对照论文：用户上传的 *Beyond Caption-Based Queries for Video Moment Retrieval*，尤其 PDF 第 6–8 页（SA / QD、Matching、Table 5–8）。

**本协议的隔离前提：在全新 `LAB_ROOT` 内开发并从头训练/Val/Test/诊断，原项目与所有外部资产只读；此要求优先于其他所有实验便利性。**

**本协议的科学目标：通过快捷但严谨的持续实验，在不改原生 Decoder SA、Query 数量和 one-to-one matcher 的前提下，找到能够提高真实事件独立覆盖、减少固定 Slot 依赖且复杂度合理的最小层间机制；允许实验最后否定该设想。**
