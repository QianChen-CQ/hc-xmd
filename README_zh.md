# HC-XMD 代码与结果复现说明

论文：**Human-Centric Explainable Mechanism Design: From Incentive Compatibility
to Human Alignment**。

本仓库将实际发送到服务器的三个 V5 代码包合并为一个源码目录，原训练和评估
Python 文件保持不变。三阶段反馈包保存的代码副本与原 V5 对应文件一致。
本次仅新增便于公开使用的启动、外部依赖获取和打包入口，并加入论文后处理脚本；
重新打包不代表重新训练。

## 1. 安装环境

以下命令在 Linux 服务器的 Bash 终端执行，并先进入本仓库根目录。
服务器记录的环境为 Python 3.12.14、PyTorch 2.5.1+cu121、NumPy 1.26.4、
CUDA 运行时 12.1；Phase I/II 的记录还明确标注 NVIDIA A100-PCIE-40GB。

```bash
conda create -n hcxmd_revision_v5 python=3.12 -y
conda activate hcxmd_revision_v5
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

没有 NVIDIA GPU 时，将上面的 PyTorch 安装命令替换为：

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

环境只需创建一次；重新登录服务器后执行 `conda activate hcxmd_revision_v5`。

## 2. 首次准备外部网络定义

PreferenceNet 上游固定版本没有明确的仓库许可证，因此其三个原文件没有放进
本公开包。运行 Phase III 或全部阶段前，先从作者仓库取回并逐文件校验：

```bash
python tools/fetch_preferencenet.py
python tools/fetch_preferencenet.py --check-only
```

下载脚本固定到原实验使用的 commit，不会改用最新主分支，也不会自动提交到 Git。
如果服务器不能访问 GitHub，但已有原始服务器完整包，可离线恢复：

```bash
python tools/fetch_preferencenet.py --source-dir /原代码包路径/phase3_author/vendor/PreferenceNet
```

该目录应包含原始 `README.md` 和 `learnable_preferences/` 下的两个 Python 文件。
安装好依赖并完成这一步后，实验运行无需再访问 GitHub。
取回的三个文件只用于本地运行，不要公开上传；`.gitignore` 仅对 Git 提交生效，
网页直接拖拽上传不会自动过滤这些文件。

## 3. 先短测试，再正式运行

先用 CPU、一个种子检查三阶段流程：

```bash
bash run_hcxmd.sh smoke all --device cpu --seeds 0
```

`smoke` 是缩小预算的软件测试，不能作为论文结果。输出单独保存在
`runs/smoke/phase1`、`phase2`、`phase3`。

正式运行全部阶段，默认使用种子 0、1、2：

```bash
bash run_hcxmd.sh run all --device cuda
```

正式输出保存在 `runs/run/phase1`、`phase2`、`phase3`。

| 阶段 | 实际执行内容 | 依赖 |
| --- | --- | --- |
| Phase I | 训练 3×10 的 RegretNet、AMenuNet，再做公共审计 | 无前置实验 |
| Phase II | 重评冻结的 Phase I 模型，评估作者发布的基线权重；另训练 2×2 的 AMenuNet、GemNet，完成价格变换和审计 | 同一输出根目录、同一 smoke/run 模式的 Phase I 结果 |
| Phase III | 训练并审计 CITransNet、PreferenceNet、ProportionNet 及对应对照 | 外部 PreferenceNet 文件；不依赖 Phase I/II 结果 |

因此，Phase II 包含新增 2×2 训练，不能简单写成“只评估、不训练”。

也可分别运行：

```bash
bash run_hcxmd.sh run phase1 --device cuda
bash run_hcxmd.sh run phase2 --device cuda
bash run_hcxmd.sh run phase3 --device cuda
```

希望关闭 SSH 后继续运行，可用：

```bash
mkdir -p runs
nohup bash run_hcxmd.sh run all --device cuda > runs/run_all.log 2>&1 &
tail -f runs/run_all.log
```

后台命令立即返回不代表运行完成。查看每阶段 `suite_status.json`，成功状态为
`"state": "COMPLETED"`；每阶段另有 `console.log` 和类似
`runs/run/phase1_feedback.tar.gz` 的反馈包。

## 4. 更换输出位置与继续运行

需要把结果放到独立数据目录时，所有阶段使用同一个输出根目录：

```bash
export HCXMD_OUTPUT_ROOT="$HOME/hcxmd_outputs"
bash run_hcxmd.sh run phase1 --device cuda
bash run_hcxmd.sh run phase2 --device cuda
bash run_hcxmd.sh run phase3 --device cuda
```

结果会进入 `$HCXMD_OUTPUT_ROOT/run/phaseN`。Phase II 会自动读取对应 Phase I
目录，无需手工修改原代码中的 `/root/` 默认值。

中断后先确认原进程已停止，再执行相同命令。程序保留完成项，并从已有恢复状态
继续未完成训练。不要让两个进程同时写同一个结果目录；改过代码或参数后要使用
新的输出根目录。若只有评估时显存不足，可在新目录下加 `--eval-batch-size 4`。
修改训练 batch 会改变参考协议，不能不记录就继续称为相同训练设置。

## 5. 结果范围与论文图表

结果目录包含选中模型、配置、来源记录、训练曲线、测试输入、逐样本指标和
`bidder_gains.csv`。公共审计默认每模型 512 个测试剖面；PMR 的分母是投标人与
测试剖面的组合，阈值为偏离收益大于 `1e-6`。

反馈包保留选中模型与核查材料，主动省略大型恢复缓存、重复模型和可重建训练
数组，不是整台服务器的完整镜像。相同文件只存一次，
`DEDUPLICATION_MAP.json` 指明逻辑路径对应哪份保留文件。需要继续训练时应保留
原结果目录，不能把反馈包当作完整恢复快照。

有限偏离搜索、有限网格检查不是连续域全局证明；无效候选保留为诊断。
合成偏好实验不是人类受试者实验。不同设置、训练种子和作者 checkpoint 的来源
不能混作同一种公平排名。

只重新生成论文表格和图片时，使用 `postprocessing/`，无需重训。
它与训练环境的 NumPy、Matplotlib 版本不同，必须单独建环境：

```bash
conda create -n hcxmd_post python=3.12 -y
conda activate hcxmd_post
python -m pip install -r postprocessing/requirements-postprocessing.txt
```

后续核查与绘图命令见 `postprocessing/README.md`。回到训练时执行
`conda activate hcxmd_revision_v5`。

第三方来源、固定版本和许可证范围见 `THIRD_PARTY_NOTICES.md`。
根目录 MIT 许可证只适用于新编写的 HC-XMD 文件，不覆盖上游代码、权重或另行
提供的实验工件。
