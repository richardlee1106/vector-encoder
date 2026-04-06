# Vector Encoder

`vector-encoder` 是从原 `vite-project/spatial_encoder` 独立整理出来的空间编码器研究仓库，保留重点不是“全部搬家”，而是把真正有研究价值、能继续训练、能继续推理、能继续做实验复盘的核心资产抽出来。

当前本地独立仓库目录：

- `D:\AAA_Edu\vector-encoder`

GitHub：

- `https://github.com/richardlee1106/vector-encoder`

## 保留了哪些核心内容

- 双模型 + 双塔主线实现
- POI 级编码核心逻辑
- cell / town 级编码核心逻辑
- 训练入口、评估入口、推理入口
- 历史版本 `v23 / v24 / v26 / v26_GLM`
- 实验日志、演进文档、图表和里程碑结果
- 为 `geoloom-agent` 提供真实空间编码器服务的运行入口

## 目录概览

```text
vector-encoder/
├─ api/                    # 编码器 API 与向量检索入口
├─ docs/                   # 实验报告、开发计划、分析结果
├─ models/                 # 通用模型层
├─ python/services/        # 真正供 geoloom-agent 调用的服务入口
├─ tests/                  # Python 测试
├─ utils/                  # 通用工具
├─ v23/                    # 历史版本
├─ v24/                    # 历史版本
├─ v26/                    # 单塔 / 导出 / 验证链路
└─ v26_GLM/                # 当前双模型 + 双塔主线
```

## 真实服务入口

如果你要把它作为 `geoloom-agent` 的真实空间编码器服务跑起来，现在已经有统一入口：

```bash
python run.py serve --port 8100
```

Windows 下也可以直接：

```bat
start.bat
```

这两个入口做的是同一件事，不需要再先起一个脚本、再起第二个服务。

默认健康检查：

```bash
curl http://127.0.0.1:8100/health
```

我这次本地实测的健康结果是：

- `encoder_loaded = true`
- `device = cuda`
- `models.poi.loaded = true`
- `models.town.loaded = true`

## 运行时权重怎么处理

这个仓库的真实服务依赖本地运行时检查点，但这些内容故意不进 Git：

- `runtime_assets/saved_models/poi_encoder/best_model.pt`
- `runtime_assets/saved_models/town_encoder/best_model.pt`

当前这台机器上，这两个检查点已经准备好了，实际路径就是：

- `D:\AAA_Edu\vector-encoder\runtime_assets\saved_models\poi_encoder\best_model.pt`
- `D:\AAA_Edu\vector-encoder\runtime_assets\saved_models\town_encoder\best_model.pt`

所以对“你现在这台机器能不能直接跑”这个问题，答案是：可以，已经就绪。

之所以 README 还要强调它们不进 Git，是因为它们属于本地运行资产，不属于适合上传 GitHub 的源码资产，所以：

- 仓库支持真实服务运行
- 当前机器本地权重已经准备好
- 其他新机器如果重新 clone，需要你自己把本地运行资产补上
- `.gitignore` 已明确排除 `runtime_assets/`

## 常用命令

```bash
# 建议先建虚拟环境
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 真实编码器服务
python run.py serve --port 8100

# 训练 / 评估
python v26_GLM/train_v26_mlp.py --sample 0.1 --epochs 5
python v26_GLM/evaluate_v26_pro.py
python v26_GLM/experiment_dual_tower.py

# 测试
python -m pytest tests/test_v26_config.py
```

## 我这次实际验证了什么

2026-04-06 我在这个仓库里重新跑了：

```bash
python -m pytest tests/test_v26_config.py tests/test_run_v26_entrypoint.py tests/test_v26_run_manifest.py tests/test_output_manager.py tests/test_quick_validate_report.py tests/test_export_contract.py
python -m compileall .
python python/services/geoloom_encoder_service.py --port 8100
```

实际结果：

- `10` 个测试全部通过
- `compileall` 成功
- 真实服务健康检查通过
- `geoloom-agent` 已经实际把它当成远端空间编码器接起来了

## 有意排除的内容

为了保证仓库可上传 GitHub、体积可控，这些内容故意不带：

- 模型权重
- 检查点
- `.npy` / `.pt` / `.pth` / `.npz` / `.ckpt` 等大产物
- embedding 导出 bundle
- `runtime_assets/`
- `__pycache__`
- 临时缓存、运行日志、环境文件

## 说明

这个仓库的目标是：

- 保留研究主线
- 保留实验脉络
- 保留图表和日志
- 保留真实编码器服务入口
- 去掉不适合进 GitHub 的大体积运行资产

详细迁移清单与验证结果见 `MIGRATION_REPORT.md`。
