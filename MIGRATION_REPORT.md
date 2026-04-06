# Vector Encode Migration Report

日期：2026-04-06

## 1. 从原仓库迁移了哪些关键文件 / 模块

### 根级入口与说明

- `README.md`
- `requirements.txt`
- `run.py`
- `config.py`
- `GeoVeX.md`
- `__init__.py`

### 研究与实现目录

- `api/`
- `archive/legacy_experiments/`
- `docs/`
- `models/`
- `tests/`
- `utils/`
- `python/services/`
- `v23/`
- `v24/`
- `v26/`
- `v26_GLM/`

### 实验日志、历史版本、图表

- `v26_GLM/p1c_full_log.txt`
- `v26_GLM/p1d_log.txt`
- `v26_GLM/weight_tuning_results.json`
- `v26_GLM/plots/L3/*.png`
- `docs/*.md`
- `docs/*.json`
- `docs/*.png`

### 为兼容旧导入路径补齐的内容

- `spatial_encoder/__init__.py`
- `spatial_encoder/run.py`

### 为 GeoLoom 独立运行补齐的新服务入口

- `python/services/geoloom_encoder_service.py`

这个文件是从原体系里抽取并适配出来的真实空间编码服务入口，负责：

- POI 级编码
- 方向与区域相关编码
- town / cell 上下文
- `/encode-text`
- `/cell/search`
- `/health`

## 2. 哪些内容被有意排除

- `saved_models/`
- `runtime_assets/`
- `__pycache__/`
- `*.pt / *.pth / *.npy / *.npz / *.pkl / *.bin / *.ckpt / *.onnx / *.h5 / *.joblib`
- `v26/outputs/exports/`
- `v26_GLM/outputs/exports/`
- `v26_GLM/p1d_output/` 中的大型数组产物
- 环境文件、缓存、日志、临时目录

保留的少量结果资产：

- `v26/outputs/metrics/**`
- `v26/outputs/reports/**`
- `v26_GLM/outputs/metrics/**`
- `v26_GLM/outputs/reports/**`
- `v26_GLM/p1d_output/p1d_report.json`

## 3. 实际运行命令与结果

### 测试与编译

- `python -m pytest tests/test_v26_config.py tests/test_run_v26_entrypoint.py tests/test_v26_run_manifest.py tests/test_output_manager.py tests/test_quick_validate_report.py tests/test_export_contract.py`
  - 结果：`10/10` 通过
- `python -m compileall .`
  - 结果：成功

### 真实服务启动与健康检查

- `python python/services/geoloom_encoder_service.py --port 8100`
  - 结果：成功
- `GET http://127.0.0.1:8100/health`
  - 结果：成功
  - 关键值：
    - `encoder_loaded = true`
    - `device = cuda`
    - `models.poi.loaded = true`
    - `models.town.loaded = true`

### 被 GeoLoom 真实调用的验证

配合 `geoloom-agent` 实测：

- `GET http://127.0.0.1:3210/api/geo/health`
  - `dependencies.spatial_encoder.mode = remote`
- `npm run smoke:dev`
  - 结果：成功
  - 说明：`geoloom-agent` 已经把本仓库服务当成真实远端空间编码器使用，而不是 fallback

## 4. 是否还存在对原仓库的隐式依赖

### 代码与路径依赖

结论：已消除对原 `vite-project/spatial_encoder` 目录层级的直接路径依赖。

已修复的隐式依赖包括：

- 测试仍假设项目根目录是旧层级
- `spatial_encoder.*` 命名空间导入仍依赖旧包布局
- 部分脚本曾写死旧仓库绝对路径

### 仍然存在但不属于“原仓库隐式依赖”的外部条件

- 完整训练和评估仍需要用户自行准备数据集或数据库
- 真实服务运行需要本地运行时检查点
- 训练 / 推理依赖 `torch` 等 Python 包

### 还需要继续消除吗

就“脱离原仓库可独立交付”这个标准来说，不需要继续消除。

剩下的是正常的外部运行条件，不是旧仓库路径依赖。

## 5. GitHub 上传准备状态

`.gitignore` 已明确排除：

- 权重 / 检查点
- `runtime_assets/`
- 常见缓存目录
- 日志与临时文件
- 环境文件

因此当前仓库已经满足“源码、文档、图表可上传 GitHub，运行时大文件不上传”的目标。
