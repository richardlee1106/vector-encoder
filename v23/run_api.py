# -*- coding: utf-8 -*-
"""
启动空间编码器API服务

用法:
    python run_api.py

或者先训练模型:
    python train_and_export.py
    python run_api.py
"""

import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("V2.3 空间编码器 API服务")
    print("=" * 60)
    print()
    print("API端点:")
    print("  GET  /              - 服务信息")
    print("  GET  /stats         - 服务统计")
    print("  POST /spatial_search - 空间相似性查询")
    print("  POST /encode        - 编码新POI")
    print("  POST /batch_search  - 批量查询")
    print("  GET  /poi/{id}      - 获取POI信息")
    print()
    print("启动服务: http://localhost:8100")
    print("API文档: http://localhost:8100/docs")
    print()
    print("=" * 60)

    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8100,
        reload=False,
        log_level="info"
    )
