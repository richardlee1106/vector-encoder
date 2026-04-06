# -*- coding: utf-8 -*-
"""
空间编码器 - 入口脚本

用法:
    python run.py train                # 训练并导出 V2.3 模型
    python run.py api                  # 启动 V2.3 API 服务
    python run.py test                 # 测试 V2.3 API 服务
    python run.py demo                 # 运行 V2.3 Demo 演示
    python run.py serve                # 启动 GeoLoom 编码器服务
    python run.py train_v26            # 运行 V2.6 统一实验
    python run.py preprocess_v26       # 运行 V2.6 预处理骨架
    python run.py validate_v26         # 验证 V2.6 配置与结构
    python run.py export_v26           # 导出 V2.6 LLM/RAG 契约
    python run.py quick_validate_v26   # 运行 V2.6 小样本快速验证
"""

import os
import subprocess
import sys


def parse_port(argv):
    default_port = "8100"
    if "--port" in argv:
        index = argv.index("--port")
        if index + 1 < len(argv):
            return str(argv[index + 1])
    return os.environ.get("GEOLOOM_ENCODER_PORT", default_port)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("可用命令:")
        print("  train  - 训练并导出模型")
        print("  api    - 启动API服务")
        print("  test   - 测试API服务")
        print("  demo   - 运行Demo演示")
        print("  serve  - 启动 GeoLoom 编码器服务")
        print("  train_v26       - 运行V2.6统一实验")
        print("  preprocess_v26  - 运行V2.6预处理骨架")
        print("  validate_v26    - 验证V2.6配置与结构")
        print("  export_v26      - 导出V2.6契约")
        print("  quick_validate_v26 - 运行V2.6小样本快速验证")
        return

    cmd = sys.argv[1]
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if cmd == "train":
        print("运行训练...")
        subprocess.run(
            [sys.executable, os.path.join(base_dir, "v23", "train_and_export.py")]
        )

    elif cmd == "api":
        print("启动API服务...")
        subprocess.run([sys.executable, os.path.join(base_dir, "v23", "run_api.py")])

    elif cmd == "test":
        print("测试API服务...")
        subprocess.run([sys.executable, os.path.join(base_dir, "v23", "test_api.py")])

    elif cmd == "demo":
        print("运行Demo演示...")
        subprocess.run(
            [
                sys.executable,
                os.path.join(base_dir, "v23", "experiments", "spatial_demo.py"),
            ]
        )

    elif cmd == "serve":
        port = parse_port(sys.argv[2:])
        print(f"启动 GeoLoom 编码器服务，端口 {port} ...")
        subprocess.run(
            [
                sys.executable,
                os.path.join(base_dir, "python", "services", "geoloom_encoder_service.py"),
                "--port",
                port,
            ]
        )

    elif cmd == "train_v26":
        print("运行 V2.6 统一实验...")
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from spatial_encoder.v26.train_v26 import run_train_v26; print(run_train_v26())",
            ]
        )

    elif cmd == "preprocess_v26":
        print("运行 V2.6 预处理骨架...")
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from spatial_encoder.v26.preprocess_v26 import run_preprocess_v26; print(run_preprocess_v26())",
            ]
        )

    elif cmd == "validate_v26":
        print("验证 V2.6 配置与结构...")
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from spatial_encoder.v26.train_v26 import run_validate_v26; print(run_validate_v26())",
            ]
        )

    elif cmd == "export_v26":
        print("导出 V2.6 LLM/RAG 契约...")
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from spatial_encoder.v26.train_v26 import run_export_v26; print(run_export_v26())",
            ]
        )

    elif cmd == "quick_validate_v26":
        print("运行 V2.6 小样本快速验证...")
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from spatial_encoder.v26.quick_validate_v26 import run_quick_validate_v26; print(run_quick_validate_v26())",
            ]
        )

    else:
        print(f"未知命令: {cmd}")
        print(
            "可用命令: train, api, test, demo, serve, train_v26, preprocess_v26, validate_v26, export_v26, quick_validate_v26"
        )


if __name__ == "__main__":
    main()
