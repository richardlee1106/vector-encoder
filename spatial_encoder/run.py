# -*- coding: utf-8 -*-
"""
空间编码器 - 兼容入口脚本

保留原仓库 `spatial_encoder/run.py` 的命令分发文本，方便旧测试与旧调用方式继续工作。
"""

import os
import subprocess
import sys


def main():
    if len(sys.argv) < 2:
        return

    cmd = sys.argv[1]
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if cmd == "train":
        subprocess.run([sys.executable, os.path.join(base_dir, "v23", "train_and_export.py")])

    elif cmd == "api":
        subprocess.run([sys.executable, os.path.join(base_dir, "v23", "run_api.py")])

    elif cmd == "test":
        subprocess.run([sys.executable, os.path.join(base_dir, "v23", "test_api.py")])

    elif cmd == "demo":
        subprocess.run([sys.executable, os.path.join(base_dir, "v23", "experiments", "spatial_demo.py")])

    elif cmd == "serve":
        port = os.environ.get("GEOLOOM_ENCODER_PORT", "8100")
        argv = sys.argv[2:]
        if "--port" in argv:
            index = argv.index("--port")
            if index + 1 < len(argv):
                port = str(argv[index + 1])
        subprocess.run([
            sys.executable,
            os.path.join(base_dir, "python", "services", "geoloom_encoder_service.py"),
            "--port",
            port,
        ])

    elif cmd == "train_v26":
        subprocess.run([
            sys.executable,
            "-c",
            "from spatial_encoder.v26.train_v26 import run_train_v26; print(run_train_v26())",
        ])

    elif cmd == "preprocess_v26":
        subprocess.run([
            sys.executable,
            "-c",
            "from spatial_encoder.v26.preprocess_v26 import run_preprocess_v26; print(run_preprocess_v26())",
        ])

    elif cmd == "validate_v26":
        subprocess.run([
            sys.executable,
            "-c",
            "from spatial_encoder.v26.train_v26 import run_validate_v26; print(run_validate_v26())",
        ])

    elif cmd == "export_v26":
        subprocess.run([
            sys.executable,
            "-c",
            "from spatial_encoder.v26.train_v26 import run_export_v26; print(run_export_v26())",
        ])

    elif cmd == "quick_validate_v26":
        subprocess.run([
            sys.executable,
            "-c",
            "from spatial_encoder.v26.quick_validate_v26 import run_quick_validate_v26; print(run_quick_validate_v26())",
        ])


if __name__ == "__main__":
    main()
