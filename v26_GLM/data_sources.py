# -*- coding: utf-8 -*-
"""
数据源模块 - 简化版

提供 PostgreSQL/PostGIS 数据连接

Author: Claude
Date: 2026-03-18
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import numpy as np


@dataclass
class PostGISConfig:
    """PostGIS 连接配置"""
    host: str = "localhost"
    port: int = 15432  # 三镇数据集端口
    database: str = "geoloom"
    user: str = "postgres"
    password: str = "123456"
    tables: Dict[str, str] = None  # 表名映射

    def __post_init__(self):
        if self.tables is None:
            self.tables = {
                "point": "pois",       # POI 数据表
                "line": "roads",       # 道路数据表
                "polygon": "landuse",  # 土地利用数据表
                "cells": "cells",
            }

    @classmethod
    def from_env(cls) -> "PostGISConfig":
        """从环境变量加载配置"""
        return cls(
            host=os.getenv("POSTGRES_HOST", os.getenv("DB_HOST", "localhost")),
            port=int(os.getenv("POSTGRES_PORT", os.getenv("DB_PORT", "15432"))),
            database=os.getenv("POSTGRES_DATABASE", os.getenv("DB_NAME", "geoloom")),
            user=os.getenv("POSTGRES_USER", os.getenv("DB_USER", "postgres")),
            password=os.getenv("POSTGRES_PASSWORD", os.getenv("DB_PASSWORD", "123456")),
        )


class PostGISSource:
    """PostGIS 数据源"""

    def __init__(self, config: Optional[PostGISConfig] = None):
        self.config = config or PostGISConfig.from_env()
        self._conn = None

    # 代理 config 属性
    @property
    def host(self) -> str:
        return self.config.host

    @property
    def port(self) -> int:
        return self.config.port

    @property
    def database(self) -> str:
        return self.config.database

    @property
    def user(self) -> str:
        return self.config.user

    @property
    def password(self) -> str:
        return self.config.password

    @property
    def tables(self) -> Dict[str, str]:
        return self.config.tables

    def connect(self):
        """建立连接"""
        try:
            import psycopg2
            self._conn = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
            )
            return True
        except Exception as e:
            print(f"Failed to connect to PostGIS: {e}")
            return False

    def disconnect(self):
        """断开连接"""
        if self._conn:
            self._conn.close()
            self._conn = None

    def is_connected(self) -> bool:
        """检查连接状态"""
        return self._conn is not None

    def query(self, sql: str, params: tuple = None) -> List[Dict]:
        """执行查询"""
        if not self._conn:
            if not self.connect():
                return []

        try:
            from psycopg2.extras import RealDictCursor
            with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            print(f"Query error: {e}")
            return []


# 默认数据源实例
default_postgis_source = PostGISSource()
