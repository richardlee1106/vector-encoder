# -*- coding: utf-8 -*-
"""
V2.6 数据源配置。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PostGISSource:
    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = "123456"
    database: str = "geoloom"
    tables: dict[str, str] = field(
        default_factory=lambda: {
            "point": "pois",
            "line": "wuhan_roads",
            "polygon": "wuhan_landuse",
        }
    )

    def as_dict(self) -> dict:
        return {
            "mode": "postgis",
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "database": self.database,
            "tables": dict(self.tables),
        }


def default_postgis_source() -> PostGISSource:
    return PostGISSource()


def resolve_sample_source(source: PostGISSource | None = None) -> dict:
    resolved = source or default_postgis_source()
    return resolved.as_dict()


def sanitize_data_source(source: dict) -> dict:
    redacted = dict(source)
    if "password" in redacted and redacted["password"]:
        redacted["password"] = "***"
    return redacted
