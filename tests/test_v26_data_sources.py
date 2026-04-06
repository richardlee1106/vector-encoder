# -*- coding: utf-8 -*-

from spatial_encoder.v26.data_sources import (
    default_postgis_source,
    resolve_sample_source,
    sanitize_data_source,
)


def test_default_postgis_source_contains_required_tables():
    source = default_postgis_source().as_dict()

    assert source["mode"] == "postgis"
    assert source["tables"]["point"] == "pois"
    assert source["tables"]["line"] == "wuhan_roads"
    assert source["tables"]["polygon"] == "wuhan_landuse"


def test_resolve_sample_source_returns_dict_payload():
    payload = resolve_sample_source()

    assert payload["mode"] == "postgis"
    assert "tables" in payload


def test_sanitize_data_source_redacts_password():
    payload = resolve_sample_source()
    sanitized = sanitize_data_source(payload)

    assert sanitized["password"] == "***"
