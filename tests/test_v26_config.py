# -*- coding: utf-8 -*-

from spatial_encoder.v26.config_v26 import V26Config


def test_v26_config_defaults():
    cfg = V26Config()

    assert cfg.h3_resolution in {8, 9, 10}
    assert cfg.embedding_dim == 64
    assert cfg.enable_direction_head is True
    assert cfg.enable_region_head is True
