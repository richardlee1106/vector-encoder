# V2.5 开发计划 - 语义增强

## 目标

**从L2（空间查询）升级到L3（空间理解）**

| 能力 | V2.4 | V2.5目标 | 提升 |
|------|------|----------|------|
| 距离保持 | >0.90 | >0.90 | 保持 |
| 空间查询 | >40% | >50% | +25% |
| 方向理解 | 无 | >60% | 新增 |
| 功能区语义 | 无 | 可区分 | 新增 |

---

## 核心问题回顾

V2.3实验发现的问题：

| 问题 | 根本原因 | 解决方案 |
|------|----------|----------|
| 空间查询差 | 高维映射损失 | ✅ V2.4解决 |
| 稳定性差 | 目标不匹配 | ✅ V2.4解决 |
| 方向未学到 | 无方向监督 | V2.5解决 |
| 功能区未学到 | 无区域上下文 | V2.5解决 |

---

## V2.5 技术方案

### 1. 方向语义编码

**问题**：当前模型不知道"东/南/西/北"

**解决方案**：极坐标编码

```python
def compute_direction_features(coords, center=None):
    """计算方向特征"""
    if center is None:
        center = coords.mean(axis=0)

    dx = coords[:, 0] - center[0]
    dy = coords[:, 1] - center[1]

    # 极坐标
    angle = np.arctan2(dy, dx)  # 方位角 [-π, π]
    dist = np.sqrt(dx**2 + dy**2)

    # 周期性编码（避免边界问题）
    dir_features = np.stack([
        np.sin(angle),      # y分量
        np.cos(angle),      # x分量
        dist,               # 距离
        np.sin(2*angle),    # 二倍角（区分东北/西南等）
        np.cos(2*angle)
    ], axis=1)

    return dir_features
```

**辅助任务**：方向分类

```python
# 将方向分为8个象限
def direction_to_class(angle):
    """角度转方向类别"""
    # 东=0, 东北=1, 北=2, 西北=3, 西=4, 西南=5, 南=6, 东南=7
    return int((angle + np.pi) / (np.pi/4)) % 8

# 添加方向分类损失
direction_labels = direction_to_class(angles)
l_direction = F.cross_entropy(direction_pred, direction_labels)
```

---

### 2. 功能区语义编码

**问题**：无法区分商业区、住宅区、工业区

**解决方案**：区域聚合特征 + 对比学习

```python
def compute_region_features(coords, features, k=20):
    """计算区域聚合特征"""
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
    _, indices = nbrs.kneighbors(coords)

    region_features = []
    for i in range(len(coords)):
        neighbors = indices[i, 1:]
        neighbor_feats = features[neighbors]

        # 邻域统计
        region_features.append([
            neighbor_feats[:, 4].mean(),    # 平均密度
            neighbor_feats[:, 4].std(),     # 密度标准差
            neighbor_feats[:, 5].mean(),    # 平均熵
            neighbor_feats[:, 5].std(),     # 熵标准差
            len(np.unique(neighbor_feats[:, 0])),  # 类别多样性
            neighbor_feats[:, 6].mean(),    # 平均道路距离
        ])

    return np.array(region_features)
```

**对比学习**：让相同功能区的POI靠近

```python
# 功能区标签（来自AOI类型）
aoi_labels = features[:, 2]  # aoi_type_id

# InfoNCE Loss
def info_nce_loss(embeddings, labels, temperature=0.1):
    """对比学习损失"""
    # 正样本：相同功能区
    # 负样本：不同功能区
    ...
```

---

### 3. V2.5模型架构

```python
class EncoderV25(nn.Module):
    """V2.5 编码器 - 方向+功能区语义"""

    def __init__(self, nc, nl, na, nr, embed_dim=64, hidden_dim=128):
        super().__init__()

        # 基础特征编码（继承V2.4）
        self.base_encoder = BaseFeatureEncoder(...)

        # 方向编码器
        self.direction_encoder = nn.Sequential(
            nn.Linear(5, 16),  # sin, cos, dist, sin2, cos2
            nn.ReLU(),
            nn.Linear(16, 16)
        )

        # 区域编码器
        self.region_encoder = nn.Sequential(
            nn.Linear(6, 16),  # 密度统计、熵统计、类别多样性等
            nn.ReLU(),
            nn.Linear(16, 16)
        )

        # 融合编码器
        self.fusion = nn.Sequential(
            nn.Linear(64 + 16 + 16, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

        # 辅助任务头
        self.direction_head = nn.Linear(embed_dim, 8)  # 8方向分类
        self.region_head = nn.Linear(embed_dim, num_aoi_types)  # 功能区分类

    def encode(self, f, c, knn_feat, dir_feat, region_feat):
        base = self.base_encoder(f, c, knn_feat)
        dir_enc = self.direction_encoder(dir_feat)
        region_enc = self.region_encoder(region_feat)

        x = torch.cat([base, dir_enc, region_enc], dim=-1)
        return F.normalize(self.fusion(x), p=2, dim=-1)
```

---

## 训练策略

### 多任务学习

```python
# 总损失
loss = (
    l1 +              # 坐标重构
    2.0 * l2 +        # 距离保持
    0.1 * l3 +        # 邻域一致性
    0.5 * l_dir +     # 方向分类
    0.5 * l_region    # 功能区对比
)
```

### 数据增强

1. **空间旋转**：随机旋转坐标，增强方向不变性
2. **区域采样**：采样不同大小的邻域，增强多尺度理解

---

## 验收标准

| 指标 | V2.4 | V2.5目标 | 测试方法 |
|------|------|----------|----------|
| Pearson | >0.90 | >0.90 | 距离保持测试 |
| 重叠率 | >40% | >50% | 空间查询测试 |
| 方向识别 | N/A | >60% | 8方向分类测试 |
| 功能区F1 | N/A | >0.5 | AOI类型分类 |

---

## 时间线

```
Week 1:
  Day 1-2: 方向特征编码实现
  Day 3-4: 方向分类辅助任务
  Day 5:   方向能力验证

Week 2:
  Day 1-2: 区域聚合特征实现
  Day 3-4: 功能区对比学习
  Day 5:   功能区能力验证

Week 3:
  Day 1-3: 多任务训练调优
  Day 4-5: 综合评估与部署
```

---

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 方向特征与距离冲突 | 可能降低距离保持 | 调整损失权重 |
| AOI标签不完整 | 功能区学习受限 | 使用半监督学习 |
| 计算资源不足 | 训练时间过长 | 使用小规模数据验证 |

---

## 文件结构

```
spatial_encoder/
├── v24/
│   └── train_v24.py          # V2.4训练（空间查询增强）
│
├── v25/
│   ├── train_v25.py          # V2.5训练（语义增强）
│   ├── direction_encoder.py  # 方向编码模块
│   ├── region_encoder.py     # 区域编码模块
│   └── evaluate_v25.py       # 评估脚本
│
└── docs/
    ├── V24_DEVELOPMENT_PLAN.md
    └── V25_DEVELOPMENT_PLAN.md
```

---

**最后更新**：2026-03-15
