# V2.4 开发计划

## 当前状态（V2.3）

**已达成能力**：
- ✅ 空间距离保持 (Pearson=0.92)
- ✅ 全局空间结构保持

**待改进能力**：
- ⚠️ 空间查询 (重叠率~20%)
- ❌ 方向理解
- ❌ 功能区语义

**智能等级**：L1（空间感知）

---

## V2.4 目标

**目标**：从L1（空间感知）升级到L2（空间查询）

| 能力 | V2.3 | V2.4目标 | 提升 |
|------|------|----------|------|
| 距离保持 | 0.92 | >0.90 | 保持 |
| 空间查询 | ~20% | >40% | +100% |
| 稳定性 | 波动大 | 稳定 | +稳定性 |

---

## 开发任务

### Phase 1: 稳定性提升（2天）

**问题**：模型表现波动大（达成率12%-91%）

**方案**：
1. 固定随机种子
2. 多轮训练取最佳
3. 增加训练轮数（500→800）

**验收**：达成率稳定在80%+

### Phase 2: 空间查询增强（3天）

**问题**：邻居重叠率低（~20%）

**方案**：
1. 引入KNN邻居特征
   - 计算每个POI的K=10最近邻居
   - 将邻居特征作为额外输入
2. 局部邻域损失
   - 添加邻居一致性损失权重
3. 多尺度编码
   - 分别编码局部(500m)和全局特征

**验收**：重叠率提升到40%+

### Phase 3: 方向编码（2天）

**问题**：未学到方向语义

**方案**：
1. 计算相对角度特征
   - 以中心点为参考，计算方位角
2. 方向分类辅助任务
   - 预测POI相对于中心的方向

**验收**：方向识别准确率>60%

### Phase 4: 功能区语义（3天）

**问题**：无法区分商业区/住宅区

**方案**：
1. 引入AOI类型Embedding
2. 引入landuse类型Embedding
3. 区域一致性损失

**验收**：能区分不同功能区

---

## 技术方案

### 2.4.1 稳定性提升

```python
# 固定随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 多轮训练取最佳
best_sil = -1
for run in range(3):  # 3次运行
    model = train(...)
    sil = evaluate(model)
    if sil > best_sil:
        best_sil = sil
        save_model(model)
```

### 2.4.2 空间查询增强

```python
# 计算KNN邻居特征
def compute_knn_features(coords, k=10):
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    # 排除自己
    return distances[:, 1:], indices[:, 1:]

# 模型增加邻居分支
class EncoderV24(nn.Module):
    def __init__(self, ...):
        # 原有编码器
        self.base_encoder = Encoder(...)
        # 邻居编码器
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(k, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

    def forward(self, f, c, neighbor_dists):
        z_base = self.base_encoder.encode(f, c)
        z_neighbor = self.neighbor_encoder(neighbor_dists)
        z = torch.cat([z_base, z_neighbor], dim=-1)
        return F.normalize(z, p=2, dim=-1)
```

---

## 时间线

```
Week 1:
  Day 1-2: Phase 1 稳定性提升
  Day 3-5: Phase 2 空间查询增强

Week 2:
  Day 1-2: Phase 3 方向编码
  Day 3-5: Phase 4 功能区语义
```

---

## 验收标准

| 指标 | V2.3 | V2.4目标 | 验收方法 |
|------|------|----------|----------|
| Pearson | 0.92 | >0.90 | 距离保持测试 |
| 重叠率 | ~20% | >40% | 空间查询测试 |
| 达成率 | 波动 | >80%稳定 | 聚类测试 |
| 方向识别 | N/A | >60% | 方向分类测试 |

---

**最后更新**：2026-03-14
