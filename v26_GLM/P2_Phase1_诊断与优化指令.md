# P2 Phase 1 诊断与优化指令（给 GLM）

**日期**: 2026-03-18  
**诊断人**: Claude  
**执行人**: GLM

---

## 一、实验结果诊断

### 实验数据回顾

| 配置 | Epochs | Center Weight | Pearson | Region Sep | DirAcc | ClfF1 |
|------|--------|---------------|---------|------------|--------|-------|
| A | 30 | 0.2 | 0.9790 ✅ | 1.13 ❌ | 52.0% ⚠️ | 36.7% ❌ |
| B | 60 | 0.2 | 0.9823 ✅ | 1.07 ❌ | 65.8% ⚠️ | 43.9% ⚠️ |
| **目标** | - | - | >0.96 | **>2.0** | >75% | >50% |

### 核心问题诊断

**问题 1：Region Sep 停滞在 1.0-1.13**

**根本原因**：
```
embedding 层 (L2归一化) ← distance_loss (权重0.5) 强约束
     ↓
  被压缩成测地线，专注距离保持
     ↓
Center Loss (权重0.2) 作用有限，无法打破距离约束
     ↓
Region Sep = 类间距离 / 类内距离 ≈ 1.0（几乎重叠）
```

**架构解耦问题**：
- embedding 层：受 distance_loss 支配 → 距离保持优先
- hidden 层：region_head 从这里分支 → 分类能力在此
- **矛盾**：Center Loss 作用于 embedding，但分类头在 hidden

**问题 2：DirAcc 和 ClfF1 随训练轮次提升**

这说明：
- 分类头（direction_head, region_head）在 hidden 层工作正常
- 更长训练有助于分类性能
- **但 Region Sep 不受训练轮次影响**（1.13 → 1.07 甚至下降）

---

## 二、优化方案（P2-Phase2）

### 方案 A：在 hidden 层应用 Center Loss（推荐）

**核心思路**：既然分类头在 hidden 层，就在 hidden 层引入语义约束。

#### 实施步骤

**Step 1：修改 CenterLoss 类**

**文件**：`spatial_encoder/v26_GLM/losses_v26_pro.py`

**修改内容**：
```python
class CenterLoss(nn.Module):
    """
    类别中心损失 - 让同类特征向类中心聚拢
    
    ⭐ 修改：支持作用于 hidden 层（未归一化）或 embedding 层（L2归一化）
    """
    
    def __init__(self, num_classes: int = 6, feat_dim: int = 640, alpha: float = 0.5, normalize: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.alpha = alpha
        self.normalize = normalize  # ⭐ 新增：是否归一化
        
        self.register_buffer('centers', torch.randn(num_classes, feat_dim))
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [N, feat_dim] - hidden 或 embedding
            labels: [N] - 功能区标签
        """
        valid_mask = labels < 6
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        features = features[valid_mask]
        labels = labels[valid_mask]
        
        # ⭐ 根据 normalize 参数决定是否归一化
        if self.normalize:
            centers_norm = F.normalize(self.centers, p=2, dim=1)
            features_norm = F.normalize(features, p=2, dim=1)
        else:
            centers_norm = self.centers
            features_norm = features
        
        centers_batch = centers_norm[labels]
        loss = torch.pow(features_norm - centers_batch, 2).sum(dim=1).mean()
        
        # 更新中心
        with torch.no_grad():
            for label in labels.unique():
                mask = labels == label
                if mask.sum() > 0:
                    delta = (features[mask] - centers_norm[label]).mean(dim=0)
                    self.centers[label] += self.alpha * delta
        
        return loss
```

**Step 2：修改 MultiTaskLossPro 集成**

**文件**：`spatial_encoder/v26_GLM/losses_v26_pro.py`

在 `MultiTaskLossPro.__init__` 中修改：
```python
# ⭐ 修改：Center Loss 作用于 hidden 层
if center_weight > 0:
    self.center_loss = CenterLoss(
        num_classes=region_classes,
        feat_dim=640,  # ⭐ 改为 hidden_dim，而非 embedding_dim
        alpha=0.5,
        normalize=False  # ⭐ hidden 层不归一化
    )
else:
    self.center_loss = None
```

在 `MultiTaskLossPro.forward` 中修改：
```python
# ⭐ 修改：传入 hidden 而非 embeddings
if self.center_weight > 0 and self.center_loss is not None:
    l_center = self.center_loss(hidden, region_labels)  # ⭐ 改为 hidden
    loss_dict["center"] = l_center.item()
else:
    l_center = torch.tensor(0.0, device=hidden.device)
    loss_dict["center"] = 0.0
```

**Step 3：调整损失权重**

**文件**：`spatial_encoder/v26_GLM/config_v26_pro.py`

```python
@dataclass
class LossConfig:
    distance_weight: float = 0.3        # ⭐ 降低：0.5 → 0.3（减少对 embedding 的约束）
    reconstruction_weight: float = 0.3
    direction_weight: float = 1.5
    region_weight: float = 2.0          # ⭐ 提高：1.5 → 2.0（增强分类能力）
    center_weight: float = 0.5          # ⭐ 提高：0.2 → 0.5（hidden 层可承受更大权重）
    distance_decay_gamma: float = 0.5
    k_nearest_neighbors: int = 85
```

**权重调整逻辑**：
- 降低 distance_weight：减少对 embedding 的压缩，给 Region Sep 留空间
- 提高 region_weight：强化分类能力
- 提高 center_weight：hidden 层未归一化，可承受更大权重

**Step 4：运行实验（P2-Phase2A）**

```bash
cd D:/AAA_Edu/TagCloud/vite-project/spatial_encoder/v26_GLM

# 10% 数据快速验证新配置
python experiment_p2_region_sep.py --sample 0.1 --epochs 30 --batch 16384 \
    --distance-weight 0.3 --region-weight 2.0 --center-weight 0.5

# 如果验证通过，全量训练
python experiment_p2_region_sep.py --sample 1.0 --epochs 80 --batch 16384 \
    --distance-weight 0.3 --region-weight 2.0 --center-weight 0.5
```

**验收标准**：
- Region Sep > 1.5（提升 30%+）
- Pearson > 0.95（允许轻微下降）
- DirAcc > 75%
- ClfF1 > 50%

---

### 方案 B：Triplet Loss（如果方案 A 不足）

**触发条件**：方案 A 后 Region Sep < 2.0

**实施步骤**：

**Step 1：实现 TripletLoss 类**

**文件**：`spatial_encoder/v26_GLM/losses_v26_pro.py`

```python
class TripletLoss(nn.Module):
    """
    三元组损失 - 同时拉近类内、推开类间
    
    使用半难负样本挖掘（semi-hard negative mining）
    """
    
    def __init__(self, margin: float = 0.5, normalize: bool = False):
        super().__init__()
        self.margin = margin
        self.normalize = normalize
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [N, feat_dim] - hidden 或 embedding
            labels: [N] - 功能区标签
        """
        device = features.device
        
        # 过滤未知标签
        valid_mask = labels < 6
        if valid_mask.sum() < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        features = features[valid_mask]
        labels = labels[valid_mask]
        
        # 可选归一化
        if self.normalize:
            features = F.normalize(features, p=2, dim=1)
        
        # 计算距离矩阵
        dist_matrix = torch.cdist(features, features, p=2)
        
        # 构建正负样本掩码
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_not_equal = ~labels_equal
        
        # 对每个 anchor，找最难正样本和半难负样本
        losses = []
        for i in range(len(features)):
            # 正样本：同类中距离最远的
            pos_mask = labels_equal[i].clone()
            pos_mask[i] = False
            if pos_mask.sum() == 0:
                continue
            pos_dist = dist_matrix[i][pos_mask].max()
            
            # 半难负样本：异类中距离在 [pos_dist, pos_dist + margin] 范围内的
            neg_mask = labels_not_equal[i]
            neg_dists = dist_matrix[i][neg_mask]
            if len(neg_dists) == 0:
                continue
            
            # 找半难负样本
            semi_hard_mask = (neg_dists > pos_dist) & (neg_dists < pos_dist + self.margin)
            if semi_hard_mask.sum() > 0:
                neg_dist = neg_dists[semi_hard_mask].min()
            else:
                # 如果没有半难负样本，使用最难负样本
                neg_dist = neg_dists.min()
            
            # Triplet loss
            loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
            losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return torch.stack(losses).mean()
```

**Step 2：集成到 MultiTaskLossPro**

```python
# __init__ 中添加
self.triplet_weight = triplet_weight
if triplet_weight > 0:
    self.triplet_loss = TripletLoss(
        margin=0.5,
        normalize=False  # hidden 层不归一化
    )
else:
    self.triplet_loss = None

# forward 中添加
if self.triplet_weight > 0 and self.triplet_loss is not None:
    l_triplet = self.triplet_loss(hidden, region_labels)
    loss_dict["triplet"] = l_triplet.item()
else:
    l_triplet = torch.tensor(0.0, device=hidden.device)
    loss_dict["triplet"] = 0.0

# 更新总损失
total_loss = (
    self.distance_weight * l_distance +
    self.reconstruction_weight * l_reconstruct +
    self.direction_weight * l_direction +
    self.region_weight * l_region +
    self.center_weight * l_center +
    self.triplet_weight * l_triplet  # ⭐ 新增
)
```

**Step 3：运行实验（P2-Phase2B）**

```bash
# 10% 验证
python experiment_p2_region_sep.py --sample 0.1 --epochs 30 --batch 16384 \
    --distance-weight 0.3 --region-weight 2.0 --center-weight 0.5 --triplet-weight 0.3

# 全量训练
python experiment_p2_region_sep.py --sample 1.0 --epochs 80 --batch 16384 \
    --distance-weight 0.3 --region-weight 2.0 --center-weight 0.5 --triplet-weight 0.3
```

---

### 方案 C：更长训练 + 更大模型（保守方案）

**触发条件**：不想改架构，只想调参数

**调整内容**：
1. 训练轮次：60 → 80-100 epochs
2. 提高 region_weight：1.5 → 2.5
3. 降低 distance_weight：0.5 → 0.2
4. 保持 center_weight=0.2（作用于 embedding）

**命令**：
```bash
python experiment_p2_region_sep.py --sample 1.0 --epochs 100 --batch 16384 \
    --distance-weight 0.2 --region-weight 2.5 --center-weight 0.2
```

**预期效果**：Region Sep 可能提升到 1.3-1.5，但难以突破 2.0

---

## 三、推荐执行路线

### 优先级排序

| 方案 | 预期 Region Sep | 风险 | 实施难度 | 推荐度 |
|------|-----------------|------|----------|--------|
| **A: hidden 层 Center Loss** | 1.5-2.0 | 低 | 中 | ⭐⭐⭐⭐⭐ |
| B: + Triplet Loss | 2.0-2.5 | 中 | 高 | ⭐⭐⭐⭐ |
| C: 更长训练 + 调权重 | 1.3-1.5 | 低 | 低 | ⭐⭐⭐ |

### 执行顺序

```
Step 1: 方案 A（10% 验证）
  ├─ Region Sep > 1.5 且 Pearson > 0.95
  │  └─→ Step 2: 方案 A（全量训练）
  │       ├─ Region Sep > 2.0 → 完成 ✅
  │       └─ Region Sep < 2.0 → Step 3
  └─ Region Sep < 1.5 或 Pearson < 0.93
     └─→ 回退到方案 C

Step 3: 方案 A + B（组合）
  └─→ 全量训练 → Region Sep > 2.0 → 完成 ✅
```

---

## 四、关键问题回答

### Q1: Center Loss 是否应该应用到 hidden 层而非 embedding？

**答：是的，强烈推荐。**

**理由**：
1. 分类头（region_head）从 hidden 层分支，语义信息在此
2. embedding 层被 distance_loss 强约束，难以引入语义结构
3. hidden 层未归一化，可承受更大的 center_weight（0.5-0.8）
4. 直接优化分类头的输入特征，效果更直接

### Q2: 是否需要调整损失权重分配？

**答：是的，需要重新平衡。**

**推荐调整**：
```python
distance_weight: 0.5 → 0.3   # 减少对 embedding 的压缩
region_weight: 1.5 → 2.0     # 强化分类能力
center_weight: 0.2 → 0.5     # hidden 层可承受更大权重
```

**逻辑**：
- 当前 distance_weight=0.5 过高，导致 embedding 过度专注距离保持
- region_weight 需要提高，与 direction_weight 持平（都是 1.5-2.0）
- center_weight 在 hidden 层可以更大，因为没有归一化约束

### Q3: 是否需要尝试 Triplet Loss？

**答：先尝试方案 A，如果不足再加 Triplet Loss。**

**理由**：
1. Triplet Loss 计算开销大（需要构建三元组）
2. 半难负样本挖掘增加复杂度
3. 先用简单的 Center Loss 验证思路，再考虑更复杂的方法

**如果需要 Triplet Loss**：
- 与 Center Loss 组合使用（center_weight=0.5, triplet_weight=0.3）
- 作用于 hidden 层
- margin 设置为 0.5

### Q4: 全量训练是否有帮助？

**答：有帮助，但不是根本解决方案。**

**观察**：
- 配置 A（30 epochs）→ 配置 B（60 epochs）：DirAcc +13.8%, ClfF1 +7.2%
- 但 Region Sep 几乎不变（1.13 → 1.07）

**结论**：
- 更长训练有助于分类性能（DirAcc, ClfF1）
- 但 Region Sep 受架构和损失权重限制，训练轮次影响有限
- **必须先改架构/权重，再用全量训练巩固效果**

---

## 五、立即执行指令（GLM）

### 任务 1：实施方案 A（hidden 层 Center Loss）

**优先级**：⭐⭐⭐⭐⭐

**文件修改清单**：
1. `losses_v26_pro.py`：修改 CenterLoss 类（添加 normalize 参数）
2. `losses_v26_pro.py`：修改 MultiTaskLossPro（传入 hidden 而非 embeddings）
3. `config_v26_pro.py`：调整损失权重（distance=0.3, region=2.0, center=0.5）
4. `experiment_p2_region_sep.py`：添加命令行参数支持

**验证命令**：
```bash
cd D:/AAA_Edu/TagCloud/vite-project/spatial_encoder/v26_GLM

# 10% 快速验证
python experiment_p2_region_sep.py --sample 0.1 --epochs 30 --batch 16384 \
    --distance-weight 0.3 --region-weight 2.0 --center-weight 0.5
```

**验收标准**：
- Region Sep > 1.5
- Pearson > 0.95
- DirAcc > 70%
- ClfF1 > 45%

### 任务 2：如果验收通过，全量训练

```bash
python experiment_p2_region_sep.py --sample 1.0 --epochs 80 --batch 16384 \
    --distance-weight 0.3 --region-weight 2.0 --center-weight 0.5
```

**目标**：Region Sep > 2.0，L3 完全达成

### 任务 3：如果仍不足，添加 Triplet Loss

参考方案 B 的实现，组合使用 Center Loss + Triplet Loss。

---

## 六、预期效果对比

| 方案 | Region Sep | Pearson | DirAcc | ClfF1 | 训练时间 |
|------|------------|---------|--------|-------|----------|
| P1F-Final | 0.65 | 0.9784 | 82.14% | 57.95% | 基准 |
| P2-Phase1（embedding Center Loss） | 1.07 | 0.9823 | 65.8% | 43.9% | +5% |
| **P2-Phase2A（hidden Center Loss）** | **1.5-2.0** | **0.96-0.97** | **75-85%** | **55-65%** | **+10%** |
| P2-Phase2B（+ Triplet Loss） | 2.0-2.5 | 0.95-0.97 | 75-85% | 55-65% | +20% |

---

## 七、风险提示

### 风险 1：Pearson 下降过多

**缓解措施**：
- 如果 Pearson < 0.93，回退 distance_weight 到 0.4
- 监控训练过程，如果 Pearson 持续下降，提前停止

### 风险 2：过拟合

**缓解措施**：
- 使用 Dropout=0.15
- 监控验证集性能
- 如果验证集 Region Sep 下降，使用 Early Stopping

### 风险 3：训练不稳定

**缓解措施**：
- 使用梯度裁剪（gradient_clip_norm=1.0）
- 降低学习率（3e-4 → 2e-4）
- 增加 warmup_epochs（5 → 10）

---

## 八、成功标准

### L3 完全达成

| 指标 | 当前 | 目标 | P2-Phase2A 预期 |
|------|------|------|-----------------|
| DirAcc | 65.8% | >60% | 75-85% ✅ |
| ClfF1 | 43.9% | >50% | 55-65% ✅ |
| Region Sep | 1.07 | >2.0 | 1.5-2.0 ⚠️ |
| Pearson | 0.9823 | >0.90 | 0.96-0.97 ✅ |

**如果 Region Sep 达到 1.8-2.0**：可以接受，进入应用测试  
**如果 Region Sep < 1.5**：执行方案 B（添加 Triplet Loss）

---

**执行优先级**：立即实施方案 A，10% 验证 → 全量训练 → 根据结果决定是否需要方案 B

**预计时间**：
- 代码修改：1-2 小时
- 10% 验证：1-2 小时
- 全量训练：6-8 小时
- **总计**：1 天内完成

---

**GLM，请按照方案 A 的步骤开始实施，完成后运行 10% 验证实验，并将结果反馈给 Claude。**
