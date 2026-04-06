# V2.3 应用落地计划

## 快速开始

### 1. 训练并导出模型

```bash
cd spatial_encoder
python train_and_export.py
```

训练完成后会在 `saved_models/` 目录生成模型文件。

### 2. 启动API服务

```bash
python run_api.py
```

服务启动后访问 http://localhost:8100/docs 查看API文档。

### 3. API调用示例

```bash
# 空间相似性查询
curl -X POST http://localhost:8100/spatial_search \
  -H "Content-Type: application/json" \
  -d '{"poi_id": 12345, "top_k": 10}'

# 获取服务状态
curl http://localhost:8100/stats
```

---

## 文件结构

```
spatial_encoder/
├── train_and_export.py   # 训练并导出模型
├── run_api.py            # 启动API服务
├── api/
│   ├── server.py         # FastAPI服务
│   └── vector_index.py   # 向量索引服务
├── saved_models/         # 训练好的模型
│   └── v23_YYYYMMDD_HHMMSS/
│       ├── model.pt      # 模型权重
│       ├── mappings.json # 特征映射
│       ├── embeddings.npy # POI向量
│       └── coords.npy    # POI坐标
└── requirements.txt      # 依赖
```

---

## 阶段1：模型导出与验证（已完成）

### 1.1 保存训练好的模型

```python
# train_and_export.py 自动完成
# 输出: saved_models/v23_YYYYMMDD_HHMMSS/
```

### 1.2 验证空间查询能力

```python
# 验证脚本
def test_spatial_query(model, coords, features, query_poi_idx, top_k=10):
    """
    测试空间相似性查询

    问题：给定一个POI，能否找到空间上相似的其他POI？
    """
    from sklearn.neighbors import NearestNeighbors

    # 获取embeddings
    embeddings = model.encode(features, coords)

    # 查询POI
    query_emb = embeddings[query_poi_idx]

    # 找最近的邻居
    nbrs = NearestNeighbors(n_neighbors=top_k+1).fit(embeddings)
    distances, indices = nbrs.kneighbors([query_emb.detach().cpu().numpy()])

    return indices[0][1:]  # 排除自己
```

## 阶段2：构建向量数据库（2-3天）

### 2.1 选择向量数据库

| 选项 | 优势 | 劣势 |
|------|------|------|
| **FAISS** | 快速、轻量 | 需要自己管理 |
| Milvus | 功能丰富 | 部署复杂 |
| Pinecone | 云服务 | 需要联网 |

推荐先用**FAISS**，后续可迁移到Milvus。

### 2.2 构建索引

```python
import faiss

# 构建FAISS索引
dimension = 64  # embedding维度
index = faiss.IndexFlatL2(dimension)

# 添加向量
embeddings_np = embeddings.detach().cpu().numpy()
index.add(embeddings_np)

# 保存索引
faiss.write_index(index, "spatial_encoder/saved_models/faiss_index.bin")
```

## 阶段3：API服务（3-5天）

### 3.1 FastAPI服务

```python
# api_server.py
from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import torch

app = FastAPI()

# 加载模型和索引
model = load_model()
index = faiss.read_index("faiss_index.bin")

class SpatialQuery(BaseModel):
    poi_id: int
    top_k: int = 10

@app.post("/spatial_search")
def spatial_search(query: SpatialQuery):
    """空间相似性查询"""
    # 获取查询向量
    query_vec = get_embedding(query.poi_id)

    # 向量搜索
    distances, indices = index.search(query_vec, query.top_k + 1)

    return {
        "query_poi_id": query.poi_id,
        "similar_pois": indices[0][1:].tolist(),  # 排除自己
        "distances": distances[0][1:].tolist()
    }

@app.post("/encode_poi")
def encode_poi(features: list, coords: list):
    """编码新POI"""
    # ...
```

### 3.2 与V1后端集成

```javascript
// V1后端调用空间编码服务
async function findSpatialSimilarPOIs(poiId, topK = 10) {
    const response = await fetch('http://localhost:8000/spatial_search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ poi_id: poiId, top_k: topK })
    });
    return response.json();
}
```

## 阶段4：前端集成（1周）

### 4.1 地图交互

```javascript
// 用户点击地图上的POI
map.on('click', 'poi-layer', async (e) => {
    const poiId = e.features[0].properties.id;

    // 调用空间相似性API
    const similarPOIs = await findSpatialSimilarPOIs(poiId);

    // 高亮显示相似POI
    highlightSimilarPOIs(similarPOIs.similar_pois);
});
```

### 4.2 AI对话集成

```javascript
// 用户问：找出与这个POI空间相似的POI
async function askSpatialQuestion(userQuery, contextPOI) {
    // 1. 获取空间相似的POI
    const similarPOIs = await findSpatialSimilarPOIs(contextPOI.id);

    // 2. 构建提示词
    const prompt = `
        用户问: ${userQuery}

        当前POI: ${contextPOI.name}
        空间相似的POI:
        ${similarPOIs.map(p => `- ${p.name} (距离: ${p.distance}米)`).join('\n')}

        请回答用户的问题。
    `;

    // 3. 调用LLM
    return await callLLM(prompt);
}
```

## 阶段5：评估与迭代

### 5.1 评估指标

| 指标 | 方法 | 目标 |
|------|------|------|
| 空间查询准确率 | 邻居重叠率 | > 60% |
| 用户满意度 | A/B测试 | 提升 |
| 响应时间 | API延迟 | < 100ms |

### 5.2 迭代方向

如果V2.3应用效果不理想，再考虑：
1. 引入功能区语义（对比学习）
2. 扩展到大规模数据（84万+ POI）
3. 多任务学习框架

---

## 时间线

```
Week 1: 阶段1-2（模型导出 + 向量数据库）
Week 2: 阶段3（API服务）
Week 3: 阶段4（前端集成）
Week 4: 阶段5（评估与迭代）
```

## 立即可以做的事

1. 运行完整V2.3训练并保存模型
2. 构建FAISS索引
3. 编写简单的空间查询API

需要我帮你开始哪一步？
