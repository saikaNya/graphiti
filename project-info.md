# Graphiti 知识图谱项目详细分析

## 项目概述

**Graphiti** 是一个专为 AI Agent 设计的时序感知知识图谱框架，由 Zep Software 开发并开源。它通过持续集成用户交互、结构化和非结构化数据，构建动态、可查询的知识图谱，为 AI Agent 提供强大的记忆和推理能力。

### 核心特点

- **实时增量更新**：无需批量重新计算，新数据立即可查询
- **双时态数据模型**：区分事件发生时间和数据摄取时间
- **高效混合检索**：结合语义、关键词和图遍历，实现亚秒级查询
- **自定义实体定义**：支持通过 Pydantic 模型定义自定义实体类型
- **智能去重机制**：自动检测和合并重复实体与关系
- **矛盾处理**：时序边失效机制处理知识更新和矛盾

---

## 一、知识图谱记忆实现机制

### 1.1 核心数据模型

#### 三层节点体系

```
┌─────────────────────────────────────────────────────────┐
│                    知识图谱层级结构                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  第1层：EpisodicNode（情节节点）                          │
│  ├─ 存储原始交互内容（消息、文本、JSON）                   │
│  ├─ 记录时间戳（valid_at 和 created_at）                 │
│  ├─ 作为知识提取的源头                                    │
│  └─ 属性：name, content, source, source_description      │
│                                                         │
│  第2层：EntityNode（实体节点）                            │
│  ├─ 从情节中提取的实体（人、地点、概念等）                │
│  ├─ 包含名称、摘要、属性、嵌入向量                        │
│  ├─ 支持自定义实体类型（通过 labels）                     │
│  └─ 属性：name, summary, labels, name_embedding         │
│                                                         │
│  第3层：CommunityNode（社区节点）                         │
│  ├─ 通过标签传播算法发现的实体社区                        │
│  ├─ 存储社区的摘要信息                                    │
│  └─ 属性：name, summary, name_embedding                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
                        # 社区摘要
```
#### 节点代码

```python
# 基础抽象类
class Node(BaseModel, ABC):
    uuid: str                    # 唯一标识
    name: str                    # 节点名称
    group_id: str                # 分组/分区ID
    labels: list[str]            # 节点标签
    created_at: datetime         # 创建时间

# 情节节点（第295行开始）
class EpisodicNode(Node):
    source: EpisodeType          # 来源类型（message/json/text）
    source_description: str      # 来源描述
    content: str                 # 原始内容
    valid_at: datetime           # 有效时间
    entity_edges: list[str]      # 关联的实体边UUID列表

# 实体节点（第435行开始）
class EntityNode(Node):
    name_embedding: list[float] | None   # 名称的嵌入向量
    summary: str                         # 实体摘要
    attributes: dict[str, Any]           # 自定义属性（支持扩展）

# 社区节点（第591行开始）
class CommunityNode(Node):
    name_embedding: list[float] | None   # 名称的嵌入向量
    summary: str 
```

#### 三类边关系

```python
# 1. EpisodicEdge（情节边）
class EpisodicEdge:
    source_node_uuid: str  # 情节节点 UUID
    target_node_uuid: str  # 实体节点 UUID
    # 语义：表示"哪个情节提到了哪个实体"（MENTIONS关系）

# 2. EntityEdge（实体边）
class EntityEdge:
    source_node_uuid: str      # 源实体 UUID
    target_node_uuid: str      # 目标实体 UUID
    name: str                  # 关系类型（如：WORKS_AT, FOUNDED）
    fact: str                  # 事实描述
    fact_embedding: list       # 事实的嵌入向量
    episodes: list[str]        # 支持该事实的情节列表
    created_at: datetime       # 边创建时间（摄取时间）
    valid_at: datetime         # 事实生效时间（事件时间）
    invalid_at: datetime       # 事实失效时间（结束时间）
    # 语义：表示"实体之间的事实关系"（RELATES_TO关系）

# 3. CommunityEdge（社区边）
class CommunityEdge:
    source_node_uuid: str  # 社区节点 UUID
    target_node_uuid: str  # 实体节点 UUID
    # 语义：表示"哪些实体属于哪个社区"（HAS_MEMBER关系）
```

### 1.2 双时态模型（Bi-temporal Model）

Graphiti 的核心创新之一是双时态追踪：

```
┌──────────────────────────────────────────────────────────┐
│                    双时态时间轴                            │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  事件时间轴（valid_at → invalid_at）                      │
│  ────────────────────────────────────────────────>       │
│  T1        T2        T3        T4        T5              │
│  │         │         │         │         │               │
│  事实A     事实A      事实B     事实B      事实B            │
│  开始      更新       开始      失效       继续             │
│                                                          │
│  摄取时间轴（created_at）                                 │
│  ────────────────────────────────────────────────>       │
│  2024-01   2024-02   2024-03   2024-04   2024-05        │
│  │         │         │         │         │               │
│  摄取      摄取       摄取      摄取      摄取              │
│  Episode1  Episode2  Episode3  Episode4  Episode5        │
│                                                          │
└──────────────────────────────────────────────────────────┘

特性：
1. valid_at：事实在真实世界生效的时间
2. invalid_at：事实在真实世界失效的时间
3. created_at：数据被摄取到系统的时间
4. 支持历史时点查询（"在2024年1月时，用户住在哪里？"）
```

---

## 二、知识图谱的创建与更新流程

### 2.1 添加情节（add_episode）完整流程

```
┌─────────────────────────────────────────────────────────────────┐
│                     add_episode 处理流程                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  输入：                                                          │
│  ├─ episode_body: "张三昨天去了北京参加会议"                     │
│  ├─ reference_time: 2025-10-29T00:00:00Z                        │
│  ├─ source: EpisodeType.text                                    │
│  └─ group_id: "user_123"                                        │
│                                                                 │
│  ▼                                                              │
│  【第1步：准备阶段】                                             │
│  ├─ 创建 EpisodicNode                                           │
│  └─ 检索最近10个历史情节（用于上下文）                           │
│                                                                 │
│  ▼                                                              │
│  【第2步：实体提取】extract_nodes()                              │
│  ├─ LLM 提取实体                                                 │
│  │   ├─ 提取："张三"、"北京"、"会议"                            │
│  │   └─ 分类：Person、Location、Event                           │
│  ├─ 反思机制（reflexion）                                        │
│  │   └─ LLM 检查是否有遗漏的实体                                 │
│  └─ 生成临时 EntityNode 列表                                     │
│                                                                 │
│  ▼                                                              │
│  【第3步：实体去重】resolve_extracted_nodes()                     │
│  ├─ 快速匹配：字符串完全相同 → 直接合并                          │
│  ├─ 相似度匹配：embedding cosine > 0.95 → 合并                  │
│  ├─ LLM 判断：                                                   │
│  │   ├─ 搜索候选节点（语义搜索）                                │
│  │   ├─ LLM 判断是否指向同一对象                                │
│  │   └─ 返回 UUID 映射表                                        │
│  └─ 输出：已去重的 EntityNode + UUID 映射                        │
│                                                                 │
│  ▼                                                              │
│  【第4步：关系提取】extract_edges()                              │
│  ├─ LLM 提取事实三元组                                           │
│  │   ├─ 实体对：(张三, 北京)                                     │
│  │   ├─ 关系类型：TRAVELED_TO                                   │
│  │   ├─ 事实描述："张三去了北京参加会议"                         │
│  │   └─ 时间解析：valid_at = 2025-10-28 (昨天)                  │
│  ├─ 反思机制                                                     │
│  │   └─ LLM 检查是否有遗漏的关系                                 │
│  └─ 生成临时 EntityEdge 列表                                     │
│                                                                 │
│  ▼                                                              │
│  【第5步：关系去重与失效】resolve_extracted_edges()               │
│  ├─ 快速去重：完全相同的 fact → 合并                             │
│  ├─ 语义搜索：查找相似的事实边                                   │
│  ├─ LLM 判断重复：                                               │
│  │   └─ "张三去了北京" vs "张三在北京" → 是否重复？              │
│  ├─ LLM 判断矛盾：                                               │
│  │   ├─ 新："张三现在在上海"                                     │
│  │   ├─ 旧："张三在北京"                                         │
│  │   └─ → 设置旧边的 invalid_at = 当前时间                      │
│  └─ 输出：已去重的 EntityEdge + 失效的 EntityEdge                │
│                                                                 │
│  ▼                                                              │
│  【第6步：属性提取】extract_attributes_from_nodes()              │
│  ├─ 提取摘要（summary）                                          │
│  │   ├─ 综合当前情节和历史信息                                   │
│  │   └─ 生成简洁描述（限制1000字符）                             │
│  └─ 提取自定义属性（如果定义了 Pydantic 模型）                   │
│                                                                 │
│  ▼                                                              │
│  【第7步：嵌入生成】                                             │
│  ├─ EntityNode.name_embedding ← embed(node.name)               │
│  └─ EntityEdge.fact_embedding ← embed(edge.fact)               │
│                                                                 │
│  ▼                                                              │
│  【第8步：批量存储】add_nodes_and_edges_bulk()                   │
│  ├─ 创建 EpisodicEdge（情节 → 实体）                            │
│  ├─ MERGE 操作存储所有节点和边                                   │
│  └─ 更新数据库索引                                               │
│                                                                 │
│  ▼                                                              │
│  【第9步：社区更新（可选）】update_community()                   │
│  ├─ 标签传播算法                                                 │
│  ├─ LLM 生成社区摘要                                             │
│  └─ 保存 CommunityNode 和 CommunityEdge                         │
│                                                                 │
│  ▼                                                              │
│  输出：AddEpisodeResults                                        │
│  ├─ episode: EpisodicNode                                      │
│  ├─ episodic_edges: list[EpisodicEdge]                         │
│  ├─ nodes: list[EntityNode]                                    │
│  ├─ edges: list[EntityEdge]                                    │
│  ├─ communities: list[CommunityNode]                           │
│  └─ community_edges: list[CommunityEdge]                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 关键代码位置

```
graphiti_core/
├── graphiti.py                           # 主入口类
│   ├── add_episode()                     # 添加单个情节
│   ├── add_episode_bulk()                # 批量添加情节
│   └── search()                          # 检索接口
│
├── nodes.py                              # 节点定义
│   ├── class EpisodicNode
│   ├── class EntityNode
│   └── class CommunityNode
│
├── edges.py                              # 边定义
│   ├── class EpisodicEdge
│   ├── class EntityEdge
│   └── class CommunityEdge
│
├── utils/maintenance/
│   ├── node_operations.py               # 节点操作
│   │   ├── extract_nodes()              # 提取实体
│   │   ├── resolve_extracted_nodes()    # 实体去重
│   │   └── extract_attributes_from_nodes() # 提取属性
│   │
│   ├── edge_operations.py               # 边操作
│   │   ├── extract_edges()              # 提取关系
│   │   ├── resolve_extracted_edges()    # 关系去重
│   │   └── resolve_extracted_edge()     # 单个边解析
│   │
│   ├── community_operations.py          # 社区操作
│   │   ├── build_communities()          # 构建社区
│   │   └── update_community()           # 更新社区
│   │
│   └── dedup_helpers.py                 # 去重辅助函数
│       ├── _resolve_with_similarity()   # 相似度匹配
│       └── _resolve_with_llm()          # LLM 判断
│
├── prompts/                             # Prompt 模板
│   ├── extract_nodes.py                 # 实体提取 prompts
│   ├── extract_edges.py                 # 关系提取 prompts
│   ├── dedupe_nodes.py                  # 节点去重 prompts
│   ├── dedupe_edges.py                  # 边去重 prompts
│   ├── invalidate_edges.py              # 边失效 prompts
│   └── summarize_nodes.py               # 摘要生成 prompts
│
└── search/                              # 检索模块
    ├── search.py                        # 搜索主逻辑
    ├── search_config.py                 # 搜索配置
    ├── search_config_recipes.py         # 预定义搜索配方
    └── search_utils.py                  # 搜索工具函数
```

---

## 三、智能去重机制详解

### 3.1 节点去重（Node Deduplication）

```python
# 位置：graphiti_core/utils/maintenance/node_operations.py

async def resolve_extracted_nodes(
    clients: GraphitiClients,
    extracted_nodes: list[EntityNode],
    episode: EpisodicNode,
    previous_episodes: list[EpisodicNode],
    entity_types: dict[str, type[BaseModel]],
) -> tuple[list[EntityNode], dict[str, str], list[tuple[EntityNode, EntityNode]]]:
    """
    三阶段去重策略：
    1. 快速路径：精确匹配
    2. 中速路径：相似度匹配
    3. 慢速路径：LLM 判断
    """
    
    # 阶段1：精确匹配
    # - 字符串完全相同 → 直接认定为重复
    
    # 阶段2：相似度匹配
    # - embedding cosine similarity > 0.95 → 认定为重复
    # - 快速筛选候选节点
    
    # 阶段3：LLM 判断
    # - 语义搜索候选节点（基于 name_embedding）
    # - LLM 根据上下文判断是否指向同一对象
    # - 返回最佳名称和 UUID 映射
```

**节点去重 Prompt（dedupe_nodes.py）：**

```
系统角色：
你是一个判断实体是否重复的 AI 助手

用户输入：
<PREVIOUS MESSAGES>
[历史对话，用于理解上下文]
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
张三：我明天要去见李总
</CURRENT MESSAGE>

<NEW ENTITY>
{id: 0, name: "李总", entity_type: ["Person"]}
</NEW ENTITY>

<EXISTING ENTITIES>
[
  {idx: 0, name: "李明", entity_type: ["Person"], summary: "公司CEO"},
  {idx: 1, name: "CEO李明", entity_type: ["Person"]},
  {idx: 2, name: "李四", entity_type: ["Person"]}
]
</EXISTING ENTITIES>

任务：
判断 NEW ENTITY 是否与 EXISTING ENTITIES 中的任何一个
指向同一真实世界对象。

规则：
- 仅当指向同一对象时标记为重复
- 相关但不同的实体不算重复
- 语义等价：描述性标签可以与具名实体匹配

返回格式：
{
  "entity_resolutions": [
    {
      "id": 0,
      "name": "李明",  // 最完整的名称
      "duplicate_idx": 0,  // 最佳匹配的索引
      "duplicates": [0, 1]  // 所有重复项的索引列表
    }
  ]
}
```

### 3.2 边去重与矛盾检测（Edge Deduplication & Contradiction）

```python
# 位置：graphiti_core/utils/maintenance/edge_operations.py

async def resolve_extracted_edge(
    llm_client: LLMClient,
    extracted_edge: EntityEdge,
    related_edges: list[EntityEdge],      # 相似的边（重复候选）
    existing_edges: list[EntityEdge],     # 可能矛盾的边
    episode: EpisodicNode,
    edge_types: dict[str, type[BaseModel]],
) -> tuple[EntityEdge, list[EntityEdge], list[EntityEdge]]:
    """
    三项任务：
    1. 检测重复：新边是否与现有边表达相同信息
    2. 分类边类型：判断应归属哪个自定义边类型
    3. 检测矛盾：新边是否与现有边矛盾，需要失效旧边
    """
```

**边去重 Prompt（dedupe_edges.py）：**

```
系统角色：
你是一个判断事实是否重复和矛盾的 AI 助手

用户输入：
<EXISTING FACTS>
[
  {idx: 0, fact: "张三在北京工作", valid_at: "2024-01-01"},
  {idx: 1, fact: "张三是软件工程师", valid_at: "2024-01-01"},
  {idx: 2, fact: "张三住在朝阳区", valid_at: "2024-01-01"}
]
</EXISTING FACTS>

<FACT INVALIDATION CANDIDATES>
[
  {idx: 0, fact: "张三在北京工作", valid_at: "2024-01-01"},
  {idx: 1, fact: "张三住在朝阳区", valid_at: "2024-01-01"}
]
</FACT INVALIDATION CANDIDATES>

<NEW FACT>
{
  fact: "张三搬到了上海",
  valid_at: "2025-10-01"
}
</NEW FACT>

任务1：重复检测
- 从 EXISTING FACTS 中找出与 NEW FACT 表达相同信息的事实
- 注意：相似但有关键差异的不算重复

任务2：事实类型分类
- 判断 NEW FACT 属于哪个预定义的 FACT TYPE
- 如果不属于任何类型，返回 DEFAULT

任务3：矛盾检测
- 从 FACT INVALIDATION CANDIDATES 中找出与 NEW FACT 矛盾的事实
- 这些事实将被设置 invalid_at

返回格式：
{
  "duplicate_facts": [],        // 重复的 idx（来自 EXISTING FACTS）
  "contradicted_facts": [1],    // 矛盾的 idx（来自 INVALIDATION CANDIDATES）
  "fact_type": "RELOCATION"     // 事实类型
}
```

**时序失效示例：**

```
场景：用户居住地变更

时间线：
2024-01-01: 添加情节 "我住在北京"
  → 创建边：(User, 北京) fact="住在北京" valid_at=2024-01-01

2025-10-01: 添加情节 "我搬到上海了"
  → 检测到矛盾
  → 更新旧边：invalid_at = 2025-10-01
  → 创建新边：(User, 上海) fact="住在上海" valid_at=2025-10-01

图数据库状态：
Edge1: (User)-[LIVES_IN {valid_at: 2024-01-01, invalid_at: 2025-10-01}]->(北京)
Edge2: (User)-[LIVES_IN {valid_at: 2025-10-01, invalid_at: null}]->(上海)

查询：
- "用户现在住哪？" → 上海（invalid_at = null）
- "用户2024年住哪？" → 北京（查询时间在 valid_at 和 invalid_at 之间）
```

---

## 四、提取 Prompt 工程详解

### 4.1 实体提取 Prompt

**消息类型（extract_nodes.py - extract_message）：**

```
系统角色：
你是从对话消息中提取实体节点的 AI 助手。
主要任务是提取并分类说话者和其他重要实体。

用户输入：
<ENTITY TYPES>
[
  {entity_type_id: 0, entity_type_name: "Entity", 
   entity_type_description: "默认实体类型"},
  {entity_type_id: 1, entity_type_name: "Person", 
   entity_type_description: "人物"},
  {entity_type_id: 2, entity_type_name: "Organization", 
   entity_type_description: "组织机构"}
]
</ENTITY TYPES>

<PREVIOUS MESSAGES>
用户：我最近在看一本书
助手：什么书？
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
用户：《三体》，刘慈欣写的，真的很棒
</CURRENT MESSAGE>

指令：
1. 说话者提取：始终提取说话者（冒号前的部分）作为第一个实体
   - 如果说话者在消息中再次提到，视为同一实体

2. 实体识别：
   - 提取当前消息中显式或隐式提到的所有重要实体
   - 排除仅在历史消息中提到的实体（历史仅用于上下文）

3. 实体分类：
   - 使用 ENTITY TYPES 中的描述分类每个实体
   - 分配适当的 entity_type_id

4. 排除项：
   - 不提取关系或动作
   - 不提取日期、时间等时间信息（将单独处理）

5. 格式要求：
   - 使用明确无歧义的名称（如使用全名）
   - 代词（he/she/they）应解析为具体名称

返回格式：
{
  "extracted_entities": [
    {"name": "用户", "entity_type_id": 1},
    {"name": "三体", "entity_type_id": 0},
    {"name": "刘慈欣", "entity_type_id": 1}
  ]
}
```

**文本类型（extract_nodes.py - extract_text）：**

```
系统角色：
你是从文本中提取实体节点的 AI 助手。

用户输入：
<ENTITY TYPES>
[实体类型定义]
</ENTITY TYPES>

<TEXT>
2024年，OpenAI 发布了 GPT-4，标志着人工智能领域的重大突破。
该模型由 Sam Altman 领导的团队开发，展示了强大的多模态能力。
</TEXT>

指令：
1. 提取文本中显式或隐式提到的重要实体
2. 根据 ENTITY TYPES 分类每个实体
3. 避免创建关系或动作节点
4. 避免创建时间信息节点（日期、时间、年份）
5. 使用完整名称，避免缩写

返回格式：
{
  "extracted_entities": [
    {"name": "OpenAI", "entity_type_id": 2},
    {"name": "GPT-4", "entity_type_id": 0},
    {"name": "Sam Altman", "entity_type_id": 1}
  ]
}
```

**JSON 类型（extract_nodes.py - extract_json）：**

```
系统角色：
你是从 JSON 数据中提取实体节点的 AI 助手。

用户输入：
<SOURCE DESCRIPTION>
电商订单数据
</SOURCE DESCRIPTION>

<JSON>
{
  "order_id": "12345",
  "customer": {
    "name": "张三",
    "email": "zhangsan@example.com"
  },
  "products": [
    {"name": "iPhone 15", "quantity": 1, "price": 5999}
  ],
  "shipping_address": "北京市朝阳区",
  "order_date": "2024-10-15"
}
</JSON>

指令：
1. 提取 JSON 代表的主要实体（通常是 "name" 或 "user" 字段）
2. 提取所有其他属性中提到的实体
3. 不提取包含日期的属性
4. 根据 ENTITY TYPES 分类每个实体

返回格式：
{
  "extracted_entities": [
    {"name": "张三", "entity_type_id": 1},
    {"name": "iPhone 15", "entity_type_id": 0},
    {"name": "北京市朝阳区", "entity_type_id": 0}
  ]
}
```

### 4.2 关系提取 Prompt

**关系提取（extract_edges.py - edge）：**

```
系统角色：
你是从文本中提取事实三元组的专家。
1. 提取的事实三元组应包含相关的日期信息
2. 将 CURRENT TIME 视为 CURRENT MESSAGE 的发送时间
   所有时间信息应相对于此时间提取

用户输入：
<FACT TYPES>
[
  {
    fact_type_name: "EMPLOYMENT",
    fact_type_signature: ("Person", "Organization"),
    fact_type_description: "雇佣关系"
  }
]
</FACT TYPES>

<PREVIOUS_MESSAGES>
用户：我在找新工作
</PREVIOUS_MESSAGES>

<CURRENT_MESSAGE>
用户：我上周加入了谷歌，担任软件工程师
</CURRENT_MESSAGE>

<ENTITIES>
[
  {id: 0, name: "用户", entity_types: ["Person"]},
  {id: 1, name: "谷歌", entity_types: ["Organization"]},
  {id: 2, name: "软件工程师", entity_types: ["Entity"]}
]
</ENTITIES>

<REFERENCE_TIME>
2025-10-29T00:00:00Z
</REFERENCE_TIME>

任务：
基于 CURRENT MESSAGE 提取 ENTITIES 之间的所有事实关系。
仅提取满足以下条件的事实：
- 涉及 ENTITIES 列表中的两个不同实体
- 在 CURRENT MESSAGE 中明确陈述或明确暗示
- 可以表示为知识图谱中的边

规则：

1. **实体 ID 验证**：source_entity_id 和 target_entity_id 必须仅使用
   ENTITIES 列表中的 id 值
   - 关键：使用不在列表中的 ID 将导致边被拒绝

2. 每个事实必须涉及两个不同的实体

3. 使用 SCREAMING_SNAKE_CASE 字符串作为 relation_type
   （例如：FOUNDED、WORKS_AT）

4. 不输出重复或语义冗余的事实

5. fact 应紧密释义原始句子，不要逐字引用原文

6. 使用 REFERENCE_TIME 解析模糊或相对的时间表达（如"上周"）

7. 不要凭空臆测或推断时间边界

日期时间规则：
- 使用 ISO 8601 格式，带 "Z" 后缀（UTC）（例如：2025-04-30T00:00:00Z）
- 如果事实正在进行中（现在时），将 valid_at 设置为 REFERENCE_TIME
- 如果表达了变更/终止，将 invalid_at 设置为相关时间戳
- 如果没有明确或可解析的时间，两个字段都留为 null
- 如果仅提到日期（无时间），假定为 00:00:00
- 如果仅提到年份，使用1月1日 00:00:00

返回格式：
{
  "edges": [
    {
      "relation_type": "WORKS_AT",
      "source_entity_id": 0,
      "target_entity_id": 1,
      "fact": "用户在谷歌担任软件工程师",
      "valid_at": "2025-10-22T00:00:00Z",  // 上周
      "invalid_at": null
    }
  ]
}
```

**反思机制（Reflexion）：**

```
系统角色：
你是一个判断哪些事实未被提取的 AI 助手

用户输入：
<PREVIOUS MESSAGES>
[历史消息]
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
用户：我昨天见了李明，他是我们公司的 CEO，我们讨论了新项目
</CURRENT MESSAGE>

<EXTRACTED ENTITIES>
["用户", "李明", "公司", "CEO", "新项目"]
</EXTRACTED ENTITIES>

<EXTRACTED FACTS>
["用户见了李明", "讨论了新项目"]
</EXTRACTED FACTS>

任务：
根据上述消息、已提取的实体和已提取的事实，
判断是否有任何事实未被提取。

分析：
- 已提取："用户见了李明"、"讨论了新项目"
- 遗漏："李明是 CEO"、"李明在公司工作"

返回格式：
{
  "missing_facts": [
    "李明是公司的 CEO",
    "李明在公司工作"
  ]
}
```

---

## 五、检索机制详解

### 5.1 混合搜索策略

```
┌────────────────────────────────────────────────────────────┐
│                   混合搜索架构（Hybrid Search）               │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  查询："张三在哪里工作？"                                    │
│                                                            │
│  ┌──────────────────────────────────────────────────┐    │
│  │              第1层：边搜索（Edge Search）           │    │
│  ├──────────────────────────────────────────────────┤    │
│  │                                                  │    │
│  │  方法1：语义搜索（Cosine Similarity）             │    │
│  │  ├─ 查询向量化                                    │    │
│  │  ├─ 与 fact_embedding 计算余弦相似度             │    │
│  │  └─ 返回 Top-K 相似边                            │    │
│  │                                                  │    │
│  │  方法2：关键词搜索（BM25 Fulltext）               │    │
│  │  ├─ 对 fact 字段全文搜索                         │    │
│  │  ├─ TF-IDF 加权                                  │    │
│  │  └─ 返回 Top-K 匹配边                            │    │
│  │                                                  │    │
│  │  方法3：图遍历搜索（BFS）                         │    │
│  │  ├─ 从中心节点开始广度优先遍历                    │    │
│  │  ├─ 限制最大跳数                                  │    │
│  │  └─ 返回邻域内的边                                │    │
│  │                                                  │    │
│  │  融合：RRF（Reciprocal Rank Fusion）             │    │
│  │  ├─ 合并多个搜索结果                              │    │
│  │  ├─ 公式：score = Σ 1/(k + rank_i)              │    │
│  │  └─ 输出融合后的边列表                            │    │
│  │                                                  │    │
│  └──────────────────────────────────────────────────┘    │
│                          ▼                               │
│  ┌──────────────────────────────────────────────────┐    │
│  │             第2层：节点搜索（Node Search）          │    │
│  ├──────────────────────────────────────────────────┤    │
│  │                                                  │    │
│  │  方法1：语义搜索（基于 name_embedding）           │    │
│  │  方法2：关键词搜索（基于 name 和 summary）        │    │
│  │  方法3：图遍历（BFS从边中提到的节点）             │    │
│  │  融合：RRF                                        │    │
│  │                                                  │    │
│  └──────────────────────────────────────────────────┘    │
│                          ▼                               │
│  ┌──────────────────────────────────────────────────┐    │
│  │            第3层：情节搜索（Episode Search）        │    │
│  ├──────────────────────────────────────────────────┤    │
│  │                                                  │    │
│  │  方法：关键词搜索（基于 content 字段）             │    │
│  │  过滤：时间范围、group_id                         │    │
│  │                                                  │    │
│  └──────────────────────────────────────────────────┘    │
│                          ▼                               │
│  ┌──────────────────────────────────────────────────┐    │
│  │           第4层：社区搜索（Community Search）       │    │
│  ├──────────────────────────────────────────────────┤    │
│  │                                                  │    │
│  │  方法1：语义搜索（基于社区摘要）                   │    │
│  │  方法2：关键词搜索                                 │    │
│  │                                                  │    │
│  └──────────────────────────────────────────────────┘    │
│                          ▼                               │
│  ┌──────────────────────────────────────────────────┐    │
│  │              重排序（Reranking）                    │    │
│  ├──────────────────────────────────────────────────┤    │
│  │                                                  │    │
│  │  方法1：节点距离（Node Distance）                  │    │
│  │  ├─ 计算到中心节点的图距离                         │    │
│  │  └─ 距离越近排名越高                               │    │
│  │                                                  │    │
│  │  方法2：交叉编码器（Cross Encoder）                │    │
│  │  ├─ LLM 判断查询与每个结果的相关性                 │    │
│  │  └─ 更精确但更昂贵                                 │    │
│  │                                                  │    │
│  │  方法3：MMR（Maximal Marginal Relevance）         │    │
│  │  ├─ 平衡相关性和多样性                             │    │
│  │  └─ 避免返回冗余结果                               │    │
│  │                                                  │    │
│  └──────────────────────────────────────────────────┘    │
│                          ▼                               │
│  输出：SearchResults                                     │
│  ├─ edges: list[EntityEdge]                            │
│  ├─ nodes: list[EntityNode]                            │
│  ├─ episodes: list[EpisodicNode]                       │
│  └─ communities: list[CommunityNode]                   │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 5.2 预定义搜索配方（Search Recipes）

```python
# 位置：graphiti_core/search/search_config_recipes.py

# 配方1：边混合搜索 + RRF
EDGE_HYBRID_SEARCH_RRF = SearchConfig(
    edge_config=EdgeSearchConfig(
        search_methods=[
            EdgeSearchMethod.cosine_similarity,  # 语义搜索
            EdgeSearchMethod.bm25,               # 关键词搜索
        ],
        reranker=EdgeReranker.rrf,              # RRF 融合
        limit=10,
    )
)

# 配方2：边混合搜索 + 节点距离重排序
EDGE_HYBRID_SEARCH_NODE_DISTANCE = SearchConfig(
    edge_config=EdgeSearchConfig(
        search_methods=[
            EdgeSearchMethod.cosine_similarity,
            EdgeSearchMethod.bm25,
        ],
        reranker=EdgeReranker.node_distance,    # 基于图距离
        limit=10,
    )
)

# 配方3：综合搜索 + 交叉编码器
COMBINED_HYBRID_SEARCH_CROSS_ENCODER = SearchConfig(
    edge_config=EdgeSearchConfig(
        search_methods=[EdgeSearchMethod.cosine_similarity, EdgeSearchMethod.bm25],
        reranker=EdgeReranker.cross_encoder,    # LLM 重排序
        limit=10,
    ),
    node_config=NodeSearchConfig(
        search_methods=[NodeSearchMethod.cosine_similarity, NodeSearchMethod.bm25],
        reranker=NodeReranker.cross_encoder,
        limit=10,
    ),
)
```

### 5.3 搜索过滤器（Search Filters）

```python
# 位置：graphiti_core/search/search_filters.py

class SearchFilters(BaseModel):
    # 时间过滤
    start_date: datetime | None = None        # 开始时间
    end_date: datetime | None = None          # 结束时间
    
    # 实体过滤
    center_node_uuid: str | None = None       # 中心节点
    entity_uuids: list[str] = []              # 实体白名单
    excluded_entity_uuids: list[str] = []     # 实体黑名单
    
    # 边过滤
    edge_uuids: list[str] = []                # 边白名单
    excluded_edge_uuids: list[str] = []       # 边黑名单
    
    # 情节过滤
    episode_uuids: list[str] = []             # 情节白名单
    excluded_episode_uuids: list[str] = []    # 情节黑名单
    
    # 分组过滤
    group_ids: list[str] | None = None        # 分组ID

# 使用示例
filter = SearchFilters(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    group_ids=["user_123"],
)

results = await graphiti.search_(
    query="张三在哪工作？",
    config=COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
    search_filter=filter,
)
```

---

## 六、社区检测算法

### 6.1 标签传播算法（Label Propagation）

```python
# 位置：graphiti_core/utils/maintenance/community_operations.py

def label_propagation(projection: dict[str, list[Neighbor]]) -> list[list[str]]:
    """
    标签传播社区检测算法
    
    步骤：
    1. 初始化：每个节点分配唯一的社区标签（自己的ID）
    2. 迭代传播：
       - 每个节点采用其邻居中最多的社区标签
       - 平局时选择最大的社区
    3. 收敛：当没有节点改变社区时停止
    4. 输出：返回社区列表
    """
    
    # 初始化：每个节点自成一个社区
    community_map = {uuid: i for i, uuid in enumerate(projection.keys())}
    
    while True:
        no_change = True
        new_community_map = {}
        
        for uuid, neighbors in projection.items():
            curr_community = community_map[uuid]
            
            # 统计邻居的社区标签
            community_counts = defaultdict(int)
            for neighbor in neighbors:
                neighbor_community = community_map[neighbor.node_uuid]
                # 加权：边的数量作为权重
                community_counts[neighbor_community] += neighbor.edge_count
            
            # 选择最多的社区（平局时选最大的）
            if community_counts:
                new_community = max(
                    community_counts.items(),
                    key=lambda x: (x[1], x[0])  # 按计数和ID排序
                )[0]
            else:
                new_community = curr_community
            
            if new_community != curr_community:
                no_change = False
            
            new_community_map[uuid] = new_community
        
        community_map = new_community_map
        
        # 收敛检查
        if no_change:
            break
    
    # 将社区标签转换为节点列表
    communities = defaultdict(list)
    for uuid, community in community_map.items():
        communities[community].append(uuid)
    
    return list(communities.values())
```

### 6.2 社区摘要生成

```python
# LLM 生成社区摘要
async def build_communities(
    driver: GraphDriver,
    llm_client: LLMClient,
    group_ids: list[str] | None,
) -> tuple[list[CommunityNode], list[CommunityEdge]]:
    # 1. 获取社区聚类
    clusters = await get_community_clusters(driver, group_ids)
    
    # 2. 为每个社区生成摘要
    for cluster in clusters:
        context = {
            'entities': [
                {'name': node.name, 'summary': node.summary}
                for node in cluster
            ]
        }
        
        # LLM 提取社区的共同特征
        summary = await llm_client.generate_response(
            prompt_library.summarize_nodes.summarize(context),
            response_model=Summary,
        )
        
        # 3. 创建 CommunityNode
        community_node = CommunityNode(
            name=f"Community_{i}",
            summary=summary.summary,
            group_id=cluster[0].group_id,
        )
        
        # 4. 创建 CommunityEdge（连接社区和成员实体）
        community_edges = build_community_edges(
            cluster, community_node, utc_now()
        )
```

---

## 七、数据库操作

### 7.1 支持的图数据库

```
1. Neo4j（推荐）
   - 成熟的图数据库
   - 丰富的查询语言（Cypher）
   - 可视化工具完善

2. FalkorDB
   - 基于 Redis 的图数据库
   - 高性能、低延迟
   - 适合实时应用

3. Kuzu
   - 嵌入式图数据库
   - 轻量级、易部署
   - 适合本地开发

4. Amazon Neptune
   - 完全托管的图数据库服务
   - 自动扩展、高可用
   - 集成 OpenSearch 全文搜索
```

### 7.2 核心查询模式

**节点保存（MERGE 操作）：**

```cypher
# Neo4j 示例
MERGE (n:Entity {uuid: $uuid})
ON CREATE SET
    n.name = $name,
    n.summary = $summary,
    n.labels = $labels,
    n.name_embedding = $name_embedding,
    n.group_id = $group_id,
    n.created_at = $created_at
ON MATCH SET
    n.name = $name,
    n.summary = $summary,
    n.labels = $labels,
    n.name_embedding = $name_embedding
RETURN n
```

**边保存（MERGE 操作）：**

```cypher
# 创建或更新边
MATCH (source:Entity {uuid: $source_uuid})
MATCH (target:Entity {uuid: $target_uuid})
MERGE (source)-[r:RELATES_TO {uuid: $uuid}]->(target)
ON CREATE SET
    r.name = $name,
    r.fact = $fact,
    r.fact_embedding = $fact_embedding,
    r.episodes = $episodes,
    r.created_at = $created_at,
    r.valid_at = $valid_at,
    r.invalid_at = $invalid_at,
    r.group_id = $group_id
ON MATCH SET
    r.fact = $fact,
    r.episodes = $episodes,
    r.invalid_at = $invalid_at
RETURN r
```

**语义搜索（向量相似度）：**

```cypher
# Neo4j 向量索引查询
CALL db.index.vector.queryNodes(
    'entity_name_embedding_index',  # 索引名称
    $limit,                          # 返回数量
    $query_embedding                 # 查询向量
)
YIELD node, score
WHERE node.group_id IN $group_ids
RETURN node, score
ORDER BY score DESC
```

**全文搜索（BM25）：**

```cypher
# Neo4j 全文索引查询
CALL db.index.fulltext.queryNodes(
    'entity_fact_fulltext_index',  # 索引名称
    $query                         # 查询字符串
)
YIELD node, score
WHERE node.group_id IN $group_ids
RETURN node, score
ORDER BY score DESC
LIMIT $limit
```

---

## 八、API 接口

### 8.1 主要方法

```python
class Graphiti:
    # 初始化
    def __init__(
        self,
        uri: str,                              # 数据库连接URI
        user: str,                             # 用户名
        password: str,                         # 密码
        llm_client: LLMClient | None,          # LLM 客户端
        embedder: EmbedderClient | None,       # 嵌入模型
        cross_encoder: CrossEncoderClient | None,  # 重排序器
        graph_driver: GraphDriver | None,      # 自定义驱动
    )
    
    # 索引和约束
    async def build_indices_and_constraints(
        self,
        delete_existing: bool = False,
    )
    
    # 添加情节
    async def add_episode(
        self,
        name: str,                             # 情节名称
        episode_body: str,                     # 情节内容
        source_description: str,               # 来源描述
        reference_time: datetime,              # 参考时间
        source: EpisodeType = EpisodeType.message,  # 类型
        group_id: str | None = None,           # 分组ID
        uuid: str | None = None,               # 可选UUID
        update_communities: bool = False,      # 是否更新社区
        entity_types: dict | None = None,      # 自定义实体类型
        excluded_entity_types: list[str] | None = None,  # 排除类型
        edge_types: dict | None = None,        # 自定义边类型
        edge_type_map: dict | None = None,     # 边类型映射
    ) -> AddEpisodeResults
    
    # 批量添加情节
    async def add_episode_bulk(
        self,
        bulk_episodes: list[RawEpisode],       # 情节列表
        group_id: str | None = None,
        entity_types: dict | None = None,
        excluded_entity_types: list[str] | None = None,
        edge_types: dict | None = None,
        edge_type_map: dict | None = None,
    ) -> AddBulkEpisodeResults
    
    # 添加三元组（直接添加实体和关系）
    async def add_triplet(
        self,
        source_node: EntityNode,
        edge: EntityEdge,
        target_node: EntityNode,
    ) -> AddTripletResults
    
    # 检索
    async def search(
        self,
        query: str,                            # 查询字符串
        center_node_uuid: str | None = None,   # 中心节点
        group_ids: list[str] | None = None,    # 分组过滤
        num_results: int = 10,                 # 结果数量
        search_filter: SearchFilters | None = None,  # 过滤器
    ) -> list[EntityEdge]
    
    # 高级检索
    async def search_(
        self,
        query: str,
        config: SearchConfig = COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
        group_ids: list[str] | None = None,
        center_node_uuid: str | None = None,
        bfs_origin_node_uuids: list[str] | None = None,
        search_filter: SearchFilters | None = None,
    ) -> SearchResults
    
    # 构建社区
    async def build_communities(
        self,
        group_ids: list[str] | None = None,
    ) -> tuple[list[CommunityNode], list[CommunityEdge]]
    
    # 检索历史情节
    async def retrieve_episodes(
        self,
        reference_time: datetime,
        last_n: int = 10,
        group_ids: list[str] | None = None,
        source: EpisodeType | None = None,
    ) -> list[EpisodicNode]
    
    # 根据情节获取节点和边
    async def get_nodes_and_edges_by_episode(
        self,
        episode_uuids: list[str],
    ) -> SearchResults
    
    # 删除情节
    async def remove_episode(
        self,
        episode_uuid: str,
    )
    
    # 关闭连接
    async def close(self)
```

### 8.2 使用示例

```python
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from datetime import datetime

# 初始化
graphiti = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
)

# 构建索引（首次使用）
await graphiti.build_indices_and_constraints()

# 添加情节
result = await graphiti.add_episode(
    name="对话1",
    episode_body="用户：我叫张三，在北京工作\n助手：很高兴认识你",
    source_description="用户对话",
    reference_time=datetime.now(),
    source=EpisodeType.message,
    group_id="user_123",
)

print(f"提取的实体：{[node.name for node in result.nodes]}")
print(f"提取的关系：{[edge.fact for edge in result.edges]}")

# 检索
edges = await graphiti.search(
    query="张三在哪工作？",
    group_ids=["user_123"],
    num_results=5,
)

for edge in edges:
    print(f"{edge.source_node_uuid} -> {edge.target_node_uuid}: {edge.fact}")

# 关闭连接
await graphiti.close()
```

---

## 九、自定义实体和边类型

### 9.1 自定义实体类型

```python
from pydantic import BaseModel, Field

# 定义自定义实体类型
class Person(BaseModel):
    """人物实体，表示个人"""
    age: int | None = Field(None, description="年龄")
    occupation: str | None = Field(None, description="职业")
    location: str | None = Field(None, description="所在地")

class Company(BaseModel):
    """公司实体"""
    industry: str | None = Field(None, description="行业")
    founded_year: int | None = Field(None, description="成立年份")
    headquarters: str | None = Field(None, description="总部位置")

# 使用
entity_types = {
    "Person": Person,
    "Company": Company,
}

result = await graphiti.add_episode(
    name="对话",
    episode_body="张三今年30岁，在北京的某科技公司工作",
    source_description="用户输入",
    reference_time=datetime.now(),
    entity_types=entity_types,
)

# 提取的实体会包含自定义属性
for node in result.nodes:
    if "Person" in node.labels:
        print(f"{node.name}: {node.attributes}")
        # 输出：张三: {"age": 30, "location": "北京"}
```

### 9.2 自定义边类型

```python
class Employment(BaseModel):
    """雇佣关系"""
    position: str | None = Field(None, description="职位")
    start_date: str | None = Field(None, description="入职日期")

class Partnership(BaseModel):
    """合作关系"""
    project: str | None = Field(None, description="合作项目")

# 定义边类型映射
edge_types = {
    "EMPLOYMENT": Employment,
    "PARTNERSHIP": Partnership,
}

edge_type_map = {
    ("Person", "Company"): ["EMPLOYMENT"],      # 人-公司：雇佣关系
    ("Company", "Company"): ["PARTNERSHIP"],    # 公司-公司：合作关系
}

result = await graphiti.add_episode(
    name="对话",
    episode_body="张三在谷歌担任软件工程师",
    source_description="用户输入",
    reference_time=datetime.now(),
    entity_types=entity_types,
    edge_types=edge_types,
    edge_type_map=edge_type_map,
)

# 提取的边会包含自定义属性
for edge in result.edges:
    print(f"{edge.name}: {edge.attributes}")
    # 输出：EMPLOYMENT: {"position": "软件工程师"}
```

### 9.3 排除实体类型

```python
# 排除默认的 Entity 类型和自定义的 Event 类型
result = await graphiti.add_episode(
    name="对话",
    episode_body="张三参加了昨天的会议",
    source_description="用户输入",
    reference_time=datetime.now(),
    entity_types={"Event": Event},
    excluded_entity_types=["Entity", "Event"],  # 不提取这些类型
)
```

---

## 十、性能优化

### 10.1 并发控制

```python
import os

# 设置并发限制（默认10，避免 LLM API 限流）
os.environ['SEMAPHORE_LIMIT'] = '20'

# 或在代码中设置
graphiti = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
    max_coroutines=20,  # 覆盖环境变量
)
```

### 10.2 批量处理

```python
# 批量添加情节（比逐个添加快得多）
from graphiti_core.utils.bulk_utils import RawEpisode

bulk_episodes = [
    RawEpisode(
        name="对话1",
        content="张三在北京工作",
        source_description="对话",
        reference_time=datetime(2024, 1, 1),
        source=EpisodeType.text,
    ),
    RawEpisode(
        name="对话2",
        content="李四在上海工作",
        source_description="对话",
        reference_time=datetime(2024, 1, 2),
        source=EpisodeType.text,
    ),
]

result = await graphiti.add_episode_bulk(
    bulk_episodes=bulk_episodes,
    group_id="user_123",
)
```

### 10.3 禁用原始内容存储

```python
# 不存储 episode.content（节省空间）
graphiti = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
    store_raw_episode_content=False,  # 仅保存提取的知识
)
```

---

## 十一、监控和追踪

### 11.1 OpenTelemetry 集成

```python
from opentelemetry import trace
from graphiti_core.tracer import Tracer

# 配置 OpenTelemetry
tracer = trace.get_tracer(__name__)

graphiti = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
    tracer=tracer,                      # 启用追踪
    trace_span_prefix="my_app",         # Span 名称前缀
)

# 自动记录各个操作的 Span
# - add_episode
# - extract_nodes
# - extract_edges
# - search
# 等
```

### 11.2 遥测（Telemetry）

Graphiti 收集匿名使用统计（可禁用）：

```python
# 禁用遥测
import os
os.environ['GRAPHITI_TELEMETRY_ENABLED'] = 'false'
```

收集的信息（不包含敏感数据）：
- 操作系统、Python 版本
- Graphiti 版本
- LLM 提供商类型（OpenAI/Anthropic/Gemini等）
- 数据库类型（Neo4j/FalkorDB/Kuzu等）
- 嵌入模型类型

---

## 十二、项目结构

```
graphiti/
├── graphiti_core/                        # 核心库
│   ├── __init__.py
│   ├── graphiti.py                       # 主入口类
│   ├── nodes.py                          # 节点定义
│   ├── edges.py                          # 边定义
│   ├── graphiti_types.py                 # 类型定义
│   ├── helpers.py                        # 辅助函数
│   ├── errors.py                         # 异常定义
│   │
│   ├── driver/                           # 数据库驱动
│   │   ├── driver.py                     # 驱动接口
│   │   ├── neo4j_driver.py               # Neo4j 驱动
│   │   ├── falkordb_driver.py            # FalkorDB 驱动
│   │   ├── kuzu_driver.py                # Kuzu 驱动
│   │   └── neptune_driver.py             # Neptune 驱动
│   │
│   ├── llm_client/                       # LLM 客户端
│   │   ├── client.py                     # 客户端接口
│   │   ├── openai_client.py              # OpenAI
│   │   ├── anthropic_client.py           # Anthropic
│   │   ├── gemini_client.py              # Gemini
│   │   └── groq_client.py                # Groq
│   │
│   ├── embedder/                         # 嵌入模型
│   │   ├── client.py                     # 嵌入器接口
│   │   ├── openai.py                     # OpenAI
│   │   ├── gemini.py                     # Gemini
│   │   └── voyage.py                     # Voyage
│   │
│   ├── cross_encoder/                    # 重排序器
│   │   ├── client.py                     # 重排序器接口
│   │   ├── openai_reranker_client.py     # OpenAI
│   │   ├── bge_reranker_client.py        # BGE
│   │   └── gemini_reranker_client.py     # Gemini
│   │
│   ├── prompts/                          # Prompt 模板
│   │   ├── extract_nodes.py              # 实体提取
│   │   ├── extract_edges.py              # 关系提取
│   │   ├── dedupe_nodes.py               # 节点去重
│   │   ├── dedupe_edges.py               # 边去重
│   │   ├── invalidate_edges.py           # 边失效
│   │   ├── summarize_nodes.py            # 摘要生成
│   │   └── models.py                     # Prompt 模型
│   │
│   ├── search/                           # 检索模块
│   │   ├── search.py                     # 搜索主逻辑
│   │   ├── search_config.py              # 搜索配置
│   │   ├── search_config_recipes.py      # 预定义配方
│   │   ├── search_filters.py             # 过滤器
│   │   ├── search_utils.py               # 搜索工具
│   │   └── search_helpers.py             # 辅助函数
│   │
│   ├── utils/                            # 工具函数
│   │   ├── bulk_utils.py                 # 批量处理
│   │   ├── datetime_utils.py             # 时间工具
│   │   ├── text_utils.py                 # 文本工具
│   │   │
│   │   ├── maintenance/                  # 维护操作
│   │   │   ├── node_operations.py        # 节点操作
│   │   │   ├── edge_operations.py        # 边操作
│   │   │   ├── community_operations.py   # 社区操作
│   │   │   ├── graph_data_operations.py  # 图数据操作
│   │   │   ├── temporal_operations.py    # 时序操作
│   │   │   └── dedup_helpers.py          # 去重辅助
│   │   │
│   │   └── ontology_utils/               # 本体工具
│   │       └── entity_types_utils.py     # 实体类型工具
│   │
│   ├── models/                           # 数据库查询
│   │   ├── nodes/
│   │   │   └── node_db_queries.py        # 节点查询
│   │   └── edges/
│   │       └── edge_db_queries.py        # 边查询
│   │
│   ├── telemetry/                        # 遥测
│   │   └── telemetry.py
│   │
│   └── tracer.py                         # OpenTelemetry 追踪
│
├── server/                               # FastAPI 服务
│   ├── graph_service/
│   │   ├── main.py                       # 主入口
│   │   ├── config.py                     # 配置
│   │   ├── dto/                          # 数据传输对象
│   │   └── routers/                      # 路由
│   └── README.md
│
├── mcp_server/                           # MCP 服务器
│   ├── graphiti_mcp_server.py            # MCP 入口
│   └── README.md
│
├── examples/                             # 示例
│   ├── quickstart/                       # 快速入门
│   ├── langgraph-agent/                  # LangGraph 集成
│   ├── podcast/                          # 播客解析
│   └── wizard_of_oz/                     # 文本分析
│
├── tests/                                # 测试
│   ├── test_graphiti_int.py              # 集成测试
│   ├── test_node_int.py                  # 节点测试
│   ├── test_edge_int.py                  # 边测试
│   └── ...
│
├── pyproject.toml                        # 项目配置
├── README.md                             # 项目文档
├── LICENSE                               # 许可证
└── Makefile                              # 构建命令
```

---

## 十三、与其他框架的对比

### 13.1 Graphiti vs GraphRAG

| 特性 | GraphRAG | Graphiti |
|------|----------|----------|
| **主要用途** | 静态文档摘要 | 动态数据管理 |
| **数据处理** | 批量处理 | 持续增量更新 |
| **知识结构** | 实体聚类 + 社区摘要 | 情节数据 + 语义实体 + 社区 |
| **检索方法** | 顺序 LLM 摘要 | 混合语义/关键词/图搜索 |
| **适应性** | 低（需重建） | 高（实时更新） |
| **时序处理** | 基本时间戳 | 双时态追踪 |
| **矛盾处理** | LLM 摘要判断 | 时序边失效 |
| **查询延迟** | 秒级到数十秒 | 亚秒级 |
| **自定义类型** | 否 | 是（Pydantic 模型） |
| **可扩展性** | 中等 | 高（优化大规模数据集） |

### 13.2 Graphiti vs 传统 RAG

| 特性 | 传统 RAG | Graphiti |
|------|----------|----------|
| **知识表示** | 文本块向量 | 结构化知识图谱 |
| **关系建模** | 无 | 显式边关系 |
| **时序追踪** | 无 | 双时态模型 |
| **去重机制** | 无 | 智能去重 |
| **矛盾处理** | 无 | 自动失效 |
| **推理能力** | 有限 | 图遍历推理 |
| **更新成本** | 低 | 中（需 LLM 提取） |

---

## 十四、最佳实践

### 14.1 设计建议

1. **分组管理（group_id）**
   - 为不同用户/租户分配独立的 group_id
   - 确保数据隔离和高效检索

2. **情节粒度**
   - 单次对话轮次或单个文档段落
   - 避免过长的情节内容（影响提取质量）

3. **自定义实体类型**
   - 根据领域定义 3-5 个核心实体类型
   - 提供清晰的 docstring 描述

4. **时间戳准确性**
   - 使用实际事件时间作为 reference_time
   - 避免使用当前时间作为历史数据的参考时间

5. **社区更新策略**
   - 定期批量构建社区（如每天一次）
   - 避免每次 add_episode 都更新（性能开销大）

### 14.2 常见陷阱

1. **LLM API 限流**
   - 默认并发为10，根据 API 限额调整 SEMAPHORE_LIMIT
   - 批量处理时注意速率限制

2. **嵌入维度不匹配**
   - 确保 embedder 配置的维度与索引一致
   - OpenAI text-embedding-3-small: 1536 维

3. **去重阈值过高**
   - 相似度阈值 0.95 可能导致不同实体被合并
   - 根据实际情况调整

4. **历史上下文不足**
   - 默认检索最近10个情节
   - 复杂场景可能需要增加 RELEVANT_SCHEMA_LIMIT

### 14.3 调试技巧

```python
import logging

# 启用详细日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('graphiti_core')
logger.setLevel(logging.DEBUG)

# 查看提取的实体和关系
result = await graphiti.add_episode(...)
print("实体:", [(n.name, n.labels) for n in result.nodes])
print("关系:", [(e.fact, e.name) for e in result.edges])

# 检查 UUID 映射（去重信息）
# 在 node_operations.py 中添加日志查看
```

---

## 十五、部署建议

### 15.1 开发环境

```bash
# 安装依赖
uv sync --extra dev

# 运行本地 Neo4j（Docker）
docker-compose up

# 运行测试
make test

# 代码格式化
make format

# 类型检查和 Lint
make lint
```

### 15.2 生产环境

```yaml
# docker-compose.yml 示例
version: '3.8'
services:
  neo4j:
    image: neo4j:5.26
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_PLUGINS=["apoc"]
    ports:
      - "7687:7687"
      - "7474:7474"
    volumes:
      - neo4j_data:/data
  
  graphiti_api:
    build: ./server
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=password
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    ports:
      - "8000:8000"
    depends_on:
      - neo4j

volumes:
  neo4j_data:
```

### 15.3 扩展性考虑

1. **数据库扩展**
   - Neo4j 集群模式（企业版）
   - 读写分离

2. **LLM 服务**
   - 使用专用 API 密钥
   - 考虑自托管模型（Ollama）

3. **缓存策略**
   - 缓存 embedding 结果
   - 缓存常见查询结果

---

## 十六、资源和链接

### 16.1 官方资源

- **GitHub**: https://github.com/getzep/graphiti
- **文档**: https://help.getzep.com/graphiti
- **论文**: [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](https://arxiv.org/abs/2501.13956)
- **Discord**: https://discord.com/invite/W8Kw6bsgXQ

### 16.2 相关项目

- **Zep Cloud**: 完整的托管 Agent 记忆平台
- **LangGraph**: Agent 编排框架（支持 Graphiti 集成）
- **Neo4j**: 推荐的图数据库
- **FalkorDB**: 高性能图数据库

### 16.3 示例和教程

- **快速入门**: `examples/quickstart/`
- **LangGraph Agent**: `examples/langgraph-agent/`
- **播客解析**: `examples/podcast/`
- **电商应用**: `examples/ecommerce/`

---

## 十七、总结

Graphiti 是一个强大的、专为 AI Agent 设计的时序知识图谱框架。其核心优势在于：

1. **实时性**：增量更新，无需批量重建
2. **时序性**：双时态模型，支持历史查询
3. **智能性**：LLM 驱动的提取、去重和矛盾检测
4. **灵活性**：自定义实体和边类型
5. **高效性**：混合搜索，亚秒级检索
6. **可扩展性**：支持多种数据库和 LLM

通过 Graphiti，AI Agent 可以构建持续学习、动态演化的记忆系统，实现更智能的对话、推理和决策能力。

---

**文档版本**: 1.0  
**最后更新**: 2025-10-29  
**作者**: heyi  
**基于 Graphiti 版本**: 0.17.0+

