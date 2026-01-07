"""
9大模块内部数据对象定义

包含各个模块内部使用的数据对象定义：
1. Instrumented Browser 模块
2. Page Representation Builder 模块
3. LLM Orchestrator 模块
4. Action Translator 模块
5. Element-Matcher Service 模块
6. Sandbox Runner & Verifier 模块
7. Execution Engine 模块
8. Human-in-loop & Learning Pipeline 模块
9. Wrapper Registry 模块
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .global_models import (
    PageSnapshot,
    PageRepresentation,
    PrimitivePlan,
    SelectorCandidate,
    ArtifactBundle,
    ErrorDetail,
    SafetyPolicy,
    CanaryResult,
    WrapperMetadata,
    RPAWorkflow,
    CredentialRef,
    HitlTask,
    ConstraintSpec,
    PrimitiveHint,
    AssertionRule,
    AssertionResult,
    TransactionSummary,
)


# ==================== 1. Instrumented Browser 模块 ====================

@dataclass
class BrowserSessionConfig:
    """浏览器会话配置 - 用于创建浏览器会话的配置"""
    browser_type: str  # 浏览器类型（如：chromium，firefox，webkit）
    headless: bool = True  # 是否以无头模式启动浏览器
    user_agent: Optional[str] = None  # 浏览器的 User-Agent 字符串
    proxy_profile: Optional[str] = None  # 代理配置文件
    cookie_policy: Optional[str] = None  # cookie 策略
    start_url: Optional[str] = None  # 启动时访问的起始 URL
    viewport: Optional[Dict[str, int]] = None  # 浏览器视窗配置


@dataclass
class BrowserSessionStatus:
    """会话状态 - 查看浏览器会话的健康状态"""
    session_id: str  # 会话的唯一标识符
    state: str  # 会话状态：running，idle，closed，crashed
    open_pages: int = 0  # 当前会话中打开的页面数量
    cpu_usage: Optional[float] = None  # 当前会话的 CPU 使用率（百分比）
    memory_usage: Optional[float] = None  # 当前会话的内存使用量（MB）


@dataclass
class CaptureOptions:
    """采集选项 - 控制采集的力度和深度"""
    include_dom: bool = True  # 是否采集 DOM 数据
    include_a11y: bool = True  # 是否采集辅助功能（a11y）数据
    include_network: bool = True  # 是否采集网络请求数据
    include_console: bool = True  # 是否采集控制台输出数据
    include_full_screenshot: bool = True  # 是否截取完整页面截图
    element_screenshot_mode: Optional[str] = None  # 元素截图模式：topN 或 by_filter


@dataclass
class PageSnapshotRequest:
    """页面快照请求 - 指定如何抓取页面的请求"""
    session_id: str  # 关联的浏览器会话 ID
    target: str = "current_tab"  # 捕获目标：current_tab，specific_url，frame
    capture_options: Optional[CaptureOptions] = None  # 采集选项
    timeout_ms: Optional[int] = None  # 超时时间（毫秒）


@dataclass
class PageSnapshotChunk:
    """页面快照块 - 流式采集时一块的数据"""
    chunk_id: str  # 快照块的唯一标识符
    parent_snapshot_id: str  # 父级快照 ID
    elements: List[Dict[str, Any]] = field(default_factory=list)  # 包含页面元素的描述
    scroll_offset: int = 0  # 当前页面的滚动偏移量


@dataclass
class SensitiveFieldMarking:
    """敏感字段标记 - 标记哪些 DOM 区域是敏感字段"""
    eid: str  # 元素 ID
    field_type: str  # 字段类型：password，card，email，personal_id
    mask_strategy: str = "mask"  # 掩码策略：mask、redact


# ==================== 2. Page Representation Builder 模块 ====================

@dataclass
class PageReprBuildRequest:
    """页面表示构建请求 - 向 Page Representation Builder 模块发起构建请求"""
    snapshot_id: Optional[str] = None  # 需要构建页面表示的页面快照的唯一标识符
    snapshot: Optional[PageSnapshot] = None  # 页面快照对象（和snapshot_id二选一）
    top_k_elements: Optional[int] = None  # 页面中需要保留的最重要元素的数量
    chunking_strategy: str = "visual"  # 分块策略：视觉块、DOM 结构块等
    token_budget: Optional[int] = None  # 控制生成页面表示的最大 token 数量
    include_visual_summary: bool = True  # 是否包含视觉摘要
    include_network_hints: bool = True  # 是否包含网络端点提示
    text_snippet_max_len:Optional[int] = None  # 文本片段最大长度


@dataclass
class ElementImportanceScore:
    """元素重要性评分 - 表示构建过程中元素的重要性评分"""
    eid: str  # 元素的唯一标识符
    score: float  # 元素的重要性评分
    reasons: Optional[str] = None  # 评分的原因


@dataclass
class ChunkingConfig:
    """分块配置 - 定义如何按视觉或 DOM 结构将页面分块"""
    max_elements_per_chunk: int = 50  # 每个块中最多包含的元素数量
    group_by: str = "visual_layout"  # 定义分块的依据：dom_tree、visual_layout、heuristic
    merge_small_chunks: bool = True  # 是否将小块合并


@dataclass
class ChunkSummary:
    """块摘要 - 为每个块生成的概要文本"""
    cid: str  # 每个块的唯一标识符
    summary_text: str  # 块的简短自然语言描述
    key_terms: List[str] = field(default_factory=list)  # 块中出现的关键术语


# ==================== 3. LLM Orchestrator 模块 ====================

@dataclass
class Example:
    """示例 - 用于 few-shot 学习的示例"""
    input: Dict[str, Any]  # 示例输入
    output: Dict[str, Any]  # 示例输出


@dataclass
class ContextOverrides:
    """LLM 上下文覆盖项 - 用于调整某些参数或行为"""
    temperature: Optional[float] = None  # 温度参数
    max_tokens: Optional[int] = None  # 最大 token 数
    model: Optional[str] = None  # 模型名称


@dataclass
class LLMPrimitiveRequest:
    """LLM primitive 生成请求 - 一次 LLM 调用的完整上下文"""
    task_id: str  # 当前任务的唯一标识符
    user_intent: Dict[str, Any]  # 用户的意图结构化表达
    page_reprs: List[PageRepresentation] = field(default_factory=list)  # 页面表示列表
    few_shot_examples: List[Example] = field(default_factory=list)  # 少量示例
    safety_policy: Optional[SafetyPolicy] = None  # 安全策略
    llm_context_overrides: Optional[ContextOverrides] = None  # LLM 上下文覆盖项


@dataclass
class LLMPrimitiveResponse:
    """LLM primitive 生成响应 - 包含 LLM 调用的结果以及验证信息"""
    primitive_plan: PrimitivePlan  # LLM 生成的原子操作计划
    raw_llm_output: Optional[str] = None  # LLM 调用的原始输出
    validation_errors: List[ErrorDetail] = field(default_factory=list)  # 验证过程中发现的错误详情
    trace_id: Optional[str] = None  # LLM 调用的唯一追踪标识符


@dataclass
class OrchestratorTrace:
    """编排轨迹 - 记录和跟踪 LLM Orchestrator 的决策过程"""
    trace_id: str  # 编排轨迹的唯一标识符
    task_id: str  # 任务的唯一标识符
    llm_model: Optional[str] = None  # 使用的 LLM 模型的名称
    prompts: List[str] = field(default_factory=list)  # 生成 LLM 请求的 prompts 列表
    plan_versions: List[str] = field(default_factory=list)  # 使用的操作计划版本信息


# ==================== 4. Action Translator 模块 ====================

@dataclass
class WrapperHint:
    """Wrapper 提示 - 提供给 Wrapper 的线索"""
    domain: str  # 目标域名
    capability: Optional[str] = None  # 推荐的能力
    confidence: float = 0.0  # 置信度


@dataclass
class TranslateRequest:
    """转换请求 - 从 Primitive 到 Workflow 的转换请求"""
    task_id: str  # 当前任务的唯一标识符
    primitive_plan: PrimitivePlan  # 原子操作计划
    page_repr: PageRepresentation  # 页面表示对象
    wrapper_hints: List[WrapperHint] = field(default_factory=list)  # 提供给 Wrapper 的线索
    safety_policy: Optional[SafetyPolicy] = None  # 安全策略


@dataclass
class WorkflowStats:
    """工作流统计信息"""
    total_actions: int = 0  # 总操作数量
    mutating_actions: int = 0  # 变更操作数量
    network_api_actions: int = 0  # 网络 API 操作数量
    wrapper_actions: int = 0  # Wrapper 操作数量


@dataclass
class TranslateResponse:
    """转换响应 - TranslateRequest 请求的响应结果"""
    workflow: RPAWorkflow  # 转换后的 RPA 工作流
    warnings: List[str] = field(default_factory=list)  # 转换过程中产生的警告信息
    stats: Optional[WorkflowStats] = None  # 工作流的统计信息


@dataclass
class WorkflowMetadata:
    """工作流元信息 - 记录与 RPAWorkflow 相关的附加信息"""
    translator_version: Optional[str] = None  # 使用的 Translator 版本
    llm_prompt_hash: Optional[str] = None  # 与 LLM 调用相关的 prompt 哈希值
    domain: Optional[str] = None  # 工作流的目标领域或应用域
    candidate_stats: Dict[str, Any] = field(default_factory=dict)  # 候选选择器的统计信息
    created_at: Optional[str] = None  # 工作流元信息的创建时间


@dataclass
class SelectorScore:
    """选择器分数"""
    candidate_id: str  # 候选 ID
    score: float  # 分数
    strategy: str  # 策略类型


@dataclass
class SelectorExplain:
    """选择器可解释信息 - 为每个 selector_bundle 输出为何选择这个 primary 的简短说明"""
    primary_candidate_id: Optional[str] = None  # 主要选择器的候选 ID
    scores: List[SelectorScore] = field(default_factory=list)  # 每种策略/候选的分数
    rationale: Optional[str] = None  # 选择该选择器的自然语言简短说明


# ==================== 5. Element-Matcher Service 模块 ====================

@dataclass
class ElementMatchRequest:
    """匹配请求 - 一次元素匹配请求"""
    task_id: str  # 当前任务的唯一标识符
    page_repr: Optional[PageRepresentation] = None  # 页面表示对象
    page_repr_ref: Optional[str] = None  # 页面表示的引用 ID
    hint: Optional[PrimitiveHint] = None  # 用于提供额外的线索
    candidate_limit: int = 5  # 限制返回的候选选择器数量
    search_scope: str = "full_page"  # 搜索范围：全页面或某个特定块


@dataclass
class MatchDebugInfo:
    """匹配调试信息 - 包含匹配过程中的调试信息"""
    text_score: Optional[float] = None  # 文本匹配得分
    struct_score: Optional[float] = None  # 结构匹配得分
    visual_score: Optional[float] = None  # 视觉匹配得分
    net_score: Optional[float] = None  # 网络特征匹配得分
    feature_vectors_refs: List[str] = field(default_factory=list)  # 特征向量的引用 ID


@dataclass
class ElementMatchResponse:
    """匹配结果 - 匹配请求的结果"""
    candidates: List[SelectorCandidate] = field(default_factory=list)  # 选择器候选列表
    best_candidate: Optional[SelectorCandidate] = None  # 最佳选择器
    debug_features: Optional[MatchDebugInfo] = None  # 匹配调试信息


@dataclass
class EmbeddingVector:
    """嵌入向量 - 表示元素、图像或文本的嵌入向量"""
    dim: int  # 向量的维度
    values: List[float] = field(default_factory=list)  # 向量的数值
    model_version: Optional[str] = None  # 嵌入模型的版本


# ==================== 6. Sandbox Runner & Verifier 模块 ====================

@dataclass
class SandboxRunMode:
    """沙箱运行模式 - 约束 dry-run 的行为"""
    mode: str = "read_only"  # 运行模式：read_only，read_with_limited_mutation


@dataclass
class AssertionResult:
    """断言执行结果 - 对某个 assertion 规则的执行结果（模块特定版本）"""
    assertion: AssertionRule  # 断言规则
    status: str  # 执行结果：pass 或 fail
    actual_value: Optional[str] = None  # 执行时实际获取的值
    message: Optional[str] = None  # 执行结果的附加信息


@dataclass
class SafetyViolation:
    """安全违规记录 - 记录在 dry-run 过程中检测到的可能违反安全策略的行为"""
    policy: SafetyPolicy  # 违反的安全策略
    violation_type: str  # 违规类型：mutate 或 access
    details: Optional[str] = None  # 违规行为的详细描述


@dataclass
class SandboxRiskEstimate:
    """风险评估结果 - 对 workflow 进行静态和动态风险评估的结果"""
    risk_level: str  # 风险级别：low，medium，high
    reasons: List[str] = field(default_factory=list)  # 风险评估的原因
    recommended_actions: List[str] = field(default_factory=list)  # 推荐的操作


# ==================== 7. Execution Engine 模块 ====================

@dataclass
class ExecutionHandle:
    """运行中的任务句柄 - 表示执行任务的句柄"""
    execution_id: str  # 唯一的执行 ID
    task_id: str  # 关联的任务 ID


@dataclass
class ExecutionStatus:
    """执行状态 - 该任务的状态信息"""
    state: str  # 当前状态：queued，running，completed，failed，cancelled
    progress: float = 0.0  # 当前进度，范围从 0 到 100
    current_action_id: Optional[str] = None  # 当前执行的操作 ID


@dataclass
class RetryPolicy:
    """重试策略 - 定义对某个操作失败时的重试控制策略"""
    max_attempts: int  # 最大重试次数
    backoff_strategy: str = "exponential"  # 回退策略：fixed、exponential、jitter
    retry_on: List[str] = field(default_factory=list)  # 触发重试的错误类型


@dataclass
class RateLimitProfile:
    """限速配置 - 定义针对某个域名或租户的限速策略"""
    domain: str  # 目标域名或租户
    max_concurrent: int = 1  # 最大并发请求数
    min_interval_ms: int = 1000  # 每次请求的最小间隔，单位：毫秒
    burst_limit: Optional[int] = None  # 最大突发请求数


@dataclass
class TelemetryRecord:
    """遥测记录 - 记录 Execution Engine 采集的监控数据"""
    timestamp: str  # 时间戳
    task_id: str  # 关联的任务 ID
    workflow_id: Optional[str] = None  # 关联的工作流 ID
    action_id: Optional[str] = None  # 当前执行的操作 ID
    latency_ms: Optional[int] = None  # 当前操作的延迟时间，单位：毫秒
    attempts: int = 1  # 当前操作的尝试次数
    result_status: Optional[str] = None  # 当前操作的执行状态
    extra_metrics: Dict[str, Any] = field(default_factory=dict)  # 其他自定义的监控数据


# ==================== 8. Human-in-loop & Learning Pipeline 模块 ====================

@dataclass
class CreateHitlTaskRequest:
    """创建 HITL 任务请求"""
    task_id: str  # 关联的任务 ID
    workflow_id: Optional[str] = None  # 关联的工作流 ID
    action_id: Optional[str] = None  # 关联的操作 ID
    reason: Optional[str] = None  # 任务生成的原因说明
    artifacts: Optional[ArtifactBundle] = None  # 任务相关的工件数据


@dataclass
class DateRange:
    """日期范围"""
    start: str  # 开始时间
    end: str  # 结束时间


@dataclass
class HitlTaskFilter:
    """HITL 任务查询过滤器"""
    status: Optional[str] = None  # 任务状态：pending，completed，in_progress
    domain: Optional[str] = None  # 任务所属的域名或目标
    created_time_range: Optional[DateRange] = None  # 任务的创建时间范围
    assignee: Optional[str] = None  # 任务的指派人


@dataclass
class HitlTaskList:
    """HITL 任务列表"""
    items: List[HitlTask] = field(default_factory=list)  # HITL 任务的列表


@dataclass
class ReplayRequest:
    """重放请求 - 触发重放某个操作的 dry-run"""
    hitl_task_id: str  # 关联的 HITL 任务 ID
    task_id: str  # 关联的任务 ID
    session_id: str  # 关联的浏览器会话 ID
    workflow_patch: Dict[str, Any] = field(default_factory=dict)  # 修正后的工作流数据


@dataclass
class LabelStoreRef:
    """标签存储引用 - 指向训练样本存储的引用"""
    store_id: str  # 存储的唯一 ID
    path: str  # 存储路径
    format: str = "jsonl"  # 存储格式：parquet，jsonl 等


@dataclass
class ModelTrainingRequest:
    """模型训练请求"""
    model_name: str  # 模型的名称
    data_filter: Dict[str, Any] = field(default_factory=dict)  # 过滤器
    hyper_params: Dict[str, Any] = field(default_factory=dict)  # 超参数设置


@dataclass
class ModelTrainingResult:
    """模型训练结果"""
    job_id: str  # 训练任务的唯一 ID
    status: str  # 训练任务的状态：completed，failed
    new_model_version: Optional[str] = None  # 新模型的版本号
    metrics_summary: Dict[str, Any] = field(default_factory=dict)  # 训练结果摘要


@dataclass
class WrapperCandidate:
    """Wrapper 候选 - 表示某个站点在经过人工修正后适合被生成 wrapper 的候选站点"""
    domain: str  # 站点的域名
    evidence: Dict[str, Any] = field(default_factory=dict)  # 证据，包括修正次数、失败率等信息
    suggested_capabilities: List[str] = field(default_factory=list)  # 推荐的 wrapper 能力


# ==================== 9. Wrapper Registry 模块 ====================

@dataclass
class WrapperFilter:
    """Wrapper 查询过滤器"""
    domain_contains: Optional[str] = None  # 过滤条件：包含的域名
    status: Optional[str] = None  # 过滤条件：wrapper 的状态
    owner: Optional[str] = None  # 过滤条件：wrapper 的所有者
    capability_name: Optional[str] = None  # 过滤条件：wrapper 提供的能力名称


@dataclass
class WrapperList:
    """Wrapper 列表"""
    items: List[WrapperMetadata] = field(default_factory=list)  # 返回的 wrapper 元数据列表


@dataclass
class WrapperContract:
    """Wrapper 契约定义 - 定义 wrapper 的输入输出 schema"""
    capability_name: str  # wrapper 提供的能力名称
    input_json_schema: Dict[str, Any] = field(default_factory=dict)  # 输入数据的 JSON schema
    output_json_schema: Dict[str, Any] = field(default_factory=dict)  # 输出数据的 JSON schema


@dataclass
class WrapperVersionHistory:
    """Wrapper 版本历史 - 记录 wrapper 的版本变更和更新历史"""
    wrapper_id: str  # 对应的 wrapper ID
    version: str  # 版本号
    released_at: Optional[str] = None  # 版本发布时间
    changelog: Optional[str] = None  # 版本变更记录
    rollback_to: Optional[str] = None  # 如果此版本不可用，可回滚到的版本号


@dataclass
class WrapperHealthStatus:
    """Wrapper 健康状态 - 记录了一个 wrapper 的健康状况"""
    wrapper_id: str  # 对应的 wrapper ID
    status: str  # wrapper 的健康状态：healthy，degraded，unhealthy
    last_canary: Optional[CanaryResult] = None  # 最后一次健康检查的结果
    error_rate: float = 0.0  # 错误率
    latency_p95: Optional[float] = None  # 95% 请求的延迟时间


@dataclass
class TenantScope:
    """租户范围 - 控制多租户环境下哪些客户能够使用哪些 wrapper"""
    tenant_id: str  # 租户 ID
    allowed_domains: List[str] = field(default_factory=list)  # 允许使用的域名列表


@dataclass
class AccessControlPolicy:
    """访问控制策略 - 定义某个 wrapper 的访问控制策略"""
    wrapper_id: str  # 对应的 wrapper ID
    allowed_tenants: List[str] = field(default_factory=list)  # 允许使用此 wrapper 的租户列表
    roles: List[str] = field(default_factory=list)  # 允许访问的角色列表

