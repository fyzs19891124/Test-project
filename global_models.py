"""
全局数据对象定义

包含所有模块共享的基础数据对象定义，包括：
- 0.1 任务和会话基础
- 0.2 页面与DOM基础
- 0.3 Page Representation基础
- 0.4 Primitive & Workflow基础
- 0.5 验证与执行结果基础
- 0.6 人机协同&学习基础
- 0.7 Wrapper基础
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ==================== 0.1 任务和会话基础 ====================

@dataclass
class TaskId:
    """任务标识 - 平台内所有一次"自动化任务"的唯一ID"""
    value: str  # 任务的唯一标识符（UUID 或雪花 ID）


@dataclass
class UserIntent:
    """用户意图 - 用户一句/多句自然语言指令的结构化表达"""
    raw_text: str  # 用户原始话语（自然语言指令）
    language: str  # 用户意图的语言（如：英语、中文等）
    goal: str  # 归纳后的任务目标
    constraints: Dict[str, Any] = field(default_factory=dict)  # 约束条件
    priority: Optional[str] = None  # 任务的优先级
    deadline: Optional[str] = None  # 任务的截止时间


@dataclass
class ExecutionEnv:
    """执行环境描述 - 描述任务运行所处的基础环境"""
    browser_type: str  # 浏览器类型（如：chrome，firefox）
    os: str  # 操作系统类型（如：Windows，macOS）
    proxy_profile: Optional[str] = None  # 代理配置文件
    locale: Optional[str] = None  # 本地化设置（如：en-US，zh-CN）
    time_zone: Optional[str] = None  # 时区信息
    tenant_id: Optional[str] = None  # 租户 ID


@dataclass
class BrowserSessionHandle:
    """浏览器会话句柄 - 代表一个受控浏览器实例/上下文"""
    session_id: str  # 会话的唯一标识符
    browser_type: str  # 浏览器类型（如：chromium，firefox）
    profile_id: Optional[str] = None  # 配置文件 ID
    status: str = "running"  # 会话状态：running，closed，crashed
    last_active_at: Optional[str] = None  # 最后一次活动时间


@dataclass
class TaskContext:
    """任务上下文 - 承载从用户指令到执行环境的一揽子上下文"""
    task_id: TaskId  # 关联的任务 ID
    user_intent: UserIntent  # 用户意图
    env: ExecutionEnv  # 执行环境
    session_refs: List[BrowserSessionHandle] = field(default_factory=list)  # 浏览器会话句柄列表
    created_at: Optional[str] = None  # 任务创建时间


# ==================== 0.2 页面与DOM基础 ====================

@dataclass
class A11yNode:
    """无障碍树节点 - 可访问性树上的节点"""
    node_id: str  # 节点 ID
    role: str  # 节点的角色（如：button，textbox）
    name: Optional[str] = None  # 节点名称
    label: Optional[str] = None  # 可访问性标签
    children_ids: List[str] = field(default_factory=list)  # 子节点 ID 列表
    bounding_box: Optional[Dict[str, Any]] = None  # 节点的边界框


@dataclass
class NetworkTrace:
    """网络追踪条目 - 页面内 XHR/fetch 等请求的摘要"""
    request_id: str  # 请求 ID
    url: str  # 请求的 URL
    method: str  # 请求方法（如：GET，POST）
    status: Optional[int] = None  # 响应状态码
    request_headers: Dict[str, str] = field(default_factory=dict)  # 请求头
    response_headers: Dict[str, str] = field(default_factory=dict)  # 响应头
    request_payload_summary: Optional[str] = None  # 请求负载的摘要
    response_sample: Optional[str] = None  # 响应内容样本


@dataclass
class ConsoleLogEntry:
    """控制台日志条目 - 收集 console.log、错误等信息"""
    level: str  # 日志级别（如：info，warn，error）
    message: str  # 日志信息
    timestamp: Optional[str] = None  # 日志生成时间
    stack: Optional[str] = None  # 错误栈信息（如果是错误日志）


@dataclass
class ElementRawDescriptor:
    """原始页面元素描述 - DOM 上一个元素的完整描述（未精简版）"""
    eid: str  # 元素 ID
    tag: str  # HTML 标签名
    text_full: Optional[str] = None  # 元素的完整文本内容
    text_snippet: Optional[str] = None  # 元素的文本片段
    outer_html: Optional[str] = None  # 元素的外部 HTML
    attributes: Dict[str, str] = field(default_factory=dict)  # 元素的属性
    bbox: Optional[Dict[str, Any]] = None  # 元素的视口或页面坐标信息
    xpath: Optional[str] = None  # 元素的 XPath 路径
    css_path: Optional[str] = None  # 元素的 CSS 路径
    screenshot_crop_base64: Optional[str] = None  # 截取的元素截图的 Base64 编码
    a11y_role: Optional[str] = None  # 元素的可访问性角色
    a11y_label: Optional[str] = None  # 元素的可访问性标签
    is_interactive: bool = False  # 元素是否可交互


@dataclass
class PageSnapshot:
    """页面原始快照 - 由 Instrumented Browser 抓取的原始大快照"""
    snapshot_id: str  # 页面快照的唯一标识符
    url: str  # 页面 URL
    title: Optional[str] = None  # 页面标题
    timestamp: Optional[str] = None  # 快照时间戳
    dom_snapshot: Optional[str] = None  # 页面 DOM 快照（文本）
    a11y_tree: List[A11yNode] = field(default_factory=list)  # 可访问性树的节点信息
    elements: List[ElementRawDescriptor] = field(default_factory=list)  # 页面元素的原始描述列表
    network_traces: List[NetworkTrace] = field(default_factory=list)  # 网络请求追踪条目
    full_screenshot_base64: Optional[str] = None  # 页面截图的 Base64 编码
    console_logs: List[ConsoleLogEntry] = field(default_factory=list)  # 控制台日志条目列表


# ==================== 0.3 Page Representation 基础 ====================

@dataclass
class PageMeta:
    """页面元信息 - 页面的基础元信息"""
    url: str  # 页面 URL
    title: Optional[str] = None  # 页面标题
    language: Optional[str] = None  # 页面语言
    viewport_size: Optional[Dict[str, int]] = None  # 页面视口大小（宽度和高度）
    meta_tags: Dict[str, str] = field(default_factory=dict)  # 页面 Meta 标签


@dataclass
class ElementView:
    """精简元素视图 - 从 ElementRawDescriptor 抽取出的 LLM 友好版本"""
    eid: str  # 元素的唯一标识符
    tag: str  # HTML 标签名
    text_snippet: Optional[str] = None  # 元素的文本片段（<= 140 字符）
    aria: Optional[str] = None  # 元素的 ARIA 属性
    role: Optional[str] = None  # 元素的角色（如：button，input）
    xpath: Optional[str] = None  # 元素的 XPath 路径
    css_path: Optional[str] = None  # 元素的 CSS 路径
    bbox: Optional[Dict[str, Any]] = None  # 元素的视口坐标（位置和大小）
    screenshot_crop_base64: Optional[str] = None  # 元素截图的 Base64 编码


@dataclass
class VisualHotspot:
    """视觉热点 - 页面中关键区域的小图"""
    bbox: Dict[str, Any]  # 热点区域的边界框
    image_base64: str  # 热点区域的截图 Base64 编码


@dataclass
class VisualSummary:
    """视觉摘要 - 页面的全页缩略图和关键区域小图"""
    full_page_small: Optional[str] = None  # 全页缩略图的 Base64 编码
    hotspots: List[VisualHotspot] = field(default_factory=list)  # 页面中关键区域的小图


@dataclass
class NetworkHint:
    """网络端点提示 - 对重要 XHR/fetch 端点的摘要"""
    endpoint_url: str  # 网络请求的端点 URL
    method: str  # 请求方法（如：GET，POST）
    param_pattern: Optional[str] = None  # 请求参数的模式
    sample_response_schema: Optional[str] = None  # 响应的样本数据格式


@dataclass
class PageChunk:
    """页面块 - 一个语义、视觉或结构上的子区域"""
    cid: str  # 页面块的唯一标识符
    elements: List[ElementView] = field(default_factory=list)  # 页面块包含的元素列表
    summary: Optional[str] = None  # 页面块的简短自然语言描述
    visual_preview_base64: Optional[str] = None  # 页面块的视觉预览图的 Base64 编码
    importance_score: float = 0.0  # 页面块的重要性评分


@dataclass
class PageRepresentation:
    """页面表示 - Page Builder 生成的为 LLM 优化的页面压缩表示"""
    repr_id: str  # 页面表示的唯一标识符
    meta: PageMeta  # 页面元信息
    chunks: List[PageChunk] = field(default_factory=list)  # 页面块的列表
    visual_summary: Optional[VisualSummary] = None  # 页面缩略图及关键区域小图
    network_hints: List[NetworkHint] = field(default_factory=list)  # 网络端点提示
    token_budget_estimate: Optional[int] = None  # 页面表示的 token 预算估算


# ==================== 0.4 Primitive & Workflow 基础 ====================

@dataclass
class PrimitiveHint:
    """primitive 提示 - 帮助 Element-Matcher 找元素的线索"""
    type: str  # 提示类型（如：text，visual_ref，network_hint）
    value: str  # 提示的具体内容
    context: Optional[str] = None  # 提示的上下文


@dataclass
class ConstraintSpec:
    """约束规格 - 定义某个 primitive 的约束描述"""
    limit: Optional[int] = None  # 任务的最大限制
    timeout_ms: Optional[int] = None  # 设置的最大超时，单位为毫秒
    filters: List[Dict[str, Any]] = field(default_factory=list)  # 用于筛选的条件


@dataclass
class AssertionRule:
    """断言规则 - 验证提取内容或状态的规则"""
    type: str  # 断言类型（如：regex，enum，json_schema）
    value: str  # 断言的值
    severity: str = "warning"  # 断言的严重性等级（如：critical，warning）
    message: Optional[str] = None  # 断言失败时的错误消息


@dataclass
class PrimitiveOp:
    """原子操作 - 语义动作，如 find_list、click、extract 等"""
    op: str  # 操作类型（如：find_list、click、extract）
    target: Optional[str] = None  # 操作目标
    hints: List[PrimitiveHint] = field(default_factory=list)  # 提示信息
    body: List["PrimitiveOp"] = field(default_factory=list)  # 支持复合操作（如：for_each）
    constraints: Optional[ConstraintSpec] = None  # 约束条件
    assertions: List[AssertionRule] = field(default_factory=list)  # 断言规则


@dataclass
class PrimitivePlan:
    """原子操作计划 - LLM 输出的高层语义动作列表"""
    plan_id: str  # 计划的唯一标识符
    task_id: TaskId  # 关联的任务 ID
    ops: List[PrimitiveOp] = field(default_factory=list)  # 原子操作列表
    assumptions: Optional[str] = None  # 任务的假设条件
    constraints: Optional[str] = None  # 任务的约束条件


@dataclass
class SelectorCandidate:
    """选择器候选 - 单一定位策略"""
    type: str  # 定位策略类型（如：xpath、text_fuzzy、visual_bbox、network_api）
    value: str  # 定位策略的具体值
    score: float = 0.0  # 定位策略的得分
    provenance: str = ""  # 定位策略的来源（如：text、struct、visual、net）
    extra: Dict[str, Any] = field(default_factory=dict)  # 附加信息


@dataclass
class SelectorExplain:
    """选择器可解释信息 - 为何选择这个 primary 的简短说明"""
    primary_candidate_id: Optional[str] = None  # 主要选择器的候选 ID
    scores: List[Dict[str, Any]] = field(default_factory=list)  # 每种策略/候选的分数
    rationale: Optional[str] = None  # 选择该选择器的自然语言简短说明


@dataclass
class SelectorBundle:
    """选择器候选包 - 解决一个动作可能有多种定位方式的统一结构"""
    candidates: List[SelectorCandidate] = field(default_factory=list)  # 定位候选列表
    selection_strategy: str = "first_success"  # 选择策略（first_success / weighted / parallel_probe）
    explainability: Optional[SelectorExplain] = None  # 解释信息


@dataclass
class FailurePolicy:
    """失败策略 - 描述当动作执行失败时应采取的处理策略"""
    policy_type: str  # 失败策略类型（如：try_next_candidate / retry / skip / abort / escalate_to_human）
    max_retries: int = 0  # 最大重试次数
    backoff_strategy: Optional[str] = None  # 重试的退避策略（如：exponential）
    escalation_channel: Optional[str] = None  # 人工干预渠道


@dataclass
class RPAAction:
    """RPA 动作 - 最低可执行单元"""
    id: str  # 动作的唯一标识符
    action: str  # 动作类型（如：click，input）
    description: Optional[str] = None  # 动作描述
    selector_bundle: Optional[SelectorBundle] = None  # 动作使用的选择器候选包
    params: Dict[str, Any] = field(default_factory=dict)  # 动作的参数
    assertions: List[AssertionRule] = field(default_factory=list)  # 动作执行时的断言规则
    on_failure: Optional[FailurePolicy] = None  # 动作失败时的处理策略
    is_mutating: bool = False  # 动作是否为变更性操作
    tags: List[str] = field(default_factory=list)  # 动作标签


@dataclass
class TransactionGroup:
    """事务组 - 一组动作被视为一个事务单元"""
    group_id: str  # 事务组的唯一标识符
    action_ids: List[str] = field(default_factory=list)  # 包含在事务中的动作 ID 列表
    compensation_plan: Optional[str] = None  # 事务回滚计划
    status: str = "pending"  # 当前事务组的状态（如：pending、completed、failed）


@dataclass
class WorkflowMetadata:
    """工作流元数据"""
    translator_version: Optional[str] = None  # 使用的 Translator 版本
    llm_prompt_hash: Optional[str] = None  # 与 LLM 调用相关的 prompt 哈希值
    domain: Optional[str] = None  # 工作流的目标领域或应用域
    candidate_stats: Dict[str, Any] = field(default_factory=dict)  # 候选选择器的统计信息
    created_at: Optional[str] = None  # 工作流元信息的创建时间


@dataclass
class RPAWorkflow:
    """RPA 工作流 - 由 Translator 生成，Execution Engine 和 Sandbox 的核心输入"""
    workflow_id: str  # 工作流的唯一标识符
    task_id: TaskId  # 关联的任务 ID
    steps: List[RPAAction] = field(default_factory=list)  # 工作流的动作步骤
    transaction_groups: List[TransactionGroup] = field(default_factory=list)  # 事务组
    metadata: Optional[WorkflowMetadata] = None  # 工作流元数据


# ==================== 0.5 验证与执行结果基础 ====================

@dataclass
class ErrorDetail:
    """错误详情 - 统一的错误对象"""
    code: str  # 错误码
    message: str  # 错误信息
    module: Optional[str] = None  # 错误发生的模块
    retryable: bool = False  # 是否可重试
    raw_error: Optional[Dict[str, Any]] = None  # 原始错误数据


@dataclass
class ArtifactBundle:
    """工件包 - 执行或验证过程中采集的截图、DOM片段、日志等"""
    screenshots: List[str] = field(default_factory=list)  # 截图列表（Base64 编码）
    dom_snippets: List[str] = field(default_factory=list)  # DOM 片段列表
    console_logs: List[str] = field(default_factory=list)  # 控制台日志列表
    network_logs: List[str] = field(default_factory=list)  # 网络日志列表


@dataclass
class AssertionResult:
    """断言执行结果 - 对某个 assertion 规则的执行结果"""
    assertion: AssertionRule  # 断言规则
    status: str  # 执行结果：pass 或 fail
    actual_value: Optional[str] = None  # 执行时实际获取的值
    message: Optional[str] = None  # 执行结果的附加信息


@dataclass
class DryRunActionResult:
    """单个动作的 Dry Run 执行结果"""
    action_id: str  # 动作 ID
    status: str  # 动作状态（如：success，failed）
    selected_candidate: Optional[SelectorCandidate] = None  # 选择的定位候选项
    assertion_results: List[AssertionResult] = field(default_factory=list)  # 断言验证结果列表
    artifacts: Optional[ArtifactBundle] = None  # 采集的工件


@dataclass
class DryRunRequest:
    """沙箱执行的输入 - 用于模拟任务的执行流程"""
    task_id: str  # 任务 ID
    workflow: RPAWorkflow  # 工作流定义
    session_id: str  # 会话 ID
    mode: str = "dry-run"  # 模式
    max_steps: Optional[int] = None  # 最大步骤数


@dataclass
class DryRunResult:
    """沙箱执行的输出 - 用于检查问题"""
    task_id: str  # 任务 ID
    workflow_id: str  # 工作流 ID
    action_results: List[DryRunActionResult] = field(default_factory=list)  # 动作执行结果列表
    overall_status: str = "pending"  # 总体执行状态
    needs_human_review: bool = False  # 是否需要人工审核
    summary: Optional[str] = None  # 执行摘要


@dataclass
class ExecutionActionResult:
    """表示某一步骤在 Execution Engine 中的实际执行结果"""
    action_id: str  # 动作 ID
    status: str  # 执行状态（如：success，failed）
    attempts: int = 1  # 执行尝试次数
    latency_ms: Optional[int] = None  # 执行延迟，单位毫秒
    selected_candidate: Optional[SelectorCandidate] = None  # 选择的定位候选项
    output_data: Optional[Dict[str, Any]] = None  # 执行结果数据
    error_info: Optional[ErrorDetail] = None  # 错误信息


@dataclass
class TransactionSummary:
    """事务摘要 - 描述一次事务组的执行结果"""
    group_id: str  # 事务组的唯一 ID
    status: str  # 当前事务的状态（如：success，failed）
    attempted_actions: int = 0  # 尝试执行的操作数量
    compensation_triggered: bool = False  # 是否触发了补偿操作
    notes: Optional[str] = None  # 事务的备注信息


@dataclass
class ExecutionRequest:
    """Execution Engine 的输入"""
    task_id: str  # 任务 ID
    workflow: RPAWorkflow  # 工作流定义
    session_id: str  # 会话 ID
    execution_mode: str = "real-time"  # 执行模式（如：real-time，batch）
    safety_overrides: Dict[str, Any] = field(default_factory=dict)  # 安全覆盖配置
    rate_limit_profile: Optional[Dict[str, Any]] = None  # 限速配置


@dataclass
class ExecutionResult:
    """Execution Engine 的输出"""
    task_id: str  # 任务 ID
    workflow_id: str  # 工作流 ID
    action_results: List[ExecutionActionResult] = field(default_factory=list)  # 动作执行结果列表
    transaction_summaries: List[TransactionSummary] = field(default_factory=list)  # 事务摘要列表
    outputs: Dict[str, Any] = field(default_factory=dict)  # 最终业务数据


@dataclass
class SafetyPolicy:
    """安全策略 - 定义任务的敏感操作的约束"""
    allow_login: bool = False  # 是否允许登录操作
    allow_mutation: bool = False  # 是否允许变更操作
    max_rate_per_domain: Optional[int] = None  # 每个域名的最大请求速率
    blocked_actions: List[str] = field(default_factory=list)  # 被阻止的操作列表
    pii_masking_rules: List[str] = field(default_factory=list)  # 敏感数据屏蔽规则


@dataclass
class CredentialRef:
    """凭证引用 - 用于存储与任务相关的凭证的引用"""
    vault_key: str  # 凭证存储的密钥
    scopes: List[str] = field(default_factory=list)  # 凭证的访问权限范围
    expires_at: Optional[str] = None  # 凭证的过期时间


# ==================== 0.6 人机协同&学习基础 ====================

@dataclass
class HitlTask:
    """人工干预任务 - 当某个操作或计划需要人工决策或修正时生成"""
    hitl_task_id: str  # 人工干预任务的唯一标识符
    task_id: str  # 关联的任务 ID
    workflow_id: Optional[str] = None  # 关联的工作流 ID
    action_id: Optional[str] = None  # 关联的动作 ID
    reason: Optional[str] = None  # 人工干预的原因
    status: str = "pending"  # 人工干预任务的状态（如：pending，completed）
    artifacts: Optional[ArtifactBundle] = None  # 收集的工件


@dataclass
class CorrectionPayload:
    """人工修正内容 - 人类标注的正确动作或选择器"""
    hitl_task_id: str  # 关联的人工干预任务 ID
    selected_candidate: Optional[SelectorCandidate] = None  # 人工标注的选择器或动作
    new_selector: Optional[str] = None  # 新的选择器（如果标注的选择器有误）
    comments: Optional[str] = None  # 备注说明
    promote_to_wrapper: bool = False  # 是否将修正的选择器提升到 Wrapper 级别


@dataclass
class TrainingExample:
    """训练样本 - 用于训练 Element-Matcher 或 Translator 的样本数据"""
    example_id: str  # 训练样本的唯一标识符
    hint: str  # 给定的提示或线索
    correct_selector: str  # 标注的正确选择器
    negative_candidates: List[SelectorCandidate] = field(default_factory=list)  # 负样本候选选择器
    page_repr_snapshot: Optional[str] = None  # 页面表示的快照
    labels: List[str] = field(default_factory=list)  # 样本的标签


@dataclass
class ModelVersion:
    """模型版本 - 记录某个模型的版本与其性能效果"""
    model_name: str  # 模型的名称
    version: str  # 模型版本号
    trained_at: Optional[str] = None  # 模型训练时间
    metrics: Dict[str, Any] = field(default_factory=dict)  # 模型的性能指标
    changelog: Optional[str] = None  # 模型更新的变更记录


# ==================== 0.7 Wrapper 基础 ====================

@dataclass
class WrapperCapability:
    """Wrapper 能力 - 描述 wrapper 提供的单一功能"""
    name: str  # 能力的名称（如：login、list_items、detail_extract）
    input_schema: Dict[str, Any] = field(default_factory=dict)  # 输入参数的结构描述
    output_schema: Dict[str, Any] = field(default_factory=dict)  # 输出结果的结构描述
    side_effects: List[str] = field(default_factory=list)  # 该能力可能引起的副作用


@dataclass
class WrapperMetadata:
    """Wrapper 元数据 - 描述一个站点 wrapper"""
    wrapper_id: str  # wrapper 的唯一标识符
    domain_pattern: str  # 匹配的域名模式
    capabilities: List[WrapperCapability] = field(default_factory=list)  # wrapper 提供的能力列表
    version: str = "1.0.0"  # wrapper 的版本号
    owner: Optional[str] = None  # wrapper 的所有者
    status: str = "active"  # wrapper 当前状态（如：active，inactive）


@dataclass
class WrapperInvocationRequest:
    """Wrapper 调用请求"""
    wrapper_id: str  # 目标 wrapper 的唯一标识符
    capability_name: str  # 要调用的 wrapper 能力的名称
    session_id: str  # 当前会话的唯一标识符
    payload: Dict[str, Any] = field(default_factory=dict)  # 调用该能力所需的输入数据
    auth_context: Optional[CredentialRef] = None  # 授权凭证的引用


@dataclass
class WrapperInvocationResponse:
    """Wrapper 调用响应"""
    status: str  # 调用的结果状态，如 success 或 failure
    data: Optional[Dict[str, Any]] = None  # 调用结果的数据
    latency_ms: Optional[int] = None  # 调用延迟，单位毫秒
    error_info: Optional[ErrorDetail] = None  # 错误信息


@dataclass
class CanaryResult:
    """Canary 检查结果 - 定期健康检查 wrapper 的结果"""
    wrapper_id: str  # 被检查的 wrapper 的唯一标识符
    ok: bool  # 检查是否通过
    sample_output_summary: Optional[str] = None  # 健康检查的样本输出摘要
    latency_ms: Optional[int] = None  # 健康检查延迟，单位毫秒
    checked_at: Optional[str] = None  # 健康检查的时间

