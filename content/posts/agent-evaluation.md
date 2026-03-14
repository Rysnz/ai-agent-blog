---
title: "AI Agent 评估与测试方法论"
date: 2026-03-11T09:00:00+08:00
draft: false
tags: ["评估", "测试", "基准测试", "AI Agent"]
categories: ["AI Agent"]
description: "系统化地评估和测试 AI Agent 系统的方法论，包括评估指标设计、测试框架构建和持续监控策略。"
---

# AI Agent 评估与测试方法论

构建 AI Agent 系统容易，构建**可信赖**的 AI Agent 系统却很困难。评估和测试是确保 Agent 系统在生产环境中可靠运行的关键。本文将系统介绍 AI Agent 的评估体系和测试方法。

## 一、为什么 Agent 评估比普通软件更复杂

传统软件测试基于确定性逻辑——给定输入 A，总会产生输出 B。而 AI Agent 面临：

- **非确定性**：相同输入可能产生不同但都合理的输出
- **长链路**：多步骤规划中任一环节出错都影响最终结果  
- **工具依赖**：外部 API 调用结果不可控
- **主观性**：许多任务的"好坏"难以量化
- **涌现行为**：Agent 可能产生设计者未预期的行为

## 二、评估维度与指标

### 2.1 任务完成评估

```python
from dataclasses import dataclass
from typing import Optional
from enum import Enum

class CompletionStatus(Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    ERROR = "error"

@dataclass
class TaskResult:
    task_id: str
    status: CompletionStatus
    output: str
    expected: str
    steps_taken: int
    tokens_used: int
    latency_ms: float
    tool_calls: list
    error: Optional[str] = None

def evaluate_task_completion(result: TaskResult) -> dict:
    """评估任务完成质量"""
    metrics = {}
    
    # 1. 完成率
    metrics["completion_rate"] = 1.0 if result.status == CompletionStatus.SUCCESS else \
                                  0.5 if result.status == CompletionStatus.PARTIAL else 0.0
    
    # 2. 效率指标
    metrics["steps_efficiency"] = min(1.0, 5 / max(result.steps_taken, 1))
    metrics["token_efficiency"] = min(1.0, 1000 / max(result.tokens_used, 1))
    
    # 3. 延迟
    metrics["latency_score"] = 1.0 if result.latency_ms < 2000 else \
                                0.5 if result.latency_ms < 5000 else 0.1
    
    # 4. 工具使用
    metrics["tool_call_count"] = len(result.tool_calls)
    
    return metrics
```

### 2.2 输出质量评估（LLM-as-Judge）

使用强大的 LLM（如 GPT-4）作为评判者评估输出质量：

```python
from openai import OpenAI

client = OpenAI()

JUDGE_SYSTEM_PROMPT = """
你是一位严格、公正的 AI 系统评估专家。
你的职责是评估 AI Agent 的输出质量，给出客观评分和改进建议。
评分范围：1-10，必须给出具体理由。
"""

def llm_judge_evaluation(
    task: str,
    agent_output: str,
    reference_output: str = None,
    criteria: list = None
) -> dict:
    
    if criteria is None:
        criteria = ["准确性", "完整性", "相关性", "清晰度", "有用性"]
    
    criteria_str = "\n".join([f"- {c}" for c in criteria])
    reference_str = f"\n\n参考答案：\n{reference_output}" if reference_output else ""
    
    eval_prompt = f"""
请评估以下 AI Agent 的输出质量：

任务：{task}

Agent 输出：
{agent_output}{reference_str}

评估维度：
{criteria_str}

请返回 JSON 格式的评估结果：
{{
    "overall_score": <1-10的总分>,
    "dimension_scores": {{
        <维度名>: <1-10的分数>
    }},
    "strengths": ["优点列表"],
    "weaknesses": ["不足列表"],
    "improvement_suggestions": ["改进建议"],
    "reasoning": "评分理由（100字以内）"
}}
"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": eval_prompt}
        ],
        response_format={"type": "json_object"}
    )
    
    import json
    return json.loads(response.choices[0].message.content)
```

### 2.3 轨迹评估（Trajectory Evaluation）

评估 Agent 的决策路径，而不仅仅是最终输出：

```python
@dataclass
class AgentStep:
    step_id: int
    thought: str           # Agent 的推理过程
    action: str            # 选择的动作
    action_input: dict     # 动作参数
    observation: str       # 动作结果
    is_correct: bool = None  # 人工标注

class TrajectoryEvaluator:
    """轨迹评估器"""
    
    def evaluate_trajectory(self, steps: list[AgentStep]) -> dict:
        metrics = {
            "total_steps": len(steps),
            "efficiency_score": 0,
            "redundancy_rate": 0,
            "error_rate": 0,
            "backtrack_count": 0
        }
        
        if not steps:
            return metrics
        
        # 检测回溯（重复相同动作）
        action_history = []
        for step in steps:
            if step.action in action_history[-3:]:
                metrics["backtrack_count"] += 1
            action_history.append(step.action)
        
        # 效率分 = 1 / (1 + 额外步骤数)
        expected_steps = max(1, len(steps) // 2)
        metrics["efficiency_score"] = expected_steps / len(steps)
        
        # 错误率（如有标注）
        annotated = [s for s in steps if s.is_correct is not None]
        if annotated:
            metrics["error_rate"] = sum(1 for s in annotated if not s.is_correct) / len(annotated)
        
        return metrics
    
    def detect_hallucination(self, steps: list[AgentStep]) -> list:
        """检测幻觉：Agent 声称执行了实际未执行的操作"""
        issues = []
        for step in steps:
            # 如果思考中提到了工具调用但实际动作不同
            if "我已经查询了" in step.thought and "search" not in step.action:
                issues.append({
                    "step": step.step_id,
                    "type": "hallucination",
                    "description": f"Agent 声称已查询但实际未执行搜索"
                })
        return issues
```

## 三、测试框架设计

### 3.1 单元测试：工具测试

```python
import pytest
from unittest.mock import AsyncMock, patch

class TestSearchTool:
    """搜索工具单元测试"""
    
    @pytest.fixture
    def search_tool(self):
        from tools.search import WebSearchTool
        return WebSearchTool(api_key="test-key")
    
    @pytest.mark.asyncio
    async def test_basic_search(self, search_tool):
        """测试基本搜索功能"""
        with patch.object(search_tool, '_call_api') as mock_api:
            mock_api.return_value = {
                "results": [
                    {"title": "Test Result", "url": "https://example.com", "snippet": "Test content"}
                ]
            }
            result = await search_tool.execute(query="test query")
        
        assert result["success"] is True
        assert len(result["results"]) > 0
        assert "title" in result["results"][0]
    
    @pytest.mark.asyncio
    async def test_empty_query(self, search_tool):
        """测试空查询处理"""
        result = await search_tool.execute(query="")
        assert result["success"] is False
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_api_timeout(self, search_tool):
        """测试 API 超时处理"""
        with patch.object(search_tool, '_call_api', side_effect=TimeoutError):
            result = await search_tool.execute(query="test")
        
        assert result["success"] is False
        assert result["error_type"] == "timeout"
```

### 3.2 集成测试：端到端流程

```python
class AgentIntegrationTest:
    """Agent 端到端集成测试"""
    
    TEST_SCENARIOS = [
        {
            "name": "简单信息查询",
            "task": "今天北京的天气如何？",
            "expected_tools": ["weather_api"],
            "success_criteria": lambda r: "天气" in r or "temperature" in r.lower(),
            "max_steps": 3,
            "timeout_seconds": 10
        },
        {
            "name": "复杂研究任务",
            "task": "比较 LangChain 和 AutoGen 框架的主要区别",
            "expected_tools": ["web_search"],
            "success_criteria": lambda r: len(r) > 200 and "LangChain" in r,
            "max_steps": 8,
            "timeout_seconds": 30
        },
        {
            "name": "代码生成任务",
            "task": "写一个 Python 函数，计算斐波那契数列前 n 项",
            "expected_tools": [],
            "success_criteria": lambda r: "def" in r and "fibonacci" in r.lower(),
            "max_steps": 2,
            "timeout_seconds": 15
        }
    ]
    
    async def run_scenario(self, agent, scenario: dict) -> dict:
        import asyncio
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            result = await asyncio.wait_for(
                agent.run(scenario["task"]),
                timeout=scenario["timeout_seconds"]
            )
            
            elapsed = asyncio.get_event_loop().time() - start_time
            success = scenario["success_criteria"](result.output)
            
            return {
                "scenario": scenario["name"],
                "success": success,
                "output": result.output[:500],
                "steps": result.step_count,
                "elapsed": elapsed,
                "tools_used": result.tools_called
            }
        
        except asyncio.TimeoutError:
            return {
                "scenario": scenario["name"],
                "success": False,
                "error": "timeout",
                "elapsed": scenario["timeout_seconds"]
            }
        except Exception as e:
            return {
                "scenario": scenario["name"],
                "success": False,
                "error": str(e)
            }
```

### 3.3 回归测试与 Golden Dataset

```python
import json
from pathlib import Path

class GoldenDatasetEvaluator:
    """基于黄金数据集的回归评估"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.dataset = self._load_dataset()
    
    def _load_dataset(self) -> list:
        with open(self.dataset_path) as f:
            return json.load(f)
    
    async def evaluate(self, agent, judge_client=None) -> dict:
        results = []
        
        for item in self.dataset:
            agent_output = await agent.run(item["task"])
            
            # 规则评估（快速、确定性）
            rule_score = self._rule_based_eval(
                agent_output.output,
                item.get("required_keywords", []),
                item.get("forbidden_keywords", [])
            )
            
            # LLM 评估（慢但准确）
            llm_score = None
            if judge_client and item.get("reference_output"):
                eval_result = llm_judge_evaluation(
                    task=item["task"],
                    agent_output=agent_output.output,
                    reference_output=item["reference_output"]
                )
                llm_score = eval_result["overall_score"] / 10
            
            results.append({
                "id": item["id"],
                "task": item["task"],
                "rule_score": rule_score,
                "llm_score": llm_score,
                "passed": rule_score > 0.7
            })
        
        return self._aggregate_results(results)
    
    def _rule_based_eval(
        self,
        output: str,
        required_keywords: list,
        forbidden_keywords: list
    ) -> float:
        score = 1.0
        
        for kw in required_keywords:
            if kw.lower() not in output.lower():
                score -= 1.0 / max(len(required_keywords), 1)
        
        for kw in forbidden_keywords:
            if kw.lower() in output.lower():
                score -= 0.3
        
        return max(0.0, score)
    
    def _aggregate_results(self, results: list) -> dict:
        passed = sum(1 for r in results if r["passed"])
        rule_scores = [r["rule_score"] for r in results]
        llm_scores = [r["llm_score"] for r in results if r["llm_score"] is not None]
        
        return {
            "total": len(results),
            "passed": passed,
            "pass_rate": passed / len(results),
            "avg_rule_score": sum(rule_scores) / len(rule_scores),
            "avg_llm_score": sum(llm_scores) / len(llm_scores) if llm_scores else None,
            "results": results
        }
```

## 四、持续监控

### 4.1 生产监控指标

```python
from prometheus_client import Counter, Histogram, Gauge
import time

# 关键监控指标
agent_requests_total = Counter(
    'agent_requests_total',
    'Total agent requests',
    ['status', 'task_type']
)

agent_latency_seconds = Histogram(
    'agent_latency_seconds',
    'Agent task latency',
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

agent_token_usage = Counter(
    'agent_token_usage_total',
    'Total tokens consumed',
    ['model', 'usage_type']
)

agent_tool_calls = Counter(
    'agent_tool_calls_total',
    'Tool call counts',
    ['tool_name', 'status']
)

quality_score = Gauge(
    'agent_quality_score',
    'Rolling average quality score'
)


class MonitoredAgent:
    """带监控的 Agent 包装器"""
    
    def __init__(self, base_agent, quality_evaluator=None):
        self.agent = base_agent
        self.quality_evaluator = quality_evaluator
    
    async def run(self, task: str, task_type: str = "general") -> dict:
        start_time = time.time()
        status = "success"
        
        try:
            result = await self.agent.run(task)
            
            # 异步质量评估
            if self.quality_evaluator:
                score = await self.quality_evaluator.quick_score(task, result.output)
                quality_score.set(score)
            
            # 记录指标
            for tool_call in result.tool_calls:
                agent_tool_calls.labels(
                    tool_name=tool_call.name,
                    status="success"
                ).inc()
            
            return result
            
        except Exception as e:
            status = "error"
            raise
        finally:
            elapsed = time.time() - start_time
            agent_requests_total.labels(status=status, task_type=task_type).inc()
            agent_latency_seconds.observe(elapsed)
```

### 4.2 异常检测与告警

```python
import statistics
from collections import deque

class AnomalyDetector:
    """Agent 行为异常检测"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.latency_window = deque(maxlen=window_size)
        self.quality_window = deque(maxlen=window_size)
        self.error_window = deque(maxlen=window_size)
    
    def record(self, latency: float, quality: float, is_error: bool):
        self.latency_window.append(latency)
        self.quality_window.append(quality)
        self.error_window.append(1 if is_error else 0)
    
    def check_anomalies(self) -> list:
        alerts = []
        
        if len(self.latency_window) < 10:
            return alerts
        
        # 延迟异常
        avg_latency = statistics.mean(self.latency_window)
        std_latency = statistics.stdev(self.latency_window) if len(self.latency_window) > 1 else 0
        recent_latency = statistics.mean(list(self.latency_window)[-10:])
        
        if recent_latency > avg_latency + 2 * std_latency:
            alerts.append({
                "type": "latency_spike",
                "severity": "warning",
                "message": f"近期延迟({recent_latency:.2f}s)显著高于平均值({avg_latency:.2f}s)"
            })
        
        # 错误率异常
        recent_error_rate = sum(list(self.error_window)[-20:]) / 20
        if recent_error_rate > 0.2:
            alerts.append({
                "type": "high_error_rate",
                "severity": "critical" if recent_error_rate > 0.5 else "warning",
                "message": f"近期错误率: {recent_error_rate:.1%}"
            })
        
        # 质量下降
        if len(self.quality_window) >= 20:
            recent_quality = statistics.mean(list(self.quality_window)[-10:])
            baseline_quality = statistics.mean(list(self.quality_window)[:10])
            
            if recent_quality < baseline_quality * 0.8:
                alerts.append({
                    "type": "quality_degradation",
                    "severity": "warning",
                    "message": f"输出质量下降：{baseline_quality:.2f} → {recent_quality:.2f}"
                })
        
        return alerts
```

## 五、评估最佳实践

### 评估流水线

```
开发阶段评估
├── 单元测试（工具、组件）
├── 提示词测试（少样本验证）
└── 小规模人工评估

集成测试
├── 端到端场景测试
├── 边界条件测试
└── 压力测试

上线前评估
├── 黄金数据集回归测试
├── A/B 测试（新旧版本对比）
└── 安全红队测试

生产监控
├── 实时指标监控
├── 异常检测与告警
└── 定期人工审查样本
```

### 关键原则总结

| 原则 | 描述 |
|------|------|
| **多维评估** | 不依赖单一指标，结合规则、模型、人工多种方式 |
| **持续回归** | 建立黄金数据集，每次更新后运行完整评估 |
| **生产监控** | 线上实时监控，及早发现性能退化 |
| **人机结合** | 自动化评估提效，人工审查保证质量 |
| **失败分析** | 深入分析失败案例，提取改进洞察 |
| **可复现性** | 固定随机种子，确保评估结果可重现 |

## 六、总结

AI Agent 的评估是一个系统工程，需要在开发、测试、上线、运营各阶段构建完整的质量保障体系。从单元测试到生产监控，每个环节都不可忽视。只有建立科学的评估体系，才能在不断迭代中保持 Agent 系统的高质量和可靠性。
