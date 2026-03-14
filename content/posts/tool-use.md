---
title: "Tool Use 与外部工具集成"
date: 2026-03-09T20:30:00+08:00
draft: false
tags: ["Tool Use", "工具调用", "API", "AI Agent"]
categories: ["AI Agent"]
description: "深入探讨 AI Agent 的工具调用机制，包括工具定义、注册、执行和错误处理的最佳实践。"
---

# Tool Use 与外部工具集成

现代 Agent 的核心能力是调用外部工具。通过集成搜索引擎、API、数据库等工具，Agent 从"仅能对话"变为"能行动"。本文详细介绍 Tool Use 的设计与实现。

## 一、Tool 定义与注册

### 1.1 工具接口标准

```python
from typing import Protocol, Dict, Any, Optional
from abc import ABC, abstractmethod

class Tool(ABC):
    """工具基类"""
    name: str
    description: str
    parameters: Dict[str, Any]

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """执行工具"""
        pass

    async def validate_input(self, **kwargs) -> bool:
        """验证输入参数"""
        required = [k for k, v in self.parameters.items()
                   if v.get("required", False)]
        for param in required:
            if param not in kwargs:
                return False
        return True
```

### 1.2 工具注册中心

```python
class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        """注册工具"""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get_tool(self, name: str) -> Optional[Tool]:
        """获取工具"""
        return self.tools.get(name)

    async def execute_tool(
        self,
        name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """执行工具"""
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool {name} not found")

        # 验证输入
        if not await tool.validate_input(**kwargs):
            raise ValueError(f"Invalid parameters for {name}")

        # 执行工具
        return await tool.execute(**kwargs)

    async def list_tools(self) -> List[Dict]:
        """列出所有可用工具"""
        return [{
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters
        } for tool in self.tools.values()]
```

## 二、常见工具实现

### 2.1 搜索引擎工具

```python
class SearchEngineTool(Tool):
    name = "search"
    description = "Search the web for information"

    parameters = {
        "query": {
            "type": "string",
            "description": "Search query",
            "required": True
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum results",
            "default": 5
        }
    }

    def __init__(self, api_key: str = None):
        self.engine = SearchEngine(api_key=api_key)

    async def execute(self, query: str, max_results: int = 5) -> Dict:
        results = await self.engine.search(query, limit=max_results)
        return {
            "success": True,
            "results": results
        }
```

### 2.2 代码执行工具

```python
class CodeExecutionTool(Tool):
    name = "execute_code"
    description = "Execute Python code safely"

    parameters = {
        "code": {
            "type": "string",
            "description": "Python code to execute",
            "required": True
        },
        "timeout": {
            "type": "integer",
            "description": "Timeout in seconds",
            "default": 30
        }
    }

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    async def execute(self, code: str, timeout: int = 30) -> Dict:
        # 安全执行（沙箱环境）
        result = await safe_execute_code(code, timeout)
        return {
            "success": result.success,
            "output": result.output,
            "error": result.error
        }
```

### 2.3 数据库查询工具

```python
class DatabaseTool(Tool):
    name = "query_database"
    description = "Query SQL database"

    parameters = {
        "query": {
            "type": "string",
            "description": "SQL query",
            "required": True
        },
        "database": {
            "type": "string",
            "description": "Database name",
            "default": "default"
        }
    }

    def __init__(self, connection_string: str):
        self.db = DatabaseClient(connection_string)

    async def execute(self, query: str, database: str = "default") -> Dict:
        try:
            results = await self.db.execute(query, database)
            return {
                "success": True,
                "columns": results.columns,
                "rows": results.rows
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
```

## 三、工具选择与调用

### 3.1 工具选择器

```python
class ToolSelector:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    async def select_tools(
        self,
        task: str,
        available_tools: List[Tool]
    ) -> List[Tool]:
        """
        根据任务选择合适的工具
        """
        # 1. 解析任务关键词
        keywords = extract_keywords(task)

        # 2. 匹配工具描述
        scored = []
        for tool in available_tools:
            score = self.calculate_match_score(
                keywords,
                tool.description,
                tool.name
            )
            scored.append((score, tool))

        # 3. 选择最高分工具
        scored.sort(key=lambda x: x[0], reverse=True)
        top_k = [t for s, t in scored if s > 0.5][:3]

        return top_k

    def calculate_match_score(self, keywords, description, name) -> float:
        """计算工具匹配分数"""
        score = 0.0

        # 关键词匹配
        desc_words = set(description.lower().split())
        keyword_set = set(keywords)
        score += len(keyword_set & desc_words) / len(keyword_set)

        # 名称匹配
        name_words = set(name.lower().split())
        score += len(name_words & keyword_set) / len(keyword_set)

        return score
```

### 3.2 工具调用编排

```python
class ToolCallOrchestrator:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.history = []

    async def plan_tool_calls(
        self,
        task: str,
        max_calls: int = 5
    ) -> List[ToolCall]:
        """
        规划工具调用序列
        """
        # 1. LLM 规划
        planner_prompt = f"""
        Task: {task}

        Available tools:
        {[t.description for t in self.registry.list_tools()]}

        Plan step-by-step tool calls:
        """

        plan = await llm.generate(planner_prompt)

        # 2. 解析工具调用
        calls = parse_tool_calls(plan, max_calls)

        return calls

    async def execute_tool_calls(
        self,
        calls: List[ToolCall]
    ) -> List[ToolResult]:
        """
        执行工具调用序列
        """
        results = []

        for call in calls:
            try:
                # 执行工具
                result = await self.registry.execute_tool(
                    call.name,
                    **call.parameters
                )
                results.append({
                    "tool": call.name,
                    "parameters": call.parameters,
                    "result": result,
                    "success": result.get("success", False)
                })

                # 如果失败，终止执行
                if not result.get("success"):
                    logger.warning(
                        f"Tool {call.name} failed: {result.get('error')}"
                    )
                    break

            except Exception as e:
                results.append({
                    "tool": call.name,
                    "error": str(e)
                })
                break

        return results
```

## 四、错误处理与安全

### 4.1 工具执行安全

```python
class SafeToolExecutor:
    async def execute_safely(self, tool: Tool, **kwargs) -> Dict:
        """安全执行工具"""
        try:
            # 1. 参数验证
            if not await tool.validate_input(**kwargs):
                return {
                    "success": False,
                    "error": "Invalid parameters"
                }

            # 2. 执行超时控制
            try:
                result = await asyncio.wait_for(
                    tool.execute(**kwargs),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                return {
                    "success": False,
                    "error": f"Tool execution timeout ({self.timeout}s)"
                }

            # 3. 结果验证
            if not self.validate_result(result):
                return {
                    "success": False,
                    "error": "Invalid result format"
                }

            return {
                "success": True,
                "result": result
            }

        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
```

### 4.2 访问控制

```python
class ToolPermissionManager:
    def __init__(self):
        self.permissions: Dict[str, List[str]] = {
            "user1": ["search", "calculator"],
            "admin": ["*"]  # 所有工具
        }

    def check_permission(self, user: str, tool_name: str) -> bool:
        """检查用户是否有权限调用工具"""
        if tool_name in self.permissions.get(user, []):
            return True

        # 检查通配符
        for role in self.permissions.get(user, []):
            if role == "*":
                return True
            if tool_name.startswith(role):
                return True

        return False
```

## 五、工具市场

### 5.1 工具生态

```python
class ToolMarketplace:
    """工具市场，管理第三方工具"""

    def __init__(self):
        self.registered_tools: Dict[str, Tool] = {}

    async def publish_tool(
        self,
        tool: Tool,
        owner: str,
        description: str
    ) -> str:
        """发布工具"""
        tool_id = f"{owner}_{tool.name}_{uuid.uuid4()}"

        self.registered_tools[tool_id] = {
            "tool": tool,
            "owner": owner,
            "description": description,
            "downloads": 0,
            "rating": 0.0
        }

        logger.info(f"Published tool: {tool_id}")
        return tool_id

    async def download_tool(self, tool_id: str) -> Tool:
        """下载工具"""
        record = self.registered_tools.get(tool_id)
        if not record:
            raise ValueError(f"Tool {tool_id} not found")

        # 更新下载计数
        record["downloads"] += 1

        return record["tool"]

    async def rate_tool(self, tool_id: str, rating: float):
        """给工具评分"""
        record = self.registered_tools.get(tool_id)
        if not record:
            raise ValueError(f"Tool {tool_id} not found")

        # 计算平均评分
        ratings = record.get("ratings", [])
        ratings.append(rating)
        record["rating"] = sum(ratings) / len(ratings)
        record["ratings"] = ratings
```

## 六、性能优化

### 6.1 工具缓存

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
async def cached_tool_call(
    tool_name: str,
    cache_key: str,
    **kwargs
) -> Dict:
    """缓存工具调用结果"""
    # 检查缓存
    cached = cache.get(cache_key)
    if cached:
        return cached

    # 执行工具
    result = await execute_tool(tool_name, **kwargs)

    # 存入缓存
    cache.set(cache_key, result, ttl=3600)  # 1小时过期
    return result
```

### 6.2 并行执行

```python
async def execute_tools_parallel(calls: List[ToolCall]) -> List[ToolResult]:
    """并行执行多个工具"""
    tasks = [
        execute_tool(call.name, **call.parameters)
        for call in calls
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 处理异常
    return [
        {
            "call": calls[i],
            "result": result if not isinstance(result, Exception) else {"error": str(result)}
        }
        for i, result in enumerate(results)
    ]
```

## 七、监控与日志

```python
class ToolMonitor:
    def __init__(self):
        self.metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "avg_duration": 0.0
        }

    def log_call(
        self,
        tool_name: str,
        success: bool,
        duration: float
    ):
        """记录工具调用"""
        self.metrics["total_calls"] += 1

        if success:
            self.metrics["successful_calls"] += 1
        else:
            self.metrics["failed_calls"] += 1

        # 更新平均时长
        count = self.metrics["total_calls"]
        self.metrics["avg_duration"] = (
            (self.metrics["avg_duration"] * (count - 1) + duration) / count
        )

        # 记录详细日志
        logger.info(
            "tool_call",
            extra={
                "tool": tool_name,
                "success": success,
                "duration_ms": duration * 1000
            }
        )
```

## 八、最佳实践

1. **工具设计原则**：单一职责、接口清晰、错误处理完善
2. **安全第一**：沙箱执行、参数验证、访问控制
3. **性能优化**：缓存、并行、延迟加载
4. **可观测性**：详细日志、性能监控、错误追踪
5. **可扩展性**：插件化架构、版本管理

## 九、参考资源

- [LangChain Tools](https://python.langchain.com/modules/tools/)
- [AutoGen Tools](https://microsoft.github.io/autogen/)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)

---

**相关文章**：
- [AI Agent 系统概览](/posts/ai-agent-概览.html)
- [Agent 工程化最佳实践](/posts/agent-工程化.html)
