---
title: "AI Agent 设计模式"
date: 2026-03-07T11:00:00+08:00
draft: false
tags: ["Design Patterns", "AI Agent", "Architecture"]
categories: ["AI Agent"]
---

# AI Agent 设计模式

## 概述

设计模式是解决常见问题的最佳实践。在 AI Agent 系统设计中，我们可以借鉴软件工程中的经典设计模式，结合 AI 的特点，构建更健壮、可维护的系统。

## 1. ReAct 模式（Reasoning + Acting）

**定义**：在推理和行动之间循环，逐步解决问题。

**工作流程**：
```
Thought（思考） → Action（行动） → Observation（观察） → 循环...
```

**Python 实现**：
```python
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent import AgentExecutor
from langchain.llms import OpenAI

tools = [
    Tool(name="Search", func=search, description="搜索网络信息"),
    Tool(name="Calculator", func=calculate, description="计算器"),
]

llm = OpenAI(temperature=0)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

result = agent.run("查询当前日期，然后计算100乘以这个数字")
```

**适用场景**：需要多步推理的任务

---

## 2. Chain of Thought（CoT）

**定义**：引导模型展示推理过程，提高复杂任务准确性。

**Prompt 示例**：
```
"请逐步思考你的推理过程，然后给出最终答案。

问题：小明有3个苹果，吃了2个，又买了5个，最后送了1个，小明现在有多少个苹果？

思考过程：
1. 小明最初有3个苹果
2. 吃了2个，剩1个
3. 买了5个，现在有6个
4. 送了1个，最终有5个

答案：5个"
```

**优势**：
- 提高复杂推理准确性
- 可解释性强
- 便于调试和优化

---

## 3. Tree of Thoughts（ToT）

**定义**：探索多个推理分支，选择最佳方案。

**实现思路**：
```python
def tree_of_thoughts(problem, max_depth=3, num_branches=4):
    """
    树状思考法
    :param problem: 要解决的问题
    :param max_depth: 最大深度
    :param num_branches: 每个节点生成多少个思考分支
    """
    # 初始状态
    thoughts = [problem]
    
    for depth in range(max_depth):
        new_thoughts = []
        for thought in thoughts:
            # 生成多个推理分支
            branches = generate_thought_branches(thought, num_branches)
            new_thoughts.extend(branches)
        thoughts = new_thoughts
    
    # 选择最佳思考路径
    return select_best_thoughts(thoughts, problem)
```

**适用场景**：需要探索多个解决方案的任务

---

## 4. 多智能体模式（Multi-Agent）

**定义**：多个 Agent 协作完成复杂任务。

### 4.1 角色-职责分离
```
Manager Agent（管理者）
  ↓ 负责任务分解和协调
Worker Agent 1（搜索专家）
Worker Agent 2（代码专家）
Worker Agent 3（验证专家）
  ↓ 负责具体任务执行
```

**示例架构**：
```python
class MultiAgentSystem:
    def __init__(self):
        self.manager = ManagerAgent()
        self.search_expert = SearchAgent()
        self.code_expert = CodeAgent()
        self.verify_agent = VerifyAgent()
    
    def execute_task(self, task):
        # 任务分解
        subtasks = self.manager.decompose(task)
        
        # 并行执行子任务
        results = []
        for subtask in subtasks:
            if subtask.type == "search":
                results.append(self.search_expert.execute(subtask))
            elif subtask.type == "code":
                results.append(self.code_expert.execute(subtask))
        
        # 验证结果
        return self.verify_agent.verify(results)
```

### 4.2 讨论-决策模式
多个 Agent 之间进行讨论和辩论，最终达成共识。

---

## 5. 反馈循环模式（Self-Reflection）

**定义**：Agent 自我评估并改进。

**实现**：
```python
def self_reflect(agent_output):
    """
    自我反思与改进
    """
    # 评估输出质量
    critique = evaluate_output(agent_output)
    
    if critique.score < 0.7:
        # 反思问题
       反思 = agent.generate_reflection(agent_output)
        
        # 重新生成
        improved_output = agent.generate_output(reflection)
        
        # 递归反思
        return self_reflect(improved_output)
    
    return agent_output
```

---

## 6. 工具使用模式（Tool Use）

**定义**：Agent 通过调用外部工具完成任务。

**标准接口**：
```python
from typing import TypedDict

class ToolCall(TypedDict):
    tool: str
    parameters: dict

class AgentResponse(TypedDict):
    content: str
    tool_calls: list[ToolCall]
    finish_reason: str
```

**工具注册机制**：
```python
class ToolRegistry:
    def __init__(self):
        self.tools = {}
    
    def register(self, name, func, description):
        self.tools[name] = {
            "func": func,
            "description": description
        }
    
    def call(self, tool_name, parameters):
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found")
        
        return self.tools[tool_name]["func"](**parameters)
```

---

## 总结

| 模式 | 用途 | 优势 | 适用场景 |
|------|------|------|----------|
| ReAct | 多步推理 | 可解释性强 | 复杂任务分解 |
| CoT | 推理过程展示 | 提高准确性 | 数学、逻辑推理 |
| ToT | 多分支探索 | 找到最优解 | 需要多方案比较 |
| Multi-Agent | 协作完成 | 复杂任务拆分 | 大型项目开发 |
| Self-Reflection | 质量保证 | 持续优化 | 关键任务执行 |
| Tool Use | 外部调用 | 扩展能力 | 工具集成 |

选择合适的设计模式是构建优秀 AI Agent 系统的关键。
