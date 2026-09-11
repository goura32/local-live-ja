from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MockTool:
    name: str
    description: str
    parameters: dict[str, Any]


class MockToolRegistry:
    """Deterministic, non-destructive tools used only by the PoC."""

    _specs = (
        MockTool(
            name="calculator",
            description="Evaluate a basic arithmetic expression.",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
                "additionalProperties": False,
            },
        ),
        MockTool(
            name="fixed_test_data",
            description="Return one item from deterministic PoC test data.",
            parameters={
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
                "additionalProperties": False,
            },
        ),
        MockTool(
            name="deterministic_time",
            description="Return a fixed time for reproducible tests.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        ),
    )
    _fixed_data = {"status": "PoC固定データ: ready", "version": "local-live-ja-poc"}
    _operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            }
            for spec in self._specs
        ]

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "calculator":
            try:
                expression = str(arguments["expression"])
                tree = ast.parse(expression, mode="eval")
                value = self._eval(tree.body)
                if isinstance(value, float) and value.is_integer():
                    return str(int(value))
                return str(value)
            except (KeyError, SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
                return f"tool_error: calculator input rejected ({type(exc).__name__})"
        if name == "fixed_test_data":
            return str(self._fixed_data.get(str(arguments.get("key")), "PoC固定データ: unknown"))
        if name == "deterministic_time":
            return "2026-01-01T00:00:00+09:00"
        return f"tool_error: unknown tool {name}"

    def _eval(self, node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._operators:
            return self._operators[type(node.op)](self._eval(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in self._operators:
            left = self._eval(node.left)
            right = self._eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("exponent too large")
            return self._operators[type(node.op)](left, right)
        raise ValueError("expression contains unsupported syntax")
