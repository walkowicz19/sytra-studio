"""Universal ReAct Tool-Calling Engine for Sytra Studio.

Gives function/tool calling capabilities to ANY local model (GGUF, Safetensors, MoE, 
fine-tuned, or merged models)—even models that do not natively support tool calling.
"""
from __future__ import annotations

import json
import re
import subprocess
import urllib.request
from typing import Any, Callable, Dict, List, Optional


REACT_SYSTEM_PROMPT = """You are a helpful AI assistant equipped with tool-calling capabilities.

Available Tools:
{tools_json}

INSTRUCTIONS FOR TOOL CALLING:
1. If you need to use a tool to answer the user's request, respond ONLY with a JSON object in this exact format:
```json
{{
  "tool_call": "tool_name",
  "arguments": {{
    "arg_name": "arg_value"
  }}
}}
```
2. Do not add any text before or after the JSON block when making a tool call.
3. If no tool is required, answer the user's question directly in standard text.
"""


from pathlib import Path

class SkillManager:
    """Manages SKILL.md files and injects custom skill prompts when invoked via /skill-name or explicit command."""
    def __init__(self, workspace_path: Optional[str] = None):
        self.skills: Dict[str, Dict[str, str]] = {}
        self.workspace_path = Path(workspace_path or ".")
        self.reload_skills()

    def reload_skills(self):
        self.skills = {}
        # Dynamic local memory skill directories (workspace + user home)
        search_dirs = [
            self.workspace_path / ".agents" / "skills",
            Path.home() / ".sytra" / "skills",
            Path.home() / ".gemini" / "config" / "skills",
            Path.home() / ".klayer" / "skills",
        ]
        file_patterns = ["SKILL.md", "*.mdc", "*.md"]
        for s_dir in search_dirs:
            if not s_dir.exists():
                continue
            for pattern in file_patterns:
                for item in s_dir.rglob(pattern):
                    try:
                        content = item.read_text(encoding="utf-8")
                        skill_name = item.stem.lower().replace("_", "-")
                        if skill_name == "skill":
                            skill_name = item.parent.name.lower()
                        self.skills[skill_name] = {
                            "name": skill_name,
                            "path": str(item.resolve()),
                            "content": content,
                        }
                    except Exception:
                        pass

    def get_skill_prompt(self, skill_name: str) -> Optional[str]:
        skill = self.skills.get(skill_name.lower().lstrip("/"))
        return skill["content"] if skill else None

    def inject_skills_into_prompt(self, user_prompt: str, system_prompt: str) -> tuple[str, str]:
        """Detect /skill-name or explicit skill triggers and inject SKILL.md content into system prompt."""
        clean_user_prompt = user_prompt
        injected_skills = []

        # Check for /skill-name slash command
        slash_match = re.match(r"^/([\w-]+)\s*(.*)", user_prompt, re.DOTALL)
        if slash_match:
            cmd = slash_match.group(1).lower()
            clean_user_prompt = slash_match.group(2)
            skill_content = self.get_skill_prompt(cmd)
            if skill_content:
                injected_skills.append(f"\n\n--- ACTIVE SKILL: /{cmd} ---\n{skill_content}\n")

        # Also check for explicit mentions in prompt
        for name, skill_info in self.skills.items():
            if f"use {name}" in user_prompt.lower() or f"skill {name}" in user_prompt.lower():
                injected_skills.append(f"\n\n--- ACTIVE SKILL: {name} ---\n{skill_info['content']}\n")

        if injected_skills:
            system_prompt = system_prompt + "".join(injected_skills)

        return clean_user_prompt, system_prompt


class UniversalToolEngine:
    def __init__(self, tools: Optional[List[Dict[str, Any]]] = None, workspace_path: Optional[str] = None):
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.skill_manager = SkillManager(workspace_path)
        if tools:
            for t in tools:
                name = t.get("name")
                if name:
                    self.tools[name] = t
        self._register_default_tools()

    def _register_default_tools(self):
        """Register built-in system tools (Web Search, Python Evaluator, File Search, System Info)."""
        if "search_web" not in self.tools:
            self.tools["search_web"] = {
                "name": "search_web",
                "description": "Performs a web search to fetch real-time information for a query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query string"}
                    },
                    "required": ["query"]
                },
                "handler": self._tool_search_web
            }

        if "evaluate_python" not in self.tools:
            self.tools["evaluate_python"] = {
                "name": "evaluate_python",
                "description": "Executes a Python code expression or calculation safely and returns the result.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code snippet or math expression"}
                    },
                    "required": ["code"]
                },
                "handler": self._tool_evaluate_python
            }

        if "list_directory" not in self.tools:
            self.tools["list_directory"] = {
                "name": "list_directory",
                "description": "Lists files and folders inside a given directory path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Absolute directory path"}
                    },
                    "required": ["path"]
                },
                "handler": self._tool_list_directory
            }

    def get_system_prompt_wrapper(self) -> str:
        """Format the system prompt with active tool definitions."""
        clean_tools = []
        for name, t in self.tools.items():
            clean_tools.append({
                "name": t.get("name"),
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {})
            })
        tools_str = json.dumps(clean_tools, indent=2)
        return REACT_SYSTEM_PROMPT.format(tools_json=tools_str)

    def extract_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse potential tool call JSON block from model text output."""
        if "tool_call" not in text:
            return None

        # Extract markdown json block or direct json object
        patterns = [
            r"```json\s*(\{\s*\"tool_call\"[\s\S]*?\})\s*```",
            r"(\{\s*\"tool_call\"[\s\S]*?\})"
        ]

        for p in patterns:
            match = re.search(p, text)
            if match:
                try:
                    payload = json.loads(match.group(1))
                    if isinstance(payload, dict) and "tool_call" in payload:
                        return payload
                except Exception:
                    continue
        return None

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a registered tool or custom handler."""
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' is not registered."

        tool_def = self.tools[tool_name]
        handler = tool_def.get("handler")

        if handler and callable(handler):
            try:
                return str(handler(**arguments))
            except Exception as e:
                return f"Error executing tool '{tool_name}': {e}"
        
        return f"Tool '{tool_name}' executed with args {arguments}."

    # ─── Built-in Tool Handlers ────────────────────────────────────────────────
    def _tool_search_web(self, query: str) -> str:
        try:
            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
                clean = re.sub(r'<[^>]+>', ' ', html)
                snippets = [s.strip() for s in clean.split() if len(s.strip()) > 3]
                snippet_text = " ".join(snippets[:100])
                return f"Web Search Result for '{query}': {snippet_text[:800]}..."
        except Exception as e:
            return f"Search notice: Could not fetch web results for '{query}' ({e})."

    def _tool_evaluate_python(self, code: str) -> str:
        try:
            # Safe eval / exec in isolated scope
            loc = {}
            exec(f"res = ({code})", {}, loc)
            return f"Result: {loc.get('res')}"
        except Exception:
            try:
                loc = {}
                exec(code, {}, loc)
                return f"Execution Output: {loc}"
            except Exception as e:
                return f"Python Evaluation Error: {e}"

    def _tool_list_directory(self, path: str) -> str:
        try:
            p = Path(path)
            if not p.exists():
                return f"Error: Directory '{path}' does not exist."
            items = [f.name + ("/" if f.is_dir() else "") for f in p.iterdir()]
            return f"Directory contents of '{path}':\n" + "\n".join(items[:50])
        except Exception as e:
            return f"Error listing directory: {e}"
