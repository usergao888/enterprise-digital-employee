"""Agent Skills loader (SKILL.md with progressive disclosure).

A skill is a folder containing ``SKILL.md``::

    ---
    name: expense-qa
    description: 回答差旅/报销制度问题
    when_to_use: 住宿标准、报销额度、发票、审批流程
    version: 1
    allowed-tools: knowledge_search, calculator
    ---

    # body: steps, tool calls, output contract

Only the metadata is always injected into the planner prompt; the body is
injected when the skill is selected. This keeps the context budget small while
still giving the executor the full procedure for the current task.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .textutil import tokenize


@dataclass
class Skill:
    name: str
    description: str
    when_to_use: str
    version: str
    allowed_tools: list[str]
    path: Path
    body: str = ""

    def metadata_line(self) -> str:
        return f"- {self.name}: {self.description}（适用：{self.when_to_use}）"


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Minimal YAML-frontmatter parser; no PyYAML dependency."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta: dict[str, str] = {}
    for raw in parts[1].splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower().replace("-", "_")
        meta[key] = value.strip().strip('"').strip("'")
    return meta, parts[2].lstrip("\n")


class SkillLibrary:
    """Scan ``<root>/*/SKILL.md`` and select skills by query overlap."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._skills: dict[str, Skill] = {}
        self.reload()

    def reload(self) -> None:
        self._skills = {}
        if not self.root.exists():
            return
        for path in sorted(self.root.glob("*/SKILL.md")):
            try:
                skill = self._parse(path)
            except (OSError, ValueError):
                continue
            if skill.name:
                self._skills[skill.name] = skill

    def _parse(self, path: Path) -> Skill:
        text = path.read_text(encoding="utf-8")
        meta, body = _split_frontmatter(text)
        name = (meta.get("name") or path.parent.name).strip()
        allowed = [
            item.strip()
            for item in (meta.get("allowed_tools") or "").split(",")
            if item.strip()
        ]
        return Skill(
            name=name,
            description=(meta.get("description") or "").strip(),
            when_to_use=(meta.get("when_to_use") or "").strip(),
            version=(meta.get("version") or "1").strip(),
            allowed_tools=allowed,
            path=path,
            body=body.strip(),
        )

    def names(self) -> list[str]:
        return sorted(self._skills)

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def describe_for_prompt(self, max_chars: int = 800) -> str:
        """Metadata only; never includes the body."""
        if not self._skills:
            return "（未配置技能）"
        text = "\n".join(
            skill.metadata_line() for skill in sorted(self._skills.values(), key=lambda s: s.name)
        )
        return text[:max_chars]

    def select(self, query: str, limit: int = 2) -> list[Skill]:
        """Pick at most ``limit`` skills by keyword overlap with metadata."""
        q = (query or "").strip()
        if not q:
            return []
        terms = {t for t in tokenize(q) if len(t) >= 2}
        if not terms:
            return []
        scored: list[tuple[int, Skill]] = []
        for skill in self._skills.values():
            haystack = f"{skill.name} {skill.description} {skill.when_to_use}".lower()
            score = sum(1 for term in terms if term.lower() in haystack)
            if score:
                scored.append((score, skill))
        scored.sort(key=lambda item: (-item[0], item[1].name))
        return [skill for _, skill in scored[: max(1, limit)]]

    def instructions_for(self, skills: list[Skill], max_chars: int = 1800) -> str:
        """Body text for selected skills only (progressive disclosure step 2)."""
        chunks = [
            f"## 技能 {skill.name}\n{skill.body}"
            for skill in skills
            if skill.body.strip()
        ]
        return "\n\n".join(chunks)[:max_chars]
