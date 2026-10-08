#!/usr/bin/env python3
"""提示文字表面的理解风险，不改写输入内容。

本工具刻意保持保守。提示需要结合上下文复核；没有提示也不能证明读者已经理解。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_MAX_CJK_SENTENCE = 70
DEFAULT_MAX_ENGLISH_SENTENCE = 32
DEFAULT_MAX_CJK_PARAGRAPH = 220
DEFAULT_MAX_ENGLISH_PARAGRAPH = 110

JARGON_TERMS = (
    "赋能",
    "抓手",
    "闭环",
    "颗粒度",
    "范式",
    "链路",
    "对齐",
    "收敛",
    "鲁棒",
    "解耦",
    "原子化",
    "底座",
    "端到端",
    "方法论",
)

VAGUE_ACTION_PATTERNS = (
    r"处理一下",
    r"进行(?:相关|相应|必要)?(?:的)?(?:处理|操作|优化|调整)",
    r"适当(?:地)?(?:处理|调整|优化|配置)",
    r"按需(?:处理|调整|优化|配置)",
    r"视情况(?:处理|调整|优化|配置)",
    r"做好(?:相关|相应)?(?:工作|处理|优化)",
    r"尽快(?:处理|跟进|完成)",
)

PATRONIZING_PATTERNS = (
    r"显而易见",
    r"显然",
    r"很简单",
    r"非常简单",
    r"轻松(?:地)?",
    r"只需(?:要)?",
    r"\b(?:obviously|trivial(?:ly)?|simply|just)\b",
)

DOUBLE_NEGATIVE_PATTERNS = (
    r"不得不",
    r"不能不",
    r"不(?:可能|会)?不",
    r"并非[^。！？!?\n]{0,30}不",
    r"\bnot\s+(?:uncommon|impossible|unable|unnecessary|incorrect|invalid|ineffective)\b",
)

PROTECTED_PATTERNS = (
    re.compile(r"(?ms)\A---[ \t]*\n.*?\n---[ \t]*(?:\n|\Z)"),
    re.compile(r"(?ms)^[ \t]*(?P<fence>`{3,}|~{3,})[^\n]*\n.*?^[ \t]*(?P=fence)[ \t]*$"),
    re.compile(r"(?s)(?<!`)`+[^`\n]+`+(?!`)"),
    re.compile(r"\b(?:https?|file)://[^\s<>()]+"),
    re.compile(r"\b[A-Za-z]:\\(?:[^\s<>:\"|?*]+\\?)+"),
    re.compile(r"(?<![\w])(?:~|\.{1,2})?/(?:[A-Za-z0-9._~%+\-]+/?)+"),
)


@dataclass(frozen=True)
class Issue:
    code: str
    message: str
    line: int
    excerpt: str
    severity: str = "warning"


def _blank_preserving_newlines(value: str) -> str:
    return "".join("\n" if char == "\n" else " " for char in value)


def mask_protected(text: str) -> str:
    """Mask code, URLs, and paths while preserving offsets and line numbers."""

    masked = text
    for pattern in PROTECTED_PATTERNS:
        masked = pattern.sub(lambda match: _blank_preserving_newlines(match.group(0)), masked)
    return masked


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def excerpt(value: str, limit: int = 96) -> str:
    compact = re.sub(r"\s+", " ", value).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def cjk_count(value: str) -> int:
    return len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", value))


def english_word_count(value: str) -> int:
    return len(re.findall(r"\b[A-Za-z]+(?:[-'][A-Za-z]+)*\b", value))


def _paragraphs(text: str) -> Iterable[tuple[int, str]]:
    for match in re.finditer(r"(?ms)(?:\A|(?<=\n))[ \t]*(\S.*?)(?=\n[ \t]*\n|\Z)", text):
        value = match.group(1).strip()
        if not value:
            continue

        lines = [line for line in value.splitlines() if line.strip()]
        if len(lines) > 1 and all(
            re.match(r"^[ \t]*(?:[-+*][ \t]+|\d+[.)][ \t]+)", line) for line in lines
        ):
            cursor = 0
            for line in lines:
                relative = value.find(line, cursor)
                yield match.start(1) + relative, line.strip()
                cursor = relative + len(line)
            continue

        yield match.start(1), value


def _sentences(text: str) -> Iterable[tuple[int, str]]:
    for match in re.finditer(
        r"[^。！？!?;；\n]+?(?:[。！？!?;；]|\.(?=\s|\Z)|\n|\Z)",
        text,
    ):
        value = match.group(0).strip()
        if value:
            yield match.start(), value


def _markdown_prefix_removed(value: str) -> str:
    return re.sub(
        r"(?m)^[ \t]{0,3}(?:#{1,6}[ \t]+|[-+*][ \t]+|\d+[.)][ \t]+|>[ \t]*)",
        "",
        value,
    )


def _acronym_is_expanded(text: str, start: int, end: int, acronym: str) -> bool:
    before = text[max(0, start - 80) : start]
    after = text[end : min(len(text), end + 80)]

    # Full term followed by the abbreviation: Large Language Model (LLM) or 大型语言模型（LLM）.
    if re.search(r"[A-Za-z\u3400-\u9fff][^。！？!?\n()（）]{2,70}[（(][ \t]*$", before) and re.match(
        r"[ \t]*[)）]", after
    ):
        return True

    # Abbreviation followed by a local explanation: LLM (large language model) or LLM（大型语言模型）.
    explanation = re.match(r"[ \t]*[（(]([^()（）\n]{2,70})[)）]", after)
    if explanation:
        content = explanation.group(1)
        if cjk_count(content) >= 2 or english_word_count(content) >= 2:
            return True

    # A prior explicit mapping in the same document counts as an expansion.
    escaped = re.escape(acronym)
    if re.search(rf"[^。！？!?\n]{{2,70}}[（(][ \t]*{escaped}[ \t]*[)）]", text[:start]):
        return True
    if re.search(rf"\b{escaped}\b[ \t]*[（(][^()（）\n]{{2,70}}[)）]", text[:start]):
        return True
    return False


def lint_text(
    text: str,
    *,
    max_cjk_sentence: int = DEFAULT_MAX_CJK_SENTENCE,
    max_english_sentence: int = DEFAULT_MAX_ENGLISH_SENTENCE,
    max_cjk_paragraph: int = DEFAULT_MAX_CJK_PARAGRAPH,
    max_english_paragraph: int = DEFAULT_MAX_ENGLISH_PARAGRAPH,
) -> list[Issue]:
    """Return advisory issues for prose in *text*."""

    masked = mask_protected(text)
    issues: list[Issue] = []

    for offset, paragraph in _paragraphs(masked):
        prose = _markdown_prefix_removed(paragraph)
        cjk = cjk_count(prose)
        words = english_word_count(prose)
        if cjk > max_cjk_paragraph or words > max_english_paragraph:
            measure = f"{cjk} 个汉字" if cjk >= words else f"{words} 个英文词"
            issues.append(
                Issue(
                    code="LONG_PARAGRAPH",
                    message=f"这一段约有 {measure}；检查它是否承担了多个任务。",
                    line=line_number(masked, offset),
                    excerpt=excerpt(paragraph),
                )
            )

        jargon = [term for term in JARGON_TERMS if term in prose]
        if len(jargon) >= 2:
            issues.append(
                Issue(
                    code="JARGON_CLUSTER",
                    message="同一段集中出现多个抽象术语；保留必要术语，并就地说明它们在这里做什么："
                    + "、".join(jargon),
                    line=line_number(masked, offset),
                    excerpt=excerpt(paragraph),
                )
            )

    for offset, sentence in _sentences(masked):
        prose = _markdown_prefix_removed(sentence)
        cjk = cjk_count(prose)
        words = english_word_count(prose)
        if cjk > max_cjk_sentence or words > max_english_sentence:
            measure = f"{cjk} 个汉字" if cjk >= words else f"{words} 个英文词"
            issues.append(
                Issue(
                    code="LONG_SENTENCE",
                    message=f"这一句约有 {measure}；检查主体、条件和结论能否更清楚地组织。",
                    line=line_number(masked, offset),
                    excerpt=excerpt(sentence),
                )
            )

    seen_acronyms: set[str] = set()
    for match in re.finditer(r"\b[A-Z][A-Z0-9]{1,}(?:-[A-Z0-9]+)*\b", masked):
        acronym = match.group(0)
        if acronym in seen_acronyms:
            continue
        seen_acronyms.add(acronym)
        if not _acronym_is_expanded(masked, match.start(), match.end(), acronym):
            issues.append(
                Issue(
                    code="UNEXPLAINED_ACRONYM",
                    message=f"“{acronym}”首次出现时可能需要全称或一句本地解释。",
                    line=line_number(masked, match.start()),
                    excerpt=excerpt(masked[max(0, match.start() - 36) : match.end() + 36]),
                )
            )

    pattern_groups = (
        (
            "VAGUE_ACTION",
            VAGUE_ACTION_PATTERNS,
            "动作不够具体；说明谁在什么条件下做什么，以及完成标准。",
        ),
        (
            "PATRONIZING_LANGUAGE",
            PATRONIZING_PATTERNS,
            "不要替读者判断任务难度；直接说明操作或风险。",
        ),
        (
            "DOUBLE_NEGATIVE",
            DOUBLE_NEGATIVE_PATTERNS,
            "双重否定可能增加理解成本；确认能否改为直接陈述且不改变逻辑。",
        ),
    )

    for code, patterns, message in pattern_groups:
        combined = re.compile("|".join(f"(?:{pattern})" for pattern in patterns), re.IGNORECASE)
        reported_lines: set[int] = set()
        for match in combined.finditer(masked):
            issue_line = line_number(masked, match.start())
            if issue_line in reported_lines:
                continue
            if code == "PATRONIZING_LANGUAGE":
                lead_in = masked[max(0, match.start() - 48) : match.start()]
                if re.search(
                    r"(?:avoid|do not use|don't use|such as|不要|避免|禁用|例如)[^。！？!?\n]{0,40}$",
                    lead_in,
                    re.IGNORECASE,
                ):
                    continue
            reported_lines.add(issue_line)
            issues.append(
                Issue(
                    code=code,
                    message=message,
                    line=issue_line,
                    excerpt=excerpt(masked[max(0, match.start() - 36) : match.end() + 36]),
                )
            )

    return sorted(issues, key=lambda issue: (issue.line, issue.code, issue.excerpt))


def _format_text(issues: list[Issue]) -> str:
    if not issues:
        return "未发现表层表达风险；这不能证明读者已经理解。"

    lines = [f"发现 {len(issues)} 条供复核的提示："]
    for issue in issues:
        lines.append(f"{issue.line}: [{issue.code}] {issue.message}")
        if issue.excerpt:
            lines.append(f"   {issue.excerpt}")
    lines.append("请结合上下文复核；这些提示不是可读性分数，也不是通过或失败的结论。")
    return "\n".join(lines)


def _format_json(issues: list[Issue]) -> str:
    counts = Counter(issue.code for issue in issues)
    payload = {
        "ok": True,
        "warningCount": len(issues),
        "warningsByCode": dict(sorted(counts.items())),
        "issues": [asdict(issue) for issue in issues],
        "limitation": "仅提示表层风险；结果不能证明事实准确或读者已经理解。",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


class ChineseArgumentParser(argparse.ArgumentParser):
    def format_help(self) -> str:
        help_text = super().format_help()
        return (
            help_text.replace("usage:", "用法:", 1)
            .replace("positional arguments:", "位置参数:")
            .replace("optional arguments:", "可选参数:")
            .replace("options:", "可选参数:")
            .replace("show this help message and exit", "显示帮助并退出")
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = ChineseArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="UTF-8 文本或 Markdown 文件；省略时从标准输入读取")
    parser.add_argument("--format", choices=("text", "json"), default="text", help="输出格式")
    parser.add_argument("--max-cjk-sentence", type=int, default=DEFAULT_MAX_CJK_SENTENCE, help="中文句子长度提示阈值")
    parser.add_argument("--max-english-sentence", type=int, default=DEFAULT_MAX_ENGLISH_SENTENCE, help="英文句子长度提示阈值")
    parser.add_argument("--max-cjk-paragraph", type=int, default=DEFAULT_MAX_CJK_PARAGRAPH, help="中文段落长度提示阈值")
    parser.add_argument("--max-english-paragraph", type=int, default=DEFAULT_MAX_ENGLISH_PARAGRAPH, help="英文段落长度提示阈值")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        text = Path(args.path).read_text(encoding="utf-8") if args.path else sys.stdin.read()
    except (OSError, UnicodeError) as error:
        print(f"说人话检查失败：{error}", file=sys.stderr)
        return 2

    issues = lint_text(
        text,
        max_cjk_sentence=args.max_cjk_sentence,
        max_english_sentence=args.max_english_sentence,
        max_cjk_paragraph=args.max_cjk_paragraph,
        max_english_paragraph=args.max_english_paragraph,
    )
    print(_format_json(issues) if args.format == "json" else _format_text(issues))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
