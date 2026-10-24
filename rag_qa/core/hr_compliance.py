# -*- coding: utf-8 -*-
"""HR 合规检查的确定性核心。

职责：
  1. PII 兜底脱敏（手机 / 邮箱 / 身份证；上游已脱敏也再做一次）；
  2. 规则扫描（13 条规则，只定位片段 + 类别 + 建议，不产生条号）；
  3. 法条取证：确定性 law_hint -> MySQL 回查；hint 回查失败时按严格领域检索兜底；
  4. 逐字回查：quote 归一化后必须是 MySQL article_text 的连续子串；
  5. LLM 闭集选条：只允许从候选集合里选，失败/超时只回退到确定性 hint。

边界：
  - mask_pii / scan_rules 是纯函数，不连 DB、不 import app.py；
  - 需要数据库或检索时通过参数注入 fetch_article / retrieve，由调用方持有连接与锁；
  - 不伪造条号：候选为空或回查失败 -> basis=[] + needs_human=true。
"""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from typing import Any, Callable, Optional

DISCLAIMER = (
    "仅覆盖全国人大及其常委会制定的法律级文本，不含行政法规、司法解释、地方规定；"
    "本结果仅作合规提示，不构成法律意见。"
)

VALID_SCOPES = ("jd", "resume_screening")
MAX_TEXT_LEN = 4000
MAX_CANDIDATES = 9
MAX_BASIS = 3
QUOTE_MAX = 400

_RULES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "compliance_rules.json",
)

# ---------------------------------------------------------------------------
# 规则库
# ---------------------------------------------------------------------------

_rules_cache: Optional[list] = None


def load_rules(path: Optional[str] = None) -> list:
    """读取规则库。默认路径做进程内缓存；显式传 path 时不缓存。"""
    global _rules_cache
    if path is None and _rules_cache is not None:
        return _rules_cache
    try:
        with open(path or _RULES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        rules = data.get("rules") or []
    except Exception:
        rules = []
    if path is None:
        _rules_cache = rules
    return rules


def _compile_rule(rule: dict) -> list:
    compiled = []
    for pattern in rule.get("patterns") or []:
        try:
            compiled.append(re.compile(pattern))
        except re.error:
            continue
    return compiled


def scan_rules(text: str, rules: Optional[list] = None) -> list:
    """按 13 条规则扫描文本，返回命中项。

    返回每项：rule_id / category / severity / span / suggestion / law_hints，
    另带 start/end 便于内部定位。纯函数，不连 DB。
    """
    if not text:
        return []
    rule_defs = rules if rules is not None else load_rules()

    hits = []
    for rule_index, rule in enumerate(rule_defs):
        for pattern in rule.get("patterns") or []:
            try:
                regex = re.compile(pattern)
            except re.error:
                continue
            for match in regex.finditer(text):
                span = match.group(0)
                if not span:
                    continue
                hits.append({
                    "rule_id": rule.get("id", ""),
                    "category": rule.get("category", ""),
                    "severity": rule.get("severity", "medium"),
                    "span": span,
                    "start": match.start(),
                    "end": match.end(),
                    "suggestion": rule.get("suggestion", ""),
                    "law_hints": list(rule.get("law_hint") or []),
                    "_rule_index": rule_index,
                })

    # 同一规则同一区间只保留一次（多个 pattern 可能命中同一片段）
    seen = set()
    unique = []
    for hit in hits:
        key = (hit["rule_id"], hit["start"], hit["end"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(hit)
    unique.sort(key=lambda item: (item["start"], item["_rule_index"], item["end"]))
    return unique


# ---------------------------------------------------------------------------
# PII 兜底脱敏
# ---------------------------------------------------------------------------

_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_ID_RE = re.compile(r"(?<!\d)(?:\d{17}[\dXx]|\d{15})(?!\d)")


def mask_pii(text: str) -> str:
    """手机 / 邮箱 / 身份证兜底脱敏；上游已脱敏也再执行一次（幂等）。"""
    if not text:
        return ""
    value = str(text)
    value = _EMAIL_RE.sub("[邮箱]", value)
    value = _ID_RE.sub("[身份证号]", value)
    value = _PHONE_RE.sub("[手机号]", value)
    return value


# ---------------------------------------------------------------------------
# 归一化与逐字回查
# ---------------------------------------------------------------------------

_CJK_PUNCT = str.maketrans({
    "，": ",", "。": ".", "；": ";", "：": ":", "！": "!", "？": "?",
    "（": "(", "）": ")", "【": "[", "】": "]", "《": "<", "》": ">",
    "、": ",", "“": '"', "”": '"', "‘": "'", "’": "'",
    "—": "-", "－": "-", "～": "~", "﹏": "_", "　": " ",
})


def normalize_for_quote(text: str) -> str:
    """引用校验归一化：NFKC（全角转半角）+ 中文标点转半角 + 去除所有空白。"""
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.translate(_CJK_PUNCT)
    value = re.sub(r"\s+", "", value)
    return value


def quote_in_article(quote: str, article_text: str) -> bool:
    """quote 归一化后是否为 article_text 的连续子串。"""
    q = normalize_for_quote(quote)
    if not q:
        return False
    return q in normalize_for_quote(article_text)


# ---------------------------------------------------------------------------
# 条号 / 法名归一化
# ---------------------------------------------------------------------------

_ARTICLE_RE = re.compile(
    r"^第?([0-9〇零一二三四五六七八九十百千万两]+)条?"
    r"(?:之([0-9〇零一二三四五六七八九十百千两]+))?$"
)


def law_key(law_name: str) -> str:
    return re.sub(r"\s+", "", str(law_name or "").replace("中华人民共和国", ""))


def article_key(article_no: str):
    """把 第二十七条 / 第27条 / 27 / 第二十七条之一 归一成可比较的 key。"""
    value = re.sub(r"\s+", "", str(article_no or ""))
    match = _ARTICLE_RE.match(value)
    if not match:
        return ("raw", value)
    try:
        from base.law_parser import cn2int

        num = cn2int(match.group(1))
        sub = cn2int(match.group(2)) if match.group(2) else None
    except Exception:
        return ("raw", value)
    if num is None:
        return ("raw", value)
    return ("num", num, sub)


def parse_law_hint(hint: str):
    """'中华人民共和国就业促进法 第二十七条' -> (法名, 条号)。"""
    value = str(hint or "").strip()
    if not value:
        return None
    if " " in value:
        law_name, article_no = value.rsplit(" ", 1)
        law_name, article_no = law_name.strip(), article_no.strip()
    else:
        match = re.match(r"^(.*?)(第?[0-9〇零一二三四五六七八九十百千万两]+条?(?:之[0-9〇零一二三四五六七八九十百千两]+)?)$", value)
        if not match:
            return None
        law_name, article_no = match.group(1).strip(), match.group(2).strip()
    if not law_name or not article_no:
        return None
    return law_name, article_no


def _detail_to_candidate(detail: Optional[dict]) -> Optional[dict]:
    if not isinstance(detail, dict):
        return None
    law_name = str(detail.get("law_name") or "").strip()
    article_no = str(detail.get("article_no") or "").strip()
    article_text = str(detail.get("article_text") or "")
    if not (law_name and article_no and article_text):
        return None
    return {
        "law_name": law_name,
        "article_no": article_no,
        "quote": article_text[:QUOTE_MAX],
        "article_text": article_text,
    }


# ---------------------------------------------------------------------------
# 取证：候选法条集合
# ---------------------------------------------------------------------------

def collect_candidates(
    hits: list,
    *,
    fetch_article: Optional[Callable[[str, str], Optional[dict]]] = None,
    retrieve: Optional[Callable[[str, int], list]] = None,
    max_candidates: int = MAX_CANDIDATES,
) -> dict:
    """为每个 rule_id 建候选法条集合（最多 max_candidates 条）。

    - law_hint 直接按法律名 + 条号回查 MySQL，不受 domain 过滤；
    - 只有「有 hint 但回查失败」时才用 retrieve 兜底（调用方保证严格领域、无 FAQ）；
    - 没有 law_hint 的规则（如薪资面议）不检索兜底，basis 留空并标 needs_human。
    """
    result: dict[str, list] = {}

    for hit in hits:
        rule_id = hit.get("rule_id", "")
        bucket = result.setdefault(rule_id, [])
        seen = {(law_key(c["law_name"]), article_key(c["article_no"])) for c in bucket}
        hints = hit.get("law_hints") or []
        resolved_any = False

        # 1) 确定性 hint 回查
        for hint in hints:
            if len(bucket) >= max_candidates:
                break
            parsed = parse_law_hint(hint)
            if not parsed or fetch_article is None:
                continue
            try:
                detail = fetch_article(parsed[0], parsed[1])
            except Exception:
                detail = None
            candidate = _detail_to_candidate(detail)
            if not candidate:
                continue
            key = (law_key(candidate["law_name"]), article_key(candidate["article_no"]))
            if key in seen:
                continue
            seen.add(key)
            bucket.append(candidate)
            resolved_any = True

        # 2) 严格领域检索兜底（只补 hint 回查失败的情况）
        if hints and not resolved_any and retrieve is not None and len(bucket) < max_candidates:
            query = str(hit.get("span") or hit.get("suggestion") or "").strip()
            if not query:
                query = str(hit.get("category") or "").strip()
            try:
                docs = retrieve(query, 5) or []
            except Exception:
                docs = []
            for doc in docs:
                if len(bucket) >= max_candidates:
                    break
                metadata = getattr(doc, "metadata", None) or {}
                law_name = str(metadata.get("law_name") or "").strip()
                article_no = str(metadata.get("article_no") or "").strip()
                if not (law_name and article_no) or fetch_article is None:
                    continue
                key = (law_key(law_name), article_key(article_no))
                if key in seen:
                    continue
                try:
                    detail = fetch_article(law_name, article_no)
                except Exception:
                    detail = None
                candidate = _detail_to_candidate(detail)
                if not candidate:
                    continue
                seen.add(key)
                bucket.append(candidate)

    return result


# ---------------------------------------------------------------------------
# LLM 闭集选条
# ---------------------------------------------------------------------------

def extract_first_json_object(text: str) -> Optional[dict]:
    """从模型输出里提取第一个完整 JSON 对象；容忍前后解释文字。"""
    if not text:
        return None
    start = None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if start is None:
            if char == "{":
                start = index
                depth = 1
            continue
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:index + 1])
                except Exception:
                    return None
    return None


def _build_llm_prompt(text: str, hits: list, candidate_map: dict) -> str:
    lines = [
        "你是企业招聘合规助手。只允许从候选法条中选择依据，不得编造条号。",
        "输出严格 JSON，不要解释："
        '{"findings":[{"rule_id":"...","basis":[{"law_name":"...","article_no":"..."}]}]}',
        "要求：",
        "1) 每个 rule_id 只能从该 rule_id 的候选法条中选择；",
        "2) 候选为空则该 rule_id 输出 basis: []；",
        "3) 不得新增 rule_id，不得修改 span / category / severity / suggestion；",
        "4) 不确定就不选。",
        "",
        "[文本]",
        text,
        "",
        "[规则命中与候选法条]",
    ]
    emitted = set()
    for hit in hits:
        rule_id = hit.get("rule_id", "")
        if rule_id in emitted:
            continue
        emitted.add(rule_id)
        lines.append(f"- rule_id: {rule_id} | category: {hit.get('category')} | "
                     f"severity: {hit.get('severity')} | span: {hit.get('span')}")
        candidates = candidate_map.get(rule_id) or []
        if not candidates:
            lines.append("  candidates: []")
            continue
        lines.append("  candidates:")
        for candidate in candidates:
            quote = str(candidate.get("quote") or "")[:300].replace("\n", " ")
            lines.append(f"  - 《{candidate['law_name']}》{candidate['article_no']} | quote: {quote}")
    return "\n".join(lines)


class LLMSelector:
    """OpenAI 兼容客户端上的闭集选条器；失败一律返回 None，由调用方回退。"""

    def __init__(self, model: Optional[str] = None, timeout: float = 5.0):
        from base.config import config

        self.model = model or config.LLM_MODEL
        self.api_key = config.DASHSCOPE_API_KEY
        self.base_url = config.DASHSCOPE_BASE_URL
        self.timeout = timeout
        self._client = None

    def _client_or_raise(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=0,
            )
        return self._client

    def _complete(self, prompt: str) -> str:
        client = self._client_or_raise()
        messages = [
            {"role": "system", "content": "你只输出 JSON，不输出其他内容。"},
            {"role": "user", "content": prompt},
        ]
        started = time.monotonic()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                max_tokens=800,
                # qwen3.x 默认开思考链；闭集选条是短 JSON 任务，关掉思考把延迟压到 1-2s
                extra_body={"enable_thinking": False},
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content or ""
        except Exception:
            elapsed = time.monotonic() - started
            remaining = self.timeout - elapsed
            if remaining <= 1.0:
                raise
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                max_tokens=800,
                extra_body={"enable_thinking": False},
                timeout=remaining,
            )
            return response.choices[0].message.content or ""

    def select(self, text: str, hits: list, candidate_map: dict) -> Optional[dict]:
        """返回解析后的 JSON；任何失败返回 None（调用方回退到确定性 hint）。"""
        prompt = _build_llm_prompt(text[:MAX_TEXT_LEN], hits, candidate_map)
        try:
            content = self._complete(prompt)
        except Exception:
            return None
        return extract_first_json_object(content)


def validate_selection(parsed: Optional[dict], candidate_map: dict) -> dict:
    """只保留「rule_id 在命中里 + 条号在候选集合内」的选择。"""
    selected: dict[str, list] = {}
    if not isinstance(parsed, dict):
        return selected
    items = parsed.get("findings")
    if not isinstance(items, list):
        return selected

    for item in items:
        if not isinstance(item, dict):
            continue
        rule_id = str(item.get("rule_id") or "").strip()
        if rule_id not in candidate_map:
            continue
        index = {
            (law_key(c["law_name"]), article_key(c["article_no"])): c
            for c in candidate_map.get(rule_id) or []
        }
        picked = []
        for raw in item.get("basis") or []:
            if not isinstance(raw, dict):
                continue
            key = (law_key(raw.get("law_name") or ""), article_key(raw.get("article_no") or ""))
            candidate = index.get(key)
            if candidate is None or candidate in picked:
                continue
            picked.append(candidate)
        if picked:
            selected[rule_id] = picked
    return selected


# ---------------------------------------------------------------------------
# 输出组装
# ---------------------------------------------------------------------------

def risk_level_for(findings: list) -> str:
    severities = {str(item.get("severity") or "").lower() for item in findings}
    if "high" in severities:
        return "high"
    if "medium" in severities:
        return "medium"
    return "low"


def build_findings(hits: list, candidate_map: dict, llm_basis: Optional[dict] = None) -> tuple:
    """组装 findings，并逐条做 quote 回查。

    返回 (findings, all_verified)：all_verified 只针对非空 basis。
    """
    llm_basis = llm_basis or {}
    findings = []
    all_verified = True

    for hit in hits:
        rule_id = hit.get("rule_id", "")
        chosen = llm_basis.get(rule_id) or candidate_map.get(rule_id) or []
        basis = []
        verified = True
        for candidate in chosen[:MAX_BASIS]:
            if not quote_in_article(candidate.get("quote", ""), candidate.get("article_text", "")):
                # 引用不通过 -> 丢弃该 basis，并标记需人工复核
                verified = False
                continue
            basis.append({
                "law_name": candidate["law_name"],
                "article_no": candidate["article_no"],
                "quote": candidate["quote"],
            })
        needs_human = (not basis) or (not verified)
        findings.append({
            "category": hit.get("category", ""),
            "span": hit.get("span", ""),
            "severity": hit.get("severity", "medium"),
            "suggestion": hit.get("suggestion", ""),
            "basis": basis,
            "needs_human": bool(needs_human),
        })
    return findings, all_verified


def analyze_compliance(
    text: str,
    *,
    scope: str = "jd",
    fetch_article: Optional[Callable[[str, str], Optional[dict]]] = None,
    retrieve: Optional[Callable[[str, int], list]] = None,
    llm: Optional[LLMSelector] = None,
    rules: Optional[list] = None,
) -> dict:
    """完整确定性 + 可选 LLM 的合规检查；返回响应 dict（不含 scope/domain）。"""
    masked_text = mask_pii(text)
    hits = scan_rules(masked_text, rules=rules)
    candidate_map = collect_candidates(hits, fetch_article=fetch_article, retrieve=retrieve)

    llm_used = False
    llm_basis: dict = {}
    # 候选集全空时 LLM 无从选择，跳过调用（省延迟，也让 salary_vague 走确定性路径）
    if llm is not None and hits and any(candidate_map.values()):
        parsed = llm.select(masked_text, hits, candidate_map)
        if parsed is not None:
            llm_used = True
            llm_basis = validate_selection(parsed, candidate_map)

    findings, all_verified = build_findings(hits, candidate_map, llm_basis)
    return {
        "risk_level": risk_level_for(findings),
        "findings": findings,
        "disclaimer": DISCLAIMER,
        "citation_verified": bool(all_verified),
        "llm_used": bool(llm_used),
    }


__all__ = [
    "DISCLAIMER",
    "VALID_SCOPES",
    "MAX_TEXT_LEN",
    "load_rules",
    "scan_rules",
    "mask_pii",
    "normalize_for_quote",
    "quote_in_article",
    "law_key",
    "article_key",
    "parse_law_hint",
    "collect_candidates",
    "extract_first_json_object",
    "LLMSelector",
    "validate_selection",
    "risk_level_for",
    "build_findings",
    "analyze_compliance",
]
