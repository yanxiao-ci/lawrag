# -*- coding: utf-8 -*-
"""HR 合规检查评测：直接调用真实 HTTP 端点 POST /api/check_hr_compliance。

指标：
  - finding 级 Precision / Recall / F1（category 相同 + span 归一化后相等或互相包含）
  - 引用准确率（basis 条号可通过 /api/get_article 取到，且 quote 归一化后是原文连续子串）
  - 干净 JD 误报率（干净样本里出现任意 finding 的比例）
  - 干净 JD risk_level=low 的比例
  - risk_level 准确率（按 gold severity 聚合：任一 high -> high；任一 medium -> medium；否则 low）

口径：
  - dev / test 拆分来自数据集 split 字段；规则与 Prompt 只用 dev 调，test 只跑一次；
  - 固定 seed=0（脚本不做随机采样，顺序固定），记录数据集 sha256、模型与时间；
  - 结果只落 id 与指标，不落 JD 原文。

用法：
  D:/software/Anaconda/envs/edu_rag/python.exe tools/eval_hr_compliance.py --split dev
  D:/software/Anaconda/envs/edu_rag/python.exe tools/eval_hr_compliance.py --split test
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from base.config import config  # noqa: E402
from rag_qa.core.hr_compliance import normalize_for_quote, quote_in_article  # noqa: E402

DATA_DEFAULT = os.path.join(ROOT, "rag_qa", "data", "hr_eval", "jd_compliance_140.jsonl")
OUT_DIR = os.path.join(ROOT, "rag_qa", "data", "hr_eval")


def load_rows(path, split):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if split == "all" or item.get("split") == split:
                rows.append(item)
    return rows


def post_json(url, payload, timeout=30):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8")), time.time() - started
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8"))
        except Exception:
            body = {"detail": str(exc)}
        return exc.code, body, time.time() - started


def get_json(url, payload, timeout=30):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, None


def norm_span(span):
    return normalize_for_quote(span).lower()


def span_match(predicted, gold):
    left, right = norm_span(predicted), norm_span(gold)
    if not left or not right:
        return False
    return left == right or left in right or right in left


def expected_risk(gold_findings):
    severities = {str(item.get("severity") or "").lower() for item in gold_findings}
    if "high" in severities:
        return "high"
    if "medium" in severities:
        return "medium"
    return "low"


def run_one(base_url, item):
    status, body, elapsed = post_json(
        f"{base_url}/api/check_hr_compliance",
        {"text": item["text"], "scope": item.get("scope", "jd"), "domain": "labor"},
    )
    return {
        "id": item["id"],
        "status": status,
        "body": body if status == 200 else None,
        "elapsed": elapsed,
        "error": None if status == 200 else str((body or {}).get("detail", ""))[:200],
    }


def verify_citations(base_url, responses, cache):
    """返回 (basis_total, basis_valid, findings_with_basis, findings_basis_valid)。"""
    basis_total = basis_valid = findings_with_basis = findings_basis_valid = 0
    for response in responses:
        body = response.get("body") or {}
        for finding in body.get("findings") or []:
            basis = finding.get("basis") or []
            if not basis:
                continue
            findings_with_basis += 1
            all_valid = True
            for entry in basis:
                basis_total += 1
                key = (entry.get("law_name", ""), entry.get("article_no", ""))
                if key not in cache:
                    status, detail = get_json(f"{base_url}/api/get_article", {
                        "law_name": key[0], "article_no": key[1]})
                    cache[key] = (status == 200 and bool(detail), detail)
                ok, detail = cache[key]
                valid = bool(ok and detail and quote_in_article(
                    entry.get("quote", ""), detail.get("article_text", "")))
                if valid:
                    basis_valid += 1
                else:
                    all_valid = False
            if all_valid:
                findings_basis_valid += 1
    return basis_total, basis_valid, findings_with_basis, findings_basis_valid


def score(rows, responses, citation_stats):
    tp = fp = fn = 0
    risk_correct = 0
    clean_items = clean_fp = clean_low = 0
    per_category = {}

    pred_by_id = {response["id"]: response for response in responses}

    for item in rows:
        response = pred_by_id.get(item["id"], {})
        body = response.get("body") or {}
        predictions = body.get("findings") or []
        gold = item.get("gold_findings") or []

        matched = set()
        for prediction in predictions:
            category = prediction.get("category", "")
            bucket = per_category.setdefault(category, {"tp": 0, "fp": 0, "fn": 0})
            hit = None
            for index, gold_item in enumerate(gold):
                if index in matched:
                    continue
                if prediction.get("category") == gold_item.get("category") and \
                        span_match(prediction.get("span", ""), gold_item.get("span", "")):
                    hit = index
                    break
            if hit is None:
                fp += 1
                bucket["fp"] += 1
            else:
                matched.add(hit)
                tp += 1
                bucket["tp"] += 1
        fn += max(len(gold) - len(matched), 0)
        for index, gold_item in enumerate(gold):
            if index not in matched:
                bucket = per_category.setdefault(gold_item.get("category", ""), {"tp": 0, "fp": 0, "fn": 0})
                bucket["fn"] += 1

        if body.get("risk_level") == expected_risk(gold):
            risk_correct += 1

        if item.get("clean"):
            clean_items += 1
            if predictions:
                clean_fp += 1
            if body.get("risk_level") == "low":
                clean_low += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    basis_total, basis_valid, findings_with_basis, findings_basis_valid = citation_stats
    total = len(rows)
    return {
        "n": total,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "risk_level_accuracy": round(risk_correct / total, 4) if total else 0.0,
        "citation_accuracy": round(basis_valid / basis_total, 4) if basis_total else 1.0,
        "basis_total": basis_total,
        "basis_valid": basis_valid,
        "findings_with_basis": findings_with_basis,
        "findings_basis_valid": findings_basis_valid,
        "clean_items": clean_items,
        "clean_false_positive_rate": round(clean_fp / clean_items, 4) if clean_items else 0.0,
        "clean_risk_low_rate": round(clean_low / clean_items, 4) if clean_items else 0.0,
        "per_category": per_category,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_DEFAULT)
    parser.add_argument("--split", choices=["dev", "test", "all"], default="all")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    rows = load_rows(args.data, args.split)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("没有可评测样本")

    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        responses = list(pool.map(lambda item: run_one(args.base_url, item), rows))

    cache = {}
    citation_stats = verify_citations(args.base_url, responses, cache)
    metrics = score(rows, responses, citation_stats)
    latencies = sorted(response["elapsed"] for response in responses)
    p95 = latencies[min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1))))]

    errors = [{"id": r["id"], "status": r["status"], "error": r["error"]}
              for r in responses if r["status"] != 200]

    digest = hashlib.sha256(open(args.data, "rb").read()).hexdigest()
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": config.LLM_MODEL,
        "seed": 0,
        "split": args.split,
        "dataset": os.path.relpath(args.data, ROOT),
        "dataset_sha256": digest,
        "base_url": args.base_url,
        "elapsed_seconds": round(time.time() - started, 2),
        "latency": {
            "avg": round(sum(latencies) / len(latencies), 3),
            "p95": round(p95, 3),
            "max": round(latencies[-1], 3),
        },
        "llm_used_rate": round(
            sum(1 for r in responses
                if (r.get("body") or {}).get("llm_used")) / len(responses), 4),
        "errors": errors,
        "metrics": metrics,
        "acceptance": {
            "precision_ge_0.85": metrics["precision"] >= 0.85,
            "recall_ge_0.80": metrics["recall"] >= 0.80,
            "f1_ge_0.82": metrics["f1"] >= 0.82,
            "citation_ge_0.95": metrics["citation_accuracy"] >= 0.95,
            "clean_fp_le_0.10": metrics["clean_false_positive_rate"] <= 0.10,
            "clean_low_ge_0.90": metrics["clean_risk_low_rate"] >= 0.90,
        },
    }
    result["all_passed"] = all(result["acceptance"].values())

    out = args.out or os.path.join(OUT_DIR, f"eval_hr_compliance_{args.split}_result.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps({k: result[k] for k in
                      ("split", "model", "elapsed_seconds", "latency", "llm_used_rate",
                       "metrics", "acceptance", "all_passed")},
                     ensure_ascii=False, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
