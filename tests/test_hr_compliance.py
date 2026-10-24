# -*- coding: utf-8 -*-
"""hr_compliance 纯函数单测（不依赖 DB / Milvus / LLM）。"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rag_qa.core import hr_compliance as hc  # noqa: E402


# 13 条规则各一条最小命中样例
RULE_SAMPLES = {
    "gender_restriction": "招聘后端工程师，限男性，负责服务端开发。",
    "marital_fertility": "招聘行政专员，已婚已育优先。",
    "age_restriction": "招聘运营主管，35岁以下。",
    "hukou_restriction": "招聘财务会计，要求本市户口。",
    "ethnicity_religion": "招聘市场专员，少数民族优先。",
    "health_disability": "招聘生产操作员，要求无残疾。",
    "height_appearance": "招聘安保人员，身高170cm以上。",
    "illegal_probation": "招聘项目助理，试用期9个月。",
    "no_social_insurance": "招聘客服，入职后不缴社保。",
    "deposit_id": "招聘司机，入职需交押金。",
    "illegal_hours": "招聘程序员，接受996工作制。",
    "false_promise": "招聘课程顾问，入职保证录用。",
    "salary_vague": "招聘设计师，薪资面议。",
}

CLEAN_TEXTS = [
    "招聘后端工程师，本科及以上学历，3年以上 Python 经验，熟悉 FastAPI 与 PostgreSQL。",
    "招聘健康管理师，需持健康管理师证书，负责健康档案管理。",
    "招聘销售专员，底薪 6000-9000 元加提成，缴纳五险一金。",
    "招聘人事专员，负责劳动合同管理、入离职办理与员工关系维护。",
]

BOUNDARY_TEXTS = [
    "优先考虑有经验者。",
    "能适应加班。",
    "年龄不限。",
    "不限性别、年龄、户籍。",
    "能接受弹性工作安排。",
    "薪资 15-25K/月。",
]


class RuleLibraryTest(unittest.TestCase):
    def test_rule_count_and_scope(self):
        rules = hc.load_rules()
        self.assertEqual(len(rules), 13)
        with open(hc._RULES_PATH, encoding="utf-8") as f:
            import json

            data = json.load(f)
        self.assertEqual(data.get("law_scope"), "labor")
        for rule in rules:
            self.assertTrue(rule.get("id"))
            self.assertTrue(rule.get("patterns"))
            self.assertIn(rule.get("severity"), {"high", "medium", "low"})

    def test_every_rule_has_at_least_one_hit(self):
        for rule_id, text in RULE_SAMPLES.items():
            hits = hc.scan_rules(text)
            ids = [hit["rule_id"] for hit in hits]
            self.assertIn(rule_id, ids, f"{rule_id} 未命中: {text}")
            for hit in hits:
                self.assertTrue(hit["span"], f"{rule_id} span 为空")
                self.assertIn(hit["span"], text, f"{rule_id} span 不在原文")


class ScanRulesTest(unittest.TestCase):
    def test_clean_text_has_no_hits(self):
        for text in CLEAN_TEXTS:
            self.assertEqual(hc.scan_rules(text), [], text)

    def test_boundary_text_has_no_high_finding(self):
        for text in BOUNDARY_TEXTS:
            hits = hc.scan_rules(text)
            high = [hit for hit in hits if hit["severity"] == "high"]
            self.assertEqual(high, [], f"边界文本误报 high: {text} -> {high}")

    def test_spans_are_in_source(self):
        text = "招聘后端工程师，限男性，35岁以下，试用期9个月，薪资面议。"
        hits = hc.scan_rules(text)
        self.assertGreaterEqual(len(hits), 4)
        for hit in hits:
            self.assertTrue(hit["span"])
            self.assertIn(hit["span"], text)
            self.assertLess(hit["start"], hit["end"])

    def test_boundary_prefix_does_not_trigger_age(self):
        # “年龄不限”不能被 “年龄要求” 规则命中
        self.assertEqual(hc.scan_rules("年龄不限，经验优先。"), [])


class PiiMaskTest(unittest.TestCase):
    def test_phone_email_id_masked(self):
        text = ("联系人张三，手机13812345678，邮箱zhangsan@example.com，"
                "身份证110101199001011234。")
        masked = hc.mask_pii(text)
        self.assertNotIn("13812345678", masked)
        self.assertNotIn("zhangsan@example.com", masked)
        self.assertNotIn("110101199001011234", masked)
        self.assertIn("[手机号]", masked)
        self.assertIn("[邮箱]", masked)
        self.assertIn("[身份证号]", masked)

    def test_mask_is_idempotent(self):
        text = "手机13812345678"
        once = hc.mask_pii(text)
        self.assertEqual(once, hc.mask_pii(once))

    def test_short_number_not_masked(self):
        self.assertEqual(hc.mask_pii("底薪 6000-9000 元"), "底薪 6000-9000 元")


class QuoteCheckTest(unittest.TestCase):
    def test_quote_with_punctuation_and_whitespace(self):
        article = "用人单位招用人员，不得以性别为由拒绝录用妇女或者提高对妇女的录用标准。"
        quote = "不得以性别为由 拒绝录用妇女"
        self.assertTrue(hc.quote_in_article(quote, article))

    def test_full_width_punctuation(self):
        article = "劳动合同期限三个月以上不满一年的，试用期不得超过一个月。"
        quote = "试用期不得超过一个月"
        self.assertTrue(hc.quote_in_article(quote, article))

    def test_missing_quote_rejected(self):
        self.assertFalse(hc.quote_in_article("这句话不在原文里", "劳动合同期限三个月以上。"))
        self.assertFalse(hc.quote_in_article("", "任何原文"))


class ArticleKeyTest(unittest.TestCase):
    def test_article_number_forms(self):
        keys = {hc.article_key(value) for value in ("第二十七条", "第27条", "27")}
        self.assertEqual(len(keys), 1)
        self.assertNotEqual(hc.article_key("第二十七条"), hc.article_key("第二十七条之一"))

    def test_parse_law_hint(self):
        self.assertEqual(
            hc.parse_law_hint("中华人民共和国就业促进法 第二十七条"),
            ("中华人民共和国就业促进法", "第二十七条"),
        )


class AssembleTest(unittest.TestCase):
    def _hit(self, text="限男性"):
        return hc.scan_rules(text)[0]

    def test_verified_basis(self):
        hit = self._hit()
        article = "用人单位招用人员，不得以性别为由拒绝录用妇女。"
        candidates = {
            "gender_restriction": [{
                "law_name": "中华人民共和国就业促进法",
                "article_no": "第二十七条",
                "quote": article,
                "article_text": article,
            }]
        }
        findings, verified = hc.build_findings([hit], candidates)
        self.assertTrue(verified)
        self.assertEqual(len(findings[0]["basis"]), 1)
        self.assertFalse(findings[0]["needs_human"])

    def test_unverified_basis_dropped_and_needs_human(self):
        hit = self._hit()
        candidates = {
            "gender_restriction": [{
                "law_name": "中华人民共和国就业促进法",
                "article_no": "第二十七条",
                "quote": "伪造的引文",
                "article_text": "原文完全不包含这句话。",
            }]
        }
        findings, verified = hc.build_findings([hit], candidates)
        # 不合格的 basis 被丢弃；保留的 basis 为空，citation_verified 按定义仍为 true
        self.assertTrue(verified)
        self.assertEqual(findings[0]["basis"], [])
        self.assertTrue(findings[0]["needs_human"])

    def test_empty_candidates_needs_human(self):
        hit = self._hit("薪资面议")
        findings, verified = hc.build_findings([hit], {})
        self.assertTrue(verified)  # 没有非空 basis，citation_verified 视为 true
        self.assertEqual(findings[0]["basis"], [])
        self.assertTrue(findings[0]["needs_human"])

    def test_risk_level_aggregation(self):
        self.assertEqual(hc.risk_level_for([]), "low")
        self.assertEqual(hc.risk_level_for([{"severity": "medium"}]), "medium")
        self.assertEqual(hc.risk_level_for([{"severity": "medium"}, {"severity": "high"}]), "high")
        self.assertEqual(hc.risk_level_for([{"severity": "low"}]), "low")


class LlmParsingTest(unittest.TestCase):
    def test_extract_first_json_object(self):
        text = '说明文字 {"findings": [{"rule_id": "x", "basis": []}]} 结尾'
        data = hc.extract_first_json_object(text)
        self.assertEqual(data["findings"][0]["rule_id"], "x")

    def test_extract_handles_braces_in_strings(self):
        text = '{"a": "包含 } 的字符串", "b": 1}'
        self.assertEqual(hc.extract_first_json_object(text), {"a": "包含 } 的字符串", "b": 1})

    def test_validate_rejects_unknown_rule_and_article(self):
        candidates = {
            "gender_restriction": [{
                "law_name": "中华人民共和国就业促进法",
                "article_no": "第二十七条",
                "quote": "q",
                "article_text": "q",
            }]
        }
        parsed = {"findings": [
            {"rule_id": "unknown_rule", "basis": [{"law_name": "中华人民共和国就业促进法",
                                                   "article_no": "第二十七条"}]},
            {"rule_id": "gender_restriction", "basis": [{"law_name": "中华人民共和国劳动法",
                                                         "article_no": "第十二条"}]},
            {"rule_id": "gender_restriction", "basis": [{"law_name": "中华人民共和国就业促进法",
                                                         "article_no": "第27条"}]},
        ]}
        selected = hc.validate_selection(parsed, candidates)
        self.assertEqual(list(selected.keys()), ["gender_restriction"])
        self.assertEqual(selected["gender_restriction"][0]["article_no"], "第二十七条")


class CandidateCollectionTest(unittest.TestCase):
    def test_hint_resolves_without_retrieval(self):
        calls = []

        def fetch(law_name, article_no):
            calls.append((law_name, article_no))
            return {"law_name": law_name, "article_no": article_no,
                    "article_text": "用人单位招用人员，不得以性别为由拒绝录用妇女。"}

        def retrieve(query, k):
            raise AssertionError("hint 已命中，不应触发检索兜底")

        hit = {
            "rule_id": "gender_restriction",
            "span": "限男性",
            "law_hints": ["中华人民共和国就业促进法 第二十七条"],
        }
        result = hc.collect_candidates([hit], fetch_article=fetch, retrieve=retrieve)
        self.assertEqual(len(result["gender_restriction"]), 1)
        self.assertEqual(calls, [("中华人民共和国就业促进法", "第二十七条")])

    def test_retrieval_fallback_when_hint_missing(self):
        class Doc:
            def __init__(self, law_name, article_no):
                self.metadata = {"law_name": law_name, "article_no": article_no}

        def fetch(law_name, article_no):
            if article_no == "第九十九条":
                return None
            return {"law_name": law_name, "article_no": article_no,
                    "article_text": "劳动者享有平等就业和选择职业的权利。"}

        def retrieve(query, k):
            return [Doc("中华人民共和国劳动法", "第三条")]

        hit = {
            "rule_id": "age_restriction",
            "span": "35岁以下",
            "law_hints": ["中华人民共和国就业促进法 第九十九条"],
        }
        result = hc.collect_candidates([hit], fetch_article=fetch, retrieve=retrieve)
        self.assertEqual([c["article_no"] for c in result["age_restriction"]], ["第三条"])

    def test_no_hint_no_retrieval(self):
        def retrieve(query, k):
            raise AssertionError("无 hint 的规则不应检索兜底")

        hit = {"rule_id": "salary_vague", "span": "薪资面议", "law_hints": []}
        result = hc.collect_candidates([hit], fetch_article=lambda *a: None, retrieve=retrieve)
        self.assertEqual(result["salary_vague"], [])


class AnalyzeFallbackTest(unittest.TestCase):
    def test_llm_failure_falls_back_to_hint(self):
        article = "用人单位招用人员，不得以性别为由拒绝录用妇女。"

        def fetch_article(law_name, article_no):
            return {
                "law_name": law_name,
                "article_no": article_no,
                "article_text": article,
                "chapter": "",
                "publish_date": "",
                "effective_date": "",
                "status": "有效",
            }

        class BrokenSelector:
            def select(self, text, hits, candidate_map):
                return None

        result = hc.analyze_compliance(
            "招聘后端工程师，限男性。",
            fetch_article=fetch_article,
            llm=BrokenSelector(),
        )
        self.assertFalse(result["llm_used"])
        self.assertEqual(result["risk_level"], "high")
        self.assertTrue(result["findings"][0]["basis"])
        self.assertTrue(result["citation_verified"])
        self.assertEqual(result["disclaimer"], hc.DISCLAIMER)


if __name__ == "__main__":
    unittest.main()
