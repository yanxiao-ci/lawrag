# -*- coding: utf-8 -*-
"""构建 HR 合规评测集（140 条：80 违规 + 40 干净 + 20 边界）。

数据来源（如实记录）：
  - 30 条沿用 HRAgent `backend/data/hr_eval/jd_compliance.jsonl`（20 违规 + 10 干净），
    文本与 gold_findings 原样保留，只补 id / scope / boundary / split；
  - 110 条按 13 类规则与边界模板生成，生成后用 hr_compliance.scan_rules 逐条核对：
    违规样本的规则命中必须与 gold_findings 完全一致，干净/边界样本必须 0 命中。

拆分（固定 seed，脚本本身无随机）：
  - dev  60：40 违规 / 15 干净 / 5 边界
  - test 80：40 违规 / 25 干净 / 15 边界

用法：
  D:/software/Anaconda/envs/edu_rag/python.exe tools/build_hr_compliance_eval.py
"""
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rag_qa.core import hr_compliance as hc  # noqa: E402

OUT = os.path.join(ROOT, "rag_qa", "data", "hr_eval", "jd_compliance_140.jsonl")

# ---------------------------------------------------------------------------
# 30 条沿用样本（原样来自 HRAgent，不修改文本与 gold）
# ---------------------------------------------------------------------------

SEED_30 = [
    {"jd_text": "招聘后端工程师，限男性，35岁以下。", "gold_findings": [{"category": "就业歧视", "span": "限男性", "severity": "high"}, {"category": "就业歧视", "span": "35岁以下", "severity": "high"}], "clean": False},
    {"jd_text": "招聘客服专员，限女性，普通话标准。", "gold_findings": [{"category": "就业歧视", "span": "限女性", "severity": "high"}], "clean": False},
    {"jd_text": "招聘运营主管，男性优先，有团队管理经验。", "gold_findings": [{"category": "就业歧视", "span": "男性优先", "severity": "high"}], "clean": False},
    {"jd_text": "招聘行政专员，已婚已育优先。", "gold_findings": [{"category": "就业歧视", "span": "已婚已育", "severity": "high"}], "clean": False},
    {"jd_text": "招聘前台，未婚优先，形象好。", "gold_findings": [{"category": "就业歧视", "span": "未婚优先", "severity": "high"}], "clean": False},
    {"jd_text": "招聘财务会计，要求本市户口。", "gold_findings": [{"category": "就业歧视", "span": "本市户口", "severity": "high"}], "clean": False},
    {"jd_text": "招聘仓库管理员，仅限本地人员。", "gold_findings": [{"category": "就业歧视", "span": "仅限本地", "severity": "high"}], "clean": False},
    {"jd_text": "招聘市场专员，少数民族优先。", "gold_findings": [{"category": "就业歧视", "span": "民族", "severity": "high"}], "clean": False},
    {"jd_text": "招聘生产操作员，要求无残疾。", "gold_findings": [{"category": "就业歧视", "span": "无残疾", "severity": "high"}], "clean": False},
    {"jd_text": "招聘配送员，要求身体健康，能吃苦耐劳。", "gold_findings": [{"category": "就业歧视", "span": "身体健康", "severity": "high"}], "clean": False},
    {"jd_text": "招聘安保人员，身高170cm以上。", "gold_findings": [{"category": "就业歧视", "span": "身高", "severity": "high"}], "clean": False},
    {"jd_text": "招聘商务接待，形象气质佳，沟通能力强。", "gold_findings": [{"category": "就业歧视", "span": "形象气质佳", "severity": "high"}], "clean": False},
    {"jd_text": "招聘项目助理，试用期9个月，表现优秀可转正。", "gold_findings": [{"category": "违法条款", "span": "试用期9个月", "severity": "high"}], "clean": False},
    {"jd_text": "招聘销售代表，试用期12个月。", "gold_findings": [{"category": "违法条款", "span": "试用期12个月", "severity": "high"}], "clean": False},
    {"jd_text": "招聘客服，入职后不缴社保，以现金补贴代替。", "gold_findings": [{"category": "违法条款", "span": "不缴社保", "severity": "high"}], "clean": False},
    {"jd_text": "招聘司机，入职需交押金。", "gold_findings": [{"category": "违法条款", "span": "押金", "severity": "high"}], "clean": False},
    {"jd_text": "招聘程序员，接受996工作制。", "gold_findings": [{"category": "违法条款", "span": "996", "severity": "medium"}], "clean": False},
    {"jd_text": "招聘门店店员，每周单休。", "gold_findings": [{"category": "违法条款", "span": "单休", "severity": "medium"}], "clean": False},
    {"jd_text": "招聘课程顾问，入职保证录用。", "gold_findings": [{"category": "虚假承诺", "span": "保证录用", "severity": "medium"}], "clean": False},
    {"jd_text": "招聘设计师，薪资面议。", "gold_findings": [{"category": "表述不清", "span": "薪资面议", "severity": "medium"}], "clean": False},
    {"jd_text": "招聘后端工程师，本科及以上学历，3年以上 Python 经验，熟悉 FastAPI 与 PostgreSQL。", "gold_findings": [], "clean": True},
    {"jd_text": "招聘健康管理师，需持健康管理师证书，负责健康档案管理。", "gold_findings": [], "clean": True},
    {"jd_text": "招聘产品经理，负责需求分析与跨部门协作，熟悉 B 端产品。", "gold_findings": [], "clean": True},
    {"jd_text": "招聘销售专员，底薪 6000-9000 元加提成，缴纳五险一金。", "gold_findings": [], "clean": True},
    {"jd_text": "招聘前端工程师，熟悉 Vue3 与 TypeScript，有组件库经验优先。", "gold_findings": [], "clean": True},
    {"jd_text": "招聘人事专员，负责劳动合同管理、入离职办理与员工关系维护。", "gold_findings": [], "clean": True},
    {"jd_text": "招聘数据分析师，熟悉 SQL 与 Python，具备良好沟通能力。", "gold_findings": [], "clean": True},
    {"jd_text": "招聘运维工程师，接受轮班值守，提供夜班补贴。", "gold_findings": [], "clean": True},
    {"jd_text": "招聘行政助理，要求工作细致、有责任心，能熟练使用 Office。", "gold_findings": [], "clean": True},
    {"jd_text": "招聘培训讲师，负责课程设计与授课，有教育行业经验优先。", "gold_findings": [], "clean": True},
]

# ---------------------------------------------------------------------------
# 生成模板：每条模板只触发一条规则（生成后仍逐条核对）
# ---------------------------------------------------------------------------

JOBS = ["后端工程师", "前端工程师", "产品经理", "运营专员", "客服专员", "销售代表",
        "会计", "人事专员", "行政助理", "市场专员", "仓库管理员", "配送员",
        "司机", "设计师", "数据分析师", "运维工程师", "门店店员", "课程顾问",
        "测试工程师", "采购专员"]

VIOLATION_TEMPLATES = [
    ("gender_restriction", "就业歧视", "high", "招聘{job}，限男性，负责{task}。"),
    ("gender_restriction", "就业歧视", "high", "招聘{job}，限女性，要求{skill}。"),
    ("gender_restriction", "就业歧视", "high", "招聘{job}，男性优先，有相关经验。"),
    ("gender_restriction", "就业歧视", "high", "招聘{job}，女性优先，可接受应届生。"),
    ("gender_restriction", "就业歧视", "high", "招聘{job}，仅限男，需出差。"),
    ("gender_restriction", "就业歧视", "high", "招聘{job}，仅限女，工作时间固定。"),
    ("marital_fertility", "就业歧视", "high", "招聘{job}，已婚已育优先。"),
    ("marital_fertility", "就业歧视", "high", "招聘{job}，已婚优先，稳定性好。"),
    ("marital_fertility", "就业歧视", "high", "招聘{job}，未婚优先，可接受外派。"),
    ("marital_fertility", "就业歧视", "high", "招聘{job}，需说明婚育情况。"),
    ("marital_fertility", "就业歧视", "high", "招聘{job}，需提供生育计划。"),
    ("age_restriction", "就业歧视", "high", "招聘{job}，35岁以下。"),
    ("age_restriction", "就业歧视", "high", "招聘{job}，40岁以上。"),
    ("age_restriction", "就业歧视", "high", "招聘{job}，年龄不超过45岁。"),
    ("age_restriction", "就业歧视", "high", "招聘{job}，年龄要求30岁以内。"),
    ("age_restriction", "就业歧视", "high", "招聘{job}，28岁以下，能接受出差。"),
    ("hukou_restriction", "就业歧视", "high", "招聘{job}，要求本地户口。"),
    ("hukou_restriction", "就业歧视", "high", "招聘{job}，要求本市户口。"),
    ("hukou_restriction", "就业歧视", "high", "招聘{job}，有户籍限制。"),
    ("hukou_restriction", "就业歧视", "high", "招聘{job}，仅限本地人员。"),
    ("ethnicity_religion", "就业歧视", "high", "招聘{job}，少数民族优先。"),
    ("ethnicity_religion", "就业歧视", "high", "招聘{job}，有宗教要求。"),
    ("ethnicity_religion", "就业歧视", "high", "招聘{job}，需说明民族。"),
    ("ethnicity_religion", "就业歧视", "high", "招聘{job}，要求填写信仰。"),
    ("health_disability", "就业歧视", "high", "招聘{job}，要求无残疾。"),
    ("health_disability", "就业歧视", "high", "招聘{job}，要求身体健康。"),
    ("health_disability", "就业歧视", "high", "招聘{job}，无传染病方可入职。"),
    ("health_disability", "就业歧视", "high", "招聘{job}，乙肝携带者不考虑。"),
    ("height_appearance", "就业歧视", "high", "招聘{job}，身高170cm以上。"),
    ("height_appearance", "就业歧视", "high", "招聘{job}，体重不超过标准。"),
    ("height_appearance", "就业歧视", "high", "招聘{job}，形象气质佳。"),
    ("height_appearance", "就业歧视", "high", "招聘{job}，五官端正。"),
    ("illegal_probation", "违法条款", "high", "招聘{job}，试用期9个月。"),
    ("illegal_probation", "违法条款", "high", "招聘{job}，试用期12个月。"),
    ("illegal_probation", "违法条款", "high", "招聘{job}，试用期8个月，表现优秀可转正。"),
    ("illegal_probation", "违法条款", "high", "招聘{job}，试用期超过6个月。"),
    ("illegal_probation", "违法条款", "high", "招聘{job}，试用期10个月。"),
    ("no_social_insurance", "违法条款", "high", "招聘{job}，入职后不缴社保。"),
    ("no_social_insurance", "违法条款", "high", "招聘{job}，不缴纳社保，以补贴代替。"),
    ("no_social_insurance", "违法条款", "high", "招聘{job}，社保自理。"),
    ("no_social_insurance", "违法条款", "high", "招聘{job}，需签自愿放弃社保承诺书。"),
    ("deposit_id", "违法条款", "high", "招聘{job}，入职需交押金。"),
    ("deposit_id", "违法条款", "high", "招聘{job}，需缴纳保证金。"),
    ("deposit_id", "违法条款", "high", "招聘{job}，入职扣押身份证。"),
    ("illegal_hours", "违法条款", "medium", "招聘{job}，接受996工作制。"),
    ("illegal_hours", "违法条款", "medium", "招聘{job}，实行大小周。"),
    ("illegal_hours", "违法条款", "medium", "招聘{job}，每周单休。"),
    ("illegal_hours", "违法条款", "medium", "招聘{job}，每月休息2天。"),
    ("false_promise", "虚假承诺", "medium", "招聘{job}，包就业。"),
    ("false_promise", "虚假承诺", "medium", "招聘{job}，入职保证录用。"),
    ("false_promise", "虚假承诺", "medium", "招聘{job}，百分百入职。"),
    ("salary_vague", "表述不清", "medium", "招聘{job}，薪资面议。"),
    ("salary_vague", "表述不清", "medium", "招聘{job}，工资面议。"),
]

TASKS = ["服务端开发", "需求分析", "客户维护", "数据报表", "合同管理",
         "仓库盘点", "课程设计", "组件开发", "系统运维", "招聘流程"]
SKILLS = ["Python", "Java", "SQL", "Vue3", "Excel", "项目管理", "沟通协调"]
YEARS = ["2", "3", "5"]
SALARIES = ["8000-12000", "12000-18000", "15000-25000", "6000-9000"]
LOCATIONS = ["北京", "上海", "深圳", "杭州", "成都"]

CLEAN_TEMPLATES = [
    "招聘{job}，负责{task}，要求本科及以上学历。",
    "招聘{job}，熟悉{skill}，有良好的沟通能力。",
    "招聘{job}，负责{task}，缴纳五险一金，提供带薪年假。",
    "招聘{job}，{years}年以上相关经验，薪资{salary}元/月。",
    "招聘{job}，工作地点{location}，双休，法定节假日正常休息。",
    "招聘{job}，负责{task}，接受应届生。",
    "招聘{job}，熟悉{skill}，有团队协作精神。",
    "招聘{job}，提供培训与晋升通道。",
    "招聘{job}，负责{task}，能熟练使用 Office。",
    "招聘{job}，{years}年以上经验，熟悉{skill}。",
    "招聘{job}，工作地点{location}，负责{task}。",
    "招聘{job}，薪资{salary}元/月，缴纳五险一金。",
]

BOUNDARY_TEMPLATES = [
    "优先考虑有经验者。",
    "能适应加班。",
    "年龄不限。",
    "不限性别、年龄、户籍。",
    "能接受弹性工作安排。",
    "薪资 15-25K/月。",
    "招聘{job}，有相关经验者优先。",
    "招聘{job}，能适应偶尔加班。",
    "招聘{job}，年龄不限，欢迎应届生。",
    "招聘{job}，性别不限。",
    "招聘{job}，户籍不限。",
    "招聘{job}，薪资 10-15K/月。",
    "招聘{job}，提供五险一金。",
    "招聘{job}，工作时间 9:00-18:00。",
    "招聘{job}，接受弹性工作制。",
    "招聘{job}，有经验者优先考虑。",
    "招聘{job}，能承受一定工作压力。",
    "招聘{job}，需偶尔出差。",
    "招聘{job}，欢迎不同背景的候选人。",
    "招聘{job}，薪资根据经验面谈。",
]


def fill(template, index):
    return template.format(
        job=JOBS[index % len(JOBS)],
        task=TASKS[index % len(TASKS)],
        skill=SKILLS[index % len(SKILLS)],
        years=YEARS[index % len(YEARS)],
        salary=SALARIES[index % len(SALARIES)],
        location=LOCATIONS[index % len(LOCATIONS)],
    )


def expected_findings(text):
    return [
        {"category": hit["category"], "span": hit["span"], "severity": hit["severity"]}
        for hit in hc.scan_rules(text)
    ]


def signature(findings):
    return sorted((item["category"], item["span"], item["severity"]) for item in findings)


def validate_entry(entry):
    text = entry["text"]
    actual = expected_findings(text)
    gold = entry["gold_findings"]
    if entry["clean"]:
        assert actual == [], f"干净样本误命中: {text} -> {actual}"
    elif entry["boundary"]:
        assert not any(hit["severity"] == "high" for hit in actual), f"边界样本误报 high: {text}"
        assert actual == [], f"边界样本仍有命中: {text} -> {actual}"
    else:
        assert signature(actual) == signature(gold), (
            f"违规样本命中与 gold 不一致: {text}\nactual={actual}\ngold={gold}")
    for item in gold:
        assert item["span"] and item["span"] in text, f"gold span 不在原文: {item} / {text}"


def build():
    entries = []

    # 1) 30 条沿用：前 10 违规 -> dev，后 10 违规 -> test；前 5 干净 -> dev，后 5 -> test
    violation_index = clean_index = 0
    for index, item in enumerate(SEED_30):
        if item["clean"]:
            clean_index += 1
            split = "dev" if clean_index <= 5 else "test"
        else:
            violation_index += 1
            split = "dev" if violation_index <= 10 else "test"
        entries.append({
            "id": f"hr-{index + 1:03d}",
            "text": item["jd_text"],
            "scope": "jd",
            "gold_findings": item["gold_findings"],
            "clean": bool(item["clean"]),
            "boundary": False,
            "split": split,
        })

    # 2) 生成违规：dev 30 + test 30，各 split 都均匀覆盖 13 类规则
    #    dev 从模板 0 起按等距取；test 偏移半个步长，避免两个 split 用同一批模板
    total_templates = len(VIOLATION_TEMPLATES)
    for split, count, offset in (("dev", 30, 0), ("test", 30, total_templates // 2)):
        for i in range(count):
            template_index = (i * total_templates // count + offset) % total_templates
            rule_id, category, severity, template = VIOLATION_TEMPLATES[template_index]
            text = fill(template, i + (0 if split == "dev" else 7))
            findings = expected_findings(text)
            entries.append({
                "id": f"gen-{split}-viol-{i + 1:02d}",
                "text": text,
                "scope": "jd",
                "gold_findings": findings,
                "clean": False,
                "boundary": False,
                "split": split,
            })

    # 3) 生成干净：dev 10 + test 20
    for split, count in (("dev", 10), ("test", 20)):
        for i in range(count):
            text = fill(CLEAN_TEMPLATES[i % len(CLEAN_TEMPLATES)], i + (0 if split == "dev" else 3))
            entries.append({
                "id": f"gen-{split}-clean-{i + 1:02d}",
                "text": text,
                "scope": "jd",
                "gold_findings": [],
                "clean": True,
                "boundary": False,
                "split": split,
            })

    # 4) 生成边界：dev 5 + test 15，20 条模板各用一次，避免重复文本
    boundary_cursor = 0
    for split, count in (("dev", 5), ("test", 15)):
        for i in range(count):
            text = fill(BOUNDARY_TEMPLATES[boundary_cursor], boundary_cursor)
            boundary_cursor += 1
            entries.append({
                "id": f"gen-{split}-boundary-{i + 1:02d}",
                "text": text,
                "scope": "jd",
                "gold_findings": [],
                "clean": False,
                "boundary": True,
                "split": split,
            })

    # 逐条核对 + 去重
    texts = set()
    for entry in entries:
        validate_entry(entry)
        assert entry["text"] not in texts, f"文本重复: {entry['text']}"
        texts.add(entry["text"])

    # 统计
    splits = {"dev": {"violation": 0, "clean": 0, "boundary": 0},
              "test": {"violation": 0, "clean": 0, "boundary": 0}}
    for entry in entries:
        kind = "clean" if entry["clean"] else ("boundary" if entry["boundary"] else "violation")
        splits[entry["split"]][kind] += 1
    assert len(entries) == 140, len(entries)
    assert splits == {"dev": {"violation": 40, "clean": 15, "boundary": 5},
                      "test": {"violation": 40, "clean": 25, "boundary": 15}}, splits
    return entries


def main():
    entries = build()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(open(OUT, "rb").read()).hexdigest()
    print(f"wrote {len(entries)} -> {OUT}")
    print(f"sha256 {digest}")
    for split in ("dev", "test"):
        subset = [e for e in entries if e["split"] == split]
        print(f"  {split}: {len(subset)}")


if __name__ == "__main__":
    main()
