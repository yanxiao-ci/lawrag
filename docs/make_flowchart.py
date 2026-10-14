# -*- coding: utf-8 -*-
"""
生成 lawrag（法律法规智能问答）项目流程图（标准流程图符号 / SVG）。
仅使用 Python 标准库，不依赖第三方包。

画的是**代码里真实执行的流程**（2026-10-14 法律方向改造后的版本）。

输出目录：docs/figures/
  - overview.svg  总体流程图（离线准备 / 在线问答 / 评估迭代）
  - indexing.svg  离线知识库构建详细流程
  - faq.svg       在线问答通道一「条文号精确路由 + 快问快答（MySQL + Redis + BM25）」
  - rag.svg       在线问答通道二「RAG 检索增强（法条 + 问答双召回）」
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
FONT = "Microsoft YaHei, PingFang SC, Segoe UI, sans-serif"

# 语义配色：(填充色, 描边色)
C = {
    "term":  ("#e8f5e9", "#2e7d32"),   # 起止
    "proc":  ("#eef4ff", "#3b6fd4"),   # 处理
    "dec":   ("#fff8e1", "#d99b00"),   # 判断
    "store": ("#f3e8ff", "#7c3aed"),   # 数据 / 存储
    "io":    ("#fff1e6", "#e2650a"),   # 输入 / 输出
    "ext":   ("#e0f7fa", "#0e7490"),   # 外部服务（大模型）
    "sub":   ("#f1f5f9", "#475569"),   # 子流程
    "note":  ("#ffffff", "#94a3b8"),   # 说明
}


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def poly(pts, fill, stroke, lw, dash=None):
    d = " ".join("%.1f,%.1f" % (p[0], p[1]) for p in pts)
    s = ('<polygon points="%s" fill="%s" stroke="%s" stroke-width="%s" stroke-linejoin="round"'
         % (d, fill, stroke, lw))
    if dash:
        s += ' stroke-dasharray="%s"' % dash
    return s + "/>"


def text_block(cx, cy, lines, size=16, weight="normal", fill="#1e293b", lh=1.32):
    if isinstance(lines, str):
        lines = lines.split("\n")
    step = size * lh
    y0 = cy - (len(lines) - 1) * step / 2.0
    out = []
    for i, ln in enumerate(lines):
        out.append(
            '<text x="%.1f" y="%.1f" font-size="%s" font-weight="%s" fill="%s" '
            'text-anchor="middle" dominant-baseline="middle" font-family="%s">%s</text>'
            % (cx, y0 + i * step, size, weight, fill, FONT, esc(ln)))
    return "\n".join(out)


def label(x, y, s, size=16, weight="bold", fill="#0f172a", anchor="middle"):
    return ('<text x="%.1f" y="%.1f" font-size="%s" font-weight="%s" fill="%s" '
            'text-anchor="%s" dominant-baseline="middle" font-family="%s" '
            'style="paint-order:stroke;stroke:#ffffff;stroke-width:5px;stroke-linejoin:round">%s</text>'
            % (x, y, size, weight, fill, anchor, FONT, esc(s)))


def node(x, y, w, h, lines, kind="proc", size=16, lw=2.0, dash=None, sub=False):
    fill, stroke = C[kind]
    cx, cy = x + w / 2.0, y + h / 2.0
    body = []
    ty = cy
    if kind == "dec":
        body.append(poly([(cx, y), (x + w, cy), (cx, y + h), (x, cy)], fill, stroke, lw, dash))
    elif kind == "term":
        r = min(h / 2.0, w / 2.0)
        body.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="%.1f" ry="%.1f" '
                    'fill="%s" stroke="%s" stroke-width="%s"/>' % (x, y, w, h, r, r, fill, stroke, lw))
    elif kind == "store":
        ry = 15.0
        body.append('<path d="M %.1f %.1f A %.1f %.1f 0 0 1 %.1f %.1f L %.1f %.1f '
                    'A %.1f %.1f 0 0 1 %.1f %.1f Z" fill="%s" stroke="%s" stroke-width="%s"/>'
                    % (x, y + ry, w / 2.0, ry, x + w, y + ry, x + w, y + h - ry,
                       w / 2.0, ry, x, y + h - ry, fill, stroke, lw))
        body.append('<ellipse cx="%.1f" cy="%.1f" rx="%.1f" ry="%.1f" fill="none" '
                    'stroke="%s" stroke-width="%s"/>' % (cx, y + ry, w / 2.0, ry, stroke, lw))
        ty = cy + ry * 0.55
    elif kind == "io":
        sk = min(24.0, w * 0.08)
        body.append(poly([(x + sk, y), (x + w, y), (x + w - sk, y + h), (x, y + h)],
                         fill, stroke, lw, dash))
    elif kind == "note":
        body.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="8" ry="8" '
                    'fill="%s" stroke="%s" stroke-width="1.5" stroke-dasharray="7 5"/>'
                    % (x, y, w, h, fill, stroke))
    else:
        body.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="9" ry="9" '
                    'fill="%s" stroke="%s" stroke-width="%s"%s/>'
                    % (x, y, w, h, fill, stroke, lw,
                       (' stroke-dasharray="%s"' % dash) if dash else ""))
    if sub:
        body.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="%s"/>'
                    % (x + 13, y + 6, x + 13, y + h - 6, stroke, lw))
        body.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="%s"/>'
                    % (x + w - 13, y + 6, x + w - 13, y + h - 6, stroke, lw))
    body.append(text_block(cx, ty, lines, size,
                           fill="#1e293b" if kind != "note" else "#475569"))
    return "\n".join(body)


def group(x, y, w, h, title, color="#64748b", fill="#fbfcfe"):
    s = ('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="16" ry="16" fill="%s" '
         'stroke="%s" stroke-width="2" stroke-dasharray="9 7"/>' % (x, y, w, h, fill, color))
    s += ('<text x="%.1f" y="%.1f" font-size="20" font-weight="bold" fill="%s" '
          'font-family="%s">%s</text>' % (x + 22, y + 30, color, FONT, esc(title)))
    return s


MARKERS = """
  <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#334155"/></marker>
  <marker id="ahb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#2563eb"/></marker>
  <marker id="ahg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#16a34a"/></marker>
  <marker id="ahr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#dc2626"/></marker>
  <marker id="ahgray" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8"/></marker>
  <marker id="ahB" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#2563eb"/></marker>
"""


def arrow(pts, color="#334155", dash=None, lw=2.2, marker="#334155"):
    d = "M " + " L ".join("%.1f %.1f" % (p[0], p[1]) for p in pts)
    s = ('<path d="%s" fill="none" stroke="%s" stroke-width="%s" stroke-linejoin="round" '
         'marker-end="url(#%s)"' % (d, color, lw, marker))
    if dash:
        s += ' stroke-dasharray="%s"' % dash
    return s + "/>"


def line(pts, color="#334155", dash=None, lw=2.2):
    """只画连线、不带箭头（用于汇入同一节点的支路）。"""
    d = "M " + " L ".join("%.1f %.1f" % (p[0], p[1]) for p in pts)
    s = ('<path d="%s" fill="none" stroke="%s" stroke-width="%s" stroke-linejoin="round"'
         % (d, color, lw))
    if dash:
        s += ' stroke-dasharray="%s"' % dash
    return s + "/>"


def svg(w, h, layers):
    head = ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">'
            % (w, h, w, h))
    defs = "<defs>%s</defs>" % MARKERS
    bg = '<rect x="0" y="0" width="%d" height="%d" fill="#ffffff"/>' % (w, h)
    return head + defs + bg + "\n".join(layers) + "</svg>"


def title(cx, main, sub=None):
    out = ('<text x="%.1f" y="52" font-size="33" font-weight="bold" fill="#0f172a" '
           'text-anchor="middle" font-family="%s">%s</text>' % (cx, FONT, esc(main)))
    if sub:
        out += ('<text x="%.1f" y="88" font-size="17" fill="#64748b" text-anchor="middle" '
                'font-family="%s">%s</text>' % (cx, FONT, esc(sub)))
    return out


def legend(x, y, items, size=15):
    """items: [(kind, 中文名, 是否子流程)]"""
    out = []
    cx = x
    for kind, name, *rest in items:
        sub = bool(rest and rest[0])
        out.append(node(cx, y, 74, 34, [""], kind=kind, size=1, lw=1.6, sub=sub))
        out.append(label(cx + 44, y + 17, name, size=size, weight="normal",
                         fill="#475569", anchor="start"))
        cx += 44 + len(name) * size + 28
    return "\n".join(out)


W1, H1 = 1880, 1500
W2, H2 = 1800, 1010
W3, H3 = 1880, 1350
W4, H4 = 1920, 1760


def fig_overview():
    g, a, n = [], [], []
    g.append(title(W1 / 2.0, "lawrag 法律法规智能问答项目 · 总体流程图",
                   "离线准备（一次性） → 在线问答（每次请求） → 效果评估（迭代优化）"))

    # ---------------- A 离线准备 ----------------
    g.append(group(40, 118, 1800, 382, "A · 离线准备：知识库构建（法规 / 问答数据更新时重跑）"))
    yA1 = 180
    a1 = (70, yA1, 250, 84, "法律法规原文 .txt\n292 部 · 4.2 MB", "io", 14)
    a2 = (350, yA1, 250, 84, "法条解析 + 边界切分\n子块 = 单条法条\n父块 = 同法连续 5 条", "proc", 13)
    a3 = (630, yA1, 250, 84, "BGE-M3 向量化\n稠密 1024 维 + 稀疏向量", "proc", 13)
    a4 = (910, yA1, 250, 84, "写入 Milvus\n集合 lawrag · 55,847 块", "store", 13)
    a5 = (1190, yA1, 250, 84, "法条结构化入库\nMySQL law_articles · 21,565 条", "store", 13)
    a6 = (1470, yA1, 340, 84, "字段：法律名 · 条号 · 编/章/节\n公布日 · 施行日 · 时效状态", "note", 13)
    for t in (a1, a2, a3, a4, a5, a6):
        n.append(node(*t))
    for i in range(3):
        a.append(arrow([(a1[0] + i * 280 + 250, yA1 + 42), (a1[0] + (i + 1) * 280, yA1 + 42)]))
    a.append(arrow([(475, yA1 + 84), (475, 276), (1315, 276), (1315, yA1 + 84)],
                   color="#64748b", dash="6 5", marker="#ahgray"))
    a.append(label(895, 286, "结构化入库（与向量化并行）", size=13, weight="normal", fill="#64748b"))
    a.append(arrow([(1440, yA1 + 42), (1470, yA1 + 42)], color="#94a3b8", dash="6 5", marker="#ahgray"))

    yA2 = 290
    b1 = (70, yA2, 250, 84, "法律问答 CSV\nlaw_qa.csv · 20,000 条", "io", 13)
    b2 = (350, yA2, 250, 84, "写入 MySQL\n表 jpkb", "store", 13)
    b3 = (630, yA2, 250, 84, "Redis 问题列表\n原文 + Jieba 分词", "store", 13)
    b4 = (910, yA2, 250, 84, "BM25Okapi 双索引\n法条正文 + 问答问题", "proc", 13)
    b5 = (1190, yA2, 620, 84, "命中判定 = 词面判据：问答看词集合 Jaccard ≥ 0.85，法条看查询词覆盖率 ≥ 0.85\n阈值由 tools/tune_fast_channel_gate.py 标定；命中后写 Redis 缓存 answer:{问题}（TTL 7 天）", "note", 13)
    for t in (b1, b2, b3, b4, b5):
        n.append(node(*t))
    for i in range(3):
        a.append(arrow([(b1[0] + i * 280 + 250, yA2 + 42), (b1[0] + (i + 1) * 280, yA2 + 42)]))
    a.append(arrow([(1160, yA2 + 42), (1190, yA2 + 42)], color="#94a3b8", dash="6 5", marker="#ahgray"))

    yA3 = 400
    c1 = (70, yA3, 250, 84, "jpkb 法律问答\n20,000 条", "io", 13)
    c2 = (350, yA3, 250, 84, "BGE-M3 向量化\n（问题 + 答案）", "proc", 13)
    c3 = (630, yA3, 250, 84, "写入 Milvus\n集合 lawfaq · 20,000 条", "store", 13)
    c4 = (910, yA3, 900, 84, "为什么单独一个集合：法条集合的检索强制 status ==「有效」，问答条目没有这个字段，混进去会被整条过滤掉", "note", 13)
    for t in (c1, c2, c3, c4):
        n.append(node(*t))
    a.append(arrow([(320, yA3 + 42), (350, yA3 + 42)]))
    a.append(arrow([(600, yA3 + 42), (630, yA3 + 42)]))
    a.append(arrow([(880, yA3 + 42), (910, yA3 + 42)], color="#94a3b8", dash="6 5", marker="#ahgray"))
    a.append(arrow([(475, yA2 + 84), (475, 392), (195, 392), (195, yA3)],
                   color="#64748b", dash="6 5", marker="#ahgray"))
    a.append(label(500, 392, "读 jpkb 的问答", size=12, weight="normal",
                   fill="#64748b", anchor="start"))

    # ---------------- B 在线问答 ----------------
    g.append(group(40, 520, 1800, 556, "B · 在线问答：一次请求的完整链路（实时，FastAPI + WebSocket 流式）"))
    yB1 = 590
    p1 = (70, yB1, 200, 84, "用户提问\n(前端页面)", "term", 14)
    p2 = (300, yB1, 320, 84, "FastAPI 统一入口\nPOST /api/query · WS /api/stream", "proc", 13)
    p3 = (650, yB1, 230, 84, "命中日常问候语？\n（正则模板）", "dec", 13)
    p4 = (910, yB1, 280, 84, "模板回复\n（法律问答场景人设）", "proc", 13)
    p5 = (1220, yB1, 260, 84, "直接返回答案\n(不触发检索)", "term", 13)
    for t in (p1, p2, p3, p4, p5):
        n.append(node(*t))
    a.append(arrow([(270, yB1 + 42), (300, yB1 + 42)]))
    a.append(arrow([(620, yB1 + 42), (650, yB1 + 42)]))
    a.append(arrow([(880, yB1 + 42), (910, yB1 + 42)]))
    a.append(label(895, yB1 + 22, "是", fill="#16a34a"))
    a.append(arrow([(1190, yB1 + 42), (1220, yB1 + 42)]))

    yB2 = 726
    q1 = (70, yB2, 240, 88, "读取会话历史\nMySQL conversations（最近 5 轮）", "store", 13)
    q2 = (330, yB2, 260, 88, "条文号精确路由\n《X法》第 Y 条 → MySQL 精确查", "proc", 13)
    q3 = (610, yB2, 220, 88, "命中条文？", "dec", 14)
    q4 = (860, yB2, 300, 88, "① 快问快答通道\nRedis 缓存 → Jieba → BM25 → 词面门控", "sub", 13)
    q5 = (1190, yB2, 240, 88, "快通道命中？\n（Jaccard / 覆盖率 ≥ 0.85）", "dec", 13)
    q6 = (1470, yB2, 340, 88, "说明：条文号路由是确定性检索，优先于语义检索；\n命中即带编 / 章 / 节与施行日期返回", "note", 13)
    for t in (q1, q2, q3, q4, q5, q6):
        n.append(node(*t))
    a.append(arrow([(765, yB1 + 84), (765, 706), (195, 706), (195, yB2)], marker="#ahgray"))
    a.append(label(745, 690, "否", fill="#dc2626"))
    a.append(arrow([(310, yB2 + 44), (330, yB2 + 44)]))
    a.append(arrow([(590, yB2 + 44), (610, yB2 + 44)]))
    a.append(arrow([(830, yB2 + 44), (860, yB2 + 44)]))
    a.append(label(845, yB2 + 22, "否", fill="#dc2626"))
    a.append(arrow([(1160, yB2 + 44), (1190, yB2 + 44)]))
    a.append(arrow([(1430, yB2 + 44), (1470, yB2 + 44)], color="#94a3b8", dash="6 5", marker="#ahgray"))

    yB3 = 846
    r1 = (610, yB3, 330, 84, "返回条文原文\n（编 / 章 / 节 + 施行日 + 历史版本提示）", "term", 13)
    r2 = (1190, yB3, 330, 84, "返回标准问答答案\n并写 Redis 缓存 answer:{问题}", "term", 13)
    for t in (r1, r2):
        n.append(node(*t))
    a.append(arrow([(720, yB2 + 88), (720, yB3)]))
    a.append(label(742, 828, "是", fill="#16a34a"))
    a.append(arrow([(1310, yB2 + 88), (1310, yB3)]))
    a.append(label(1332, 828, "是", fill="#16a34a"))

    yB4 = 960
    t1 = (350, yB4, 320, 96, "② RAG 检索增强通道\n意图识别 → 策略选择 → 法条 + 问答双召回 → 精排\n（详见 图 4）", "sub", 13)
    t2 = (750, yB4, 300, 96, "拼接 Prompt\n法条块 + 问答块 + 历史 + 问题\n（超 4096 字符截断）", "proc", 13)
    t3 = (1130, yB4, 300, 96, "大模型流式生成 + 引用校验\nDashScope（逐 token 输出）", "ext", 13)
    t4 = (1500, yB4, 300, 96, "流式返回答案\n并写入会话历史", "term", 13)
    for t in (t1, t2, t3, t4):
        n.append(node(*t))
    a.append(arrow([(1240, yB2), (1240, 700), (845, 700), (845, 944), (510, 944), (510, yB4)],
                   marker="#ahgray"))
    a.append(label(1258, 712, "否", fill="#dc2626", anchor="start"))
    a.append(arrow([(670, yB4 + 48), (750, yB4 + 48)]))
    a.append(arrow([(1050, yB4 + 48), (1130, yB4 + 48)]))
    a.append(arrow([(1430, yB4 + 48), (1500, yB4 + 48)]))

    # ---------------- C 评估 ----------------
    g.append(group(40, 1096, 1800, 300, "C · 效果评估与迭代（四类评测 + 参数回写闭环）"))
    yC1 = 1160
    u1 = (70, yC1, 260, 88, "评测数据\n检索集 · RAGAS 集 · 标注题集", "io", 13)
    u2 = (370, yC1, 400, 88, "四类评测：检索 Recall@k / 上下文覆盖率 /\n快通道门控标定 / RAGAS 四项指标", "proc", 13)
    u3 = (810, yC1, 280, 88, "结果落盘\nrag_qa/data/law_eval/", "store", 13)
    u4 = (1130, yC1, 330, 88, "定位问题并调参（切分粒度 / 门控阈值 /\n检索策略 / k · M / 提示词）", "proc", 13)
    u5 = (1500, yC1, 310, 88, "回归评估\n确认指标不下降", "term", 13)
    for t in (u1, u2, u3, u4, u5):
        n.append(node(*t))
    a.append(arrow([(330, yC1 + 44), (370, yC1 + 44)]))
    a.append(arrow([(770, yC1 + 44), (810, yC1 + 44)]))
    a.append(arrow([(1090, yC1 + 44), (1130, yC1 + 44)]))
    a.append(arrow([(1460, yC1 + 44), (1500, yC1 + 44)]))
    a.append(arrow([(1655, yC1), (1655, 1130), (1295, 1130), (1295, yC1)],
                   color="#2563eb", dash="8 6", marker="#ahb"))
    a.append(label(1475, 1114, "指标下降 → 回到调参（参数写回 config.ini / .env）",
                   size=13, weight="normal", fill="#2563eb"))

    yC2 = 1272
    v1 = (70, yC2, 860, 64, "脚本：tools/eval_recall.py · tools/eval_context_coverage.py · tools/tune_fast_channel_gate.py · rag_qa/rag_assessment/ragas_evaluate.py", "note", 13)
    v2 = (950, yC2, 860, 64, "参数：切分粒度 · RETRIEVAL_K · CANDIDATE_M · FAQ_RETRIEVAL_K / FAQ_INJECT_N · 快通道阈值", "note", 13)
    for t in (v1, v2):
        n.append(node(*t))

    g.append(legend(60, 1420, [("term", "起止", False), ("proc", "处理", False),
                               ("dec", "判断", False), ("store", "数据/存储", False),
                               ("io", "输入/输出", False), ("sub", "子流程", True),
                               ("ext", "外部服务", False), ("note", "说明", False)]))
    return svg(W1, H1, g + a + n)


def fig_indexing():
    g, a, n = [], [], []
    g.append(title(W2 / 2.0, "离线知识库构建详细流程",
                   "三条链路：① 法条向量库（Milvus lawrag） ② 法条结构化表 + 快问快答库（MySQL + Redis + BM25） ③ 问答向量库（Milvus lawfaq）"))
    g.append(group(40, 120, 1720, 330, "① RAG 法条向量库构建：法条原文 → 解析 → 切分 → 向量化 → Milvus"))

    y1 = 190
    t1 = (70, y1, 320, 86, "法条原文 rag_qa/data/law_data\n292 部 .txt（288 现行 + 4 历史）", "io", 13)
    t2 = (410, y1, 320, 86, "法条解析 base/law_parser.py\n法律名 · 条号 · 条文正文", "proc", 13)
    t3 = (750, y1, 320, 86, "法条边界切分\n子块 = 一条完整法条（不跨条）", "proc", 13)
    t4 = (1090, y1, 320, 86, "父块聚合\n同一部法律连续 5 条为一组", "proc", 13)
    t5 = (1430, y1, 320, 86, "父子关系元数据\nparent_id · parent_content · id(md5)", "proc", 13)
    for t in (t1, t2, t3, t4, t5):
        n.append(node(*t))
    for i in range(4):
        a.append(arrow([(t1[0] + i * 340 + 320, y1 + 43), (t1[0] + (i + 1) * 340, y1 + 43)]))

    y2 = 326
    u1 = (70, y2, 320, 86, "BGE-M3 向量化\n稠密 1024 维 + 稀疏向量", "proc", 13)
    u2 = (410, y2, 320, 86, "upsert 写入 Milvus\n集合 lawrag · 55,847 块", "store", 13)
    u3 = (750, y2, 1000, 86, "集合字段：id · text · dense_vector · sparse_vector · parent_id · parent_content\nsource(=law) · law_name · article_no · chapter · status（检索强制 status ==「有效」）", "note", 13)
    for t in (u1, u2, u3):
        n.append(node(*t))
    a.append(arrow([(1590, y1 + 86), (1590, 306), (230, 306), (230, y2)], color="#94a3b8", dash="6 5", marker="#ahgray"))
    a.append(arrow([(390, y2 + 43), (410, y2 + 43)]))
    a.append(arrow([(730, y2 + 43), (750, y2 + 43)], color="#94a3b8", dash="6 5", marker="#ahgray"))

    g.append(group(40, 476, 1720, 444, "② 法条结构化表 + 快问快答库   ③ 问答向量库（RAG 第三召回源）"))
    y3 = 544
    x1 = (70, y3, 340, 86, "写入 MySQL 表 law_articles\n21,565 条（法律名 · 条号 · 条文）", "store", 13)
    x2 = (450, y3, 340, 86, "补 P1 元数据\n编 / 章 / 节 · 公布日 · 施行日 · 时效状态", "proc", 13)
    x3 = (830, y3, 900, 86, "在线用途：① 条文号精确路由（《X法》第 Y 条 → 精确查表）② 快通道法条 BM25 语料\n③ 时效过滤：历史版本标注为「已修改 / 已废止」，不参与召回", "note", 13)
    for t in (x1, x2, x3):
        n.append(node(*t))
    a.append(arrow([(410, y3 + 43), (450, y3 + 43)]))
    a.append(arrow([(790, y3 + 43), (830, y3 + 43)], color="#94a3b8", dash="6 5", marker="#ahgray"))

    y4 = 670
    z1 = (70, y4, 340, 86, "law_qa.csv\n20,000 条法律问答", "io", 13)
    z2 = (450, y4, 340, 86, "MySQL 表 jpkb\nsubject_name · question · answer", "store", 13)
    z3 = (830, y4, 340, 86, "Redis 问题列表\nqa_original_questions / qa_tokenized_questions", "store", 12)
    z4 = (1210, y4, 520, 86, "BM25Okapi 双索引（法条正文 + 问答问题）\n门控：词集合 Jaccard ≥ 0.85 / 查询词覆盖率 ≥ 0.85", "proc", 12)
    for t in (z1, z2, z3, z4):
        n.append(node(*t))
    a.append(arrow([(410, y4 + 43), (450, y4 + 43)]))
    a.append(arrow([(790, y4 + 43), (830, y4 + 43)]))
    a.append(arrow([(1170, y4 + 43), (1210, y4 + 43)]))

    y5 = 798
    w1 = (70, y5, 340, 86, "jpkb 20,000 条\n（问题 + 答案）", "io", 13)
    w2 = (450, y5, 340, 86, "BGE-M3 向量化", "proc", 13)
    w3 = (830, y5, 340, 86, "写入 Milvus\n集合 lawfaq · 20,000 条", "store", 13)
    w4 = (1210, y5, 520, 86, "在线用途：RAG 第三召回源\nFAQ_RETRIEVAL_K = 3 → 精排后注入 FAQ_INJECT_N = 2 条", "note", 12)
    for t in (w1, w2, w3, w4):
        n.append(node(*t))
    a.append(arrow([(410, y5 + 43), (450, y5 + 43)]))
    a.append(arrow([(790, y5 + 43), (830, y5 + 43)]))
    a.append(arrow([(1170, y5 + 43), (1210, y5 + 43)], color="#94a3b8", dash="6 5", marker="#ahgray"))
    a.append(arrow([(620, y4 + 86), (620, 776), (240, 776), (240, y5)], color="#64748b", dash="6 5", marker="#ahgray"))

    g.append(legend(60, 946, [("term", "起止", False), ("proc", "处理", False),
                               ("store", "数据/存储", False), ("io", "输入/输出", False),
                               ("note", "说明", False)]))
    return svg(W2, H2, g + a + n)


def fig_faq():
    g, a, n = [], [], []
    g.append(title(W3 / 2.0, "在线问答 · 通道一：条文号精确路由 + 快问快答（MySQL + Redis + BM25）",
                   "确定性优先：能精确命中的问题绝不进向量检索；只有「同一个问题 / 查询本身就是那条条文」才允许抢答"))

    cx = [80, 520, 960, 1400]
    bw = 400
    dec_h, box_h = 130, 96

    def row(cy):
        return cy - box_h / 2.0, cy - dec_h / 2.0

    by, dy = row(240)
    p1 = (cx[0], by, bw, box_h, "用户提问\n(前端页面)", "term", 14)
    p2 = (cx[1], by, bw, box_h, "FastAPI 统一入口\nPOST /api/query · WS /api/stream", "proc", 14)
    p3 = (cx[2], dy, bw, dec_h, "命中日常问候语？\n（正则模板 4 类）", "dec", 15)
    p4 = (cx[3], by, bw, box_h, "直接返回模板回复\n（法律问答场景人设）", "term", 14)
    for t in (p1, p2, p3, p4):
        n.append(node(*t))
    a.append(arrow([(cx[0] + bw, by + box_h / 2), (cx[1], by + box_h / 2)]))
    a.append(arrow([(cx[1] + bw, by + box_h / 2), (cx[2], by + box_h / 2)]))
    a.append(arrow([(cx[2] + bw, 240), (cx[3], 240)]))
    a.append(label((cx[2] + bw + cx[3]) / 2, 218, "是", fill="#16a34a"))

    by2, dy2 = row(460)
    q1 = (cx[0], by2, bw, box_h, "读取会话历史\nMySQL conversations（最近 5 轮）", "store", 13)
    q2 = (cx[1], by2, bw, box_h, "条文号精确路由\nbase/law_parser.extract_citation", "proc", 13)
    q3 = (cx[2], dy2, bw, dec_h, "命中条文？\n（《X法》第 Y 条 精确查 MySQL）", "dec", 14)
    q4 = (cx[3], by2, bw, box_h, "返回条文原文\n编 / 章 / 节 + 施行日 + 历史版本提示", "term", 13)
    for t in (q1, q2, q3, q4):
        n.append(node(*t))
    a.append(arrow([(cx[2], dy + dec_h), (cx[2], 340), (cx[0] + bw / 2, 340), (cx[0] + bw / 2, by2)]))
    a.append(label(cx[2] - 40, 320, "否", fill="#dc2626"))
    a.append(arrow([(cx[0] + bw, 460), (cx[1], 460)]))
    a.append(arrow([(cx[1] + bw, 460), (cx[2], 460)]))
    a.append(arrow([(cx[2] + bw, 460), (cx[3], 460)]))
    a.append(label((cx[2] + bw + cx[3]) / 2, 438, "是", fill="#16a34a"))

    by3, dy3 = row(680)
    r1 = (cx[0], dy3, bw, dec_h, "Redis 缓存命中？\nkey = answer:{用户问题}", "dec", 14)
    r2 = (cx[1], by3, bw, box_h, "直接返回缓存答案\n（无需检索）", "term", 14)
    r3 = (cx[2], by3, bw, box_h, "Jieba 分词\n统一转小写 preprocess_text()", "proc", 14)
    r4 = (cx[3], by3, bw, box_h, "BM25Okapi 双索引打分\n法条通道 / 问答通道各取 top-1", "proc", 13)
    for t in (r1, r2, r3, r4):
        n.append(node(*t))
    a.append(arrow([(cx[2] + bw / 2, dy2 + dec_h), (cx[2] + bw / 2, 600), (cx[0] + bw / 2, 600), (cx[0] + bw / 2, dy3)]))
    a.append(label(cx[2] + bw / 2 + 34, 580, "否", fill="#dc2626"))
    a.append(arrow([(cx[0] + bw, 680), (cx[1], 680)]))
    a.append(label((cx[0] + bw + cx[1]) / 2, 658, "是", fill="#16a34a"))
    a.append(arrow([(cx[0] + bw / 2, dy3 + dec_h), (cx[0] + bw / 2, 772),
                    (cx[2] + bw / 2, 772), (cx[2] + bw / 2, by3 + box_h)]))
    a.append(label(cx[0] + bw / 2 + 24, 762, "否", fill="#dc2626", anchor="start"))
    a.append(arrow([(cx[2] + bw, 680), (cx[3], 680)]))

    by4, dy4 = row(900)
    s1 = (cx[0], dy4, bw, dec_h, "法条通道：查询词覆盖率 ≥ 0.85？\n（查询基本就是这条条文）", "dec", 13)
    s2 = (cx[1], by4, bw, box_h, "返回该法条原文\n并写 Redis 缓存", "term", 13)
    s3 = (cx[2], dy4, bw, dec_h, "问答通道：词集合 Jaccard ≥ 0.85？\n（就是同一个问题）", "dec", 13)
    s4 = (cx[3], by4, bw, box_h, "MySQL 表 jpkb 取标准答案\n并写 Redis 缓存", "proc", 13)
    for t in (s1, s2, s3, s4):
        n.append(node(*t))
    a.append(arrow([(cx[3], by3 + box_h), (cx[3], 800), (cx[0] + bw / 2, 800), (cx[0] + bw / 2, dy4)]))
    a.append(arrow([(cx[0] + bw, 900), (cx[1], 900)]))
    a.append(label((cx[0] + bw + cx[1]) / 2, 878, "是", fill="#16a34a"))
    a.append(arrow([(cx[0] + bw / 2, dy4 + dec_h), (cx[0] + bw / 2, 985),
                    (cx[2] + bw / 2, 985), (cx[2] + bw / 2, dy4 + dec_h)]))
    a.append(label(700, 975, "否", fill="#dc2626"))
    a.append(arrow([(cx[2] + bw, 900), (cx[3], 900)]))
    a.append(label((cx[2] + bw + cx[3]) / 2, 878, "是", fill="#16a34a"))

    by5 = 1080
    t1 = (cx[0], by5, bw, box_h, "旧判据为什么废掉：全库 Softmax 归一化分会随语料规模漂移\n（97 部 → 292 部后，真实问题 90% 被抢答、约一半答错）", "note", 12)
    t2 = (cx[1], by5, bw, box_h, "标定方法：P = 库内原句 200 条，N = 真实法律问题 100 条\n新判据下错答 0、库内原句 200/200 秒回", "note", 12)
    t3 = (cx[2], by5, bw, box_h, "阈值 0.85 可配置（search(query, threshold=0.85)）\n标定脚本 tools/tune_fast_channel_gate.py", "note", 12)
    s5 = (cx[3], by5, bw, box_h, "未过两道门控：回退 RAG 通道\n（详见 图 4）", "sub", 14)
    for t in (t1, t2, t3, s5):
        n.append(node(*t))
    a.append(arrow([(cx[2] + bw / 2, dy4 + dec_h), (cx[2] + bw / 2, 1005),
                    (cx[3] + bw / 2, 1005), (cx[3] + bw / 2, by5)], marker="#ahgray"))
    a.append(label(cx[2] + bw / 2 + 22, 995, "否", fill="#dc2626", anchor="start"))

    g.append(legend(60, 1226, [("term", "起止", False), ("proc", "处理", False),
                               ("dec", "判断", False), ("store", "数据/存储", False)]))
    g.append(legend(60, 1276, [("sub", "子流程", True), ("note", "说明", False)]))
    return svg(W3, H3, g + a + n)


def fig_rag():
    g, a, n = [], [], []
    g.append(title(W4 / 2.0, "在线问答 · 通道二：RAG 检索增强",
                   "意图识别 → 策略选择 → 法条 + 问答双召回 → 精排 → 提示词 → 大模型生成（流式）→ 引用校验"))

    cx = [80, 520, 960, 1400]
    bw = 400

    y1 = 200
    t1 = (cx[0], y1, bw, 96, "输入：query + source_filter(=law) + 会话历史", "io", 14)
    t2 = (cx[1], y1, bw, 96, "BERT 意图识别\nrag_qa/models/bert_query_classifier_law", "proc", 13)
    t3 = (cx[2], y1 - 20, bw, 136, "查询类别？\n（通用知识 / 专业咨询）", "dec", 15)
    t4 = (cx[3], y1, bw, 96, "通用知识：直接调用大模型\n不检索知识库", "proc", 13)
    for t in (t1, t2, t3, t4):
        n.append(node(*t))
    a.append(arrow([(cx[0] + bw, y1 + 48), (cx[1], y1 + 48)]))
    a.append(arrow([(cx[1] + bw, y1 + 48), (cx[2], y1 + 48)]))
    a.append(arrow([(cx[2] + bw, y1 + 48), (cx[3], y1 + 48)]))
    a.append(label(1320, y1 + 24, "通用知识", size=13, fill="#16a34a"))

    y2 = 420
    u1 = (cx[1], y2, bw, 96, "LLM 策略选择器 StrategySelector\n从 4 种策略中选 1 种（temperature=0）", "proc", 13)
    n.append(node(*u1))
    a.append(arrow([(cx[2] + bw / 2, y1 + 116), (cx[2] + bw / 2, 380), (cx[1] + bw / 2, 380), (cx[1] + bw / 2, y2)]))
    a.append(label(cx[2] + bw / 2 + 60, 362, "专业咨询", fill="#dc2626"))

    y3 = 640
    v1 = (cx[0], y3, bw, 130, "① 直接检索\n原始 query 直接检索\n（意图明确、要特定信息）", "proc", 14)
    v2 = (cx[1], y3, bw, 130, "② 假设问题检索 HyDE\nLLM 先生成假设答案，\n再用假设答案去检索", "proc", 14)
    v3 = (cx[2], y3, bw, 130, "③ 子查询检索\n拆成多个子问题分别检索，\n合并后去重", "proc", 14)
    v4 = (cx[3], y3, bw, 130, "④ 回溯问题检索\n把复杂问题简化成\n更基础的问题再检索", "proc", 14)
    for t in (v1, v2, v3, v4):
        n.append(node(*t))
    for i in range(4):
        a.append(arrow([(cx[1] + bw / 2, y2 + 96), (cx[1] + bw / 2, 560),
                        (cx[i] + bw / 2, 560), (cx[i] + bw / 2, y3)]))

    y4 = 880
    w1 = (cx[0], y4, 820, 140, "BGE-M3 混合检索（粗排）\n稠密向量 + 稀疏向量并行检索，WeightedRanker(1.0, 0.7)\n过滤条件：status ==「有效」· 返回 k = 5 个子块", "proc", 14)
    w2 = (cx[2], y4, bw, 140, "按 parent_content 父块去重\n_get_unique_parent_docs()", "proc", 14)
    w3 = (cx[3], y4 - 18, bw, 176, "父块数量 ≥ 2 ？", "dec", 16)
    for t in (w1, w2, w3):
        n.append(node(*t))
    for i in range(4):
        a.append(arrow([(cx[i] + bw / 2, y3 + 130), (cx[i] + bw / 2, 840),
                        (cx[0] + 410, 840), (cx[0] + 410, y4)]))
    a.append(arrow([(cx[0] + 820, y4 + 70), (cx[2], y4 + 70)]))
    a.append(arrow([(cx[2] + bw, y4 + 70), (cx[3], y4 + 70)]))

    y5 = 1120
    z1 = (cx[3], y5, bw, 96, "直接返回父块\n（不足 2 个，无需重排）", "proc", 14)
    z2 = (cx[2], y5, bw, 96, "BGE-Reranker-large 精排\n(CrossEncoder) 打分降序", "proc", 14)
    z3 = (cx[1], y5, bw, 96, "取 Top-2 作为法条块\nCANDIDATE_M = 2", "proc", 14)
    for t in (z1, z2, z3):
        n.append(node(*t))
    a.append(arrow([(cx[3] + bw / 2, y4 + 158), (cx[3] + bw / 2, y5)]))
    a.append(label(cx[3] + bw / 2 + 46, y4 + 188, "否", fill="#dc2626"))
    a.append(arrow([(cx[3], y4 + 70), (cx[2] + bw, y4 + 70)], color="#16a34a", marker="#ahg"))
    a.append(label(cx[3] - 24, y4 + 42, "是", fill="#16a34a"))
    a.append(arrow([(cx[2] + bw / 2, y4 + 140), (cx[2] + bw / 2, y5)]))
    a.append(arrow([(cx[2], y5 + 48), (cx[1] + bw, y5 + 48)]))

    y6 = 1260
    s1 = (cx[1], y6, bw, 96, "问答召回 search_faq(query, k = 3)\n集合 lawfaq（2 万条问答向量）", "proc", 13)
    s2 = (cx[2], y6, bw, 96, "用同一个精排模型排序\n取 FAQ_INJECT_N = 2 条作为问答块", "proc", 13)
    for t in (s1, s2):
        n.append(node(*t))
    a.append(arrow([(cx[0] + 410, y4 + 140), (cx[0] + 410, 1240), (cx[1] + bw / 2, 1240), (cx[1] + bw / 2, y6)],
                   color="#64748b", dash="6 5", marker="#ahgray"))
    a.append(label(cx[0] + 560, 1232, "并行：问答召回（第三召回源）", size=12, weight="normal", fill="#64748b"))
    a.append(arrow([(cx[1] + bw, y6 + 48), (cx[2], y6 + 48)]))

    y7 = 1400
    k1 = (cx[0], y7, 560, 104, "拼接 Prompt：法条块（主要依据）+ 问答块（参考依据）\n+ 对话历史 + 问题（超 4096 字符截断）", "proc", 13)
    k2 = (cx[1] + 180, y7, 460, 104, "大模型流式生成\nDashScope（yield token）", "ext", 13)
    k3 = (cx[2] + 240, y7, 400, 104, "引用校验 validate_citations\n核对答案引用的法条是否真实存在", "proc", 13)
    k4 = (cx[3] + 160, y7, 320, 104, "结束：返回答案\n（流式 token / 一次性）", "term", 13)
    for t in (k1, k2, k3, k4):
        n.append(node(*t))
    a.append(arrow([(cx[1] + bw / 2, y5 + 96), (cx[1] + bw / 2, 1360), (cx[0] + 280, 1360), (cx[0] + 280, y7)]))
    a.append(arrow([(cx[2] + bw / 2, y6 + 96), (cx[2] + bw / 2, 1382), (cx[0] + 400, 1382), (cx[0] + 400, y7)]))
    a.append(arrow([(cx[0] + 560, y7 + 52), (cx[1] + 180, y7 + 52)]))
    a.append(arrow([(cx[1] + 640, y7 + 52), (cx[2] + 240, y7 + 52)]))
    a.append(arrow([(cx[2] + 640, y7 + 52), (cx[3] + 160, y7 + 52)]))
    a.append(arrow([(cx[3] + bw, y1 + 48), (cx[3] + 460, y1 + 48), (cx[3] + 460, 1452), (cx[3] + 480, 1452)],
                   color="#16a34a", marker="#ahg"))

    y8 = 1540
    m1 = (cx[0], y8, 560, 84, "提示词模板：rag_qa/core/prompts.py\n上下文分两类块：法条块「主要依据」/ 问答块「参考依据」", "note", 13)
    m2 = (cx[1] + 180, y8, 460, 84, "策略提示词：strategy_selector.py\nHyDE / 子查询 / 回溯提示词在 prompts.py", "note", 13)
    m3 = (cx[2] + 240, y8, 720, 84, "参数来自 config.ini / .env：切分粒度 · RETRIEVAL_K · CANDIDATE_M · FAQ_RETRIEVAL_K · FAQ_INJECT_N", "note", 13)
    for t in (m1, m2, m3):
        n.append(node(*t))

    g.append(legend(60, 1670, [("term", "起止", False), ("proc", "处理", False),
                               ("dec", "判断", False), ("store", "数据/存储", False),
                               ("io", "输入/输出", False), ("ext", "外部服务", False),
                               ("note", "说明", False)]))
    return svg(W4, H4, g + a + n)


FIGURES = (("overview", fig_overview, W1, H1), ("indexing", fig_indexing, W2, H2),
           ("faq", fig_faq, W3, H3), ("rag", fig_rag, W4, H4))


def main():
    """生成 SVG，并额外输出一份用于无头浏览器渲染的 HTML 包装页（便于导出 PNG）。"""
    import tempfile
    os.makedirs(FIG, exist_ok=True)
    tmp = os.path.join(tempfile.gettempdir(), "codex_rag_flow")
    os.makedirs(tmp, exist_ok=True)
    tmp_html = []
    for name, builder, w, h in FIGURES:
        body = builder()
        with open(os.path.join(FIG, name + ".svg"), "w", encoding="utf-8") as f:
            f.write(body)
        html = ('<!doctype html><html><head><meta charset="utf-8">'
                '<style>html,body{margin:0;padding:0;background:#fff}</style></head>'
                '<body>%s</body></html>' % body)
        with open(os.path.join(tmp, name + ".html"), "w", encoding="utf-8") as f:
            f.write(html)
        tmp_html.append(os.path.join(tmp, name + ".html"))
        print("written:", name, w, h)
    print("html:", tmp)


if __name__ == "__main__":
    main()
