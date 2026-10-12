# -*- coding: utf-8 -*-
"""
生成 edu_rag 项目流程图（标准流程图符号 / SVG）。
仅使用 Python 标准库，不依赖第三方包。

输出目录：docs/figures/
  - overview.svg  总体流程图（离线准备 / 在线问答 / 评估迭代）
  - indexing.svg  离线知识库构建详细流程
  - faq.svg       在线问答之「快问快答（MySQL + Redis + BM25）」通道
  - rag.svg       在线问答之「RAG 检索增强」通道
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


W1, H1 = 1880, 1330
W2, H2 = 1800, 1150
W3, H3 = 1880, 1210
W4, H4 = 1920, 1600


def fig_overview():
    g, a, n = [], [], []
    g.append(title(W1 / 2.0, "edu_rag 智能问答项目 · 总体流程图",
                   "离线准备（一次性） → 在线问答（每次请求） → 效果评估（迭代优化）"))

    # ---------------- A 离线准备 ----------------
    g.append(group(40, 120, 1800, 268, "A · 离线准备：知识库构建（文档 / 题库更新时重跑）"))
    yA1 = 196
    n1 = (70, yA1, 235, 82, "原始教学文档\nPDF · Word · PPT\n图片 · MD · TXT", "io", 14)
    n2 = (335, yA1, 235, 82, "文档解析 + OCR\nedu_document_loaders", "proc")
    n3 = (600, yA1, 235, 82, "父子分层切分\n父块 512/128 · 子块 120/30", "proc")
    n4 = (865, yA1, 235, 82, "BGE-M3 向量化\n稠密 1024 维 + 稀疏向量", "proc")
    n5 = (1130, yA1, 245, 82, "写入 Milvus\n集合 edurag", "store")
    n6 = (1405, yA1, 405, 82, "写入字段：id(md5) · text · dense_vector\nsparse_vector · parent_id · parent_content\nsource · timestamp", "note", 14)
    for t in (n1, n2, n3, n4, n5, n6):
        n.append(node(*t))
    for i in range(4):
        a.append(arrow([(n1[0] + i * 265 + n1[2], yA1 + 41), (n1[0] + (i + 1) * 265, yA1 + 41)]))
    a.append(arrow([(1375, yA1 + 41), (1405, yA1 + 41)], color="#94a3b8", dash="6 5", marker="#94a3b8"))

    yA2 = 300
    n7 = (70, yA2, 235, 76, "快问快答题库\nJP学科知识问答.csv", "io")
    n8 = (335, yA2, 235, 76, "写入 MySQL 表 jpkb\n学科 / 问题 / 答案", "store")
    n9 = (600, yA2, 235, 76, "同步 Redis 问题列表\n原文 + Jieba 分词", "store")
    n10 = (865, yA2, 235, 76, "BM25Okapi 初始化\nrank_bm25", "proc")
    n11 = (1130, yA2, 245, 76, "BM25 检索器就绪\n(阈值 0.85)", "proc")
    n12 = (1405, yA2, 405, 76, "说明：BM25 检索器首次启动时自动同步\n缓存 key：answer:{原始问题}", "note")
    for t in (n7, n8, n9, n10, n11, n12):
        n.append(node(*t))
    for i in range(4):
        a.append(arrow([(n7[0] + i * 265 + n7[2], yA2 + 38), (n7[0] + (i + 1) * 265, yA2 + 38)]))
    a.append(arrow([(1375, yA2 + 38), (1405, yA2 + 38)], color="#94a3b8", dash="6 5", marker="#94a3b8"))

    # ---------------- B 在线问答 ----------------
    g.append(group(40, 420, 1800, 452, "B · 在线问答：一次请求的完整链路（实时，FastAPI + WebSocket 流式）"))
    yB1 = 500
    b1 = (70, yB1, 200, 80, "用户提问\n(前端页面)", "term")
    b2 = (300, yB1, 320, 80, "FastAPI 统一入口\nPOST /api/query · WS /api/stream", "proc")
    b3 = (650, yB1, 220, 80, "日常问候语？\n(正则匹配)", "dec", 15)
    b4 = (1070, yB1, 300, 80, "模板回复\n（黑马程序员人设）", "proc")
    b5 = (1400, yB1, 260, 80, "直接返回答案\n(不触发检索)", "term")
    for t in (b1, b2, b3, b4, b5):
        n.append(node(*t))
    a.append(arrow([(270, yB1 + 40), (300, yB1 + 40)]))
    a.append(arrow([(620, yB1 + 40), (650, yB1 + 40)]))
    a.append(arrow([(870, yB1 + 40), (1070, yB1 + 40)]))
    a.append(label(970, yB1 + 22, "是", fill="#16a34a"))
    a.append(arrow([(1330, yB1 + 40), (1400, yB1 + 40)]))

    yB2 = 640
    c1 = (70, yB2, 260, 84, "读取会话历史\nMySQL conversations（最近 5 轮）", "store")
    c2 = (380, yB2, 300, 84, "① 快问快答通道\nRedis 缓存 → Jieba → BM25 → 阈值 0.85", "sub", 15)
    c3 = (730, yB2, 240, 84, "快问快答命中？", "dec", 15)
    c4 = (1050, yB2, 280, 84, "返回标准答案\n并写入会话历史", "term")
    c5 = (1380, yB2, 330, 84, "详见 图 3：快问快答通道详细流程\n(MySQL + Redis + BM25)", "note", 14)
    for t in (c1, c2, c3, c4, c5):
        n.append(node(*t))
    a.append(label(733, 570, "否", fill="#dc2626"))
    a.append(arrow([(758, yB1 + 80), (758, 610), (200, 610), (200, yB2)]))
    a.append(arrow([(330, yB2 + 42), (380, yB2 + 42)]))
    a.append(arrow([(680, yB2 + 42), (730, yB2 + 42)]))
    a.append(arrow([(970, yB2 + 42), (1050, yB2 + 42)]))
    a.append(label(1010, yB2 + 24, "是", fill="#16a34a"))
    a.append(arrow([(1330, yB2 + 42), (1380, yB2 + 42)], color="#94a3b8", dash="6 5", marker="#94a3b8"))

    yB3 = 780
    d1 = (380, yB3, 320, 84, "② RAG 检索增强通道（详见 图 4）\n意图识别 → 策略选择\n混合检索 → 精排 → 取 Top-2", "sub", 14)
    d2 = (750, yB3, 280, 84, "拼接 Prompt\n上下文 + 历史 + 问题（超 4096 截断）", "proc", 15)
    d3 = (1080, yB3, 290, 84, "大模型 DashScope\nqwen3.8-27b（流式生成）", "ext", 15)
    d4 = (1420, yB3, 290, 84, "流式返回答案\n并写入会话历史", "term", 15)
    for t in (d1, d2, d3, d4):
        n.append(node(*t))
    a.append(label(850, 750, "否", fill="#dc2626"))
    a.append(arrow([(850, yB2 + 84), (850, 750), (520, 750), (520, yB3)]))
    a.append(arrow([(700, yB3 + 42), (750, yB3 + 42)]))
    a.append(arrow([(1030, yB3 + 42), (1080, yB3 + 42)]))
    a.append(arrow([(1370, yB3 + 42), (1420, yB3 + 42)]))

    # ---------------- C 评估 ----------------
    g.append(group(40, 908, 1800, 258, "C · 效果评估与迭代（RAGAS 指标闭环）"))
    yC1 = 976
    e1 = (70, yC1, 260, 84, "评测数据集\nrag_evaluate_data.json", "io")
    e2 = (370, yC1, 400, 84, "RAGAS 评估\n忠实度 · 答案相关性 · 上下文相关性 · 上下文召回率", "proc", 15)
    e3 = (810, yC1, 280, 84, "生成评测报告\nragas_evaluation_results.csv", "store", 15)
    e4 = (1130, yC1, 330, 84, "定位问题并调参\n切分粒度 / k / Top-M / 策略 / 重排模型", "proc", 15)
    e5 = (1500, yC1, 310, 84, "回归评估\n确认指标不下降", "term", 15)
    for t in (e1, e2, e3, e4, e5):
        n.append(node(*t))
    a.append(arrow([(330, yC1 + 42), (370, yC1 + 42)]))
    a.append(arrow([(770, yC1 + 42), (810, yC1 + 42)]))
    a.append(arrow([(1090, yC1 + 42), (1130, yC1 + 42)]))
    a.append(arrow([(1460, yC1 + 42), (1500, yC1 + 42)]))
    # 参数调优闭环
    a.append(arrow([(1655, yC1), (1655, 942), (1295, 942), (1295, yC1)],
                   color="#2563eb", dash="8 6", marker="#2563eb"))
    a.append(label(1475, 926, "指标下降 → 回到调参（参数写回 config.ini / .env）",
                   size=14, weight="normal", fill="#2563eb"))

    yC2 = 1080
    f1 = (70, yC2, 640, 66, "运行日志 logs/app.log：每次查询的耗时与分支记录（base/logger.py）", "note", 14)
    f2 = (750, yC2, 640, 66, "两套评测实现：DashScope 云端 ragas_evaluate.py · 本地 Ollama ollama_ragas.py", "note", 14)
    f3 = (1430, yC2, 380, 66, "评测集也可用 Ragas 自动生成", "note", 14)
    for t in (f1, f2, f3):
        n.append(node(*t))

    g.append(legend(60, 1196, [("term", "起止", False), ("proc", "处理", False),
                               ("dec", "判断", False), ("store", "数据/存储", False),
                               ("io", "输入/输出", False), ("sub", "子流程", True),
                               ("ext", "外部服务", False), ("note", "说明", False)]))
    return svg(W1, H1, g + a + n)


def fig_indexing():
    g, a, n = [], [], []
    g.append(title(W2 / 2.0, "离线知识库构建详细流程",
                   "两条并行链路：① RAG 向量知识库（Milvus） ② 快问快答库（MySQL + Redis + BM25）"))
    g.append(group(40, 120, 1720, 452, "① RAG 向量知识库构建：文档 → 切分 → 向量化 → Milvus"))

    y1 = 190
    t1 = (70, y1, 300, 86, "按学科分目录的原始文档\nai_data / java_data / ops_data ...", "io", 15)
    t2 = (400, y1, 320, 86, "按扩展名分派加载器\ndocument_loaders 映射表", "proc", 15)
    t3 = (750, y1, 380, 86, "解析 + OCR\nRapidOCR · PyMuPDF · python-docx / pptx", "proc", 15)
    t4 = (1160, y1, 260, 86, "写入元数据\nsource · file_path · timestamp", "proc", 15)
    t5 = (1450, y1, 270, 86, "得到 Document 列表\nload_documents_from_directory()", "proc", 14)
    for t in (t1, t2, t3, t4, t5):
        n.append(node(*t))
    a.append(arrow([(370, y1 + 43), (400, y1 + 43)]))
    a.append(arrow([(720, y1 + 43), (750, y1 + 43)]))
    a.append(arrow([(1130, y1 + 43), (1160, y1 + 43)]))
    a.append(arrow([(1420, y1 + 43), (1450, y1 + 43)]))
    a.append(label(1500, y1 - 22, "不支持的类型直接跳过", size=13, weight="normal", fill="#64748b"))

    y2 = 336
    u1 = (70, y2, 300, 86, "父块切分\nchunk_size=512 · overlap=128", "proc", 15)
    u2 = (400, y2, 320, 86, "子块切分\nchunk_size=120 · overlap=30", "proc", 15)
    u3 = (750, y2, 380, 86, "写入父子关系元数据\nparent_id · parent_content · id", "proc", 15)
    u4 = (1160, y2, 260, 86, "BGE-M3 向量化\n稠密向量 1024 维 + 稀疏向量", "proc", 15)
    u5 = (1450, y2, 270, 86, "upsert 写入 Milvus\n集合 edurag", "store", 15)
    for t in (u1, u2, u3, u4, u5):
        n.append(node(*t))
    a.append(arrow([(1585, y1 + 86), (1585, 310), (220, 310), (220, y2)]))
    a.append(arrow([(370, y2 + 43), (400, y2 + 43)]))
    a.append(arrow([(720, y2 + 43), (750, y2 + 43)]))
    a.append(arrow([(1130, y2 + 43), (1160, y2 + 43)]))
    a.append(arrow([(1420, y2 + 43), (1450, y2 + 43)]))

    y3 = 452
    v1 = (70, y3, 640, 96, "切分器：ChineseRecursiveTextSplitter（中文递归切分）\nMarkdown 文件改用 MarkdownTextSplitter", "note", 14)
    v2 = (750, y3, 640, 96, "Milvus 字段与索引：id(md5) · text · dense_vector · sparse_vector · source\n稠密 IVF_FLAT(IP, nlist=128) / 稀疏 SPARSE_INVERTED_INDEX(IP)", "note", 14)
    v3 = (1430, y3, 290, 96, "知识库就绪\n可被在线检索", "term", 15)
    for t in (v1, v2, v3):
        n.append(node(*t))
    a.append(arrow([(1585, y2 + 86), (1585, y3)]))
    a.append(arrow([(390, y2 + 86), (390, y3)], color="#94a3b8", dash="6 5", marker="#94a3b8"))
    a.append(arrow([(1070, y2 + 86), (1070, y3)], color="#94a3b8", dash="6 5", marker="#94a3b8"))

    g.append(group(40, 600, 1720, 420, "② 快问快答库构建：CSV → MySQL → Redis → BM25"))
    w1 = 690
    x1 = (70, w1, 300, 86, "JP学科知识问答.csv\n学科名称 · 问题 · 答案", "io", 15)
    x2 = (400, w1, 300, 86, "建表并插入数据\nMySQL 表 jpkb", "store", 15)
    x3 = (750, w1, 320, 86, "Redis 缓存问题原文列表\nqa_original_questions", "store", 14)
    x4 = (1120, w1, 320, 86, "Redis 缓存分词列表\nqa_tokenized_questions", "store", 14)
    x5 = (1490, w1, 230, 86, "BM25Okapi 初始化\nrank_bm25", "proc", 15)
    for t in (x1, x2, x3, x4, x5):
        n.append(node(*t))
    a.append(arrow([(370, w1 + 43), (400, w1 + 43)]))
    a.append(arrow([(700, w1 + 43), (750, w1 + 43)]))
    a.append(arrow([(1070, w1 + 43), (1120, w1 + 43)]))
    a.append(arrow([(1440, w1 + 43), (1490, w1 + 43)]))

    w2 = 862
    z1 = (70, w2, 380, 86, "在线检索：Jieba 分词 → BM25 打分\nSoftmax 归一化 → 取最高分", "proc", 15)
    z2 = (500, w2, 380, 86, "命中：MySQL 取答案\n并写 Redis 缓存 answer:{query}", "proc", 15)
    z3 = (930, w2, 360, 86, "未命中：need_rag = True\n交给 RAG 通道继续处理", "proc", 15)
    z4 = (1340, w2, 380, 86, "题库更新后需重新同步问题列表\n（清空 Redis 对应 key 或重跑建库）", "note", 14)
    for t in (z1, z2, z3, z4):
        n.append(node(*t))
    a.append(arrow([(1605, w1 + 86), (1605, 820), (260, 820), (260, w2)]))
    a.append(arrow([(450, w2 + 43), (500, w2 + 43)]))
    a.append(arrow([(880, w2 + 43), (930, w2 + 43)]))
    a.append(arrow([(1290, w2 + 43), (1340, w2 + 43)], color="#94a3b8", dash="6 5", marker="#94a3b8"))

    g.append(legend(60, 1060, [("term", "起止", False), ("proc", "处理", False),
                               ("store", "数据/存储", False), ("io", "输入/输出", False),
                               ("note", "说明", False)]))
    return svg(W2, H2, g + a + n)


def fig_faq():
    g, a, n = [], [], []
    g.append(title(W3 / 2.0, "在线问答 · 通道一：快问快答（MySQL + Redis + BM25）",
                   "低成本、低延迟：先用缓存与 BM25 精确匹配，命中即直接返回；未命中再回退 RAG"))

    cx = [80, 530, 980, 1430]
    bw = 370
    dec_h, box_h = 130, 96

    def row(cy):
        return cy - box_h / 2.0, cy - dec_h / 2.0

    by, dy = row(250)
    p1 = (cx[0], by, bw, box_h, "用户提问\n(前端页面)", "term")
    p2 = (cx[1], by, bw, box_h, "FastAPI 统一入口\nPOST /api/query · WS /api/stream", "proc", 15)
    p3 = (cx[2], dy, bw, dec_h, "命中日常问候语？\n（正则匹配 4 类模板）", "dec", 15)
    p4 = (cx[3], by, bw, box_h, "直接返回模板回复\n（黑马程序员人设）", "term", 15)
    for t in (p1, p2, p3, p4):
        n.append(node(*t))
    a.append(arrow([(cx[0] + bw, by + box_h / 2), (cx[1], by + box_h / 2)]))
    a.append(arrow([(cx[1] + bw, by + box_h / 2), (cx[2], by + box_h / 2)]))
    a.append(arrow([(cx[2] + bw, 250), (cx[3], 250)]))
    a.append(label((cx[2] + bw + cx[3]) / 2, 228, "是", fill="#16a34a"))

    by2, dy2 = row(470)
    q1 = (cx[0], by2, bw, box_h, "读取会话历史\nMySQL conversations 表（最近 5 轮）", "store", 15)
    q2 = (cx[1], dy2, bw, dec_h, "Redis 缓存命中？\nkey = answer:{原始问题}", "dec", 15)
    q3 = (cx[2], by2, bw, box_h, "直接返回缓存答案\n（无需检索）", "term", 15)
    for t in (q1, q2, q3):
        n.append(node(*t))
    a.append(arrow([(cx[2], dy + dec_h), (cx[2], 350), (cx[0] + bw / 2, 350), (cx[0] + bw / 2, by2)]))
    a.append(label(cx[2] - 40, 330, "否", fill="#dc2626"))
    a.append(arrow([(cx[0] + bw, 470), (cx[1], 470)]))
    a.append(arrow([(cx[1] + bw, 470), (cx[2], 470)]))
    a.append(label((cx[1] + bw + cx[2]) / 2, 448, "是", fill="#16a34a"))

    by3, dy3 = row(690)
    r1 = (cx[0], by3, bw, box_h, "Jieba 分词\n文本统一转小写 preprocess_text()", "proc", 15)
    r2 = (cx[1], by3, bw, box_h, "BM25Okapi 打分 → Softmax 归一化\n→ 取最高分 best_score", "proc", 15)
    r3 = (cx[2], dy3, bw, dec_h, "best_score > 阈值 0.85 ？", "dec", 16)
    r4 = (cx[3], by3, bw, box_h, "MySQL 表 jpkb 按原问题取答案\n并写 Redis 缓存", "proc", 15)
    for t in (r1, r2, r3, r4):
        n.append(node(*t))
    a.append(arrow([(cx[1] + bw / 2, dy2 + dec_h), (cx[1] + bw / 2, 610), (cx[0] + bw / 2, 610), (cx[0] + bw / 2, by3)]))
    a.append(label(cx[1] + bw / 2 + 34, 592, "否", fill="#dc2626"))
    a.append(arrow([(cx[0] + bw, 690), (cx[1], 690)]))
    a.append(arrow([(cx[1] + bw, 690), (cx[2], 690)]))
    a.append(arrow([(cx[2] + bw, 690), (cx[3], 690)]))
    a.append(label((cx[2] + bw + cx[3]) / 2, 668, "是", fill="#16a34a"))

    by4, dy4 = row(910)
    s1 = (cx[0], by4, bw, box_h, "阈值 0.85 可配置\nsearch(query, threshold=0.85)", "note", 14)
    s2 = (cx[1], by4, bw, box_h, "命中判定用的是 Softmax 归一化后的分数\n不是原始 BM25 分值", "note", 14)
    s3 = (cx[2], by4, bw, box_h, "回退 RAG 通道\n（详见 图 4）", "sub", 15)
    s4 = (cx[3], by4, bw, box_h, "返回标准答案\n并写入会话历史", "term", 15)
    for t in (s1, s2, s3, s4):
        n.append(node(*t))
    a.append(arrow([(cx[2] + bw / 2, dy3 + dec_h), (cx[2] + bw / 2, by4)]))
    a.append(label(cx[2] + bw / 2 + 34, 828, "否", fill="#dc2626"))
    a.append(arrow([(cx[3] + bw / 2, by3 + box_h), (cx[3] + bw / 2, by4)]))
    a.append(label(cx[3] + bw / 2 + 30, 850, "是", fill="#16a34a"))

    g.append(legend(60, 1040, [("term", "起止", False), ("proc", "处理", False),
                               ("dec", "判断", False), ("store", "数据/存储", False)]))
    g.append(legend(60, 1100, [("sub", "子流程", True), ("note", "说明", False)]))
    return svg(W3, H3, g + a + n)


def fig_rag():
    g, a, n = [], [], []
    g.append(title(W4 / 2.0, "在线问答 · 通道二：RAG 检索增强",
                   "意图识别 → 检索策略选择 → 混合检索 → 精排 → 提示词 → 大模型生成（流式）"))

    cx = [80, 520, 960, 1400]
    bw = 400

    y1 = 200
    t1 = (cx[0], y1, bw, 96, "输入：query + source_filter(学科) + 会话历史", "io", 15)
    t2 = (cx[1], y1, bw, 96, "BERT 意图识别\nrag_qa/models/bert_query_classifier1", "proc", 15)
    t3 = (cx[2], y1 - 20, 360, 136, "查询类别？\n(通用知识 / 专业咨询)", "dec", 16)
    t4 = (cx[3], y1, bw, 96, "通用知识：直接调用大模型\n不检索知识库", "proc", 15)
    for t in (t1, t2, t3, t4):
        n.append(node(*t))
    a.append(arrow([(cx[0] + bw, y1 + 48), (cx[1], y1 + 48)]))
    a.append(arrow([(cx[1] + bw, y1 + 48), (cx[2], y1 + 48)]))
    a.append(arrow([(cx[2] + 360, y1 + 48), (cx[3], y1 + 48)]))
    a.append(label((cx[2] + 360 + cx[3]) / 2, y1 + 26, "通用知识", fill="#16a34a"))

    y2 = 420
    u1 = (cx[1], y2, bw, 96, "LLM 策略选择器 StrategySelector\n从 4 种策略中选 1 种（temperature=0）", "proc", 15)
    n.append(node(*u1))
    a.append(arrow([(cx[2] + 180, y1 + 116), (cx[2] + 180, 380), (cx[1] + bw / 2, 380), (cx[1] + bw / 2, y2)]))
    a.append(label(cx[2] + 220, 362, "专业咨询", fill="#dc2626"))

    y3 = 640
    v1 = (cx[0], y3, bw, 130, "① 直接检索\n原始 query 直接检索\n（意图明确、要特定信息）", "proc", 15)
    v2 = (cx[1], y3, bw, 130, "② 假设问题检索 HyDE\nLLM 先生成假设答案，\n再用假设答案去检索", "proc", 15)
    v3 = (cx[2], y3, bw, 130, "③ 子查询检索\n拆成多个子问题分别检索，\n合并后去重", "proc", 15)
    v4 = (cx[3], y3, bw, 130, "④ 回溯问题检索\n把复杂问题简化成\n更基础的问题再检索", "proc", 15)
    for t in (v1, v2, v3, v4):
        n.append(node(*t))
    for i in range(4):
        a.append(arrow([(cx[1] + bw / 2, y2 + 96), (cx[1] + bw / 2, 560),
                        (cx[i] + bw / 2, 560), (cx[i] + bw / 2, y3)]))

    y4 = 880
    w1 = (cx[0], y4, 820, 130, "BGE-M3 混合检索（粗排）\n稠密向量 + 稀疏向量并行检索，WeightedRanker(1.0, 0.7)\n过滤条件：source == 学科 · 返回 k = 5 个子块", "proc", 15)
    w2 = (cx[2], y4, bw, 130, "按 parent_content 父块去重\n_get_unique_parent_docs()", "proc", 15)
    w3 = (cx[3], y4 - 18, bw, 166, "父块数量 ≥ 2 ？", "dec", 16)
    for t in (w1, w2, w3):
        n.append(node(*t))
    for i in range(4):
        a.append(arrow([(cx[i] + bw / 2, y3 + 130), (cx[i] + bw / 2, 840),
                        (cx[0] + 410, 840), (cx[0] + 410, y4)]))
    a.append(arrow([(cx[0] + 820, y4 + 65), (cx[2], y4 + 65)]))
    a.append(arrow([(cx[2] + bw, y4 + 65), (cx[3], y4 + 65)]))

    y5 = 1120
    z1 = (cx[3], y5, bw, 96, "直接返回父块\n（不足 2 个，无需重排）", "proc", 15)
    z2 = (cx[2], y5, bw, 96, "BGE-Reranker-large 精排\n(CrossEncoder, CPU) 打分降序", "proc", 15)
    z3 = (cx[1], y5, bw, 96, "取 Top-2 作为上下文\nCANDIDATE_M = 2", "proc", 15)
    for t in (z1, z2, z3):
        n.append(node(*t))
    a.append(arrow([(cx[3] + bw / 2, y4 + 148), (cx[3] + bw / 2, y5)]))
    a.append(label(cx[3] + bw / 2 + 46, y4 + 178, "否", fill="#dc2626"))
    a.append(arrow([(cx[3], y4 + 65), (cx[2] + bw + 0, y4 + 65)], color="#16a34a", marker="#16a34a"))
    a.append(label(cx[3] - 24, y4 + 36, "是", fill="#16a34a"))
    a.append(arrow([(cx[2] + bw / 2, y4 + 130), (cx[2] + bw / 2, y5)]))
    a.append(arrow([(cx[2], y5 + 48), (cx[1] + bw, y5 + 48)]))
    # 父块不足 2 个时，同样进入提示词拼接
    a.append(line([(cx[3] + bw / 2, y5 + 96), (cx[3] + bw / 2, 1250), (cx[1] + bw / 2, 1250)]))

    y6 = 1290
    k1 = (cx[0], y6, 560, 96, "拼接 Prompt：上下文 + 对话历史 + 问题\n超 4096 字符截断", "proc", 15)
    k2 = (cx[1] + 180, y6, 440, 96, "大模型流式生成\nDashScope qwen3.8-27b（yield token）", "ext", 15)
    k3 = (cx[2] + 220, y6, 400, 96, "写入 MySQL 会话历史\n保留最近 5 轮", "store", 15)
    k4 = (cx[3] + 120, y6, 320, 96, "结束：返回答案\n（流式 token / 一次性）", "term", 15)
    for t in (k1, k2, k3, k4):
        n.append(node(*t))
    a.append(arrow([(cx[1] + bw / 2, y5 + 96), (cx[1] + bw / 2, 1250), (cx[0] + 280, 1250), (cx[0] + 280, y6)]))
    a.append(arrow([(cx[0] + 560, y6 + 48), (cx[1] + 180, y6 + 48)]))
    a.append(arrow([(cx[1] + 620, y6 + 48), (cx[2] + 220, y6 + 48)]))
    a.append(arrow([(cx[2] + 620, y6 + 48), (cx[3] + 120, y6 + 48)]))
    # 通用知识分支直连生成
    a.append(arrow([(cx[3] + bw, y1 + 48), (1850, y1 + 48), (1850, 1338), (cx[3] + 440, 1338)],
                   color="#16a34a", marker="#16a34a"))

    y7 = 1430
    m1 = (cx[0], y7, 560, 80, "提示词模板：rag_qa/core/prompts.py\n信息不足时回复「请联系人工客服」", "note", 14)
    m2 = (cx[1] + 180, y7, 440, 80, "策略提示词：strategy_selector.py\nHyDE / 子查询 / 回溯提示词在 prompts.py", "note", 14)
    m3 = (cx[2] + 220, y7, 400, 80, "检索参数来自 config.ini\nparent/child 切分、k、M", "note", 14)
    for t in (m1, m2, m3):
        n.append(node(*t))

    g.append(legend(60, 1548, [("term", "起止", False), ("proc", "处理", False),
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
    for name, builder, w, h in FIGURES:
        body = builder()
        with open(os.path.join(FIG, name + ".svg"), "w", encoding="utf-8") as f:
            f.write(body)
        html = ('<!doctype html><html><head><meta charset="utf-8">'
                '<style>html,body{margin:0;padding:0;background:#fff}</style></head>'
                '<body>%s</body></html>' % body)
        with open(os.path.join(tmp, name + ".html"), "w", encoding="utf-8") as f:
            f.write(html)
        print("written:", name, w, h)


if __name__ == "__main__":
    main()
