from rag_qa.core.prompts import RAGPrompts
#   导入 time 模块，用于计算时间   带记忆 + 流式的版本
import time
from base.config import config
from base.domains import faq_domain_for, get_domain, law_names_for
from base.logger import logger
from rag_qa.core.query_classifier import QueryClassifier  # 导入查询分类器
from rag_qa.core.strategy_selector import StrategySelector  # 导入策略选择器
from rag_qa.core.citation_check import validate_citations  # 引用可验证（小改动 C）

# P2：第三召回源（law_faq 法律问答）的两个参数
#   召回候选数：先粗取几条，用精排模型排序
#   注入条数：最终最多补充几条问答进上下文（法条那条链路仍然是 CANDIDATE_M 条，不受影响）
FAQ_RETRIEVAL_K = 3
FAQ_INJECT_N = 2

#   定义 RAGSystem 类，封装 RAG 系统的核心逻辑
class RAGSystem:
    #   初始化方法，设置 RAG 系统的基本参数
    def __init__(self, vector_store, llm):
        #   设置向量数据库对象
        self.vector_store = vector_store
        #   设置大语言模型调用函数
        self.llm = llm
        #   获取 RAG 提示模板
        self.rag_prompt = RAGPrompts.rag_prompt()
        #   通用知识提示模板（与法条无关）
        self.general_prompt = RAGPrompts.general_prompt()
        #   初始化查询分类器
        self.query_classifier = QueryClassifier(
            model_path=f'{config.MODELS_DIR}/bert_query_classifier_law')
        #   初始化策略选择器
        self.strategy_selector = StrategySelector()
        # 最大 4096 字符
        self.max_prompt_length = 4096

    #   定义私有方法，使用假设文档进行检索（HyDE）
    def _retrieve_with_hyde(self, query, source_filter=None, law_names=None):
        logger.info(f"_retrieve_with_hyde 使用 HyDE 策略进行检索 (查询: '{query}', 过滤条件: {source_filter}, 领域法数: {len(law_names) if law_names else 0})")
        #   获取假设问题生成的 Prompt 模板
        hyde_prompt_template = RAGPrompts.hyde_prompt()  # 使用 template 后缀区分
        #   调用大语言模型生成假设答案
        try:
            hypo_answer = self.llm(hyde_prompt_template.format(query=query))
            hypo_answer = "".join(hypo_answer)
            logger.info(f"HyDE 生成的假设答案: '{hypo_answer}'")
            #   使用假设答案进行检索，并返回检索结果
            #   注意：HyDE 通常只用于生成检索向量，不一定需要 rerank 这一步，但这里复用了
            return self.vector_store.hybrid_search_with_rerank(
                hypo_answer, k=config.RETRIEVAL_K,  # 使用 K 而非 M
                source_filter=source_filter, law_names=law_names
            )
        except Exception as e:
            logger.error(f"HyDE 策略执行失败: {e}")
            return []

    #   定义私有方法，使用子查询进行检索
    def _retrieve_with_subqueries(self, query, source_filter=None, law_names=None):
        logger.info(f"_retrieve_with_subqueries 使用子查询策略进行检索 (查询: '{query}', 过滤条件: {source_filter}, 领域法数: {len(law_names) if law_names else 0})")
        #   获取子查询生成的 Prompt 模板
        subquery_prompt_template = RAGPrompts.subquery_prompt()  # 使用 template 后缀区分
        try:
            #   调用大语言模型生成子查询列表
            subqueries_text = self.llm(subquery_prompt_template.format(query=query))
            subqueries_text = "".join(subqueries_text)
            subqueries = [q.strip() for q in subqueries_text.split("\n") if q.strip()]
            logger.info(f"生成的子查询: {subqueries}")
            if not subqueries:
                logger.warning("未能生成有效的子查询")
                return []

            #   初始化空列表，用于存储所有子查询的检索结果
            all_docs = []
            #   遍历每个子查询
            for sub_q in subqueries:
                #   使用子查询进行检索，并将结果添加到列表中
                #   这里对每个子查询都执行了 hybrid search + rerank，开销可能较大
                docs = self.vector_store.hybrid_search_with_rerank(
                    sub_q, k=config.RETRIEVAL_K,  # 使用 K
                    source_filter=source_filter, law_names=law_names
                )
                all_docs.extend(docs)
                logger.info(f"子查询 '{sub_q}' 检索到 {len(docs)} 个文档")

            #   对所有检索结果进行去重 (基于对象内存地址，如果 Document 内容相同但对象不同则无法去重)
            #   更可靠的去重方式是基于文档内容或 ID
            unique_docs_dict = {doc.page_content: doc for doc in all_docs}  # 基于内容去重
            unique_docs = list(unique_docs_dict.values())

            logger.info(f"所有子查询共检索到 {len(all_docs)} 个文档, 去重后剩 {len(unique_docs)} 个")
            #   返回去重后的文档，限制数量 (是否需要在此处限制? retrieve_and_merge 末尾会限制)
            # return unique_docs[: configig.CANDIDATE_M]
            return unique_docs  # 返回所有唯一文档，让 retrieve_and_merge 处理数量

        except Exception as e:
            logger.error(f"子查询策略执行失败: {e}")
            return []

    #   定义私有方法，使用回溯问题进行检索
    def _retrieve_with_backtracking(self, query, source_filter=None, law_names=None):
        logger.info(f"_retrieve_with_backtracking 使用回溯问题策略进行检索 (查询: '{query}', 过滤条件: {source_filter}, 领域法数: {len(law_names) if law_names else 0})")
        #   获取回溯问题生成的 Prompt 模板
        backtrack_prompt_template = RAGPrompts.backtracking_prompt()  # 使用 template 后缀区分
        try:
            #   调用大语言模型生成回溯问题
            simplified_query = self.llm(backtrack_prompt_template.format(query=query))
            simplified_query = "".join(simplified_query)
            logger.info(f"生成的回溯问题: '{simplified_query}'")
            #   使用回溯问题进行检索，并返回检索结果
            return self.vector_store.hybrid_search_with_rerank(
                simplified_query, k=config.RETRIEVAL_K, source_filter=source_filter,
                law_names=law_names
            )
        except Exception as e:
            logger.error(f"回溯问题策略执行失败: {e}")
            return []

    #   定义方法，检索并合并相关文档
    def retrieve_and_merge(self, query, source_filter=None, strategy=None,
                           domain=None, domain_mode="off", allow_faq=True):
        """检索并拼接上下文。

        domain / domain_mode 是领域视图：
          off    : 不过滤（默认，行为与改造前一致）
          soft   : 先按领域法名过滤；父块不足 CANDIDATE_M 时放开过滤重试
          strict : 只按领域法名过滤，不兜底（合规取证用）
        """
        use_domain = bool(domain) and domain_mode in ("soft", "strict")
        law_names = law_names_for(domain) if use_domain else None
        faq_domain = faq_domain_for(domain) if use_domain else None
        if domain and not use_domain:
            logger.info(f"已指定领域 {domain}，但 domain_mode={domain_mode}，本次不启用领域过滤")
        logger.info(f"retrieve_and_merge 开始处理查询: '{query}', 来源过滤: {source_filter}, "
                    f"领域: {domain}, 模式: {domain_mode}, 领域法数: {len(law_names) if law_names else 0}")

        # 如果未指定检索策略，则使用策略选择器选择
        if not strategy:
            strategy = self.strategy_selector.select_strategy(query)

        #   根据检索策略选择不同的检索方式
        ranked_sub_chunks = []  # 初始化
        if strategy == "回溯问题检索":
            ranked_sub_chunks = self._retrieve_with_backtracking(query, source_filter, law_names)
        elif strategy == "子查询检索":
            ranked_sub_chunks = self._retrieve_with_subqueries(query, source_filter, law_names)
            # 注意：子查询返回的是已 rerank 过的父文档或子块列表，后续合并逻辑可能需要调整
        elif strategy == "假设问题检索":
            ranked_sub_chunks = self._retrieve_with_hyde(query, source_filter, law_names)
        else:  # 默认或“直接检索”
            logger.info(f"使用直接检索策略 (查询: '{query}')")
            ranked_sub_chunks = self.vector_store.hybrid_search_with_rerank(
                query, k=config.RETRIEVAL_K, source_filter=source_filter, law_names=law_names
            )  # 注意 hybrid_search_with_rerank 返回的是 rerank 后的父文档

        logger.info(f"策略 '{strategy}' 检索到 {len(ranked_sub_chunks)} 个候选文档 (可能已是父文档)")

        # soft 模式兜底：领域过滤命中不足时放开过滤，避免“聚焦”变成“漏答”。
        domain_fallback = False
        if law_names and domain_mode == "soft" and len(ranked_sub_chunks) < config.CANDIDATE_M:
            logger.info("领域过滤命中不足，soft 模式放开过滤重试")
            extra = self.vector_store.hybrid_search_with_rerank(
                query, k=config.RETRIEVAL_K, source_filter=source_filter, law_names=None
            )
            seen = {d.page_content for d in ranked_sub_chunks}
            for d in extra:
                if d.page_content not in seen:
                    d.metadata["domain_fallback"] = True
                    ranked_sub_chunks.append(d)
                    seen.add(d.page_content)
            domain_fallback = True
            logger.info(f"兜底后候选文档 {len(ranked_sub_chunks)} 个")

        final_context_docs = ranked_sub_chunks[:config.CANDIDATE_M]

        # P2：第三召回源——law_faq 的 2 万条法律问答。
        # 法条检索漏掉的条文，问答对常常能补上：实测条文覆盖率 62.5% → 81.9%。
        # 问答走单独一路召回，用同一个精排模型排序，最多补充 FAQ_INJECT_N 条；
        # 失败只记日志，不影响法条链路。合规场景 allow_faq=False，避免合成 FAQ 当依据。
        if allow_faq:
            try:
                faq_docs = self.vector_store.search_faq(query, k=FAQ_RETRIEVAL_K, domain=faq_domain)
                if faq_docs:
                    scores = self.vector_store.reranker.predict([[query, d.page_content] for d in faq_docs])
                    faq_docs = [d for _, d in sorted(zip(scores, faq_docs), reverse=True)][:FAQ_INJECT_N]
                    logger.info(f"FAQ 召回 {len(faq_docs)} 条问答作为补充上下文")
                    final_context_docs = final_context_docs + faq_docs
            except Exception as e:
                logger.error(f"FAQ 召回失败（不影响法条链路）: {e}")

        # 只有兜底追加的文档标 True；命中的领域内文档保持 False
        for d in final_context_docs:
            d.metadata.setdefault("domain_fallback", False)

        logger.info(f"最终选取 {len(final_context_docs)} 个文档作为上下文")
        return final_context_docs

    # 添加完历史消息后，定义方法，生成答案
    # 修改 rag_system.py 中的 generate_answer 方法
    def generate_answer(self, query, source_filter=None, history=None,
                        domain=None, domain_mode="off", allow_faq=True):
        """生成答案，支持对话历史、领域视图，限制提示长度，并支持流式输出"""
        start_time = time.time()
        logger.info(f"generate_answer 开始处理查询: '{query}', 来源过滤: {source_filter}, "
                    f"领域: {domain}, 模式: {domain_mode}")

        # 验证历史格式
        if history is not None and not isinstance(history, list):
            logger.warning(f"无效的历史格式: {type(history)}，忽略历史")
            history = []
        elif history:
            history = history[-5:]  # 限制最多5轮
            for h in history:
                if not (isinstance(h, dict) and "question" in h and "answer" in h):
                    logger.warning(f"无效的历史条目: {h}，忽略历史")
                    history = []
                    break

        # 构造历史上下文成一个完整的列表
        history_context = ""
        if history:
            history_context = "\n".join(
                [f"Q: {h['question']}\nA: {h['answer']}" for h in history]
            )
            logger.info(f"使用对话历史: {history_context[:100]}...")

        # 判断查询类型，通用知识还是专业知识
        query_category = self.query_classifier.predict_category(query)
        logger.info(f"查询分类结果: {query_category} (查询: '{query}')")

        # 准备上下文
        context = ""
        try:
            if query_category == "通用知识":
                logger.info("查询为通用知识，直接调用 LLM")
                prompt_input = self.general_prompt.format(
                    history=history_context,
                    question=query,
                )
            else:
                logger.info("查询为专业咨询，执行 RAG 流程")
                strategy = self.strategy_selector.select_strategy(query)  # 根据用户的query选择检索策略
                # 根据检索策略，返回检索上下文
                context_docs = self.retrieve_and_merge(
                    query, source_filter=source_filter, strategy=strategy,
                    domain=domain, domain_mode=domain_mode, allow_faq=allow_faq
                )
                context = "\n\n".join([self._format_context_doc(doc) for doc in context_docs]) if context_docs else ""
                logger.info(f"构建上下文完成，包含 {len(context_docs)} 个文档块")

                # 构造提示
                prompt_input = self.rag_prompt.format(
                    context=context,
                    history=history_context,
                    question=query,
                    phone=config.CUSTOMER_SERVICE_PHONE
                )

            # 截断提示（若超长）
            if len(prompt_input) > self.max_prompt_length:
                logger.warning(f"提示长度 {len(prompt_input)} 超过 {self.max_prompt_length}，进行截断")
                # 保留问题和近期历史，截断上下文或早期历史
                prompt_input = prompt_input[:self.max_prompt_length]
                logger.info(f"截断后提示长度: {len(prompt_input)}")

            processing_time = time.time() - start_time
            logger.info(f"检索时间 (耗时: {processing_time:.2f}s, 查询: '{query}')")
            start_time = time.time()

            # 调用 LLM（流式输出）
            try:
                # 使用 generator 获取流式输出，同时累积完整答案用于引用校验
                collected_answer = ""
                for token in self.llm(prompt_input):
                    collected_answer += token
                    yield token  # 直接将每个 token 传递给调用者

                # 引用可验证（小改动 C）：回答里的《X》第Y条必须能在本轮上下文里找到
                missing = validate_citations(collected_answer, context)
                if missing:
                    names = "、".join(f"《{law}》第{num}条" for law, num in missing)
                    logger.warning(f"引用校验未通过: {names}")
                    yield f"\n\n（校验提示：回答中引用的 {names} 未能在本轮检索到的条文中找到依据，请谨慎采信或补充条文号。）"

                # 记录处理时间
                processing_time = time.time() - start_time
                logger.info(f"LLM查询处理完成 (耗时: {processing_time:.2f}s, 查询: '{query}')")
            except Exception as e:
                logger.error(f"调用 LLM 失败: {e}")
                yield f"抱歉，处理您的问题时出错。请联系人工客服：{config.CUSTOMER_SERVICE_PHONE}"

        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            yield f"抱歉，处理您的问题时出错。请联系人工客服：{config.CUSTOMER_SERVICE_PHONE}"

    @staticmethod
    def _format_context_doc(doc):
        """P1：给上下文块加上「命中条文 + 编/章/节 + 时效状态」抬头。
        目的：让模型引用时能带出章节与时效说明，而不是只给一个条文号。
        """
        md = doc.metadata or {}
        # P2：第三召回源（law_faq 问答）的抬头
        if md.get("source") == "faq" and md.get("question"):
            return f"【命中问答】{md['question']}\n{md.get('answer', doc.page_content)}"
        body = md.get("parent_content") or doc.page_content
        if not md.get("law_name"):
            return body
        tags = [f"《{md['law_name']}》{md.get('article_no', '')}".strip()]
        if md.get("chapter"):
            tags.append(md["chapter"])
        if md.get("status"):
            tags.append(md["status"])
        return f"【命中条文】{' · '.join(tags)}\n{body}"


# query 意图识别 使用bert 分类
# 对 改进：
# 1. 用LLM做意图识别，考虑最近几轮历史问答
# 2. 最近几轮的query 拼接([CLS]current_query[SEP] history_query[SEP] ) 使用多轮对话数据训练Bert
