# -*- coding: utf-8 -*-
"""
用法律二分类数据训练查询分类器（《项目方案书》§4 第 7 行 / §6.2）
数据: rag_qa/classify_data/model_law_5000.json  (2500 通用知识 + 2500 法律咨询)
产物: rag_qa/models/bert_query_classifier_law
用法: python tools/train_law_classifier.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from base.config import config                              # noqa: E402
from rag_qa.core.query_classifier import QueryClassifier     # noqa: E402


def main():
    model_path = os.path.join(config.MODELS_DIR, "bert_query_classifier_law")
    data_file = os.path.join(ROOT, "rag_qa", "classify_data", "model_law_5000.json")
    print(f"训练数据: {data_file}")
    print(f"保存到  : {model_path}")
    classifier = QueryClassifier(model_path=model_path)
    classifier.train_model(data_file=data_file)
    print(f"完成，模型保存到 {model_path}")


if __name__ == "__main__":
    main()
