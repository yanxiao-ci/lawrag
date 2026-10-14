"""
    查询分类器   BERT 意图识别
"""
# 导入标准库
import json
import os
# 导入 PyTorch
import torch
# 导入日志
import sys
from base.logger import logger
from base.config import config
# 导入numpy
import numpy as np
# 导入 Transformers 库
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
# 导入train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

"""
意图识别模块，提供如下功能：
1. 数据加载：读取 5000 条 JSON 数据集，包含查询和标签（“通用知识”或“专业咨询”）
2. 模型训练：使用 bert-base-chinese 模型，微调二分类任务，准确率达 90%+
3. 评估优化：直接处理数字标签（0 或 1），生成分类报告和混淆矩阵
4. 预测接口：支持实时分类，集成到法律法规智能问答系统。

为了满足以上功能，需要实现以下需求：
1. 初始化方法：初始化预训练的分词器、 预训练的模型。 如果是在上线阶段，主要是负责加载训练好的模型
2. 数据预处理：将查询文本和预测标签转化为模型的输入数据格式
3. 构建数据集：用于模型的训练，适配模型的训练函数
4. 模型训练：基于处理好的数据集划分出来训练集，对模型进行训练
5. 模型评估：在数据集划分出来的验证集，对模型进行评估
6. 模型预测：加载训练好的模型，完成意图识别任务
"""

class QueryClassifier:

    def __init__(self, model_path='models/bert_query_classifier'):
        # 加载预训练bert模型路径
        self.pre_trained_model_path = f'{config.MODELS_DIR}/bert-base-chinese'
        # 模型训练以后保存的位置
        self.model_path = model_path
        # 加载BERT分词器：将中文文本转换为模型可处理的token序列     vocab.txt
        self.tokenizer = BertTokenizer.from_pretrained(self.pre_trained_model_path)
        # 初始化模型
        self.model = None
        # 确定设备（GPU 或 CPU）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu" )
        # 记录设备信息
        logger.info(f"使用设备: {self.device}")
        # 定义标签映射（法律方向二分类：通用知识 / 法律咨询）
        self.label_map = {"通用知识": 0, "法律咨询": 1}
        # 加载模型
        self.load_model()

    def load_model(self):
        # 检查模型路径是否存在
        if os.path.exists(self.model_path):
            # 存在：加载训练后的模型
            # 加载预训练模型
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            # 将模型移到指定设备
            self.model.to(self.device)
            # 记录加载成功的日志
            logger.info(f"加载模型: {self.model_path}")
        else:
            # 不存在：初始化原始bert模型
            # 初始化新模型，设置类别数为2
            self.model = BertForSequenceClassification.from_pretrained(self.pre_trained_model_path, num_labels=2)
            # 将模型移到指定设备
            self.model.to(self.device)
            # 记录初始化模型的日志
            logger.info("初始化新 BERT 模型")

    def save_model(self):
        """保存模型"""
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        logger.info(f"模型保存至: {self.model_path}")

    def preprocess_data(self, texts, labels):
        # texts： [问题1,问题2,...问题n] -> ["解释一下什么是RESTful API。"]
        # labels：[标签1,标签2,...标签n] -> ["通用知识"]
        """预处理数据为 BERT 输入格式"""
        # 文本转张量
        encodings = self.tokenizer(
            texts,
            truncation=True,   # 超过最大长度时截断
            padding=True,   # 批次内填充到相同长度
            max_length=128, # 最大长度
            return_tensors="pt"  # 返回PyTorch张量格式
        )
        # labels = ["通用知识", "专业咨询"]
        # 标签转为 0 和 1    # {"通用知识": 0, "专业咨询": 1}
        return encodings, [self.label_map[label] for label in labels]

    def create_dataset(self, encodings, labels):
        """创建 PyTorch 数据集"""
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            # 根据索引获取对应的值
            def __getitem__(self, idx):
                # encodings: { 'input_ids': xxx, 'attention_mask': yyy, 'token_type_ids': zzz }
                # xxx: tensor[batch_size(有几句话，几条数据)=485, seq_len(一句话多长)=<128]"   张量
                # val[idx]： 第idx条数据的编码以后的id
                item = {key: val[idx] for key, val in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                # item
                '''
                    {    
                        "labels": 第idx条数据的label, 
                        "attention_mask": 第idx条数据attention_mask, 
                        "input_ids": 第idx条数据input_ids,
                        "token_type_ids": 第idx条数据的token_type_ids
                    }
                '''
                return item

            def __len__(self):
                return len(self.labels)

        return Dataset(encodings, labels)

    def train_model(self, data_file="training_dataset_hybrid_5000.json"):
        """训练 BERT 分类模型"""
        # 加载数据集
        if not os.path.exists(data_file):
            logger.error(f"数据集文件 {data_file} 不存在")
            raise FileNotFoundError(f"数据集文件 {data_file} 不存在")
        # 读取文件，加载成JSON格式
        with open(data_file, "r", encoding="utf-8") as f:
            # value: {"query": "1024乘以768等于多少？", "label": "通用知识"}
            data = [json.loads(value) for value in f.readlines()]
        # 获取文本和标签
        # texts = [问题1,问题2,...问题n]
        texts = [item["query"] for item in data]
        # 标签 = ["通用知识", "专业咨询", ...]
        labels = [item["label"] for item in data]

        # 数据划分  80%:训练集 20%:验证集
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=5
        )

        # 预处理
        train_encodings, train_labels = self.preprocess_data(train_texts, train_labels)
        val_encodings, val_labels = self.preprocess_data(val_texts, val_labels)
        # print(f'train_encodings-->{train_encodings}')
        # 创建数据集
        train_dataset = self.create_dataset(train_encodings, train_labels)
        # print(f'train_dataset--》{train_dataset[0]}')
        val_dataset = self.create_dataset(val_encodings, val_labels)
        #
        # 设置训练参数
        training_args = TrainingArguments(
            # 为了防止模型训练中途崩溃，设置模型和检查点保存的目录路径
            output_dir="./bert_results",
            # 设置训练的总轮数为3轮
            num_train_epochs=3,
            # 设置每个设备（GPU/CPU）上的训练批次大小为8
            per_device_train_batch_size=8,
            # 设置每个设备（GPU/CPU）上的评估批次大小为8
            per_device_eval_batch_size=8,
            # 设置学习率预热步数为500步，训练初期学习率从0逐渐增加到设定值
            warmup_steps=500,
            # 设置权重衰减系数为0.01，用于防止过拟合
            weight_decay=0.01,
            # 设置日志文件保存的目录路径
            logging_dir="./bert_logs",
            # 设置每10个训练步骤记录一次日志
            logging_steps=10,
            # 设置评估策略为每个epoch结束后进行评估
            evaluation_strategy="epoch",
            # 设置模型保存策略为每个epoch结束后保存
            save_strategy="epoch",
            # 设置训练结束后加载最佳模型而非最后一个模型
            load_best_model_at_end=True,
            # 设置最多保存1个检查点文件，超出时自动删除旧的
            save_total_limit=1,
            # 设置用于判断最佳模型的指标为评估损失
            metric_for_best_model="eval_loss",
            # 禁用FP16混合精度训练，使用FP32精度
            fp16=False,
        )

        # 初始化 Trainer
        trainer = Trainer(
            # 传入要训练的模型实例
            model=self.model,
            # 传入上面定义的训练参数配置
            args=training_args,
            # 传入训练数据集
            train_dataset=train_dataset,
            # 传入验证数据集，用于训练过程中评估模型性能
            eval_dataset=val_dataset,
            # 传入计算评估指标的函数，用于在验证集上计算准确率等指标
            compute_metrics=self.compute_metrics
        )
        # 训练模型
        logger.info("开始训练 BERT 模型...")
        trainer.train()
        self.save_model()

        # 评估模型
        self.evaluate_model(val_texts, val_labels)

    def compute_metrics(self, eval_pred):
        """计算评估指标"""
        # 获取预测结果和标签
        logits, labels = eval_pred
        # 获取概率最大的预测结果
        predictions = np.argmax(logits, axis=-1)
        # 计算准确率
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    def evaluate_model(self, texts, labels):
        """评估模型性能"""
        # 仅对 texts 进行分词，labels 已为数字
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        dataset = self.create_dataset(encodings, labels)

        trainer = Trainer(model=self.model)
        # 评估、预测
        predictions = trainer.predict(dataset)
        # 获取预测标签
        pred_labels = np.argmax(predictions.predictions, axis=-1)
        # 真实标签
        true_labels = labels

        logger.info("分类报告:")
        logger.info(classification_report(
            true_labels,
            pred_labels,
            target_names=list(self.label_map.keys())
        ))
        logger.info("混淆矩阵:")
        logger.info(confusion_matrix(true_labels, pred_labels))

    def predict_category(self, query):
        # 检查模型是否加载
        if self.model is None:
            # 模型未加载，记录错误
            logger.error("模型未训练或加载")
            # 默认返回通用知识
            return "通用知识"
        # 对查询进行编码
        encoding = self.tokenizer(query, truncation=True, padding=True, max_length=128, return_tensors="pt")
        # 将编码移到指定设备
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        # 不计算梯度，进行预测
        with torch.no_grad():
            # 获取模型输出
            outputs = self.model(**encoding)
            # 获取预测结果
            prediction = torch.argmax(outputs.logits, dim=1).item()
        # 根据预测结果返回类别
        return "法律咨询" if prediction == 1 else "通用知识"


# 测试
if __name__ == "__main__":
    # 初始化分类器
    classifier = QueryClassifier(model_path=config.MODELS_DIR + "/bert_query_classifier_law")

    # 训练模型
    # classifier.train_model(data_file='../classify_data/model_law_5000.json')

    # 示例预测
    test_queries = [
        "《民法典》第一千零六十二条讲什么？",
        "劳动合同到期不续签有补偿吗？",
        "买到假货可以要求几倍赔偿？",
        "租房押金不退怎么办？",
        "5*9等于多少？",
        "今天天气怎么样？",
        "帮我写一段 Python 排序代码",
        "deepseek是大肥鱼吗?"
    ]
    for query in test_queries:
        category = classifier.predict_category(query)
        print(f"查询: {query} -> 分类: {category}")
