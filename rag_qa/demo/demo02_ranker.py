from base.config import config
from sentence_transformers import CrossEncoder

model1 = CrossEncoder(model_name=config.MODELS_DIR + "/bge-reranker-large")

#计算问题与答案的匹配度分值
pairs = [['what is panda?', "what is panda?"],
         ['what is panda?', 'hi'],
         ['what is panda?', 'The giant panda '],
         ]

print(model1.predict(pairs))  # [0.999556  0.003653  0.3362122]
