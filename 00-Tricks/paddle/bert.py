# -*- coding: utf-8 -*-
# @Time     : 2020/11/27 10:56
# @Author   : Michael_Zhouy

import numpy as np
import paddlehub as hub
module = hub.Module(name="ernie")

reader = hub.reader.ClassifyReader(
    dataset=dataset,
    vocab_path=module.get_vocab_path(),
    sp_model_path=module.get_spm_path(),
    word_dict_path=module.get_word_dict_path(),
    max_seq_len=128
)

strategy = hub.AdamWeightDecayStrategy(
    weight_decay=0.01,
    warmup_proportion=0.1,
    learning_rate=5e-5
)

config = hub.RunConfig(
    use_cuda=True,
    num_epoch=5,
    checkpoint_dir="model",
    batch_size=100,
    eval_interval=50,
    strategy=strategy
)

inputs, outputs, program = module.context(trainable=True, max_seq_len=128)

# Use "pooled_output" for classification tasks on an entire sentence.
pooled_output = outputs["pooled_output"]
feed_list = [
    inputs["input_ids"].name,
    inputs["position_ids"].name,
    inputs["segment_ids"].name,
    inputs["input_mask"].name
]
cls_task = hub.TextClassifierTask(
    data_reader=reader,
    feature=pooled_output,
    feed_list=feed_list,
    num_classes=dataset.num_labels,
    config=config
)

run_states = cls_task.finetune_and_eval()

# 写入预测的文本数据
data = [["抗击新型肺炎第一线中国加油鹤岗・绥滨县"], ["正能量青年演员朱一龙先生一起武汉祈福武汉加油中国加油"]]
index = 0
run_states = cls_task.predict(data=data)
results = [run_state.run_results for run_state in run_states]
for batch_result in results:
    # 获取预测的标签索引
    batch_result = np.argmax(batch_result, axis=2)[0]
    for result in batch_result:
        print("%s\预测值=%s" % (data[index][0], result))
        index += 1
