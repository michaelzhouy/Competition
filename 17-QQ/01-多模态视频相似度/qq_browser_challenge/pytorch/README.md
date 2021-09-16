### [Multimodal Video Similarity Challenge](https://algo.browser.qq.com/)
#### [@CIKM 2021](https://www.cikm2021.org/analyticup) 
This is the official pytorch version baseline built on our lichee framework
 
#### 1. data
Please download the 'data' directory and verify the integrity first, then prepare 'data' according to [data/README.md](data/README.md)
 
#### 2. code description
- [lichee](lichee) is our multimodal training framework
- [module](module) is the demo model dependency
- [embedding_example.yaml](embedding_example.yaml) is a demo config
- [main.py](main.py) is the main entry
- [read_tf_record_example.py](read_tf_record_example.py) is a demo for tfrecords parsing

#### 3. Install the dependency
```bash
pip install -r requirements.txt
```

#### 4. Train：
```bash
python3 main.py \
--trainer=embedding_trainer \
--model_config_file=embedding_example.yaml
```

#### 5. Validate：
```bash
python3 main.py \
--trainer=embedding_trainer \
--model_config_file=embedding_example.yaml \
--mode=eval \
--checkpoint=your_check_point.bin \
--dataset=SPEARMAN_EVAL  # the validation DATASET key in embedding_example.yaml
```

#### 6. Test
```bash
python3 main.py \
--trainer=embedding_trainer \
--model_config_file=embedding_example.yaml \
--mode=test \
--checkpoint=Epoch_3_0.7531_0.4387_0.6525.bin \
--dataset=SPEARMAN_TEST_A  # the test DATASET key in embedding_example.yaml
```
