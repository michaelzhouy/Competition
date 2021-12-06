
### 1. train
```bash
nohup python3 -u train.py \
--data_dir='../data/' \
--model_dir='./model/model_pth/' \
--epochs=20 \
--model='swin_base_patch4_window7_224' \
--batch_size=128 \
--input_size=224 \
--LR=1e-3 \
--num_workers=8 \
--cuda='1,2,3,4' >> swin_base_224_log 2>&1 &
```

### 2. inference
```bash
python3 inference.py
```