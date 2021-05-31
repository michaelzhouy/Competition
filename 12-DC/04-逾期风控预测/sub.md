python /home/workspace/project/model/run_train.py  /home/workspace/data/fk_train/ sub.csv

cd  /home/workspace/project

zip -r models.zip model

castlecli --third sany --source /home/workspace/project/models.zip --token e27acdcd74fa29945174e19b8679bf34