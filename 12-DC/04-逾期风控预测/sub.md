python /home/workspace/project/model/run_train.py  /home/workspace/project/model/fk_train/ sub.csv

cd  /home/workspace/output

zip -r models.zip model

castlecli --third sany --source /home/workspace/output/models.zip --token 5551f9d836bbd2cdeb864c4cab2b46cc

castlecli --third sany --source /home/workspace/output/models.zip --test "dirpath submit_path"