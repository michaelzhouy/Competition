python /home/workspace/project/model/run_train.py  /home/workspace/project/model/fk_train/ sub.csv

cd /home/workspace/output

zip -r models.zip model/*

castlecli --third sany --source /home/workspace/output/models.zip --token f3a1620b9e001dd51c7192f5885b91e2

castlecli --third sany --source /home/workspace/output/models.zip --test "dirpath submit_path"