python /home/workspace/project/model/run_train.py  /home/workspace/project/model/fk_train/ sub.csv

cd  /home/workspace/output

zip -r models.zip model

castlecli --third sany --source /home/workspace/output/models.zip --token def7221ad81b994458386f2506bb6b96

castlecli --third sany --source /home/workspace/output/models.zip --test "dirpath submit_path"