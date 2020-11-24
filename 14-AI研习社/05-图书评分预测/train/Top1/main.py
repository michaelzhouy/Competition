import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from catboost import CatBoostRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split
import fire
train_f = pd.read_csv("./process_train.csv")
test_f = pd.read_csv("./process_test.csv")

def process2new(df_raw,train=True):
#     'User-ID', 'ISBN', 'Book-Rating', 'Location', 'Age', 'Book-Title',
#        'Book-Author', 'Year-Of-Publication', 'Publisher'
    loc0 = list(df_raw["Loc0"])
    loc1 = list(df_raw["Loc1"])
    loc2 = list(df_raw["Loc2"])
    
    User_id = list(df_raw["User-ID"])
    ISBN = list(df_raw["ISBN"])
    Age = [int(x) if x!=None else 0 for x in list(df_raw["Age"])]
    Book_Author = list(df_raw["Book-Author"])
    Publisher = list(df_raw["Publisher"])
    
    res_d = {'User-ID':User_id, 'ISBN':ISBN, 'Loc0':loc0,'Loc1':loc1,"Loc2":loc2, 'Age':Age,
      'Book-Author':Book_Author, 'Publisher':Publisher}  
    if train:
        Book_Rating = df_raw["Book-Rating"]
        return pd.DataFrame(res_d),Book_Rating 
    return pd.DataFrame(res_d)

train_x,train_y = process2new(train_f)
test_x = process2new(test_f,False)

categorical_features_indices = [0,1,2,3,5,6]
x_train, x_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size=0.1, random_state=2200)

def train(iterations=22000,depth=10):
	model = CatBoostRegressor(
	        iterations=iterations, 
	        learning_rate=0.03,
	        depth=depth, 
	        cat_features=categorical_features_indices,
	        l2_leaf_reg=3,
	        loss_function='MAE',
	        eval_metric='MAE',
	        random_seed=2200,
	        task_type="GPU",
	        devices = "0"
	)
	model.fit(x_train,y_train,eval_set=(x_valid, y_valid),plot=True)

	result = model.predict(test_x)
	d = {"index":range(len(result)),"score":result}
	d = pd.DataFrame(d)
	d.to_csv("./submission_{0}_{1}.csv".format(str(iterations),str(depth)),header=None,index=0)

if __name__ == '__main__':
	fire.Fire()
