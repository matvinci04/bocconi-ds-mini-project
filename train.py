import argparse
from src.data import load_csv
from src.features import split_features_target
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",default="data/dataset.csv")
    args=parser.parse_args()
    df=load_csv(args.data)
    X,y=split_features_target(df)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
    models={"LinearRegression":LinearRegression(),"RandomForest":RandomForestRegressor(n_estimators=100,random_state=42)}
    for name,m in models.items():
        m.fit(X_train,y_train)
        pred=m.predict(X_test)
        mae=mean_absolute_error(y_test,pred)
        rmse=mean_squared_error(y_test,pred,squared=False)
        print(f"{name}: MAE={mae:.2f}, RMSE={rmse:.2f}")

if __name__=="__main__":
    main()
