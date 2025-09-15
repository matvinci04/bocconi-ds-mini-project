import pandas as pd
from src.features import split_features_target,FEATURES,TARGET

def test_split():
    df=pd.DataFrame({f:[1,2] for f in FEATURES+[TARGET]})
    X,y=split_features_target(df)
    assert list(X.columns)==FEATURES
    assert y.name==TARGET
