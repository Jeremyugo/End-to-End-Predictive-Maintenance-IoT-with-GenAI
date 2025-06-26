from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class DateTimeImputer(BaseEstimator, TransformerMixin):
    """
        A transformer that fills missing datetime values with the mean datetime of the column.
    """
    def __init__(self, strategy:str = "mean"):
        self.strategy = strategy
        
    def fit(self, X, y=None):
        match self.strategy:
            case "mean":
                self.fill_value_ = pd.to_datetime(X).mean()
            case "median":
                self.fill_value_ = pd.to_datetime(X).median()
            case "mode":
                self.fill_value_ = pd.to_datetime(X).mode()[0]
            case _:
                self.fill_value_ = None
        return self
        
    def transform(self, X):
        X = X.copy()
        if self.strategy in ["mean", "median", "mode"]:
            X = X.fillna(self.fill_value_)
        elif self.strategy == "ffill":
            X = X.fillna(method='ffill')
        elif self.strategy == "bfill":
            X = X.fillna(method='bfill')
            
        return X
    
    
    def get_feature_names_out(self, input_features=None):
        return input_features

    
    
class TimeStampTransformer(BaseEstimator, TransformerMixin):
    """
        Transforms datetime objects
    """
    
    def __init__(self, drop_original=True, granularity='day'):
        self.drop_original = drop_original
        self.granularity = granularity
        
        
    @staticmethod
    def truncate_timestamp(granularity:str):
        datetime_options = ['year', 'month', 'day', 'hour', 'minute', 'dayofweek']
        if granularity not in datetime_options:
            return datetime_options
        else:
            return datetime_options[:datetime_options.index(granularity) + 1]
       
        
    def fit(self, X, y=None):
            return self
    
    
    def transform(self, X):
        X = X.copy()
        ts = pd.to_datetime(X)
        X = pd.DataFrame(X)

        for part in self.truncate_timestamp(self.granularity):
            X[f'{part}_ts'] = getattr(ts.dt, part)
        
        if self.drop_original:
            X = X.filter(items=[col for col in X.columns if col.endswith('_ts')])
        
        return X
    
    
    def get_feature_names_out(self, input_features=None):
        base = input_features[0] if input_features else "timestamp"
        parts = self.truncate_timestamp(self.granularity)
        return [f"{base}_{p}_ts" for p in parts]