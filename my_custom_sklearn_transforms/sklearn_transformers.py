from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

# All sklearn Transforms must have the `transform` and `fit` methods
class RegLinInputer(BaseEstimator, TransformerMixin):
    def __init__(self, features, target):
        self.features = features
        self.target = target

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.query('{0} == {0}'.format(self.target))
        #print(data.shape)
                
        Xrl = data[self.features]
        #print(Xrl.head())

        yrl = data[self.target]
        #print(yrl.head())

        reg = LinearRegression()
        reg.fit(Xrl, yrl)
                       
                       
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
                       
        data.loc[X[self.target].isnull(), self.target] = reg.predict(X[self.features])[X[self.target].isnull()]
                               
        # Retornamos um novo dataframe com valores preenchidos
        return data
