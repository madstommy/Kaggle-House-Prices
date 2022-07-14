import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

train_raw = pd.read_csv('train.csv')
test_raw = pd.read_csv('test.csv')

LotFrontage = train_raw['LotFrontage']
LotArea = train_raw['LotArea']
YearBuilt = train_raw['YearBuilt']
YearRemodAdd = train_raw['YearRemodAdd']
MasVnrArea = train_raw['MasVnrArea']
BsmtFinSF1 = train_raw['BsmtFinSF1']
BsmtFinSF2 = train_raw['BsmtFinSF2']
BsmtUnfSF = train_raw['BsmtUnfSF']
TotalBsmtSF = train_raw['TotalBsmtSF']

FirstFlrSF = train_raw['1stFlrSF']
SecondFlrSF = train_raw['2ndFlrSF']
LowQualFinSF = train_raw['LowQualFinSF']
GrLivArea = train_raw['GrLivArea']
BsmtFullBath = train_raw['BsmtFullBath']
BsmtHalfBath = train_raw['BsmtHalfBath']
FullBath = train_raw['FullBath']
HalfBath = train_raw['HalfBath']
BedroomAbvGr = train_raw['BedroomAbvGr']

KitchenAbvGr = train_raw['KitchenAbvGr']
TotRmsAbvGrd = train_raw['TotRmsAbvGrd']
Fireplaces = train_raw['Fireplaces']
GarageYrBlt = train_raw['GarageYrBlt']
GarageCars = train_raw['GarageCars']
GarageArea = train_raw['GarageArea']
WoodDeckSF = train_raw['WoodDeckSF']
OpenPorchSF = train_raw['OpenPorchSF']
EnclosedPorch = train_raw['EnclosedPorch']

SsnPorch = train_raw['3SsnPorch']
ScreenPorch = train_raw['ScreenPorch']
PoolArea = train_raw['PoolArea']
MiscVal = train_raw['MiscVal']
MoSold = train_raw['MoSold']
YrSold = train_raw['YrSold']

sale_price = train_raw['SalePrice']
ids = train_raw['Id']

train_data = train_raw.drop(['Id', 'LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
                            '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 
                            'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
                            '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SalePrice'], axis='columns')

for column in train_data:
    train_data[column] = pd.factorize(train_data[column], na_sentinel=None)[0]


train_data['LotFrontage'] = LotFrontage
train_data['LotArea'] = LotArea
train_data['YearBuilt'] = YearBuilt
train_data['YearRemodAdd'] = YearRemodAdd
train_data['MasVnrArea'] = MasVnrArea
train_data['BsmtFinSF1'] = BsmtFinSF1
train_data['BsmtFinSF2'] = BsmtFinSF2
train_data['BsmtUnfSF'] = BsmtUnfSF
train_data['TotalBsmtSF'] = TotalBsmtSF

train_data['1stFlrSF'] = FirstFlrSF
train_data['2ndFlrSF'] = SecondFlrSF
train_data['LowQualFinSF'] = LowQualFinSF
train_data['GrLivArea'] = GrLivArea
train_data['BsmtFullBath'] = BsmtFullBath
train_data['BsmtHalfBath'] = BsmtHalfBath
train_data['FullBath'] = FullBath
train_data['HalfBath'] = HalfBath
train_data['BedroomAbvGr'] = BedroomAbvGr

train_data['KitchenAbvGr'] = KitchenAbvGr
train_data['TotRmsAbvGrd'] = TotRmsAbvGrd
train_data['Fireplaces'] = Fireplaces
train_data['GarageYrBlt'] = GarageYrBlt
train_data['GarageCars'] = GarageCars
train_data['GarageArea'] = GarageArea
train_data['WoodDeckSF'] = WoodDeckSF
train_data['OpenPorchSF'] = OpenPorchSF
train_data['EnclosedPorch'] = EnclosedPorch

train_data['3SsnPorch'] = SsnPorch
train_data['ScreenPorch'] = ScreenPorch
train_data['PoolArea'] = PoolArea
train_data['MiscVal'] = MiscVal
train_data['MoSold'] = MoSold
train_data['YrSold'] = YrSold

imp = IterativeImputer()
imp.fit(train_data)
train_data = imp.fit_transform(train_data)
train_x = pd.DataFrame(train_data)
train_y = pd.DataFrame(train_raw[['SalePrice']], columns=['SalePrice'])
train_x.to_csv('train_x.csv', index=False)
train_y.to_csv('train_y.csv', index=False)



#test_data
#########################################################################################

LotFrontage = test_raw['LotFrontage']
LotArea = test_raw['LotArea']
YearBuilt = test_raw['YearBuilt']
YearRemodAdd = test_raw['YearRemodAdd']
MasVnrArea = test_raw['MasVnrArea']
BsmtFinSF1 = test_raw['BsmtFinSF1']
BsmtFinSF2 = test_raw['BsmtFinSF2']
BsmtUnfSF = test_raw['BsmtUnfSF']
TotalBsmtSF = test_raw['TotalBsmtSF']

FirstFlrSF = test_raw['1stFlrSF']
SecondFlrSF = test_raw['2ndFlrSF']
LowQualFinSF = test_raw['LowQualFinSF']
GrLivArea = test_raw['GrLivArea']
BsmtFullBath = test_raw['BsmtFullBath']
BsmtHalfBath = test_raw['BsmtHalfBath']
FullBath = test_raw['FullBath']
HalfBath = test_raw['HalfBath']
BedroomAbvGr = test_raw['BedroomAbvGr']

KitchenAbvGr = test_raw['KitchenAbvGr']
TotRmsAbvGrd = test_raw['TotRmsAbvGrd']
Fireplaces = test_raw['Fireplaces']
GarageYrBlt = test_raw['GarageYrBlt']
GarageCars = test_raw['GarageCars']
GarageArea = test_raw['GarageArea']
WoodDeckSF = test_raw['WoodDeckSF']
OpenPorchSF = test_raw['OpenPorchSF']
EnclosedPorch = test_raw['EnclosedPorch']

SsnPorch = test_raw['3SsnPorch']
ScreenPorch = test_raw['ScreenPorch']
PoolArea = test_raw['PoolArea']
MiscVal = test_raw['MiscVal']
MoSold = test_raw['MoSold']
YrSold = test_raw['YrSold']

test_ids = test_raw['Id']

test_data = test_raw.drop(['Id', 'LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
                            '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 
                            'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
                            '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'], axis='columns')

for column in test_data:
    test_data[column] = pd.factorize(test_data[column], na_sentinel=None)[0]


test_data['LotFrontage'] = LotFrontage
test_data['LotArea'] = LotArea
test_data['YearBuilt'] = YearBuilt
test_data['YearRemodAdd'] = YearRemodAdd
test_data['MasVnrArea'] = MasVnrArea
test_data['BsmtFinSF1'] = BsmtFinSF1
test_data['BsmtFinSF2'] = BsmtFinSF2
test_data['BsmtUnfSF'] = BsmtUnfSF
test_data['TotalBsmtSF'] = TotalBsmtSF

test_data['1stFlrSF'] = FirstFlrSF
test_data['2ndFlrSF'] = SecondFlrSF
test_data['LowQualFinSF'] = LowQualFinSF
test_data['GrLivArea'] = GrLivArea
test_data['BsmtFullBath'] = BsmtFullBath
test_data['BsmtHalfBath'] = BsmtHalfBath
test_data['FullBath'] = FullBath
test_data['HalfBath'] = HalfBath
test_data['BedroomAbvGr'] = BedroomAbvGr

test_data['KitchenAbvGr'] = KitchenAbvGr
test_data['TotRmsAbvGrd'] = TotRmsAbvGrd
test_data['Fireplaces'] = Fireplaces
test_data['GarageYrBlt'] = GarageYrBlt
test_data['GarageCars'] = GarageCars
test_data['GarageArea'] = GarageArea
test_data['WoodDeckSF'] = WoodDeckSF
test_data['OpenPorchSF'] = OpenPorchSF
test_data['EnclosedPorch'] = EnclosedPorch

test_data['3SsnPorch'] = SsnPorch
test_data['ScreenPorch'] = ScreenPorch
test_data['PoolArea'] = PoolArea
test_data['MiscVal'] = MiscVal
test_data['MoSold'] = MoSold
test_data['YrSold'] = YrSold



imp = IterativeImputer()
imp.fit(test_data)
test_data = imp.fit_transform(test_data)
test_x = pd.DataFrame(test_data)
test_x['Id'] = test_ids
test_x.to_csv('test_x.csv', index=False)