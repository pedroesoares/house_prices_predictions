# 0.0 - Imports

import pandas            as pd
import numpy             as np
import seaborn           as sns
import matplotlib.pyplot as plt

from boruta                import BorutaPy
from sklearn.ensemble      import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics       import mean_absolute_error, mean_absolute_percentage_error




## 0.1 - Helper Functions


# This codes are for displaying a larger number of rowns and columns 
pd.options.display.max_columns = 200
pd.options.display.max_rows= 200

# This codes are for displaying larger pictures and graphics
plt.rcParams['figure.figsize']= [30,20]
plt.rcParams['font.size'] = 30








# 1.0 - Data management

df_train_raw = pd.read_csv('C:/Users/Pedro Enrique/Desktop/Projetos/ciencia de dados/House_prices/train.csv')
df_test_raw = pd.read_csv('C:/Users/Pedro Enrique/Desktop/Projetos/ciencia de dados/House_prices/test.csv')
df_sample_submission_raw = pd.read_csv('C:/Users/Pedro Enrique/Desktop/Projetos/ciencia de dados/House_prices/sample_submission.csv')




df_train_raw.head()

# Considering the size of our dataset
df_train_raw.shape

# Lets understand how many NA's there is in the DF
df_train_raw.isna().sum()



## 1.1 Filling Out the NA's

df1 = df_train_raw.copy()

# Well I think that a good way to solve the problem of the missing data from LotFrontage will be, filling
# it with the mean of other data
# LotFrontage = Mean of the rest of the data

# For the Alley, the NAN are not an actual problem, it is another data that is simply missing.
# Alley = replace NAN for No, meaning None  

# I believe that for the Masonry veneer type or MasVnrArea there is nothing much to do.

# BSMT Quality and Condition the same thing for the Alley, it means no Basement in the house
# BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, FireplaceQu(Also lets rename to
# FireplaceQual),GarageType, GarageYrBlt, GarageFinish, GarageQual, GarageCond, 
# PoolQC(also renaming to PoolQual),Fence and MiscFeature = None





# For the 'LotFrontage' we are considering that the mean of all data is suitable to fill the NA's
df1['LotFrontage'] = df1['LotFrontage'].fillna(df1['LotFrontage'].mean(), inplace=False)


# This columns are those which still with NA's but were simply filled with a 'None' word.
columns = ['Alley','BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu',
           'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 
            'PoolQC','Fence','MiscFeature']

# We applyed a 'for' for all the columns that were meant to have their NA's filled out
# the rest of Na's are 17 and I am just going to drop that.

for i in columns:
    print(i)
    df1[i] = df1[i].fillna('None', inplace=False)
    




#  Getting two columns renamed, for pattern reasons.

df1.columns = ['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
        'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
        'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
        'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
        'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
        'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
        'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
        'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
        'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
        'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
        'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
        'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQual', 'GarageType',
        'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
        'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
        'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQual',
        'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
        'SaleCondition', 'SalePrice']









# 1.2 - EDA

plt.subplot(2,2,1)
sns.lineplot(x=df1['YearBuilt'], y = df1['SalePrice'], estimator='mean')

plt.subplot(2,2,2)
sns.boxplot(x=df1['MSZoning'], y = df1['SalePrice'])

plt.subplot(2,2,3)
sns.boxplot(x=df1['OverallCond'], y = df1['SalePrice'])

plt.subplot(2,2,4)
sns.histplot(df1[['SalePrice']], bins=50)






# The new DataFrame is the old one but without any NA's
df2 = df1.dropna()
















## 1.3 - Encoding and Scalling


num_attributes = df2.select_dtypes(include=['int64', 'float64'])



# I am describing all the numerical attributes, for those with a large number of Outliers, I am going 
# to use a Robust Scaler, for scalling, and for those with less number of outliers, I am going to
# use a MinMaxScaler()
for i in num_attributes:
    print(i)
    print(df1[i].describe())


# Using Robust Scaler for Outliers
robust_columns = ['LotArea', 'MasVnrArea','BsmtFinSF1', 'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','LowQualFinSF',
'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal']

num_attributes.columns



from sklearn.preprocessing import LabelEncoder

# Preprocessing, we are going to encode the cattegorical Attributes
le = LabelEncoder()

# Columns to Encode
columns_encoder = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
           'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
           'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
           'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
           'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
           'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
           'Functional', 'FireplaceQual', 'GarageType',
           'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQual',
           'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

for i in columns_encoder:
    df2[i] = le.fit_transform(df2[i])

minmax_columns = df2[num_attributes.columns].drop(columns = robust_columns, axis=1, inplace= False).columns
minmax_columns

num_attributes = ['Id', 'MSSubClass', 'LotFrontage','OverallQual',
           'OverallCond', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF',
           'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
           'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
           'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
           'MoSold', 'YrSold']

# We are doing a Scaller for the numerical attributes
mms = MinMaxScaler()
for i in  minmax_columns:
    df2[i] = mms.fit_transform(df2[[i]].values)

    
rs = RobustScaler()
for i in robust_columns:
    rs(df2[i].values)
    

corrmat = df2.corr()
top_corr_features = corrmat.index[abs(corrmat['SalePrice'])>0.5]
plt.figure()
g = sns.heatmap(df2[top_corr_features].corr(),annot=True,cmap="RdYlGn")





# 2.0 - Boruta

# We are going to use the Boruta to select features that impact the most on the SalePrice
x_boruta = df2.drop(['GarageYrBlt','SalePrice'], axis=1, inplace=False)

x_boruta.columns






# cols_boruta = ['MSZoning','Street','Alley','LotShape','LandContour','LotConfig','Utilities',
#                    'LandSlope','Neighborhood','RoofStyle','RoofMatl','Exterior1st',
#                    'Exterior2nd','MasVnrType' ,'ExterQual','ExterCond','Foundation',
#                    'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating',
#                    'HeatingQC','CentralAir','KitchenQual','Functional','FireplaceQual','GarageType',
#                    'GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQual',
#                    'Fence','MiscFeature','Condition1','Condition2','BldgType','HouseStyle',
#                    'Condition1','Condition2','BldgType','HouseStyle','WoodDeckSF','OpenPorchSF',
#                    '3SsnPorch','ScreenPorch','PoolArea','GarageCars','GarageArea',
#                    'Fireplaces','TotRmsAbvGrd','1stFlrSF','2ndFlrSF','LowQualFinSF',
#                    'GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',
#                    'KitchenAbvGr','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFinSF1','OverallQual',
#                    'OverallCond','YearBuilt','YearRemodAdd']


# Boruta Feature Selection
x_train = df2[x_boruta.columns].values
y_train = df2['SalePrice'].values

# defining the Random Forest
rf = RandomForestRegressor( n_jobs=-1)


# This is the Boruta algorithms 
boruta = BorutaPy( rf, n_estimators = 'auto', verbose=2, random_state=1).fit(x_train, y_train)

# feature_ranks = list(zip(x_boruta.columns, boruta.ranking_, boruta.support_))
# feature_ranks

# This are the columns selected from Boruta
cols_boruta_selected = ['LotFrontage','LotArea','Neighborhood','OverallQual','YearBuilt',
'YearRemodAdd','BsmtFinSF1','TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea',
'FullBath','TotRmsAbvGrd','GarageType','GarageCars','GarageArea']

# These are the same columns plus the SalePrice
cols_boruta_selected_full = ['LotFrontage','LotArea','Neighborhood','OverallQual','YearBuilt',
'YearRemodAdd','BsmtFinSF1','TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea',
'FullBath','TotRmsAbvGrd','GarageType','GarageCars','GarageArea','SalePrice']

# For the next steps I will be using a Data Frame with only the features that matters

df3 = df2.copy()
df3 = df3[['Id','LotFrontage','LotArea','Neighborhood','OverallQual','YearBuilt',
            'YearRemodAdd','BsmtFinSF1','TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea',
            'FullBath','TotRmsAbvGrd','GarageType','GarageCars','GarageArea','SalePrice',
            'ExterQual','BsmtQual','KitchenQual']]



df1_test = df_test_raw[['Id','LotFrontage','LotArea','Neighborhood','OverallQual','YearBuilt',
'YearRemodAdd','BsmtFinSF1','TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea',
'FullBath','TotRmsAbvGrd','GarageType','GarageCars','GarageArea','ExterQual','BsmtQual','KitchenQual']]



df1_test_num = ['LotFrontage','LotArea','OverallQual','YearBuilt','YearRemodAdd','BsmtFinSF1',      
            'TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea','FullBath','TotRmsAbvGrd',
            'GarageCars','GarageArea'  ]

# Now everything tha is categorical from the test I am going to encode
# And everything numerical is going to be normalized

# Encoding
df1_test['GarageType'] = le.fit_transform(df1_test['GarageType']);
df1_test['Neighborhood'] = le.fit_transform(df1_test['Neighborhood']);
df1_test['ExterQual'] = le.fit_transform(df1_test['ExterQual']);
df1_test['BsmtQual'] = le.fit_transform(df1_test['BsmtQual']);
df1_test['KitchenQual'] = le.fit_transform(df1_test['KitchenQual']);


# Normalizing
for i in df1_test_num:
    df1_test[i] = mms.fit_transform(df1_test[[i]].values);






y_train = df3['SalePrice']
x_train = df3.drop(['SalePrice','Id'], axis=1, inplace=False)

x_test = df1_test.drop(['Id'], axis=1, inplace=False)

import xgboost as xgb

x_test['LotFrontage'] = x_test['LotFrontage'].fillna(x_test['LotFrontage'].mean(),inplace=False)
rf_test = x_test
rf_test = rf_test.dropna()

# XGBoost model 
model = xgb.XGBRegressor().fit(x_train, y_train)
yhat = model.predict(x_test)

# Random Forest Regressor Model
model_rf = rf.fit(x_train,y_train)
yhat_rf = model_rf.predict(rf_test)



df_yhat = pd.DataFrame(yhat, columns=['Prediction_xgb'])
df_yhat_rf = pd.DataFrame(yhat_rf, columns=['Predictions_rf'])


# df_yhat and df_yhat_rf
df_prices = df_sample_submission_raw
df_prices['Predictions_xgb'] = df_yhat
df_prices['Predictions_rf'] = df_yhat_rf

# Dropping Na from 'Predictions_RF'
df_prices = df_prices.dropna()

# These are the errors related with each modeling, here we have both the Mae and the Mape
mae_xgb = mean_absolute_error(df_prices['SalePrice'], df_prices['Predictions_xgb'])
mae_rf = mean_absolute_error(df_prices['SalePrice'], df_prices['Predictions_rf'])

mape_xgb = mae_xgb = mean_absolute_percentage_error(df_prices['SalePrice'], df_prices['Predictions_xgb'])
mape_rf = mean_absolute_percentage_error(df_prices['SalePrice'], df_prices['Predictions_rf'])
