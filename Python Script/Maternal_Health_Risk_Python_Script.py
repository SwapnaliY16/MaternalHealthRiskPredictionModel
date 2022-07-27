#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Author: Swapnali Yadav(21118122)


# ## Step1: Data Collection and Analysis 

# In[2]:


# Data wrangling
import pandas as pd

# To igrnore warnings
import warnings
warnings.filterwarnings("ignore")


# In[3]:


# Import comma-separated values (csv) file into DataFrame 
df = pd.read_csv('Maternal Health Risk Data Set.csv')
# Read first default 5 rows of DataFrame
df.head()


# In[4]:


# Check DataFrame information about index, dtype, columns, non-null values and memory usage
df.info()


# In[5]:


# Describe statistics i.e. count, mean, std, min, max with lower, upper and mid percentiles 
df.describe()


# In[6]:


# Check dimensionality of the DataFrame i.e (no of rows, no of columns)
df.shape


# In[7]:


# Check unique values of Target Variable
df['RiskLevel'].unique()


# ## Step 2: Data visualization and exploration(Exploratory Data Analysis)

# In[8]:


# Data visualisation
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


# Plot pairwise relationships i.e pair of Target with all other features in a dataset
sns.pairplot(df, hue = 'RiskLevel')


# In[10]:


# To show marginal histograms instead of layered KDE (kernel density estimate)
sns.pairplot(df, hue = 'RiskLevel', diag_kind = 'hist')


# In[11]:


# Show the counts of RiskLevel for each Age variable
plt.figure(figsize = (17,6))
sns.countplot('Age', hue = 'RiskLevel', data = df)


# In[12]:


df['RiskLevel'].value_counts()


# In[13]:


list(df['RiskLevel'].unique())


# In[14]:


# Show pie chart of RiskLevel
fig = plt.figure(figsize =(7, 7))
plt.pie(df['RiskLevel'].value_counts(), labels = list(df['RiskLevel'].unique()),  autopct = '%1.1f%%')
plt.show()


# In[15]:


# Check correlation between all features
correlation = df.corr()
plt.figure(figsize = (12,8))
sns.heatmap(correlation, xticklabels = correlation.columns, yticklabels = correlation.columns, annot = True)


# In[16]:


# replacing string target variables to numeric
df['RiskLevel'] = df['RiskLevel'].replace({'low risk':0,'mid risk':1,'high risk':2})


# In[17]:


df.to_csv('Maternal Health Risk Data Set with numeric target.csv', index=False)


# In[18]:


# Dtype has been changed from object to int64
df.info()


# In[19]:


# RiskLevel values are changed to 0, 1, 2
df.head()


# In[20]:


# Check correlation between all feature after conversion of target variable RiskLevel to numeric
# First BS is highly correlated with RiskLevel following SystolicBP, DiastolicBP and then Age.
correlation = df.corr()
plt.figure(figsize = (12,8))
sns.heatmap(correlation, xticklabels = correlation.columns, yticklabels = correlation.columns, annot =True)


# In[21]:


# Histogram plot of Age range
# we have data of age ranges from 10 to 70
sns.histplot(df['Age'])


# In[22]:


# Histogram plot of BS range
# we have data of BS ranges from 6 to 19  
# values after 8 are high risk patient's reading, hence they are valid and not considerd as outliers
plt.figure(figsize = (15,5))
sns.histplot(df['BS'])


# In[23]:


# Histogram plot of BS range
# we have data of BS ranges from 98 to 103
# values after 99 are high risk patient's reading, hence they are normal and not considerd as outliers
sns.histplot(df['BodyTemp'])


# In[24]:


# Show ScatterPlot - indicates when DBP and SBP is high then Risk is high
sns.scatterplot('DiastolicBP', 'SystolicBP', hue='RiskLevel', data = df, palette=['green','orange','red'])


# In[25]:


grid = sns.FacetGrid(df, col = "RiskLevel", hue = "RiskLevel", col_wrap=3, palette=['green','orange','red'])
grid.map(sns.scatterplot, "DiastolicBP", "SystolicBP")
grid.add_legend()
plt.show()


# In[26]:


# below scatter plot indicates total count of high risklevel in BS is high compared to low RiskLevel
# So BS is impactful factor for high risk  
sns.scatterplot('RiskLevel','BS', hue = 'RiskLevel', data = df, palette=['green','orange','red'])


# In[27]:


grid = sns.FacetGrid(df, col = "RiskLevel", hue = "RiskLevel", col_wrap=3, palette=['green','orange','red'])
grid.map(sns.scatterplot, "RiskLevel", "BS")
grid.add_legend()
plt.show()


# In[28]:


grid = sns.FacetGrid(df, col = "RiskLevel", hue = "RiskLevel", col_wrap=3, palette=['green','orange','red'])
grid.map(sns.scatterplot, "Age", "BS")
grid.add_legend()
plt.show()


# In[29]:


plt.figure(figsize=(15,6))
sns.countplot('BS', hue ='RiskLevel',data=df,  palette=['green','orange','red'])


# In[30]:


# As BS increases, Risk also increases significantly in all ages and vice versa
sns.scatterplot('Age','BS', hue = 'RiskLevel',data = df, palette = ['green','orange','red'])


# ## Step 3 – Feature Engineering

# In[31]:


# Check null values and there are no numm values
df.isna().sum()


# In[32]:


# Splitting data into input and target class
X = df.drop('RiskLevel',axis = 1)
y = df['RiskLevel']

# X, Y shape
print("X shape: ", X.shape)
print("Y shape: ", y.shape)


# In[33]:


# features span across different range of values, i.e. attributed to the different units.
# Feature scaling can help us solve this issue.
df.describe().transpose()


# In[34]:


# Feature Scaling
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# Instantiate MinMaxScaler, StandardScaler
normal = MinMaxScaler()
standard = StandardScaler()


# In[35]:


# MinMaxScaler
normalised_features = normal.fit_transform(X)
normalised_df = pd.DataFrame(normalised_features, index = X.index, columns = X.columns)

# StandardScaler
standardised_features = standard.fit_transform(X)
standardised_df = pd.DataFrame(standardised_features, index = X.index, columns = X.columns)


# In[36]:


X.index


# In[37]:


standardised_features


# In[38]:


df[X.columns]


# In[39]:


normalised_df[X.columns]


# In[40]:


standardised_df[X.columns]


# In[41]:


# Create subplots
fig, ax = plt.subplots(1, 3, figsize = (21, 5))

# Original
sns.boxplot(x = 'variable', y = 'value', data = pd.melt(df[X.columns]), ax = ax[0])
ax[0].set_title('Original')

# MinMaxScaler
sns.boxplot(x = 'variable', y = 'value', data = pd.melt(normalised_df[X.columns]), ax = ax[1])
ax[1].set_title('MinMaxScaler')

# StandardScaler
sns.boxplot(x = 'variable', y = 'value', data = pd.melt(standardised_df[X.columns]), ax = ax[2])
ax[2].set_title('StandardScaler')


# ### To Decide scalers from Normal or Standard for further steps

# In[42]:


pip install catboost


# In[43]:


# models
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost.sklearn import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge

from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
import numpy as np


# In[44]:


# Instantiate models 
knn = KNeighborsRegressor()
svr = SVR()
tree = DecisionTreeRegressor(max_depth = 10, random_state = 42)
xgb = XGBRegressor()
catb = CatBoostRegressor()
linear = LinearRegression()
sdg = SGDRegressor()
rfr = RandomForestRegressor(max_depth=2, random_state=0)
gb = GradientBoostingRegressor(random_state=0)
br = BayesianRidge()

# Create a list which contains different scalers 
scalers = [normal, standard]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# In[45]:


knn_rmse = []

# Without feature scaling
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
knn_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

# Apply different scaling techniques and make predictions using KNN 
for scaler in scalers:
    pipe = make_pipeline(scaler, knn)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    knn_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

# Show results     
knn_df = pd.DataFrame({'Root Mean Squared Error': knn_rmse}, index = ['Original', 'MinMaxScaler', 'StandardScaler'])
knn_df


# In[46]:


svr_rmse = []

# Without feature scaling
svr.fit(X_train, y_train)
pred = svr.predict(X_test)
svr_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

# Apply different scaling techniques and make predictions using SVR
for scaler in scalers:
    pipe = make_pipeline(scaler, svr)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    svr_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

# Show results
svr_df = pd.DataFrame({'Root Mean Squared Error': svr_rmse}, index = ['Original', 'MinMaxScaler', 'StandardScaler'])
svr_df


# In[47]:


xgb_rmse = []

# Without feature scaling
xgb.fit(X_train, y_train)
pred = xgb.predict(X_test)
xgb_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

# Apply different scaling techniques and make predictions using XGB
for scaler in scalers:
    pipe = make_pipeline(scaler, xgb)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    xgb_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

# Show results
xgb_df = pd.DataFrame({'Root Mean Squared Error': xgb_rmse}, index = ['Original', 'MinMaxScaler', 'StandardScaler'])
xgb_df


# In[48]:


catb_rmse = []

# Without feature scaling
catb.fit(X_train, y_train)
pred = catb.predict(X_test)
catb_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

# Apply different scaling techniques and make predictions using CatBoost
for scaler in scalers:
    pipe = make_pipeline(scaler, catb)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    catb_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

# Show results
catb_df = pd.DataFrame({'Root Mean Squared Error': catb_rmse}, index = ['Original', 'MinMaxScaler', 'StandardScaler'])
catb_df


# In[49]:


linear_rmse = []

# Without feature scaling
linear.fit(X_train, y_train)
pred = linear.predict(X_test)
linear_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

# Apply different scaling techniques and make predictions using linear
for scaler in scalers:
    pipe = make_pipeline(scaler, linear)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    linear_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

# Show results
linear_df = pd.DataFrame({'Root Mean Squared Error': linear_rmse}, index = ['Original', 'MinMaxScaler', 'StandardScaler'])
linear_df


# In[50]:


sdg_rmse = []

# Without feature scaling
sdg.fit(X_train, y_train)
pred = sdg.predict(X_test)
sdg_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

# Apply different scaling techniques and make predictions using SDG
for scaler in scalers:
    pipe = make_pipeline(scaler, sdg)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    sdg_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

# Show results
sdg_df = pd.DataFrame({'Root Mean Squared Error': sdg_rmse}, index = ['Original', 'MinMaxScaler', 'StandardScaler'])
sdg_df


# In[51]:


rfr_rmse = []

# Without feature scaling
rfr.fit(X_train, y_train)
pred = rfr.predict(X_test)
rfr_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

# Apply different scaling techniques and make predictions using RandomForest
for scaler in scalers:
    pipe = make_pipeline(scaler, rfr)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    rfr_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

# Show results
rfr_df = pd.DataFrame({'Root Mean Squared Error': rfr_rmse}, index = ['Original', 'MinMaxScaler', 'StandardScaler'])
rfr_df


# In[52]:


gb_rmse = []

# Without feature scaling
gb.fit(X_train, y_train)
pred = gb.predict(X_test)
gb_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

# Apply different scaling techniques and make predictions using GradientBoosting
for scaler in scalers:
    pipe = make_pipeline(scaler, gb)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    gb_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

# Show results
gb_df = pd.DataFrame({'Root Mean Squared Error': gb_rmse}, index = ['Original', 'MinMaxScaler', 'StandardScaler'])
gb_df


# In[53]:


br_rmse = []

# Without feature scaling
br.fit(X_train, y_train)
pred = br.predict(X_test)
br_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

# Apply different scaling techniques and make predictions using BayesianRidge
for scaler in scalers:
    pipe = make_pipeline(scaler, br)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    br_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

# Show results
br_df = pd.DataFrame({'Root Mean Squared Error': br_rmse}, index = ['Original', 'MinMaxScaler', 'StandardScaler'])
br_df


# In[54]:


# Comparison of RMSE of MinMaxScaler and StandardScaler for different machine learning models
# XGBoost, CatBoost, Linear, RandomForest, GradientBoosting, BayesianRidege gives equal RMSE for both, 
# Hence, commented in below barpolt 
df = pd.DataFrame([
    ['KNN', 'MinMaxScaler', knn_rmse[1]], ['KNN', 'StandardScaler',  knn_rmse[2]],
    ['SVR', 'MinMaxScaler', svr_rmse[1]], ['SVR', 'StandardScaler',  svr_rmse[2]],
    #['XGBoost', 'MinMaxScaler', xgb_rmse[1]], ['XGBoost', 'StandardScaler',  xgb_rmse[2]],
    #['CatBoost', 'MinMaxScaler', catb_rmse[1]], ['CatBoost', 'StandardScaler',  catb_rmse[2]],
    #['Linear', 'MinMaxScaler', linear_rmse[1]], ['Linear', 'StandardScaler',  linear_rmse[2]],
    ['SGD', 'MinMaxScaler', sdg_rmse[1]], ['SGD', 'StandardScaler',  sdg_rmse[2]],
    #['RF', 'MinMaxScaler', rfr_rmse[1]], ['RF', 'StandardScaler',  rfr_rmse[2]],
    #['GB', 'MinMaxScaler', gb_rmse[1]], ['GB', 'StandardScaler',  gb_rmse[2]],
    #['BR', 'MinMaxScaler', br_rmse[1]],  ['BR', 'StandardScaler',  br_rmse[2]]
     ], 
    columns=['Models', 'Scalers', 'Root Mean Squared Error'])

# plot with seaborn barplot
plt.figure(figsize=(20, 7))
ax = sns.barplot(data=df, x ='Scalers', y ='Root Mean Squared Error', hue ='Models', ci = None)

# for printing RMSE over each bar plot
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', padding=3)


# In[55]:


normalised_features


# In[56]:


# Splitting data into train and test data
#X_train, X_test, y_train, y_test = train_test_split(normalised_features, y, test_size = 0.30, random_state = 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)


# In[57]:


X_train.size


# In[58]:


X_test.size


# In[59]:


# Normal scaling of training dataset
X_train = normal.fit_transform(X_train)  
X_test = normal.transform(X_test)


# In[60]:


y_train.value_counts()


# In[61]:


pip install imblearn


# In[62]:


from imblearn.over_sampling import RandomOverSampler
over_sampler = RandomOverSampler(random_state = 42)
X_res, y_res = over_sampler.fit_resample(X_train, y_train)
y_res.value_counts()


# ## Step 4 : Model tuning

# In[63]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[64]:


pip install xgboost


# In[65]:


from xgboost import XGBClassifier


# In[66]:


#The LogisticRegression class can be configured for multinomial logistic regression 
#by setting the “multi_class” argument to “multinomial” and the “solver” argument to a solver 
#that supports multinomial logistic regression, such as “lbfgs“.
lr = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial')
lr.fit(X_train, y_train)
score_lr_without_Kfold = lr.score(X_test, y_test)
score_lr_without_Kfold


# In[67]:


# one-vs-one (‘ovo’) is used for multi-class strategy.
svm = SVC(decision_function_shape='ovo')
svm.fit(X_train, y_train)
score_svm_without_Kfold = svm.score(X_test, y_test)
score_svm_without_Kfold


# In[68]:


# n_estimators is a parameter for the number of trees in the forest, which is 40
rf = RandomForestClassifier(n_estimators=40)
rf.fit(X_train, y_train)
score_rf_without_Kfold = rf.score(X_test, y_test)
score_rf_without_Kfold


# In[69]:


# n_neighbors is a parameter for Number of neighbors, which is 6
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
score_knn_without_Kfold = knn.score(X_test, y_test)
score_knn_without_Kfold


# In[70]:


xgb = XGBClassifier()
xgb.fit(X_train, y_train)
score_xgb_without_Kfold = xgb.score(X_test, y_test)
score_xgb_without_Kfold


# In[71]:


# import cross_val_score
from sklearn.model_selection import cross_val_score


# In[72]:


# Check cross val scores of LogisticRegression with K-fold as 3.
score_lr_with_Kfold_imbalance = cross_val_score(LogisticRegression(solver='lbfgs',multi_class='multinomial'), 
                                                X_train, y_train, cv=3)
print(score_lr_with_Kfold_imbalance)
print("Avg :",np.average(score_lr_with_Kfold_imbalance))


# In[73]:


# Check cross val scores of SVC with one-vs-one and K-fold as 3.
score_svm_with_Kfold_imbalance = cross_val_score(SVC(decision_function_shape='ovo'), X_train, y_train, cv=3)
print(score_svm_with_Kfold_imbalance)
print("Avg :",np.average(score_svm_with_Kfold_imbalance))


# In[74]:


# Check cross val scores of RandomForestClassifier with with K-fold as 10.
score_rf_with_Kfold_imbalance = cross_val_score(RandomForestClassifier(n_estimators=40), X_train, y_train, cv=10)
print(score_rf_with_Kfold_imbalance)
print("Avg :",np.average(score_rf_with_Kfold_imbalance))


# In[75]:


# Check cross val scores of KNeighborsClassifier with with K-fold as 10.
score_knn_with_Kfold_imbalance = cross_val_score(KNeighborsClassifier(n_neighbors=6), X_train, y_train, cv=10)
print(score_knn_with_Kfold_imbalance)
print("Avg :",np.average(score_knn_with_Kfold_imbalance))


# In[76]:


# Check cross val scores of XGBClassifier with with K-fold as 3.
score_xgb_with_Kfold_imbalance = cross_val_score(XGBClassifier(), X_train, y_train, cv=3)
print(score_xgb_with_Kfold_imbalance)
print("Avg :",np.average(score_xgb_with_Kfold_imbalance))


# In[77]:


# With imbalance dataset, score of RandomForestClassifier is high 
# hence reverified with differnt estimators but n_estimators=40 gives good score
scores1 = cross_val_score(RandomForestClassifier(n_estimators=5),X_train, y_train, cv=10)
print("Avg Score for Estimators=5 and CV=10 :",np.average(scores1))
scores2 = cross_val_score(RandomForestClassifier(n_estimators=10), X_train, y_train, cv=10)
print("Avg Score for Estimators=10 and CV=10 :",np.average(scores1))
scores3 = cross_val_score(RandomForestClassifier(n_estimators=20),X_train, y_train, cv=10)
print("Avg Score for Estimators=20 and CV=10 :",np.average(scores1))
scores4 = cross_val_score(RandomForestClassifier(n_estimators=30), X_train, y_train, cv=10)
print("Avg Score for Estimators=30 and CV=10 :",np.average(scores1))


# In[78]:


# cross validation scores with balance dataset
score_lr_with_Kfold_balance = cross_val_score(LogisticRegression(solver='liblinear',multi_class='ovr'), X_res, y_res, cv=3)
print(score_lr_with_Kfold_balance)
print("Avg :",np.average(score_lr_with_Kfold_balance))
score_svm_with_Kfold_balance = cross_val_score(SVC(gamma='auto'), X_res, y_res, cv=3)
print(score_svm_with_Kfold_balance)
print("Avg :",np.average(score_svm_with_Kfold_balance))
score_rf_with_Kfold_balance = cross_val_score(RandomForestClassifier(n_estimators=40),X_res, y_res, cv=10)
print(score_rf_with_Kfold_balance)
print("Avg :",np.average(score_rf_with_Kfold_balance))
score_knn_with_Kfold_balance = cross_val_score(KNeighborsClassifier(n_neighbors=6), X_res, y_res, cv=10)
print(score_knn_with_Kfold_balance)
print("Avg :",np.average(score_knn_with_Kfold_balance))
score_xgb_with_Kfold_balance = cross_val_score(XGBClassifier(), X_res, y_res, cv=10)
print(score_xgb_with_Kfold_balance)
print("Avg :",np.average(score_xgb_with_Kfold_balance))


# In[79]:


# With balance dataset, score of RandomForestClassifier is high 
# hence reverified with differnt estimators but n_estimators=40 gives good score
scores1 = cross_val_score(RandomForestClassifier(n_estimators=5),X_res, y_res, cv=10)
print("Avg Score for Estimators=5 and CV=10 :",np.average(scores1))
scores2 = cross_val_score(RandomForestClassifier(n_estimators=10),X_res, y_res, cv=10)
print("Avg Score for Estimators=10 and CV=10 :",np.average(scores1))
scores3 = cross_val_score(RandomForestClassifier(n_estimators=20),X_res, y_res, cv=10)
print("Avg Score for Estimators=20 and CV=10 :",np.average(scores1))
scores4 = cross_val_score(RandomForestClassifier(n_estimators=30),X_res, y_res, cv=10)
print("Avg Score for Estimators=30 and CV=10 :",np.average(scores1))


# In[80]:


# Bar subplots for checking differnce between original, k-folded imbalanced and k-folded balanced data for differnt models
df = pd.DataFrame([['LogisticRegression', 'without_Kfold', score_lr_without_Kfold], 
                   ['LogisticRegression', 'with_Kfold_imbalance', score_lr_with_Kfold_imbalance], 
                   ['LogisticRegression', 'with_Kfold_balance', score_lr_with_Kfold_balance], 
                   ['SVM', 'without_Kfold', score_svm_without_Kfold], 
                   ['SVM', 'with_Kfold_imbalance', score_svm_with_Kfold_imbalance], 
                   ['SVM', 'with_Kfold_balance', score_svm_with_Kfold_balance],
                   ['RandomForest', 'without_Kfold', score_rf_without_Kfold], 
                   ['RandomForest', 'with_Kfold_imbalance', score_rf_with_Kfold_imbalance], 
                   ['RandomForest', 'with_Kfold_balance', score_rf_with_Kfold_balance],
                   ['KNN', 'without_Kfold', score_knn_without_Kfold], 
                   ['KNN', 'with_Kfold_imbalance', score_knn_with_Kfold_imbalance], 
                   ['KNN', 'with_Kfold_balance', score_knn_with_Kfold_balance],
                   ['XGBoost', 'without_Kfold', score_xgb_without_Kfold], 
                   ['XGBoost', 'with_Kfold_imbalance', score_xgb_with_Kfold_imbalance], 
                   ['XGBoost', 'with_Kfold_balance', score_xgb_with_Kfold_balance]], 
                  columns=['Models', 'Processes', 'Cross Validation Scores'])

df = df.explode('Cross Validation Scores')
df['Cross Validation Scores'] = df['Cross Validation Scores'].astype('float') * 100

# plot with seaborn barplot
plt.figure(figsize=(18, 8))
ax = sns.barplot(data=df, x ='Processes', y ='Cross Validation Scores', hue ='Models', ci = None)

for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', padding=3)


# ## Step 5 : Model prediction and evaluation

# ### RandomForestClassifier

# In[81]:


# Fitting balanced data into RandomForestClassifier
RF = RandomForestClassifier(criterion='gini')
RF.fit(X_res, y_res)
# Predicting unseen data with RandomForestClassifier
pred= RF.predict(X_test)
# Check mean_squared_error
print(np.sqrt(mean_squared_error(y_test, pred)))


# In[82]:


from sklearn.metrics import classification_report,plot_confusion_matrix,confusion_matrix,accuracy_score
plot_confusion_matrix(RF,X_test,y_test)


# In[83]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred))


# ### KNeighborsClassifier

# In[84]:


# Fitting balanced data into KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_res, y_res)
# Predicting unseen data with KNeighborsClassifier
pred= knn.predict(X_test)
# Check mean_squared_error
print(np.sqrt(mean_squared_error(y_test, pred)))


# In[85]:


plot_confusion_matrix(knn,X_test,y_test)


# In[86]:


print(classification_report(y_test,pred))


# ### XGBClassifier

# In[87]:


# Fitting balanced data into XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_res, y_res)
# Predicting unseen data with XGBClassifier
pred= xgb.predict(X_test)
# Check mean_squared_error
print(np.sqrt(mean_squared_error(y_test, pred)))


# In[88]:


plot_confusion_matrix(xgb, X_test, y_test)


# In[89]:


print(classification_report(y_test,pred))


# ### DecisionTreeClassifier

# In[90]:


# Fitting balanced data into DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_res, y_res)
# Predicting unseen data with DecisionTreeClassifier
pred= model.predict(X_test)
# Check mean_squared_error
print(np.sqrt(mean_squared_error(y_test, pred)))


# In[91]:


sns.heatmap(confusion_matrix(y_test,pred),annot=True)


# In[92]:


print(classification_report(y_test,pred))


# ### SVM

# In[93]:


# Fitting balanced data into SVC
svm = SVC(decision_function_shape='ovo')
svm.fit(X_res, y_res)
# Predicting unseen data with SVC
pred = svm.predict(X_test)


# In[94]:


plot_confusion_matrix(svm, X_test, y_test)


# In[95]:


print(classification_report(y_test, pred))


# ### SVM RBF

# In[96]:


# Fitting balanced data into SVM RBF
svm_rbf = SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo')
svm_rbf.fit(X_res, y_res)
# Predicting unseen data with SVM RBF
pred = svm_rbf.predict(X_test)


# In[97]:


plot_confusion_matrix(svm_rbf, X_test, y_test)


# In[98]:


accuracy_rbf = svm_rbf.score(X_test, y_test)
accuracy_rbf


# ### QuadraticDiscriminantAnalysis

# In[99]:


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# Fitting balanced data into QuadraticDiscriminantAnalysis
qda = QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0, store_covariance=False, tol=0.0001)
qda.fit(X_res, y_res)
# Predicting unseen data with QuadraticDiscriminantAnalysis
pred = qda.predict(X_test)


# In[100]:


plot_confusion_matrix(qda, X_test, y_test)


# In[101]:


print(classification_report(y_test, pred))


# ### GaussianNB

# In[102]:


from sklearn.naive_bayes import GaussianNB
# Fitting balanced data into GaussianNB
gnb = GaussianNB()
gnb.fit(X_res, y_res)
# Predicting unseen data with GaussianNB
pred = gnb.predict(X_test)


# In[103]:


plot_confusion_matrix(gnb, X_test, y_test)


# In[104]:


print(classification_report(y_test, pred))


# ### LogisticRegression

# In[105]:


# Fitting balanced data into LogisticRegression
lr = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial')
lr.fit(X_res, y_res)
# Predicting unseen data with LogisticRegression
pred = lr.predict(X_test)


# In[106]:


plot_confusion_matrix(lr, X_test, y_test)


# In[107]:


print(classification_report(y_test, pred))


# ## Step 6: Hyper-parameters tuning of an estimator

# In[108]:


from sklearn.model_selection import GridSearchCV


# ### XGBoost

# In[109]:


# Initialising list of paramaters for selection of best params for XGBoost Model
param_grid = {
    "learning_rate": [0.5, 1, 3, 5],
    "reg_lambda": [0, 1, 5, 10, 20]
}


# In[110]:


# Applying param_grid , k_fold as 3 and training the model
# Computations can be run in parallel by using the keyword n_jobs=-1
grid = GridSearchCV(xgb, param_grid, cv=3, n_jobs=-1)
grid.fit(X_res, y_res)


# In[111]:


# Best params for XGBoost Model
grid.best_params_


# In[112]:


# Best score
grid.best_score_


# In[113]:


# Applying Best params to XGBoost Model
xgb = XGBClassifier(colsample_bytree= 1, gamma=0, learning_rate=1, max_depth=3, subsample=0.8, reg_lambda=1)
xgb.fit(X_res, y_res)
pred= xgb.predict(X_test)


# In[114]:


plot_confusion_matrix(xgb, X_test, y_test)


# In[115]:


print(classification_report(y_test,pred))


# ### RandomForestClassifier

# In[116]:


# Initialising list of paramaters for selection of best params for RandomForestClassifier Model
# Applying param_grid , k_fold as 5 and training the model
param_grid={'n_estimators': [40,50,200,300,500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [5,10,15,20,30],
    'criterion' :['gini', 'entropy']
}
gridsearchcv = GridSearchCV(RF, param_grid, cv=5)
gridsearchcv.fit(X_res, y_res)


# In[117]:


gridsearchcv.best_params_


# In[118]:


gridsearchcv.best_score_


# In[119]:


# Applying Best params to RandomForestClassifier Model
RF2= RandomForestClassifier(criterion='gini', max_depth=20, max_features='log2', n_estimators=50)
RF2.fit(X_res, y_res)
pred= RF2.predict(X_test)


# In[120]:


plot_confusion_matrix(RF2, X_test, y_test)


# In[121]:


print(classification_report(y_test, pred))


# ### KNeighborsClassifier

# In[122]:


# Initialising list of paramaters for selection of best params for KNeighborsClassifier Model
# Applying param_grid and training the model
param_grid={'n_neighbors': [1,2,3,4,5,6,7,8]}
gridsearchcv = GridSearchCV(knn, param_grid)
gridsearchcv.fit(X_res, y_res)


# In[123]:


gridsearchcv.best_params_


# In[124]:


# Applying Best params to KNeighborsClassifier Model
knn2 = KNeighborsClassifier(n_neighbors=1)
knn2.fit(X_res, y_res)
pred= knn2.predict(X_test)


# In[125]:


plot_confusion_matrix(knn2, X_test, y_test)


# In[126]:


print(classification_report(y_test, pred))


# ### Appenedix A: Plot Decision Tree

# In[127]:


# Data wrangling
import pandas as pd

# Import comma-separated values (csv) file into DataFrame 
df = pd.read_csv('Maternal Health Risk Data Set.csv')

# Get predictor and target variables
X = df.drop('RiskLevel', axis = 1)
Y = df['RiskLevel']

# Plot Decsion tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
fn = df.columns[0:6]
cn = df["RiskLevel"].unique().tolist()
dataTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
dataTree.fit(X,Y)
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize= (20,10), dpi=300)
tree.plot_tree(dataTree, feature_names= fn, class_names= cn, filled = True)
plt.show()


# ### Appenedix B: Performance evalution after removal of highly correlated feature

# In[128]:


df.drop(columns="SystolicBP", axis=1, inplace=True)
df


# In[129]:


# Splitting data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

# Normal scaling of training dataset
normal = MinMaxScaler()  
X_train_features = normal.fit_transform(X_train)  
X_test_features = normal.transform(X_test)

# Balancing dataset
over_sampler = RandomOverSampler(random_state = 42)
X_res, y_res = over_sampler.fit_resample(X_train, y_train)
y_res.value_counts()


# In[130]:


RF2= RandomForestClassifier(criterion='gini', max_depth=20, max_features='log2', n_estimators=50)
RF2.fit(X_res, y_res)
pred= RF2.predict(X_test)


# In[131]:


plot_confusion_matrix(RF2, X_test, y_test)


# In[132]:


print(classification_report(y_test, pred))

