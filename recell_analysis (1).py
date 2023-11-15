# importing data mainpulation libraries
import pandas as pd
import numpy as np

#importing data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

#importing model bulding libraries
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import randint, zscore

#importing the dataset
data = '/content/drive/MyDrive/PORTFOLIO PROJECTS/Linear regression/used phone price prediction/data.csv'
data = pd.read_csv(data)
data.head()

#making a copy of the dataset
df = data.copy()
df.head()

#getting the shape of the data
df.shape

#checking missing values
df.isnull().sum()

#checking for duplicates
df.duplicated().sum()

#checking the data types of the columns
dtypes = df.dtypes
print('dtypes: \n', dtypes)

#getting the statistical summary of the numeric columns
df.describe().T

#getting the statistical summary of the object columns
df.describe(include = 'object').T

### Univariet analysis
df.head()

#checking the distribution of the target variable
# Plot the distribution of the 'normalized_new_price'
plt.figure(figsize=(7, 4))
plt.hist(df['normalized_new_price'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Normalized New Price')
plt.xlabel('Normalized New Price')
plt.ylabel('Frequency')
plt.show()

def histogram_boxplot(data, feature, figsize=(7, 4), kde=False, bins=None, edgecolor = 'black'):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (15,10))
    kde: whether to show the density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  # creating the 2 subplots
    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )  # boxplot will be created and a triangle will indicate the mean value of the column
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2
    )  # For histogram
    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram

#using the function above to plot histograms
histogram_boxplot(df, 'screen_size')

df.columns

#histogram of main_camera_mp
histogram_boxplot(df, 'main_camera_mp')

#histogram and box plot of selfie camera mp
histogram_boxplot(df, 'selfie_camera_mp')

#Histogram and boxplot of internal memory
histogram_boxplot(df, 'int_memory')

#plotting the distributions of 4g and 5g phones
fig, ax = plt.subplots(1, 2, figsize=(7, 4))

# Plot the distribution of 4G support
sns.countplot(x='4g', data=df, ax=ax[0])
ax[0].set_title('Distribution of 4G Support')

# Plot the distribution of 5G support
sns.countplot(x='5g', data=df, ax=ax[1])
ax[1].set_title('Distribution of 5G Support')

plt.tight_layout()
plt.show()


#histogram and boxplot of ram
histogram_boxplot(df, 'ram')


#distribution of brands of the phones
plt.figure(figsize=(10, 6))
data['brand_name'].value_counts().plot(kind='bar', color='red', edgecolor='black')
plt.title('Distribution of Phone Brands')
plt.xticks(rotation = 70, ha = 'right')
plt.xlabel('Brand Name')
plt.ylabel('Frequency')
plt.show()


#histogram and boxplot showing battery
histogram_boxplot(df, 'battery')



### Bivariet analysis


df.head()

# Create a scatter plot of RAM vs. normalized used price
plt.figure(figsize=(7, 4))
sns.scatterplot(x='ram', y='normalized_used_price', data=df)
plt.title('RAM vs. Normalized used Price')
plt.xlabel('RAM (GB)')
plt.ylabel('Normalized used Price')
plt.show()


# Create a scatter plot of int_memory vs. normalized used price
plt.figure(figsize=(7, 4))
sns.scatterplot(x='int_memory', y='normalized_used_price', data=df)
plt.title('Internal Memory vs. Normalized used Price')
plt.xlabel('Internal Memory (GB)')
plt.ylabel('Normalized used Price')
plt.show()




# Create a scatter plot of battery vs. normalized used price
plt.figure(figsize=(7, 4))
sns.scatterplot(x='battery', y='normalized_used_price', data=df)
plt.title('battery vs. Normalized used Price')
plt.xlabel('battery')
plt.ylabel('Normalized used Price')
plt.show()

# Create a scatter plot of selfie_camera_mp vs. normalized used price
plt.figure(figsize=(7, 4))
sns.scatterplot(x='selfie_camera_mp', y='normalized_used_price', data=df)
plt.title('selfie camera vs. Normalized used Price')
plt.xlabel('selfie camera (MP)')
plt.ylabel('Normalized used Price')
plt.show()

# Create a scatter plot of release year vs. normalized used price
plt.figure(figsize=(7, 4))
sns.scatterplot(x='release_year', y='normalized_used_price', data=df)
plt.title('release year vs. Normalized used Price')
plt.xlabel('year of release')
plt.ylabel('Normalized used Price')
plt.show()

# Create a scatter plot of weight vs. normalized used price
plt.figure(figsize=(7, 4))
sns.scatterplot(x='weight', y='normalized_used_price', data=df)
plt.title('weight vs. Normalized used Price')
plt.xlabel('weight')
plt.ylabel('Normalized used Price')
plt.show()

# Create a scatter plot of days used vs. normalized used price
plt.figure(figsize=(7, 4))
sns.scatterplot(x='days_used', y='normalized_used_price', data=df)
plt.title('number of days used vs. Normalized used Price')
plt.xlabel('number of days used')
plt.ylabel('Normalized used Price')
plt.show()

# Create a scatter plot of days used vs. normalized used price
plt.figure(figsize=(7, 4))
sns.barplot(x='os', y='normalized_used_price',hue= '4g', data=df)
plt.title('operating system vs. Normalized used Price')
plt.xlabel('os')
plt.ylabel('Normalized used Price')
plt.show()

# Create a scatter plot of days used vs. normalized used price
plt.figure(figsize=(7, 4))
sns.scatterplot(x='brand_name', y='normalized_used_price',data=df)
plt.title('brand name vs. Normalized used Price')
plt.xticks(rotation = 90, ha = 'left')
plt.xlabel('brand name')
plt.ylabel('Normalized used Price')
plt.show()

#creating a new feature for the number of years a phone has existed before 2021
df['years_existed_after_release'] = 2021 - df.release_year
print(df.head())

#dropping the release year and days used columns column
df = df.drop(['release_year', "days_used", 'brand_name'], axis = 1)
df.head()

#initializing the simple imputer
imputer = SimpleImputer(strategy = 'median')

#looping through the data and inputing missing values
for col in df.select_dtypes(include = 'number').columns:
  df[[col]] = imputer.fit_transform(df[[col]])

#checking for missing values
df.isnull().sum()

#getting the numeric columns in the dataset
numeric_df = df.select_dtypes(include = 'number').columns

#getting the zscore of the colums
z_scores = df[numeric_df].apply(zscore)
#getting the absolute values of the zscores
abs_zscore = z_scores.abs()

#setting the threshold
threshold = 3

#removing the outliers
non_outliers = (abs_zscore < threshold).all(axis = 1)

# Concatenate the non-outlier rows with the original DataFrame
df_clean = pd.concat([df[non_outliers], df[~non_outliers]])

# Display the cleaned DataFrame
print(df_clean.head())

#initializing the label encoder
le = LabelEncoder()
#label encoding the 4g and 5g columns
df['4g'] = le.fit_transform(df['4g'])

df.head()
#label encoding the 4g and 5g columns
df['5g'] = le.fit_transform(df['5g'])

df.head()

#getting dummy variables
df = pd.get_dummies(df, columns = ['os'], drop_first = True)
df.head()

# creating the dependent and independent variables
X = df.drop(['normalized_used_price'], axis = 1)
y = df.pop('normalized_used_price')

X.head()

#Splitting data into train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state= 1)

#checking the shape of training and test set
print('*' * 50)
print("Training Set Shape", X_train.shape)
print('*' * 50)
print("Testing set shape: ", X_test.shape)

#scaling the dataset
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)

X_train_scaled.head()


### Model Building
#initializing the linear regressor
lin = LinearRegression()

#creating a copy of data
X_train_ln = X_train_scaled.copy()
X_test_ln = X_test_scaled.copy()
y_train_ln = y_train.copy()
y_test_ln = y_test.copy()

# fitting the data on the model
lin.fit(X_train_ln, y_train_ln)

#making predictions on the data
lin_pred = lin.predict(X_test_ln)

#checking the rmse
mse = mean_squared_error(y_test_ln, lin_pred)
ln_report = np.sqrt(mse)
print('The rmse is: ', ln_report)

#initializing the lasso regression
lass = LassoCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0], cv=5)

#Creating a copy of data to use on the lasso regression model
X_train_la = X_train_scaled.copy()
X_test_la = X_test_scaled.copy()
y_train_la = y_train.copy()
y_test_la = y_test.copy()

#fitting the model
lasso_reg = lass.fit(X_train_la, y_train_la)
#predicting values for testing set
predicted_vals2 = lasso_reg.predict(X_test_la)

#getting the mse
lasso_mse = mean_squared_error(y_test_la, predicted_vals2)
lasso_rmse = np.sqrt(lasso_mse)
print("The rmse of lasso regression is: ",lasso_rmse)

#getting the best alphas and coefficients
best_alpha = lasso_reg.alpha_
print("The best alpha value is: ",best_alpha)

#using ridge regression
ridge_cv = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
#Creating a copy of data to use on the ridge regression model
X_train_ri = X_train_scaled.copy()
X_test_ri = X_test_scaled.copy()
y_train_ri = y_train.copy()
y_test_ri = y_test.copy()

#fitting the data
ridge_regression = ridge_cv.fit(X_train_ri, y_train_ri)
#predicting values for test set
predicted_vals3 = ridge_regression.predict(X_test_ri)

#getting the mse
ridge_mse = mean_squared_error(y_test_ri, predicted_vals3)
ridge_rmse = np.sqrt(ridge_mse)
print('The rmse of ridge regression is: ',ridge_rmse)

#getting the best alphas and coefficients
best_alpha_ridge = ridge_regression.alpha_
print("The best alpha value is: ",best_alpha_ridge)

#using random forest regressor
parameters = {
    'n_estimators': randint(100,1000), #number of trees,
    'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider at every split.
    'max_depth': randint(0,100),# maximum number of levels for each decision tree
    'min_samples_split': randint(2, 11),  # Minimum number of data points placed in a node before the node is split.
    'min_samples_leaf': randint(1, 11),  # Minimum number of data points allowed in a leaf node.
    'bootstrap': [True, False]  # Method of selecting samples for training each tree.
}

#creating an insatnce of the random forest regressor
rf = RandomForestRegressor()

#Instatiating randmo search cv
random_search = RandomizedSearchCV(estimator = rf, param_distributions= parameters,
                                    n_iter=100,cv = 5, n_jobs=-1,verbose= True)

#making a copy of the data
X_train_rf = X_train_scaled.copy()
X_test_rf = X_test_scaled.copy()
y_train_rf = y_train.copy()
y_test_rf = y_test.copy()

#fitting the model to the data
random_search.fit(X_train_rf, y_train_rf)

#getting the best parameters
best_params = random_search.best_params_
print("Best Parameters: ", best_params)

#getting the best model
best_model = random_search.best_estimator_

#predicting on the test set
predicted_rf = best_model.predict(X_test_rf)
#scoring the model
mse = mean_squared_error(y_test_rf, predicted_rf)
rf_result = np.sqrt(mse)
print('accuracy of the model is: ', rf_result)
