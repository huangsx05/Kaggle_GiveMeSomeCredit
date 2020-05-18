import pandas as pd

# import data
filename = 'cs-trainingV1.csv'
dataset = pd.read_csv(filename, index_col=u'id') 
dataset.info()

# rename the columns
names = {'SeriousDlqin2yrs':'target', 
		'RevolvingUtilizationOfUnsecuredLines':'unsecuredRatio', 
		'NumberOfTime30-59DaysPastDueNotWorse':'30to59Late', 
		'DebtRatio':'debtRatio', 
		'MonthlyIncome':'income', 
		'NumberOfOpenCreditLinesAndLoans':'nCredit', 
		'NumberOfTimes90DaysLate':'90Late', 
		'NumberRealEstateLoansOrLines':'nRealLoan', 
		'NumberOfTime60-89DaysPastDueNotWorse':'60to89Late', 
		'NumberOfDependents':'nDepend'}
dataset.rename(columns=names, inplace=True)

# process missing values - replace with median
dataset['income'] = dataset['income'].fillna(dataset['income'].median())
dataset['nDepend'] = dataset['nDepend'].fillna(dataset['nDepend'].median())
print('dataset after missing values are replaced: \n')
dataset.info() # 确认没有缺失值（non-null)

# split X and y
dataset.data = dataset.loc[:,'unsecuredRatio' : ]
dataset.target = dataset.loc[:,'target']
print('X: \n', dataset.data.head())
print('y:\n', dataset.target.head())

# split train / test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, random_state=0)

# make pipeline
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
pipe = Pipeline([('classifier', RandomForestClassifier())])

# parameter grid
param_grid = {'classifier__n_estimators': [100]}

# grid search
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best params:\n{}\n".format(grid.best_params_))
print("Best cross-validation score: {:.2f}".format(grid.best_score_))

# test score
print("Test-set score: {:.2f}".format(grid.score(X_test, y_test)))

### ------ Result Submission ------ ###
# import data
filename_test = 'cs-test.csv'
dataset_test = pd.read_csv(filename_test, index_col=u'id') 

# rename the columns
dataset_test.rename(columns=names, inplace=True)
dataset_test.info()

# process missing values - replace with median
dataset_test['income'] = dataset_test['income'].fillna(dataset_test['income'].median())
dataset_test['nDepend'] = dataset_test['nDepend'].fillna(dataset_test['nDepend'].median())

# extract X
dataset_test.data = dataset_test.loc[:,'unsecuredRatio' : ]
dataset_test.data.info()
print('X: \n', dataset_test.data.head())

# test score
y_predictProba = grid.predict_proba(dataset_test.data)
print(y_predictProba)
print(y_predictProba.shape)


# write result to csv file
outputpath='results.csv'
y_dataFrame = pd.DataFrame(y_predictProba)
#np.savetxt(outputpath, y_predictProba, delimiter=",") # for numpy array
y_dataFrame.to_csv(outputpath, sep=',', index=True, header=False) # for DataFrame

