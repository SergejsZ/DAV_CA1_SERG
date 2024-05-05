
"""
Created on Tue Nov 15 19:07:27 2022

@author: serge
"""

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

#To set figure size
from matplotlib.pyplot import figure

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

def decisionTree(x_train, y_train, x_test, y_test, depthIn):
    modelDecisionTree = DecisionTreeClassifier(max_depth=depthIn)
    modelDecisionTree = modelDecisionTree.fit(x_train,y_train)
    predictions = modelDecisionTree.predict(x_test)
    confusionMatrix = confusion_matrix(y_test, predictions)
    accuracy = (confusionMatrix[0,0]+confusionMatrix[1,1])/len(predictions)
    return accuracy

def neuralNetwork(x_train, y_train, x_test, y_test, layersIn, iterationsIn):
    model = MLPClassifier(hidden_layer_sizes=layersIn,max_iter=iterationsIn)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    confusionMatrix = confusion_matrix(y_test, predictions)
    accuracy = (confusionMatrix[0,0]+confusionMatrix[1,1])/len(predictions)
    return accuracy

#DATA IMPORT########################################################

import os
cwd = os.getcwd()
print(cwd)
os.chdir('C:/Users/serge/OneDrive/Desktop/spyder')

# Read the data from "data.csv" file.
data= pd.read_csv("data.csv")


data.info()
data.head()
data.describe()

#Checking if the data is clean - All Ok if 0
data.isnull().sum() 

# DATA CLEANING ########################################################


# renaming columns to remove spaces that might cause problem later
data.rename(columns = {'race/ethnicity':'race_ethnicity','parental level of education':'parent_education', 
                       'test preparation course':'test_preparation_course', 'math score':'maths_score', 
                       'reading score':'reading_score', 'writing score':'writing_score'}, inplace = True)
data.info()

######### FEATURE ENGINEERING###########################


#Feature Engineering Step 1: Identify Variables    ########################################################

 #   Column                       Non-Null Count  Dtype 
 #---  ------                       --------------  ----- 
 # 0   gender                       1000 non-null   object  Categorical, Explanatory (used to predict)
 # 1   race_ethnicity               1000 non-null   object  Categorical, Explanatory (used to predict)
 # 2   parent_education             1000 non-null   object  Categorical, Explanatory (used to predict)
 # 3   lunch                        1000 non-null   object  Categorical, Explanatory (used to predict)
 # 4   test_preparation_course      1000 non-null   object  Categorical, Explanatory (used to predict)
 # 5   math_score                   1000 non-null   int64   Numetical Data, Response (want to predict)
 # 6   reading_score                1000 non-null   int64   Numetical Data, Explanatory (used to predict)
 # 7   writing_score                1000 non-null   int64   Numetical Data, Explanatory (used to predict)
 
#########Feature Engineering Step 2: Drop certain variables if not required

# Keeping all variables


#########Feature Engineering Step 3: Construct New Variables if required
#   Changeing Continent to numerical

##printing all the unique ethnicities
print(data.race_ethnicity.unique())

##checking how many students are in each ethnicity
data.race_ethnicity.value_counts()

data['ethGroupA']=np.where(data.race_ethnicity =="group A",1,0)
data['ethGroupB']=np.where(data.race_ethnicity =="group B",1,0)
data['ethGroupC']=np.where(data.race_ethnicity =="group C",1,0)
data['ethGroupD']=np.where(data.race_ethnicity =="group D",1,0)


# changing from catigorical to numerical data, male = 1 female = 0
print(data.gender.unique())
data['genderType']=np.where(data.gender =="male",1,0)

# changing from catigorical to numerical data, standard = 1 free/reduced = 0
print(data.lunch.unique())
data['lunchType']=np.where(data.lunch =="standard",1,0)

# changing from catigorical to numerical data, completed = 1 none = 0
print(data.test_preparation_course.unique())
data['test_preparation_course']=np.where(data.test_preparation_course =="completed",1,0)

#Adding parentEducationUpward field with starting from 1 to be the lowest education field
print(data.parent_education.unique())
conditions = [data.parent_education== "high school", 
              data.parent_education== "some high school",
              data.parent_education== "some college",
              data.parent_education=="associate's degree",
              data.parent_education== "bachelor's degree",
              data.parent_education=="master's degree"]
choices = [1, 2, 3, 4, 5, 6]

data['parentEducationUpward']=np.select(conditions, choices)

#########Feature Engineering Step 4: Scale Data
# not needed here

#Produce scatter and correlation plots - Pay particular attention to the Response variable
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.pairplot(data)

#Correlations
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.heatmap(data.corr(), annot=True, cmap = 'Reds')
plt.show()

#Box plot
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
sns.boxplot(data.parentEducationUpward, data.maths_score)
plt.show()

corrVals=data.corr()
#Order of Importance for Predicting Maths Score:
# reading_score	0.8193 As rhe reading score goes up, the maths score goes up
# lunchType	0.3744 The more students had a full meal, the higher the maths score
# genderType	0.2008 Male students are bringing the maths score up
# parentEducationUpward	0.1909 As perent education goes up, the student score goes up
# test_preparation_course 0.1517 students that prepared for the exam, get a higher score
# ethGroupC	-0.1465 students in ethnicity C have a negative value, meaning the score will go down
# ethGroupD	0.1111 studets of ethnicity group D bring the maths score up
# ethGroupB	-0.1063 value is negative, meaning the score will go down
# ethGroupA	-0.0224 value is negative, meaning the score will go down



#########Feature Engineering Step 5: Multicolinearity
#writing score perfectly correlated as reading increases.
data.drop('writing_score', axis = 1, inplace = True)
corrVals=data.corr()

###############REGRESSION MODELLING######################################

#########Regression Modelling - Step 1: Split Data into Train and Test

#Set the Response and the predictor variables

x = data[['test_preparation_course', 'reading_score', 'ethGroupA', 'ethGroupB', 'ethGroupC','ethGroupD','genderType','lunchType','parentEducationUpward']] #pandas dataframe
y = data['maths_score'] #Pandas series

#Splitting the Data Set into Training Data and Test Data
from sklearn.model_selection import train_test_split

#Spliting train 66.7%, test 33.3%.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.333)

#Check size of training and test sets
print(len(y_train)) #667 values
print(len(y_test)) #333 values

#########Regression Modelling - Step 2: Model Selection


from sklearn.linear_model import LinearRegression
model1 = LinearRegression()
model2 = LinearRegression()
model3 = LinearRegression()
model4 = LinearRegression()
model5 = LinearRegression()
model6 = LinearRegression()
model7 = LinearRegression()
model8 = LinearRegression()
model9 = LinearRegression()


#Fit the variables in order of strongest correlation with maths_score and calculate adjusted R squared at each step.

#Model 1 - First adding reading_score to model
model1.fit(x_train[['reading_score']], y_train)
#Show the model parameters
print(model1.coef_)
print(model1.intercept_)
#So maths_score = 7.3652 + 0.85476965*reading_score

#A nicer way to view the coefficients is by placing them in a DataFrame. This can be done with the following statement:
Output = pd.DataFrame(model1.coef_, ['reading score Coeff'], columns = ['Coeff'])
print(Output)


#Generate predictions for the train data
predictions_train = model1.predict(x_train[['reading_score']])

raw_sum_sq_errors = sum((y_train.mean() - y_train)**2)

prediction_sum_sq_errors = sum((predictions_train - y_train)**2)

Rsquared1 = 1-prediction_sum_sq_errors/raw_sum_sq_errors

N=len(y_train) #Number of values in training set 666
p=1 # one predictor used
Rsquared_adj1 = 1 - (1-Rsquared1)*(N-1)/(N-p-1)
print("Rsquared Regression Model with reading score: "+str(Rsquared1))
print("Rsquared Adjusted Regression Model with reading score: "+str(Rsquared_adj1))

#RSquaredAdj = 0.6757 

######Model 2 - Next adding the lunchType variable
model2.fit(x_train[['reading_score', 'lunchType']], y_train)
#Show the model parameters
print(model2.coef_)
print(model2.intercept_)
#So maths_score = 4.47823036 + 0.8155283*reading_score

#A nicer way to view the coefficients is by placing them in a DataFrame. This can be done with the following statement:
Output = pd.DataFrame(model2.coef_, ['reading score Coeff', 'lunchType Coeff'], columns = ['Coeff'])
print(Output)

#Generate predictions for the train data
predictions_train = model2.predict(x_train[['reading_score', 'lunchType']])

#Raw sum of squares of errors is based on the mean of the y values without having any predictors to help.
raw_sum_sq_errors = sum((y_train.mean() - y_train)**2)

#Calculate sum of squares for prediction errors.
prediction_sum_sq_errors = sum((predictions_train - y_train)**2)

Rsquared2 = 1-prediction_sum_sq_errors/raw_sum_sq_errors

N=len(y_train) #Number of values in training set 666
p=2 #Two predictors used
Rsquared_adj2 = 1 - (1-Rsquared2)*(N-1)/(N-p-1)
print("Rsquared Regression Model with reading_score & lunchType: "+str(Rsquared2))
print("Rsquared Adjusted Regression Model with reading_score & lunchType: "+str(Rsquared_adj2))

#RSquaredAdj = 0.6934 - Model two is a  better than the first one!!


####Model 3 - Next Adding the gender
model3.fit(x_train[['reading_score', 'lunchType', 'genderType']], y_train)
#Show the model parameters
print(model3.coef_)
print(model3.intercept_)
#So maths_score = 10.983038*genderType +  4.484209*lunchType + 0.880753*reading_score

#A nicer way to view the coefficients is by placing them in a DataFrame. This can be done with the following statement:
Output = pd.DataFrame(model3.coef_, ['reading score Coeff', 'lunchType Coeff', 'genderType Coeff'], columns = ['Coeff'])
print(Output)

#Generate predictions for the train data
predictions_train = model3.predict(x_train[['reading_score', 'lunchType', 'genderType']])

#Raw sum of squares of errors is based on the mean of the y values without having any predictors to help.
raw_sum_sq_errors = sum((y_train.mean() - y_train)**2)

#Calculate sum of squares for prediction errors.
prediction_sum_sq_errors = sum((predictions_train - y_train)**2)

Rsquared3 = 1-prediction_sum_sq_errors/raw_sum_sq_errors

N=len(y_train) #Number of values in training set 666
p=3 #Three predictors used
Rsquared_adj3 = 1 - (1-Rsquared3)*(N-1)/(N-p-1)
print("Rsquared Regression Model with reading_score & writing_score & genderType: "+str(Rsquared3))
print("Rsquared Adjusted Regression Model with reading_score & writing_score & genderType: "+str(Rsquared_adj3))

#RSquaredAdj = 0.8199 - Model three is a way better than the first & second model!!

#Model 4 - Next adding the parent education variable
model4.fit(x_train[['reading_score', 'lunchType', 'genderType', 'parentEducationUpward']], y_train)
#Show the model parameters
print(model4.coef_)
print(model4.intercept_)
#So maths_score = 10.920737 +  4.495738 + 0.407710 +0.871758*reading_score

#A nicer way to view the coefficients is by placing them in a DataFrame. This can be done with the following statement:
Output = pd.DataFrame(model4.coef_, ['reading score Coeff', 'lunchType Coeff', 'genderType Coeff', 'parentEducationUpward Coeff'], columns = ['Coeff'])
print(Output)

#Generate predictions for the training data
predictions_train = model4.predict(x_train[['reading_score', 'lunchType', 'genderType', 'parentEducationUpward']])

prediction_sum_sq_errors = sum((predictions_train - y_train)**2)

Rsquared4 = 1-prediction_sum_sq_errors/raw_sum_sq_errors

N=len(y_train) #Number of values in training set 666
p=4 #Four predictors used
Rsquared_adj4 = 1 - (1-Rsquared4)*(N-1)/(N-p-1)
print("Rsquared Regression Model with reading_score & writing_score & genderType & parentEducationUpward: "+str(Rsquared4))
print("Rsquared Adjusted Regression Model with reading_score & writing_score & genderType & parentEducationUpward: "+str(Rsquared_adj4))

#RSquared Adj Value = 0.8212 - no big difference between model 3 and 4 but there is still some improvement in model 4

#Model 5 - Next adding test_preparation_course variable
model5.fit(x_train[['reading_score', 'lunchType', 'genderType', 'parentEducationUpward', 'test_preparation_course']], y_train)
#Show the model parameters
print(model5.coef_)
print(model5.intercept_)
#So maths_score = -2.460108 + 11.048219 +  4.392870 + 0.403052 +0.893305*reading_score

#A nicer way to view the coefficients is by placing them in a DataFrame. This can be done with the following statement:
Output = pd.DataFrame(model5.coef_, ['reading score Coeff', 'lunchType Coeff','parentEducationUpward', 'genderType Coeff', 'test_preparation_course Coeff'], columns = ['Coeff'])
print(Output)

#Generate predictions for the training data
predictions_train = model5.predict(x_train[['reading_score', 'lunchType', 'genderType', 'parentEducationUpward', 'test_preparation_course']])

prediction_sum_sq_errors = sum((predictions_train - y_train)**2)

Rsquared5 = 1-prediction_sum_sq_errors/raw_sum_sq_errors

N=len(y_train) #Number of values in training set 666
p=5 #Five predictors used
Rsquared_adj5 = 1 - (1-Rsquared5)*(N-1)/(N-p-1)
print("Rsquared Regression Model with reading_score & writing_score & genderType & parentEducationUpward & test_preparation_course: "+str(Rsquared5))
print("Rsquared Adjusted Regression Model with reading_score & writing_score & genderType & parentEducationUpward & test_preparation_course: "+str(Rsquared_adj5))

#RSquared Adj Value = 0.8264 - Model 5 is just a little better than Model 4

#Model 6 - Next adding ethGroupC variable
model6.fit(x_train[['reading_score', 'lunchType', 'genderType', 'parentEducationUpward', 'test_preparation_course','ethGroupC']], y_train)
#Show the model parameters
print(model6.coef_)
print(model6.intercept_)
#So maths_score = -1.469147 -2.433047 + 10.948350 +  4.436425 + 0.425747 +0.884817*reading_score

#A nicer way to view the coefficients is by placing them in a DataFrame. This can be done with the following statement:
Output = pd.DataFrame(model6.coef_, ['reading score Coeff', 'lunchType Coeff', 'genderType Coeff','parentEducationUpward Coeff', 'test_preparation_course Coeff','ethGroupC Coeff'], columns = ['Coeff'])
print(Output)

#Generate predictions for the training data
predictions_train = model6.predict(x_train[['reading_score', 'lunchType', 'genderType', 'parentEducationUpward', 'test_preparation_course', 'ethGroupC']])

prediction_sum_sq_errors = sum((predictions_train - y_train)**2)

Rsquared6 = 1-prediction_sum_sq_errors/raw_sum_sq_errors

N=len(y_train) #Number of values in training set 666
p=6 #Six predictors used
Rsquared_adj6 = 1 - (1-Rsquared6)*(N-1)/(N-p-1)
print("Rsquared Regression Model with reading_score & writing_score & genderType & parentEducationUpward & test_preparation_course & ethGroupC: "+str(Rsquared6))
print("Rsquared Adjusted Regression Model with reading_score & writing_score & genderType & parentEducationUpward & test_preparation_course & ethGroupC: "+str(Rsquared_adj6))

#RSquared Adj Value = 0.8281 - Model 6 is just a little better than the previous model, almost no difference 

#Model 7 - Next adding ethGroupD variable
model7.fit(x_train[['reading_score', 'lunchType', 'genderType', 'parentEducationUpward', 'test_preparation_course','ethGroupC','ethGroupD']], y_train)
#Show the model parameters
print(model7.coef_)
print(model7.intercept_)
#So maths_score = -0.656848 -1.709073 -2.442604 + 10.988586 +  4.443137 + 0.417767 +0.886770*reading_score

#A nicer way to view the coefficients is by placing them in a DataFrame. This can be done with the following statement:
Output = pd.DataFrame(model7.coef_, ['reading score Coeff', 'lunchType Coeff', 'genderType Coeff', 'parentEducationUpward Coeff', 'test_preparation_course Coeff', 'ethGroupC Coeff', 'ethGroupD Coeff'], columns = ['Coeff'])
print(Output)

#Generate predictions for the training data
predictions_train = model7.predict(x_train[['reading_score', 'lunchType', 'genderType', 'parentEducationUpward','test_preparation_course', 'ethGroupC', 'ethGroupD']])

prediction_sum_sq_errors = sum((predictions_train - y_train)**2)

Rsquared7 = 1-prediction_sum_sq_errors/raw_sum_sq_errors

N=len(y_train) #Number of values in training set 666
p=7 #Seven predictors used
Rsquared_adj7 = 1 - (1-Rsquared7)*(N-1)/(N-p-1)
print("Rsquared Regression Model with reading_score & writing_score & genderType & parentEducationUpward & test_preparation_course & ethGroupC & ethGroupD: "+str(Rsquared7))
print("Rsquared Adjusted Regression Model with reading_score & writing_score & genderType & parentEducationUpward & test_preparation_course & ethGroupC & ethGroupD: "+str(Rsquared_adj7))

#RSquared Adj Value = 0.8282 - Model 7 is just a little better than the previous model, almost no difference 

#Model 8 - Next adding ethGroupB variable
model8.fit(x_train[['reading_score', 'lunchType', 'genderType', 'parentEducationUpward', 'test_preparation_course','ethGroupC','ethGroupD','ethGroupB']], y_train)
#Show the model parameters
print(model8.coef_)
print(model8.intercept_)
#So maths_score = - 3.739717 - 2.504966 - 2.119256 - 3.678382 + 10.740618 + 4.637761 + 0.484044 + 0.870961*reading_score

#A nicer way to view the coefficients is by placing them in a DataFrame. This can be done with the following statement:
Output = pd.DataFrame(model8.coef_, ['reading score Coeff', 'lunchType Coeff', 'genderType Coeff', 'parentEducationUpward Coeff', 'test_preparation_course Coeff', 'ethGroupC Coeff', 'ethGroupD Coeff', 'ethGroupB Coeff'], columns = ['Coeff'])
print(Output)

#Generate predictions for the training data
predictions_train = model8.predict(x_train[['reading_score', 'lunchType', 'genderType', 'parentEducationUpward','test_preparation_course', 'ethGroupC', 'ethGroupD', 'ethGroupB']])

prediction_sum_sq_errors = sum((predictions_train - y_train)**2)

Rsquared8 = 1-prediction_sum_sq_errors/raw_sum_sq_errors

N=len(y_train) #Number of values in training set 666
p=8 #Eight predictors used
Rsquared_adj8 = 1 - (1-Rsquared8)*(N-1)/(N-p-1)
print("Rsquared Regression Model with reading_score & writing_score & genderType & parentEducationUpward & test_preparation_course & ethGroupC & ethGroupD & ethGroupB: "+str(Rsquared8))
print("Rsquared Adjusted Regression Model with reading_score & writing_score & genderType & parentEducationUpward & test_preparation_course & ethGroupC & ethGroupD & ethGroupB: "+str(Rsquared_adj8))

#RSquared Adj Value = 0.8341 - Model 8 is just little better than the previous model

#Model 9 - Next adding ethGroupA variable
model9.fit(x_train[['reading_score', 'lunchType', 'genderType', 'parentEducationUpward', 'test_preparation_course','ethGroupC','ethGroupD','ethGroupB','ethGroupA']], y_train)
#Show the model parameters
print(model9.coef_)
print(model9.intercept_)
#So maths_score = - 4.211698 -5.210687 - 3.921240 - 5.145474 - 2.188980 + 10.650269 + 4.725176 + 0.526849 + 0.863346*reading_score

#A nicer way to view the coefficients is by placing them in a DataFrame. This can be done with the following statement:
Output = pd.DataFrame(model9.coef_, ['reading score Coeff', 'lunchType Coeff', 'genderType Coeff', 'parentEducationUpward Coeff', 'test_preparation_course Coeff', 'ethGroupC Coeff', 'ethGroupD Coeff', 'ethGroupB Coeff', 'ethGroupA Coeff'], columns = ['Coeff'])
print(Output)

#Generate predictions for the training data
predictions_train = model9.predict(x_train[['reading_score', 'lunchType', 'genderType', 'parentEducationUpward','test_preparation_course', 'ethGroupC', 'ethGroupD', 'ethGroupB', 'ethGroupA']])

prediction_sum_sq_errors = sum((predictions_train - y_train)**2)

Rsquared9 = 1-prediction_sum_sq_errors/raw_sum_sq_errors

N=len(y_train) #Number of values in training set 666
p=9 #Nine predictors used
Rsquared_adj9 = 1 - (1-Rsquared9)*(N-1)/(N-p-1)
print("Rsquared Regression Model with reading_score & writing_score & genderType & parentEducationUpward & test_preparation_course & ethGroupC & ethGroupD & ethGroupB & ethGroupA: "+str(Rsquared9))
print("Rsquared Adjusted Regression Model with reading_score & writing_score & genderType & parentEducationUpward & test_preparation_course & ethGroupC & ethGroupD & ethGroupB & ethGroupA: "+str(Rsquared_adj9))

#RSquared Adj Value = 0.8373 - Model 9 is just little better than the previous model

#DECISION -----MODEL3

#Exact Model
#maths_score = -2.859395726074652 + reading_score*0.880753 + lunchType*4.484209 + genderType*10.983038

predictions_test = model3.predict(x_test[['reading_score', 'lunchType', 'genderType']])

Prediction_test_MAE = sum(abs(predictions_test - y_test))/len(y_test)
Prediction_test_MAPE = sum(abs((predictions_test - y_test)/y_test))/len(y_test)
Prediction_test_RMSE = (sum((predictions_test - y_test)**2)/len(y_test))**0.5

print(Prediction_test_MAE) #5.12. So error in prediction for the test maths_score is 5.12%. 
print(Prediction_test_MAPE)  # 8.7% Error on average, which is very high, i would use this prediction to know the requirements for a succefull exam, o would not recommend this prediction to anyone.
print(Prediction_test_RMSE)


figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(y_test, predictions_test - y_test)
plt.title("Errors v Actual Test Values")
plt.xlabel("Actual values")
plt.ylabel("Error Values")
plt.show()

figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(x_test['reading_score'], predictions_test - y_test)
plt.title("Errors v Actual Test Values")
plt.xlabel("Actual values")
plt.ylabel("Error Values")
plt.show()

# Summary : most of the points are in the middle, there are a few high outliers that i would remove to get a better prediction



#First split training - test 60-40
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4)

#Now split test set into  validation - test  equally
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size = 0.5)


############ Apply Models

#Check decision tree
print(decisionTree(x_train, y_train, x_test, y_test,6))
#Check neural network
print(neuralNetwork(x_train, y_train, x_test, y_test,(13,13,13),500))


#Apply decision tree using depths from 1 to 30
accuracyDecisionTrees=[]
for dp in range(1,31,1):
    accuracyDecisionTrees.append(decisionTree(x_train, y_train, x_val, y_val,dp))

optimalAccuracyDT = max(accuracyDecisionTrees)
optimalDepthDT=accuracyDecisionTrees.index(optimalAccuracyDT)+1
optimalAccuracyDT
optimalDepthDT

#Apply neural networks from 1 to 10 layers with 5,10,15 and 20 nodes.
accuracyNeuralNetworks=[]
ind=0
for ly in range(1,11,1):
    for nd in range(5,21,5):
        print("ly: "+str(ly)+" nd: "+str(nd)+" Index: "+str(ind))
        layers=tuple([nd for i in range(ly)])
        print(layers)
        acc=neuralNetwork(x_train, y_train, x_val, y_val,layers,500)
        print(acc)
        accuracyNeuralNetworks.append(acc)
        ind=ind+1

optimalAccuracyNN = max(accuracyNeuralNetworks)
optimalIndexNN=accuracyNeuralNetworks.index(optimalAccuracyNN)
optimalAccuracyNN
optimalIndexNN

#Optimal network is (20, 20, 20, 20)

#optimal decision tree 0.8707
#optimal neural network 0.8651

#best model is neural network (10,10,10,10,10)



############Apply Best Model to TEST data set

model = MLPClassifier(hidden_layer_sizes=(10,10,10,10,10),max_iter=500)

#Select the model using the training data
model.fit(x_train, y_train)

#Find the predicted values from the test set
predictions = model.predict(x_test)

#Calculate performance metrics Accuracy, Error Rate, Precision and Recall from the confusion matrix

confusionMatrix = confusion_matrix(y_test, predictions)
print(confusionMatrix)

#Check numbers
numberSurvivedTest = y_test.value_counts()

accuracy = (confusionMatrix[0,0]+confusionMatrix[1,1])/len(predictions)
errorRate = 1- accuracy
precision = (confusionMatrix[1,1])/(confusionMatrix[1,1] + confusionMatrix[0,1])
recall = (confusionMatrix[1,1])/(confusionMatrix[1,1] + confusionMatrix[1,0])
print("Accuracy: " + str(accuracy))
print("Error Rate: " + str(errorRate))
print("Precision: " + str(precision))
print("Recall: " + str(recall))

# Accuracy: 0.7932960893854749
# Error Rate: 0.2067039106145251
# Precision: 0.7575757575757576
# Recall: 0.704225352112676

