# ALLstate_severity_prediction


All state insurance severity prediction
Ref: https://www.kaggle.com/c/allstate-claims-severity/

Table of Content :
Business Problem
ML formulation
Understanding the Data
Existing approaches / Literature Survey
First cut solution
EDA
Feature Engineering
Understanding Autoencoders
Modeling different Machine Learning architectures/algorithms
Results
Future Work
Profile
References
Business Problem:

1.1 Overview :
This problem statement is from the Kaggle recruitment challenge, by Allstate Insurance. Allstate is an insurance services company in the USA, which provides insurance to over 16 million households in the USA. The company wants to reduce the complexity of the insurance claiming process and make it a worry-free experience for the customers by automating the predictions of claims severity.

1.2 Main Objective :
The Allstate Insurance company wants to reduce the time taking process and make it easier for the people who need insurance cover to claim it much easier.
So in order to reduce the complexity, It has given a dataset to use machine learning algorithms to predict the costs and hence the severity of the claims accurately. This would help the people claiming the insurance to better cope with the severity rather than dealing with the complex process of submitting the papers and dealing with the insurance agents.
2. ML formulation
The insurance claims can depend upon several factors, and with help of those factors ( features ), we can build a machine learning model to predict the loss amount / the severity of the claims. This would be a unidimensional regression problem as we have to only predict the costs for each given set of features.
3. Understanding the Data
First after obtaining the data from kaggle, importing the data and checking its dimensions

We see that the Train data dimensions are : (188318, 132) and Test data dimensions are : (125546, 131)
This indicates, there are 131 features. Also, there is an identification number of train and test data points.
All the features are anonymized i.e categorical features are given cat1, cat2, etc naming, and continuous features are given cont1, cont2, etc naming.
There are 14 continuous features and 116 categorical features.



4. First cut solution
With the help of the above research done from kaggle and other sources, I have come to know about different aspects of the data and also the problems which are being faced in the data either from the MAE. And also the high amount of categorical and continuous variables.
EDA:
Plot the correlation plot between each of the variables that are highly correlated with each other. The distribution of the loss is also to be observed and the log loss of the distribution is also plotted. As every reference considers log-loss transformation to train the models (because log loss generates a QQ plot which is almost parallel to the normal distribution ) the skewness of all the variables is to be calculated and transformed with box cox transformation and plotting their distribution in QQ -plots. The use of DABL plots, to plot all the variables, helps in checking the distribution of every variable with respect to the target variables and we can also obtain the top features directly.
Feature Engineering :
The most important features are obtained by using Random Forest regressor and constructing an autoencoder to create additional features on top and the skewed features are transformed for box-cox transformation to make the non-linear distribution normal. And for certain models ( for stacking ) highly correlated features, and least important features are removed and checked for the score, as some of the solutions suggested.
Models:
As given in all the almost references, Initially I had to build a random forest regressor and observe its score. Then build multiple models with XG boost, Light GBM, and also gradient boosting model from sklearn as it supports MAE. And also deploy a neural net model like this neural net stack all of them, and experiment with making additional XGboost models on top of the well-performing initial models, to get even better predictions. But one of the most crucial parts is using fair_obj for Xgboost and other related models to get around the MAE problem as discussed in the first place solution. And use k fold predictions for each of the best performing XGboost models. Although I would like to split the models according to different feature engineerings as mentioned above and experiment with them. Also, I thought of experimenting with CAT Boosting and giving only categorical features to it and seeing the performance.

5. EDA (Exploratory data analysis)
With the help of various plots, the analysis has been done on the data, initially, I have used a correlation plot on the continuous variables to analyze the correlation amongst each feature in the dataset.




we see that the distribution of both the train and test data is almost similar as they overlap on each other and hence while training the models we will have relatively no issues because of distributions of train and test datasets.
Conclusions from EDA are :
-> The distribution of loss (target variable ) is highly skewed, hence transforming it with log(x) helps in converting the target variable into a normal distribution.
-> There are certain variables in the data both in categorical and continuous features which are very highly intercorrelated amongst each corresponding class of variables (cat and cont)
->There is a small number of outliers in the data, but due to the vast amount of data, they are mostly negligible. But there would be some issues while building distance-based models like K-NN, as due to long distance from clusters, the model would be affected.
->There aren't any distribution differences between the train and test data. This indicates that while training our model in train data, there wouldn't be problems of overfitting.
->There are a lot of features and there are some features in cat116, cat110, etc which have very high labels which might cause dimensionality problems.

6. Feature Engineering
Because of the problems which we might face with the high dimensions and also the high number of labels in some of the categorical variables, I have calculated the top 30 features using RandomForestRegressor,


Here we see that there is very high feature importance to the features of CAT80, CAT79, and other features which are observed in the above graph. An important note to be observed is that almost all the categorical features present in the top 30 features have labels less than 25. That represents that the very high labeled categories like cat 116, cat 110, etc, because of their high dimensionality don't have high importance
Along with the top 30 features, I have built an autoencoder to add them as new features.
And built a linear regression model as a baseline and compared the results between the normal data and data checked with an autoencoder effect on the baseline model.
The normal baseline model produced a score of :
1323.67 (Mean Absolute Error)
The baseline model working on data with autoencoder features produced a score of and produced a score of :
1311.46 (Mean Absolute Error)

7. Understanding Autoencoders

Autoencoders are used here as they use deep-learning techniques to compress(encode) the data and perform Analysis such that when they decode again, we get the most important features like the function of an autoencoder is to do dimensionality reduction, or in other words, they try to conserve as much information as possible. Here as we have many features, this method helps us in obtaining the most information out of the data.


8. Modeling different Machine Learning architectures/algorithms
After some preprocessing steps like :
Converting skewed continuous variables using Box-Cox
Min-Maxing transform on continuous variables
Combining the categorical features together
Obtaining the autoencoder features
I have built various models like :
Linear Regression
Ridge Regression
Random Forest Regressor
Light GBM
ADA boost model
CAT Boosting Model
Custom Model
Custom Ensemble Model
The total training data is divided into two data sets D1 and D2. D1 contains 80% of the training data in the D2 contains 20% of the training data which is a holdout set and will later be used for testing the performance of the final custom ensemble model. From the D1 set, we are sampling(with Replacement) N different dataset which is used for training N base regressors ( decision trees). Using the prediction of the N base models a meta-regression model is trained which will predict our final loss for a data point. The performance of this metamodel is finally tested on the hold-out set D2.



(All scores in MAE)
9. Results
Although, because of over ensembling the models did overfit on Kaggle's private dataset, With the help of a custom model, the kaggle score improved drastically to 

10. Future Work
Adding more Features can improve the score
Doing some feature engineering on continuous features can improve the score.
Using different loss functions in Xgboost.
Trying out different stacking architecture and reducing the MAE.
Including different feature engineering ways.
Using Hyperparameter tuning using libraries like optuna.

12. Detailed Blog: https://medium.com/@mrupesh6/all-state-insurance-severity-prediction-461b7088669f


LinkedIn :
https://www.linkedin.com/in/rupesh-malla-b11359194/

13. References :
https://www.kaggle.com/c/allstate-claims-severity/overview
https://www.kaggle.com/c/allstate-claims-severity/discussion/24520#140255
https://www.kaggle.com/sharmasanthosh/exploratory-study-on-ml-algorithms
https://www.kaggle.com/c/allstate-claims-severity/discussion/26427
https://www.kaggle.com/cuijamm/allstate-claims-severity-score-1113-12994
https://www.kaggle.com/chandrimad31/claims-severity-analysis-of-models-in-depth/notebook#Adversarial-Validation-:
https://www.kaggle.com/mariusbo/xgb-lb-1106-33084
https://www.kaggle.com/c/allstate-claims-severity/discussion/26416
https://www.appliedaicourse.com/course/11/Applied-Machine-learning-course
Thank you.







