# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary ##
__Problem Statement__  Analysing the Bankmarketing  dataset and predicting whether a person will deposit or not. 

 __Solution__  The best performing model was voting ensemble model with an accuracy of **0.91754** which was performed by Auto-ML run and is slightly better than the Logistic-Regression

## Scikit-learn Pipeline ## 
__Architecture__ A compute instance is created to run on virtual machine and then assigned to run our Jupyter Notebook which involves accessing our workspace, creating experiment and run the models

__Steps Involved__

First the *workspace.config()* is defined and experiment named udacity_project is created

We have to check for the compute instance and if it is not created then the compute instance is created 

*steps in train.py*

The dataset is first loaded from the url provided

The Data is fed into the clean data to remove null values 

The data is split in training and testing data

The training data is fed into Logistic Regression

__HyperParameters:__ 
     
     *C:* -Inverse of regularisation: chosen discrete parameters choice(100, 10, 1.0, 0.1, 0.01)
                      
     *max_iter:* -Number of iterations possible :chosen discrete parameters choice(100, 120, 150,200)
                      
__RandomParameterSampling__ Similar to GridSearch,RandomSearch in python.It is used to choose the hyperparameters of a model from a set of values defined using various options such as choice, uniform ,loguniform,normal.

__Early termination policy__ Bandit policy is used as specified. Bandit Policy is based on slack factor and it terminates runs where the primary metric is not within the slack factor compared to the best performing run

slack factor: The slack allowed with respect to the best performing run

The best fitted model parameters of hyperdrive config are :
                
             *C:* - 0.1,*max_iter:* -120 and *Accuracy:* is **0.914**

![image](https://user-images.githubusercontent.com/68179281/112526686-38bfc680-8dc8-11eb-8002-6ec1aeaedcaa.png)

## AutoML pipeline ##
The AutoML run duration for 32 iterations is 32 mintutes with 6-fold cross validation. The best fitted model is _votingEnsemble_ ,giving out accuracy score of **0.91754** 

The best fitted autoML model parameters are:       min_samples_split=0.2442105263157895,
                                                   min_weight_fraction_leaf=0.0,
                                                   n_estimators=10,
                                                   n_jobs=1,
                                                   oob_score=False,
                                                   random_state=None,
                                                   verbose=0,
                                                   warm_start=False

![image](https://user-images.githubusercontent.com/68179281/112526984-8e946e80-8dc8-11eb-91ef-bfcc928a6d05.png)

## Pipeline comparison
HyperDrive run pipeline enables us to tune the hyperparameters of a model with different parameters whereas in the AutoML pipeline, we can apply different models using ensemble technique on a dataset to find the best performing model in terms of our defined metricgoal which is accuracy in our case.
## Future work
To try out other models of classification and see how they work. And also to explore all the options of hyperdrive config parameters such as median stopping policy as early termination policy and using other estimators .And I would see also reduce the number of cross folds and see how it affects accuracy as the number of folds increases , the time training for the model also increases and hence cost of training also increases. 
