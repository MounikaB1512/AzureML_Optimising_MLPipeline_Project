# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary ##
__Problem Statement__  Analysing the Bankmarketing  dataset and predicting a person will do term deposit or not. Here "Y" defines whether will invest and we need the analyse the parameters that could affect the deposit in the next bank marketing model. 

 The best performing model was voting ensemble model with an accuracy of **0.9171** which was performed by Auto-ML run and is slightly better than the Logistic-Regression

## Scikit-learn Pipeline ## 
__Architecture__ First a compute instance is created to run on virtual machine . Once the compute instance is created, we assign the compute instance to run our Jupyter Notebook and access our workspace, create an experiment and run the models

__Steps Involved__

First the *workspace.config()* is defined and experiment is created

We have to check for the compute instance and if it is not created then the compute instance is created 

*steps in train.py*

The data is first loaded from the url

The Data is fed into the clean data to remove null values 

The data is split in training and testing data

The training data is fed into Logistic Regression

__HyperParameters:__ 
                     *C:* Inverse of regularisation

                      I have chosen discrete parameters choice(100, 10, 1.0, 0.1, 0.01)
                      
                      *max_iter:* number of iterations possible
                      
                      here I have chosen discrete paramters choice(100, 110, 120)
                      
__RandomParameterSampling__ In this method the hyperparameters could be discrete and continous,both are accepted

__Early termination policy__ I have used Bandit policy. Bandit Policy is based on slack factor. Bandit terminates runs where the primary metric is not within the slack factor compared to the best performing run

slack factor: The slack allowed with respect to the best performing run


![hyperdrive result](https://user-images.githubusercontent.com/51949018/107182263-6a402680-6a02-11eb-9797-3a3cedd836cf.png)

## AutoML pipeline ##
My AutoML ran for 27 iterations in 30 mintutes. The best model is _votingEnsemble_ . The primary metric I used is __Acuraccy__ and Number of cross_validations as 6 for my AutoML configuation and I got an accuracy score of **0.9171**

And the autoML model is                                                                             min_samples_split=0.2442105263157895,
                                                                                                    min_weight_fraction_leaf=0.0,
                                                                                                    n_estimators=10,
                                                                                                    n_jobs=1,
                                                                                                    oob_score=False,
                                                                                                    random_state=None,
                                                                                                    verbose=0,
                                                                                                    warm_start=False

![automl result](https://user-images.githubusercontent.com/51949018/107182934-bb044f00-6a03-11eb-83ad-c19292e77977.png)
![auto_graph](https://user-images.githubusercontent.com/51949018/107183068-03237180-6a04-11eb-91c5-9fd0fec0755b.png)

## Pipeline comparison
In HyperDrive, I was able to tune the hyperparameters of logistic regression with different parameters whereas in the AutoML, I was able to apply different algorithm model on my dataset , ensemble model has has low bias and low variance hence the accuracy of the ensemble model is higher than the logistic regression model

## Future work
In future experiments, I want to try other models with with classification and see how they work. And I would also try median stopping policy as early termination policy and see how the accuracy of the model changes since this  policy computes running averages across all training runs and terminates runs with primary metric values worse than the median of averages.And I would see also reduce the number of cross folds and see how it affects accuracy as the number of folds increases , the time training for the model also increases and hence cost of training also increases. 

