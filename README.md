XGBoost

Implementing XGBoost algorithm in R

XGBoost is a member of the boosting ensemble machine learning algorithms family, along with AdaBoost and Gradient Tree Boosting. It's scalable, portable, fast in execution (faster than other implementations of gradient boosting), accurate in prediction and has a great overall performance.

Major characteristics:

* Grows predictive power by taking into consideration previous iteration errors and learn from them.
* Uses weak learners to create strong learners and design predictions through an iterative process by putting larger weights on misclassified observations in order to classify them correctly in the next iteration. Weak learners are usually decision trees and similar classifiers performing relatively poorly.
* The core of the algorithm is the objective function which measures the model performance by trying to find the best parameters from the training data. It always consists of two parts: training loss and regularization.
* The complexity of the model is controlled by a regularization which additionally prevents overfitting, while loss function controls the predictive power.
* Automatic computation in parallel (20 times faster than 'gbm' on Higgs Boson competition). It uses all CPU cores during the training and performs the cache optimization in order to use the hardware in best possible manner.
* Distributed computations for training large models and external memory version for very large datasets that don’t fit into memory.
* Automatic missing data management for sparse data sets.
*	Advanced customization (customized loss function, error evaluation, prediction in cross-validation, uses early stopping to prevent overfitting during early stages of cross-validation)
*	Possibility to additionally improve an already fitted model on new data.
*	XGBoost is a parameter-rich function consisting of several classes of parameters:
    -	General parameters (number of threads for parallel processing)
    -	Booster parameters (stepsize, regularization)
    - Task parameters (objective, evaluation metric)

In this example, we will demonstrate the XGBoost capability by implementing it in R. The package 'xgboost' is adapted for usage in R by Tong He. We will take the 'churn' public data set available here. The main objective of the problem is to identify customers of the telecom company who will likely to switch to another company. Here we are dealing with a binary classification problem, where the target variable is a binary decision (churn / not churn), with class 0 meaning that customer will not churn, and class 1 meaning that customer will leave the company for some other competitor. The other features represent numerical or categorical variables which are required to be converted into numeric vectors. 
