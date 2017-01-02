#########
# 0 INIT
#########

s <- read.csv("D:/Projects/R/algorithms/survival analysis/dataset/s.csv")
library(xgboost)

# removing unwanted features

s$X <- NULL
s$State <- NULL
s$Area.Code <- NULL
s$Phone <- NULL

#####################################
# 1 SPLIT DATASET TO TRAIN AND TEST
#####################################

set.seed(123)

split_rule <- floor(0.75 * nrow(s))
split_sampling <- sample(seq_len(nrow(s)), size=split_rule)

s_train <- s[split_sampling,]
s_test <- s[-split_sampling,]

#################################
# 2 Extracting response variables
#################################

label <- s_train$event
y_label <- s_test[,18]

# remove response variables from train and test

s_train <- data.matrix(s_train)
s_train <- s_train[,-18]

s_test <- data.matrix(s_test)
s_test <- s_test[,-18]

############################################### 
# 3 Cross-validation 5-folds with 10 iterations
###############################################

# assists in choosing the correct number of iterations

fit_cv <- xgb.cv(data=s_train,              # training set
             nfold=5,                       # number of cv folds
             label=label,                   # response variable as a single vector
             nround=10,                     # number of xgboost iterations
             objective='binary:logistic',   # type of problem 
             eval_metric='auc')             # accuracy/error metric

fit_cv

# we get iterative reduction of prediction error both in train and test sets
#
# [1]	 train-auc:0.925164+0.005378	test-auc:0.906091+0.019790 
# [2]	 train-auc:0.931283+0.005536	test-auc:0.911381+0.022081 
# [3]	 train-auc:0.939551+0.006609	test-auc:0.918771+0.017078 
# [4]	 train-auc:0.945848+0.004964	test-auc:0.920542+0.020892 
# [5]	 train-auc:0.952739+0.007880	test-auc:0.915511+0.019566 
# [6]	 train-auc:0.964463+0.001492	test-auc:0.919409+0.029795 
# [7]	 train-auc:0.969306+0.002919	test-auc:0.919758+0.029869 
# [8]	 train-auc:0.971027+0.003086	test-auc:0.918096+0.028246 
# [9]	 train-auc:0.974844+0.003071	test-auc:0.917955+0.029366 
# [10] train-auc:0.979697+0.003319	test-auc:0.919247+0.028786

# algorithm managed to increase train accuracy from 0.9256 to 0.9808
# algorithm managed to increase test accuracy from 0.9041 to 0.9249

################################################ 
# 4 Applying xgboost algorithm to a training set
################################################

# the problem at hand is a binary classification problem
# thus we set the objective parameter as 'binary:logistic'
# error metric set to be auc

fit <- xgboost(data = s_train,                 # training set
               label = label,                  # response variable
               eta = 0.5,                      # control of over-fitting
               max_depth = 15,                 # tree size
               nround=10,                      # number of iterations
               subsample = 0.5,                # percentage of training set to sample
               colsample_bytree = 0.5,         # percentage of features included in tree growing
               seed = 1,                       # reproducibility factor
               eval_metric = "auc",            # accuracy/error metric
               objective = "binary:logistic",  # type of problem to solve
               num_class = 1,                  # number of response variables (classes)
               nthread = 3)                    # number of chunks to run in parallel mode

###################################################
# 5 Make a prediction on test set using xgboost fit
###################################################

pred <- predict(fit, s_test)
head(pred,20)

# values computed are the probabilities of churn.

#  [1] 0.2789122 0.2673115 0.6504272 0.7333463 0.2673115 0.2673115 0.2673115 0.2673115 0.2673115
# [10] 0.2912362 0.2673115 0.2673115 0.2679214 0.7189246 0.2673115 0.2673115 0.2877418 0.3589851
# [19] 0.2673115 0.7308004

####################
# 6 Accuracy metrics
####################

pred2 <- rep(0, 834)
pred2 = ifelse(pred>0.5,1,0)
table(pred2, y_label)

#################################
# 7 Algorithm performance metrics
#################################

library(ROSE)

accuracy.meas(y_label, pred1)
roc.curve(y_label, pred1, plotit=F)

################
# early stopping
################

est =  xgb.cv(data = s_train,                  # training set
               nfold = 5,                      # number of folds in cv
               label = label,                  # response variable
               eta = 0.1,                      # over-fitting control
               max_depth = 15,                 # tree size
               nround=20,                      # number of iterations
               subsample = 0.5,                # percentage of training set to sample
               colsample_bytree = 0.5,         # percentage of features included in tree growing
               seed = 1,                       # reproducibility factor
               eval_metric = "auc",            # accuracy/error metric
               objective = "binary:logistic",  # type of problem to solve
               num_class = 1,                  # number of response variables (classes)
               nthread = 3,                    # number of chunks to run in parallel mode
               maximize = FALSE,
               early_stopping_rounds = 10)     # stop after 10 rounds if test accuracy doesn't improve
             
