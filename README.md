Predict Promotion Using Boosting Techniques On Tech Company Data 
In this case study, you will the Tech company dataset to solve a classification problem using Bagging and Boosting Techniques.

Problem Statement
Palmer Tech is a tech company supported by 90,000+ employees. They are now facing a dilemma - they do not know who their best employees are. Instead of relying on managers' perceptions and biases, they want to use machine learning to identify the right promotion candidates.

We can see that there are missing values in the previous_year_rating and education columns. For the first iteration, we will drop the missing values as they constitute only about 10% of the data.

Missing values have been handled. Now, let us analyze the numerical values in the dataset.

Accuracy is not so great here. Also, since this is an imbalanced dataset, we need to pay attention to the f1-score for class 1. It is also not so good.

This might be due to the fact that there are a lot of categorical variables in the dataset. As there is a lot of non-linearity, let us use tree algorithms.

Random Forest
We will try the Bagging technique Random Forest on the dataset. Since we do not know the optimal hyperparameters for the forest, let us use GridSearch cross-validation to identify them.


Null accuracy itself is close to 0.91. So our accuracy of 0.92 is not a big deal. Also, as discussed earlier, f1-score is more important in these scenarios. But even that is pretty low. By the way, is f1-score still the right metric in this scenario?

For imbalanced datasets, ROC_AUC is considered to be a more relevant metric than f1-score and accuracy as it is independent of threshold value.

Hurray! Xgboost gave us a higher ROC_AUC score compared to Random Forest. Let us see if we can make this better by tuning the real strengths of Xgboost - its Hyperparameters.

LightGBM performance is not upto the mark.It is unable to correctly classify even a single datapoint that belonged to class 1. We can conclude that Palmer Tech can use the XGboost model to identify the promotion candidates.


