# DecisionTree

This is an exercise I did at University to learn about Decision Trees.

The exercise consists in predicting if a credit is going to be approved or not through a Decision Tree, the exercise is divided in 3 parts (C, B, A).

## Part C

This parts consists on the creation of the Tree using the ID3 algorithm, I use the splitting criterion C4.5 (Information Gain and Gini), when the tree is contructed we validated with K-Fold and Leave-One-Out Cross Validation.

In this part there is no missing value treatment, that will be done in part B. 

### *Tree Construction*

I made a recursive function called ID3 than given the candidates and the splitting criterion constructs the Decision Tree and returns its root node.

Then to visualize the Tree I made another function that saves the tree in a file, this files are stored at the TreeRepresentation folder of this repository.

### *Cross Validation*

#### K-Fold

Here the Dataframe is divided in K folds, from this folds 1 is used for validation and the others for training, then we repeat this K times so that each fold is used in testing, essentially we are creating K models in this step 
one for every fold.

The K-Fold function I made returns the confusion matrix, from there we can obtain different metrics to evaluate the model, for K=5 the results are the following:

<Table>
  <tr>
    <td></td>
    <td>Gain</td>
    <td>Gain Ratio</td>
    <td>Gini</td>
  </tr>
  <tr>
    <td>Accuracy</td>
    <td>78.61%</td>
    <td>76.37%</td>
    <td>77.18%</td>
  </tr>
  <tr>
    <td>Precision</td>
    <td>78.41%</td>
    <td>76.13%</td>
    <td>75.82%</td>
  </tr>
  <tr>
    <td>Recall</td>
    <td>75.03%</td>
    <td>71.35%</td>
    <td>74.63%</td>
  </tr>
  <tr>
    <td>Specificity</td>
    <td>82.10%</td>
    <td>80.88%</td>
    <td>79.80%</td>
  </tr>
</Table>

Given this results Gain is the best splittingg criterion in this case.

#### Leave-One-Out

Leave One Out is similar to K-Fold but instead of choosing a K to decide the number of folds we use every single entry individually as testing.

This function is implemented in the code, however given the amount of rows that there are this method is very slow because it has to create a model for every single row.

#### Prediction in Test

In this part there is no missing value imputation, but instead of simply removing the entries with missing values I wanted to see what would happen if when a missing value appears we continue traversing the Tree through the node that has more children since it's more likely that the node is part of it.

<Table>
  <tr>
    <td></td>
    <td>Gain</td>
    <td>Gain Ratio</td>
    <td>Gini</td>
  </tr>
  <tr>
    <td>Accuracy</td>
    <td>65.89%</td>
    <td>86.71%</td>
    <td>80.35%</td>
  </tr>
  <tr>
    <td>Precision</td>
    <td>60.96%</td>
    <td>86.48%</td>
    <td>77.92%</td>
  </tr>
  <tr>
    <td>Recall</td>
    <td>64.96%</td>
    <td>86.48%</td>
    <td>77.92%</td>
  </tr>
  <tr>
    <td>Specificity</td>
    <td>66.66%</td>
    <td>89.58%</td>
    <td>82.29%</td>
  </tr>
</Table>

Given this results we can assume that the Gain criterion didn't generalze enough resulting in overfitting, we can assume this becuase of how well it did in the validation compared to the results in test.

The other 2 criterion received a significant increase in all the metrics, especially Gain Ratio that gained almost a 10% increase. This is likely due to a better generalization of the data and probably the data.

This difference in results is also affected by the way the missing values were handled, Gain did not benefit of this method, but Gain Ratio and Gini did.

## Part B

In this part we will predict the missing values using a Decision Tree. First the train dataset is divided between rows with missing values and rows without them, we use the dataset without missing values to make a model and then we make a prediction for the missing values and impute the result of the prediction on them. This imputation is also done on the Test dataset but the Test data is never user for training the model.

#### K-Fold K=5

<Table>
  <tr>
    <td></td>
    <td>Gain</td>
    <td>Gain Ratio</td>
    <td>Gini</td>
  </tr>
  <tr>
    <td>Accuracy</td>
    <td>79.89%</td>
    <td>80.65%</td>
    <td>79.69%</td>
  </tr>
  <tr>
    <td>Precision</td>
    <td>78.81%</td>
    <td>79.70%</td>
    <td>77.98%</td>
  </tr>
  <tr>
    <td>Recall</td>
    <td>77.13%</td>
    <td>76.66%</td>
    <td>77.20%</td>
  </tr>
  <tr>
    <td>Specificity</td>
    <td>82.90%</td>
    <td>84.34%</td>
    <td>82.35%</td>
  </tr>
</Table>

We can observe a general increase on all metrics compared to the results without predictingg the missingg values. This is because all the information we were loosing by not training the dataset with missing values is being taking into account in this model.

Another reason for this increase is that there is a bias given that we are using data we imputed for the testing so the validation set isn't completely unknown.

#### Prediction in test

#### *Imputing missing values into train and test*

<Table>
  <tr>
    <td></td>
    <td>Gain</td>
    <td>Gain Ratio</td>
    <td>Gini</td>
  </tr>
  <tr>
    <td>Accuracy</td>
    <td>54.55%</td>
    <td>72.73%</td>
    <td>54.54</td>%</td>
  </tr>
  <tr>
    <td>Precision</td>
    <td>33.33%</td>
    <td>66.66%</td>
    <td>40.00%</td>
  </tr>
  <tr>
    <td>Recall</td>
    <td>25.00%</td>
    <td>50.00%</td>
    <td>50.00%</td>
  </tr>
  <tr>
    <td>Specificity</td>
    <td>71.41%</td>
    <td>85.71%</td>
    <td>57.14%</td>
  </tr>
</Table>

We can see that this results are awful, this is happening beacause there is bias in the model and imputing the missing values into test is letting thiis bias make bad predictions.

We can conclude that this bad predictions are either made because the values are missing compltely at random, or because our method of imputation is not able to get the relations between feature correctly.

To improve the results of this model we would need to find another method to impute missing values that actually mangaes to represent the relations between the data.

## Part A

In this part instead of converting all data to categorical we will use the continuous values directly.

#### K-Fold

<Table>
  <tr>
    <td></td>
    <td>Gain</td>
    <td>Gain Ratio</td>
    <td>Gini</td>
  </tr>
  <tr>
    <td>Accuracy</td>
    <td>77.59%</td>
    <td>71.91%</td>
    <td>82.28%</td>%</td>
  </tr>
  <tr>
    <td>Precision</td>
    <td>78.91%</td>
    <td>76.91%</td>
    <td>80.31%</td>
  </tr>
  <tr>
    <td>Recall</td>
    <td>72.30%</td>
    <td>56.52%</td>
    <td>75.32%</td>
  </tr>
  <tr>
    <td>Specificity</td>
    <td>83.21%</td>
    <td>85.07%</td>
    <td>83.77%</td>
  </tr>
</Table>

We can see an improvement in Gini but a decrease in Gain-Ratio.

This decrease is because when we calculate slpit info we make the logarithm in base 2 of the division between the feature we are evaluating ang the total of entries. Since we are making a 2-way-partition the nuumber of rows of the feature is the same as the total, making the division 1 and the resullt of the logarithm 0. 

To avoid divisions of 0 I use 0 to the power of -5 instead and makes the Gain increase in continuous columns.

#### Prediction in Test

<Table>
  <tr>
    <td></td>
    <td>Gain</td>
    <td>Gain Ratio</td>
    <td>Gini</td>
  </tr>
  <tr>
    <td>Accuracy</td>
    <td>82.08%</td>
    <td>63.01%</td>
    <td>82.66%</td>%</td>
  </tr>
  <tr>
    <td>Precision</td>
    <td>80.26%</td>
    <td>63.83%</td>
    <td>84.06%</td>
  </tr>
  <tr>
    <td>Recall</td>
    <td>79.22%</td>
    <td>36.96%</td>
    <td>75.32%</td>
  </tr>
  <tr>
    <td>Specificity</td>
    <td>84.32%</td>
    <td>82.29%</td>
    <td>88.54%</td>
  </tr>
</Table>

Same as in validation but more clear. Here is the best results Gain and Gini have gotten until now.

### Prediction with missing value imputation

Here we can predict numeric values without discretization so the relationship between feature should be preserved unlike what happened in the previous part.

#### K-Fold

<Table>
  <tr>
    <td></td>
    <td>Gain</td>
    <td>Gain Ratio</td>
    <td>Gini</td>
  </tr>
  <tr>
    <td>Accuracy</td>
    <td>79.87%</td>
    <td>69.85%</td>
    <td>82.01%</td>%</td>
  </tr>
  <tr>
    <td>Precision</td>
    <td>75.83%</td>
    <td>61.70%</td>
    <td>79.49%</td>
  </tr>
  <tr>
    <td>Recall</td>
    <td>80.73%</td>
    <td>51.71%</td>
    <td>80.67%</td>
  </tr>
  <tr>
    <td>Specificity</td>
    <td>79.63%</td>
    <td>83.75%</td>
    <td>74.05%</td>
  </tr>
</Table>

#### Prediction in Test

<Table>
  <tr>
    <td></td>
    <td>Gain</td>
    <td>Gain Ratio</td>
    <td>Gini</td>
  </tr>
  <tr>
    <td>Accuracy</td>
    <td>82.01%</td>
    <td>65.31%</td>
    <td>84.39%</td>%</td>
  </tr>
  <tr>
    <td>Precision</td>
    <td>82.26%</td>
    <td>69.75%</td>
    <td>89.06%</td>
  </tr>
  <tr>
    <td>Recall</td>
    <td>79.22%</td>
    <td>38.96%</td>
    <td>74.03%</td>
  </tr>
  <tr>
    <td>Specificity</td>
    <td>84.38%</td>
    <td>86.46%</td>
    <td>92.70%</td>
  </tr>
</Table>

The test prediction is considerably better with the missing value treatment this time because the relations bewtween classes is being mantained.

## Conclusions

To conclude, generally the more data we have to trin the better, since it will avoid having specific data that does no represent reality accurately.

It is also really important to find a method for imputing missing values that mantains the relations between features, we need to avoid imputing values that create relatins between features that are not presented in reality.
