# Text Classification for Hate Speech
  Our goal here is to build a Naive Bayes Model and Logistic Regression model on a real-world hate speech classification dataset. The dataset is collected from Twitter online. Each example is labeled as 1 (hatespeech) or 0 (Non-hatespeech).

![1](images/1.PNG)

## Naive Bayes 
  Naive Bayes model was implemented with add-1 smoothing.
  
![2](images/2.PNG)
  
### Trends observed
  From the two tables above, it can be observed that the top 10 distinctly hatespeech words are words that are commonly used to describe people in a negative way. Some of these words are also directed at a person belonging to a specific race (e.g. asian), religion (e.g. jews), or a having certain political/moral leaning (e.g. liberal). The classifier is able to identify many hateful words correctly. On the other hand, the top 10 non-hatespeech words that were observed are random words like ”thanks”, ”information”, ”check”, or numbers like ”10.00” and ”15” which may not have much meaning in the context of a sentence.
    
## Logistic Regression model
   Accuracy results for LR model using unigram features:
   
   Train Accuracy: 0.9804
   Test Accuracy: 0.7285
   
   From the above data,we can observe that both the models gave almost similar performance, however logistic regression model performed slightly better with train and test data. This slightly better performance of logistic regression could be attributed to logistic regression being better at generalization, as it does not assume independence between words like Naive Bayes does.
   
##  Comparison of both models with Perceptron classifier
 * Naive Bayes and Logistic Regression models use separate training and test data. Perceptron classifier however, uses training data also as test data as this model learns online. 
 * The Perceptron classifier reads one sample instance at a time to learn about the data and update its understanding about it e.g If a prediction is incorrect, increases weights for features of the true label, and decreasesweights for features of the predicted label. On the other hand, Naive Bayes and Logistic Regression models read the entire sample dataset before updating their knowledge of the data.
 * Observations about \lambda values varying from [0.0001, 0.001, 0.01, 0.1, 1, 10]
    
![3](images/3.PNG)

### Observations:
  From the observations of accuracy with the different \lambda values, we can see that the test accuracy remains exactly the same for  values of 0, 0.0001, 0.001, 0.01 and 0.1
and starts dropping at \lambda value 1 and furthermore at value 10. On the other hand, train accuracy remains same for 0, 0.0001 and 0.001. It shows a sight increase for each \lambda value from 0.001 to 0.1 to 1, and drops again at value 10. For \lambda = 1, train accuracy increases, but test accuracy drops. This may be due to over-fitting of data.
As the train accuracy increases and test accuracy remain steady till 0.1, it can be concluded that \lambda = 0:1 value shows the highest accuracy.

