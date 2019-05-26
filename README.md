# id3DecisionTree

![png](question.png)

![png](question_.png)

  
  
**Note:**  

**For categorical data the number of splits are equal to the number of unique values of the attribute**  

**For continuous data if the values are 1,2,3,4 then splits are 1.5, 2.5, 3.5 (This was mentioned in class)**  

OUTPUT: 

=========================  

Training please wait ..... (takes 60 seconds) --- 60.211225509643555 seconds ---  

Training Accuracy: 94.84%  
Dev Accuracy: 76.19%  
Testing Accuracy: 74.78%  

Pruning.....  

Training Accuracy: 93.07%  
Dev Accuracy: 81.3%  
Testing Accuracy: 74.59%  

Comparing with Scikit implementation.....  
Training Accuracy: 96.72%  
Dev Accuracy: 78.51%  
Testing Accuracy: 78.37%  

=========================

The id3 algorithm gives good training accuracy of ~95%. The dev accuracy is ~75% and the test accuracy is also ~75%  

After pruning the validation accuracy increases to ~81% and training accuracy reduces to 93%. There is no significant change in test accuracy. The scikit implementation also gives ~78% accuracy, so it is comparable.  

The scikit-learn code gives ~97% accuracy on training data and ~78% accuracy on test data. Hence, the scikit-learn implementation performs better by 3%.  
