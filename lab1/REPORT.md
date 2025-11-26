# Lab 1 - Baseline Models Report

## Results Summary

| Dataset       | Model        | Accuracy | Precision (Macro) | Recall (Macro) | Note    |
| ------------- | ------------ | -------- | ----------------- | -------------- | ------- |
| **Accidents** | NaiveBayes   | 0.8213   | 0.88              | 0.80           | Success |
| **Accidents** | KNN          | 0.7800   | 0.81              | 0.73           | Success (Sampled: 1000 train) |
| **Accidents** | DecisionTree | **0.8300** | **0.87**        | **0.81**       | Success |
| **Accidents** | LogReg       | 0.8600   | 0.90              | 0.82           | Success (Sampled: 1000 train) |
| **Accidents** | MLP          | 0.8667   | 0.90              | 0.82           | Success (Sampled: 1000 train) |
| **Flights**   | NaiveBayes   | 0.9338   | 0.92              | 0.87           | Success |
| **Flights**   | KNN          | 0.7733   | 0.66              | 0.54           | Success (Sampled: 1000 train) |
| **Flights**   | DecisionTree | **0.9400** | **0.93**        | **0.90**       | Success |
| **Flights**   | LogReg       | 0.9767   | 0.97              | 0.96           | Success (Sampled: 1000 train) |
| **Flights**   | MLP          | 0.7700   | 0.39              | 0.50           | Success (Sampled: 1000 train) |

---

## Accidents Dataset

### Final Model Comparison

#### Accuracy Comparison
![Accidents Final Accuracy](images/accidents_final_accuracy.png)

#### Precision Comparison
![Accidents Final Precision](images/accidents_final_precision.png)

#### Recall Comparison
![Accidents Final Recall](images/accidents_final_recall.png)

### Naive Bayes Study

#### Accuracy
![Accidents Naive Bayes Accuracy](images/accidents_nb_accuracy.png)

#### Precision
![Accidents Naive Bayes Precision](images/accidents_nb_precision.png)

#### Recall
![Accidents Naive Bayes Recall](images/accidents_nb_recall.png)

### KNN Study

#### Accuracy
![Accidents KNN Accuracy](images/accidents_knn_accuracy.png)

#### Precision
![Accidents KNN Precision](images/accidents_knn_precision.png)

#### Recall
![Accidents KNN Recall](images/accidents_knn_recall.png)

### Decision Tree Study

#### Accuracy
![Accidents Decision Tree Accuracy](images/accidents_dt_accuracy.png)

#### Precision
![Accidents Decision Tree Precision](images/accidents_dt_precision.png)

#### Recall
![Accidents Decision Tree Recall](images/accidents_dt_recall.png)

### Logistic Regression Study

#### Accuracy
![Accidents Logistic Regression Accuracy](images/accidents_lr_accuracy.png)

#### Precision
![Accidents Logistic Regression Precision](images/accidents_lr_precision.png)

#### Recall
![Accidents Logistic Regression Recall](images/accidents_lr_recall.png)

### MLP Study

#### Adaptive Learning Rate - Accuracy
![Accidents MLP Adaptive Accuracy](images/accidents_mlp_adaptive_accuracy.png)

#### Adaptive Learning Rate - Precision
![Accidents MLP Adaptive Precision](images/accidents_mlp_adaptive_precision.png)

#### Adaptive Learning Rate - Recall
![Accidents MLP Adaptive Recall](images/accidents_mlp_adaptive_recall.png)

#### Constant Learning Rate - Accuracy
![Accidents MLP Constant Accuracy](images/accidents_mlp_constant_accuracy.png)

#### Constant Learning Rate - Precision
![Accidents MLP Constant Precision](images/accidents_mlp_constant_precision.png)

#### Constant Learning Rate - Recall
![Accidents MLP Constant Recall](images/accidents_mlp_constant_recall.png)

#### Inverse Scaling Learning Rate - Accuracy
![Accidents MLP Inverse Scaling Accuracy](images/accidents_mlp_invscaling_accuracy.png)

#### Inverse Scaling Learning Rate - Precision
![Accidents MLP Inverse Scaling Precision](images/accidents_mlp_invscaling_precision.png)

#### Inverse Scaling Learning Rate - Recall
![Accidents MLP Inverse Scaling Recall](images/accidents_mlp_invscaling_recall.png)

---

## Flights Dataset

### Final Model Comparison

#### Accuracy Comparison
![Flights Final Accuracy](images/flights_final_accuracy.png)

#### Precision Comparison
![Flights Final Precision](images/flights_final_precision.png)

#### Recall Comparison
![Flights Final Recall](images/flights_final_recall.png)

### Naive Bayes Study

#### Accuracy
![Flights Naive Bayes Accuracy](images/flights_nb_accuracy.png)

#### Precision
![Flights Naive Bayes Precision](images/flights_nb_precision.png)

#### Recall
![Flights Naive Bayes Recall](images/flights_nb_recall.png)

### KNN Study

#### Accuracy
![Flights KNN Accuracy](images/flights_knn_accuracy.png)

#### Precision
![Flights KNN Precision](images/flights_knn_precision.png)

#### Recall
![Flights KNN Recall](images/flights_knn_recall.png)

### Decision Tree Study

#### Accuracy
![Flights Decision Tree Accuracy](images/flights_dt_accuracy.png)

#### Precision
![Flights Decision Tree Precision](images/flights_dt_precision.png)

#### Recall
![Flights Decision Tree Recall](images/flights_dt_recall.png)

### Logistic Regression Study

#### Accuracy
![Flights Logistic Regression Accuracy](images/flights_lr_accuracy.png)

#### Precision
![Flights Logistic Regression Precision](images/flights_lr_precision.png)

#### Recall
![Flights Logistic Regression Recall](images/flights_lr_recall.png)

### MLP Study

#### Adaptive Learning Rate - Accuracy
![Flights MLP Adaptive Accuracy](images/flights_mlp_adaptive_accuracy.png)

#### Adaptive Learning Rate - Precision
![Flights MLP Adaptive Precision](images/flights_mlp_adaptive_precision.png)

#### Adaptive Learning Rate - Recall
![Flights MLP Adaptive Recall](images/flights_mlp_adaptive_recall.png)

#### Constant Learning Rate - Accuracy
![Flights MLP Constant Accuracy](images/flights_mlp_constant_accuracy.png)

#### Constant Learning Rate - Precision
![Flights MLP Constant Precision](images/flights_mlp_constant_precision.png)

#### Constant Learning Rate - Recall
![Flights MLP Constant Recall](images/flights_mlp_constant_recall.png)

#### Inverse Scaling Learning Rate - Accuracy
![Flights MLP Inverse Scaling Accuracy](images/flights_mlp_invscaling_accuracy.png)

#### Inverse Scaling Learning Rate - Precision
![Flights MLP Inverse Scaling Precision](images/flights_mlp_invscaling_precision.png)

#### Inverse Scaling Learning Rate - Recall
![Flights MLP Inverse Scaling Recall](images/flights_mlp_invscaling_recall.png)
