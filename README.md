# Patient Readmission Prediction

## Models 
Model Link: [prabinpanta0/Patient-Readmission-Prediction](https://huggingface.co/prabinpanta0/Patient-Readmission-Prediction)

## Dataset

* Original Source: [Kaggle/datasets/dubradave/hospital-readmissions](https://kaggle.com/datasets/dubradave/hospital-readmissions)
* Import Source: [HuggingFace/datasets/prabinpanta0/genki_hospital](https://huggingface.co/datasets/prabinpanta0/genki_hospital)

```Json
{
  "model_id": "prabinpanta0/Patient-Readmission-Prediction",
  "model_type": "sequence-classification",
  "library": {
    "random_forest": "scikit-learn",
    "logistic_regression": "scikit-learn",
    "k_nearest": "scikit-learn",
    "svc": "scikit-learn",
    "naive_bayes": "scikit-learn",
    "neural_network": "keras",
    "cross_validation_random_forest": "scikit-learn",
    "cross_validation_logistic_regression": "scikit-learn",
    "cross_validation_lightgbm": "LightGBM"
  },
  "model_architectures": {
    "random_forest": "RandomForestClassifier",
    "logistic_regression": "LogisticRegression",
    "k_nearest": "KNeighborsClassifier",
    "svc": "SVC",
    "naive_bayes": "MultinomialNB",
    "neural_network": "NeuralNetwork",
    "cross_validation_random_forest": "RandomForestClassifier",
    "cross_validation_logistic_regression": "LogisticRegression",
    "cross_validation_lightgbm": "LGBMClassifier"
  },
  "model_paths": {
    "random_forest": "model_RandomForestClassifier.pkl",
    "logistic_regression": "model_Logistic_Regression.pkl",
    "k_nearest": "model_K_nearest.pkl",
    "svc": "model_svc.pkl",
    "naive_bayes": "model_naive_bayes.pkl",
    "neural_network": "neural_network.keras",
    "cross_validation_random_forest": "model_rf.pkl",
    "cross_validation_logistic_regression": "model_lr.pkl",
    "cross_validation_lightgbm": "model_lgbm.pkl"
  },
  "model_classes": {
    "random_forest": "RandomForestClassifier",
    "logistic_regression": "LogisticRegression",
    "k_nearest": "KNeighborsClassifier",
    "svc": "SVC",
    "naive_bayes": "MultinomialNB",
    "neural_network": "NeuralNetwork",
    "cross_validation_random_forest": "RandomForestClassifier",
    "cross_validation_logistic_regression": "LogisticRegression"
  },
  "model_configs": {
    "random_forest": {
      "n_estimators": 100,
      "max_depth": 5
    },
    "logistic_regression": {
      "C": 1,
      "max_iter": 1000
    },
    "k_nearest": {
      "n_neighbors": 5
    },
    "svc": {
      "C": 1,
      "kernel": "linear"
    },
    "naive_bayes": {
      "alpha": 1
    },
    "neural_network": {
      "input_dim": 10,
      "output_dim": 1,
      "hidden_dim": 10
    },
    "cross_validation_random_forest": {
      "n_estimators": 100,
      "max_depth": 5
    },
    "cross_validation_logistic_regression": {
      "C": 1,
      "max_iter": 1000
    },
    "cross_validation_lightgbm": {
      "random_state": 42
    }
  }
}
```

## metrics

|Model|Accuracy          |Precision         |Recall            |AUC-ROC           |
|-----|------------------|------------------|------------------|------------------|
|Random Forest|0.86544           |0.8734358240972471|0.8337883959044369|0.8635809449401703|
|Logistic Regression|0.74736           |0.7493540051679587|0.6928327645051194|0.7441573461079813|
|K-Nearest Neighbors|0.84112           |0.8543724844493231|0.7969283276450512|0.838524404786381 |
|Support Vector Classifier|0.84256           |0.8492462311557789|0.8075085324232082|0.8405012541634113|
|Naive Bayes|0.74176           |0.7692307692307693|0.6416382252559727|0.7358793535918418|
|Neural Network|0.87664           |0.889009009009009 |0.8419795221843004|0.8746042189234755|
|Random Forest (Cross-Validation)|0.86544           |0.8734358240972471|0.8337883959044369|0.8635809449401703|
|Logistic Regression (Cross-Validation)|0.74736           |0.7493540051679587|0.6928327645051194|0.7441573461079813|
|LightGBM (Cross-Validation)|0.8728            |0.8773418168964299|0.847098976109215 |0.8712904519100293|


|Random Forest|Logistic Regression|K-Nearest Neighbors|Support Vector Classifier|Naive Bayes       |Neural Network    |Random Forest (Cross-Validation)|Logistic Regression (Cross-Validation)|LightGBM (Cross-Validation)|
|-------------|-------------------|-------------------|-------------------------|------------------|------------------|--------------------------------|--------------------------------------|---------------------------|
|1.0          |0.7453866666666666 |0.8901866666666667 |0.8530133333333333       |0.7455466666666667|0.88288           |1.0                             |0.7453866666666666                    |0.9045866666666667         |
|1.0          |0.7449201741654572 |0.9005328596802842 |0.8556024378809189       |0.7743332882090158|0.8964114832535885|1.0                             |0.7449201741654572                    |0.910874897792314          |
|1.0          |0.6979827742520399 |0.8618540344514959 |0.8272892112420671       |0.6482320942883046|0.849274705349048 |1.0                             |0.6979827742520399                    |0.8837262012692656         |
|1.0          |0.7427552396345833 |0.8886139001594574 |0.8515853672571407       |0.7401446588709305|0.8810145438895148|1.0                             |0.7427552396345833                    |0.9034286859660855         |

