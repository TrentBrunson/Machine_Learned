# Machine_Learned
---

Resources
---
######  
Python, Pandas, NumPy, scikit-Learn, Imbalanced-Learn, Jupyter Notebook

---

### Overview
---
This code implements several Machine Learning models to evaluate factors that may contribute credit risk and make a recommendation based on past performance at **LendingClub**, a peer-to-peer lending company.  The data is unbalanced and will be evaluated the a logarithmic classification (regression) to predict a binary recommendation as to the loan requestor being "high-risk" or "low-risk.

### Summary
---
Describe the precision, recall scores, balanced accuracy score. 
    # 	            Predicted True	Predicted False
# Actually True	    TRUE POSITIVE	FALSE NEGATIVE
# Actually False    FALSE POSITIVE	TRUE NEGATIVE

# Precision = TP/(TP + FP)...the model said it is so, so how likely is it that the model's right 
  ratio of +actual to all +predictions
# Sensitivity (recall) = TP/(TP + FN)...how likely it is that the model is will catch the right outcome
  - how many people who are x (hi or low risk) actually are high or low risk
  ratio of all +true in model to all actual trues


**Random Oversampler** had a 59.3% accuracy overall. It scored poorly predicting true positives with recall/sensitivity scores of 56% for high-risk and 62% for low-risk.  This model caught all low-risk requests (100%) but caught a mere 1% of high-risk loan request, which is to say that it fails when the recommendation is most needed, for high-risk requests.

**SMOTE** had a 59.3% accuracy overall. It scored poorly predicting true positives with recall/sensitivity scores of 56% for high-risk and 62% for low-risk.  This model caught all low-risk requests (100%) but caught a mere 1% of high-risk loan request, which is to say that it fails when the recommendation is most needed, for high-risk requests.

**Undersample** had a 59.3% accuracy overall. It scored poorly predicting true positives with recall/sensitivity scores of 56% for high-risk and 62% for low-risk.  This model caught all low-risk requests (100%) but caught a mere 1% of high-risk loan request, which is to say that it fails when the recommendation is most needed, for high-risk requests.

**SMOTEENN** had a 59.3% accuracy overall. It scored poorly predicting true positives with recall/sensitivity scores of 56% for high-risk and 62% for low-risk.  This model caught all low-risk requests (100%) but caught a mere 1% of high-risk loan request, which is to say that it fails when the recommendation is most needed, for high-risk requests.

Recommendation

#### Ensemble Classifiers

**Balanced Random Forest** classifier was the first ensemble classifier evaluated.

**Easy Ensemble AdaBoost** classifier

### Recommendation
---
######  
final recommendation on  model to use, if any. If you do not recommend any of the models, justify your reasoning.
