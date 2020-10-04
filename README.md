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
**Random Oversampler** had a 59.3% accuracy overall. It scored poorly predicting true positives with recall/sensitivity scores of 56% for high-risk and 62% for low-risk with an average of 62% between the two.  This model caught all low-risk requests (100%) but caught a mere 1% of high-risk loan request, which is to say that it fails when the recommendation is most needed, for high-risk requests despite its 99% overall precision.

**SMOTE** had a 62.6% accuracy overall, a slight improvement over the Random Oversampler model. It scored poorly predicting true positives with recall/sensitivity scores of 69% for high-risk and 56% for low-risk with an average of 56% between the two.  Likewise, this model caught all low-risk requests (100%) but caught a mere 1% of high-risk loan request, which is to say that it fails when the recommendation is most needed, for high-risk requests, despite its 99% overall precision.

**Undersample with ClusterCentroids** had a 52.9% accuracy overall, a weaker performance compared to the others. It scored poorly predicting true positives with recall/sensitivity scores of 68% for high-risk and 38% for low-risk with an average of 38% between the two.  Likewise, this model caught all low-risk requests (100%) but caught a mere 1% of high-risk loan request, which is to say that it fails when the recommendation is most needed, for high-risk requests, despite its 99% overall precision.  This model is the worst of the group.

**SMOTEENN** had a 66.2% accuracy overall, performed the best in the group with this data set. It scored well in true positives with recall/sensitivity scores of 75% for high-risk and while not the worst, it fell in the mid-range with 56% recall for low-risk with an average of 58% between the two.  Likewise, this model caught all low-risk requests (100%) but caught a mere 1% of high-risk loan request, which is to say that it fails when the recommendation is most needed, for high-risk requests, despite its 99% overall precision.

###
Interim Evaluation
######
Since **LendingClub** is a peer-to-peer lending, the assumption is that members are not mortgage brokers or bankers.  It's expected they lack the deep financial background to evaluate and manage lending risk; that's why they're in the this special club. As such, members require help to avoid high-risk situations, or when encountering one, assign a higher premium (interest rate) commensurate with the risk.  The model needs to be precise identifying those who are high-risk.  None of the models are.  None of these models are recommended from this group.  Next, will ensemble classifiers be less wrong?  They kind of have to be...99% miss rate in the key metric is awful.

#### Ensemble Classifiers

**Balanced Random Forest** classifier was the first ensemble classifier evaluated. It had a 81.3 balanced accuracy overall. It scored well with recall/sensitivity scores of 75% for high-risk and 88% recall for low-risk with an average of 88% between the two.  Likewise, this model caught all low-risk requests (100%) but only 3% of high-risk loan request, which is to say that it fails the most when the recommendation is most needed, for high-risk requests, despite its 99% overall precision.  While this showed marked improvement in precision, recall/sensitivity was not enough to move the needle on the recommendation. Perhaps Easy

**Easy Ensemble Classifer** had the best overall accuracy at 93.9%. It scored well true positives with recall/sensitivity scores of 93% for high-risk and 95% recall for low-risk with an average of 95% between the two.  Likewise, this model caught all low-risk requests (100%) but inched up to 8% of high-risk loan request, which is to say that it is not reliable enough when the recommendation is most needed, for high-risk requests, despite its 99% overall precision.

### Recommendation
---
######  
The Easy Ensemble Classifier performed the best of all the models tried with this data set. However, it was not successful enough to provide support club members seek.  The important factors impacting performance center around recent payments and it is likely that this dataset lacks key features like major life changes (move, divorce, job loss, etc.) that could have a higher correlation to loan risk.  Expanding the dataset is the most likely requirement to satisfy the abysmal high-risk precision scores.
