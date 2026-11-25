# Error analysis - Stacking model

Out-of-fold predictions on the 242 training patients: **38 mistakes** (each patient predicted by a model that never saw them). Held-out test set: **8 mistakes** out of 61.

**Check:** at least 10 wrong predictions analysed - YES.

## Types of mistake (out-of-fold)

| true_negative | false_positive | false_negative | true_positive |
| --- | --- | --- | --- |
| 117 | 14 | 24 | 87 |

## The 15 most confidently wrong predictions (out-of-fold)

| patient_row | age | sex | cp_label | trestbps | chol | thalach | exang | oldpeak | ca | thal | y_true | proba_disease | error_type | confidence_in_wrong_answer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 252 | 64.000 | 1.000 | asymptomatic | 128.000 | 263.000 | 105.000 | 1.000 | 0.200 | 1.000 | 7.000 | 0 | 0.920 | false alarm (false positive) | 0.920 |
| 145 | 47.000 | 1.000 | non-anginal pain | 108.000 | 243.000 | 152.000 | 0.000 | 0.000 | 0.000 | 3.000 | 1 | 0.089 | missed disease (false negative) | 0.911 |
| 32 | 64.000 | 1.000 | non-anginal pain | 140.000 | 335.000 | 158.000 | 0.000 | 0.000 | 0.000 | 3.000 | 1 | 0.120 | missed disease (false negative) | 0.880 |
| 176 | 52.000 | 1.000 | asymptomatic | 108.000 | 233.000 | 147.000 | 0.000 | 0.100 | 3.000 | 7.000 | 0 | 0.864 | false alarm (false positive) | 0.864 |
| 44 | 61.000 | 0.000 | asymptomatic | 130.000 | 330.000 | 169.000 | 0.000 | 0.000 | 0.000 | 3.000 | 1 | 0.146 | missed disease (false negative) | 0.854 |
| 199 | 59.000 | 1.000 | typical angina | 160.000 | 273.000 | 125.000 | 0.000 | 0.000 | 0.000 | 3.000 | 1 | 0.148 | missed disease (false negative) | 0.852 |
| 250 | 57.000 | 1.000 | asymptomatic | 110.000 | 201.000 | 126.000 | 1.000 | 1.500 | 0.000 | 6.000 | 0 | 0.842 | false alarm (false positive) | 0.842 |
| 171 | 53.000 | 1.000 | asymptomatic | 142.000 | 226.000 | 111.000 | 1.000 | 0.000 | 0.000 | 7.000 | 0 | 0.837 | false alarm (false positive) | 0.837 |
| 183 | 59.000 | 1.000 | typical angina | 178.000 | 270.000 | 145.000 | 0.000 | 4.200 | 0.000 | 7.000 | 0 | 0.832 | false alarm (false positive) | 0.832 |
| 218 | 64.000 | 0.000 | asymptomatic | 130.000 | 303.000 | 122.000 | 0.000 | 2.000 | 2.000 | 3.000 | 0 | 0.823 | false alarm (false positive) | 0.823 |
| 259 | 57.000 | 1.000 | atypical angina | 124.000 | 261.000 | 141.000 | 0.000 | 0.300 | 0.000 | 7.000 | 1 | 0.190 | missed disease (false negative) | 0.810 |
| 69 | 46.000 | 1.000 | non-anginal pain | 150.000 | 231.000 | 147.000 | 0.000 | 3.600 | 0.000 | 3.000 | 1 | 0.191 | missed disease (false negative) | 0.809 |
| 184 | 60.000 | 0.000 | asymptomatic | 158.000 | 305.000 | 161.000 | 0.000 | 0.000 | 0.000 | 3.000 | 1 | 0.197 | missed disease (false negative) | 0.803 |
| 22 | 58.000 | 1.000 | atypical angina | 120.000 | 284.000 | 160.000 | 0.000 | 1.800 | 0.000 | 3.000 | 1 | 0.204 | missed disease (false negative) | 0.796 |
| 294 | 63.000 | 0.000 | asymptomatic | 124.000 | 197.000 | 136.000 | 1.000 | 0.000 | 0.000 | 3.000 | 1 | 0.218 | missed disease (false negative) | 0.782 |

## Error rate by patient group (out-of-fold)

**Sex**

| sex_label | n | errors | error_rate |
| --- | --- | --- | --- |
| female | 77 | 8 | 0.104 |
| male | 165 | 30 | 0.182 |

**Age band**

| age_band | n | errors | error_rate |
| --- | --- | --- | --- |
| <=45 | 47 | 5 | 0.106 |
| 46-55 | 73 | 9 | 0.123 |
| 56-65 | 95 | 21 | 0.221 |
| >65 | 27 | 3 | 0.111 |

**Chest pain type**

| cp_label | n | errors | error_rate |
| --- | --- | --- | --- |
| asymptomatic | 117 | 18 | 0.154 |
| atypical angina | 42 | 6 | 0.143 |
| non-anginal pain | 64 | 7 | 0.109 |
| typical angina | 19 | 7 | 0.368 |

## Do stacking and Logistic Regression fail on the same patients? (out-of-fold)

| wrong in both | only stacking wrong | only Logistic Regression wrong |
| --- | --- | --- |
| 34 | 4 | 5 |

## Test-set mistakes

| patient_row | age | sex | cp_label | trestbps | chol | thalach | exang | oldpeak | ca | thal | y_true | proba_disease | error_type | confidence_in_wrong_answer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 92 | 62.000 | 1.000 | non-anginal pain | 130.000 | 231.000 | 146.000 | 0.000 | 1.800 | 3.000 | 7.000 | 0 | 0.912 | false alarm (false positive) | 0.912 |
| 66 | 60.000 | 1.000 | non-anginal pain | 140.000 | 185.000 | 155.000 | 0.000 | 3.000 | 0.000 | 3.000 | 1 | 0.247 | missed disease (false negative) | 0.753 |
| 33 | 59.000 | 1.000 | asymptomatic | 135.000 | 234.000 | 161.000 | 0.000 | 0.500 | 0.000 | 7.000 | 0 | 0.736 | false alarm (false positive) | 0.736 |
| 271 | 66.000 | 1.000 | asymptomatic | 160.000 | 228.000 | 138.000 | 0.000 | 2.300 | 0.000 | 6.000 | 0 | 0.606 | false alarm (false positive) | 0.606 |
| 172 | 59.000 | 0.000 | asymptomatic | 174.000 | 249.000 | 143.000 | 1.000 | 0.000 | 0.000 | 3.000 | 1 | 0.430 | missed disease (false negative) | 0.570 |
| 196 | 69.000 | 1.000 | typical angina | 160.000 | 234.000 | 131.000 | 0.000 | 0.100 | 1.000 | 3.000 | 0 | 0.547 | false alarm (false positive) | 0.547 |
| 179 | 53.000 | 1.000 | non-anginal pain | 130.000 | 246.000 | 173.000 | 0.000 | 0.000 | 3.000 | 3.000 | 0 | 0.542 | false alarm (false positive) | 0.542 |
| 287 | 58.000 | 1.000 | atypical angina | 125.000 | 220.000 | 144.000 | 0.000 | 0.400 |  | 7.000 | 0 | 0.526 | false alarm (false positive) | 0.526 |

## Interpretation (auto-written from the numbers above)

1. **Type of mistake.** The stacking model made 38 out-of-fold mistakes on 242 training patients (15.7%): 24 missed-disease cases (false negatives) and 14 false alarms (false positives), so the mistakes are mostly missed disease. A missed case is the more serious error for a doctor, because a sick patient would be sent home untreated, whereas a false alarm usually only leads to extra tests.

2. **Most confidently wrong patients.** Among the 10 most confident mistakes (model sure at 0.82-0.92), 6 are 'false alarm (false positive)'. The commonest chest-pain type is 'asymptomatic' (6 of 10); 8 are male, and their median age is 59 (all patients: 56). The model was strongly convinced of its wrong answer for these patients, so they are worth inspecting by hand.

3. **Groups.** Looking only at groups with at least 20 patients, the highest error rate is for age band '56-65': 22.1% (21 of 95) versus 15.7% overall; that is about 6 more mistakes than expected (z = 1.7), which is not clearly beyond chance, especially because several groups were compared. Smaller groups are not ranked because their rates are too unstable to trust.

4. **Shared failures.** 34 patients are wrong for both stacking and Logistic Regression; 4 are wrong only for stacking and 5 only for Logistic Regression. About 87% of Logistic Regression's mistakes are also made by stacking, so the models mostly fail on the same patients, which leaves stacking little room to help.

5. **Did stacking beat the best single model?** In cross-validation, Stacking (3 models) was numerically ahead of Logistic Regression on roc_auc (mean difference +0.0033) but the gain is NOT statistically reliable (corrected p = 0.638). On the held-out test set the difference was -0.0032 with a 95% bootstrap interval of [-0.0218, +0.0140], which includes zero, so the test set cannot separate the two models. The difference is too small to be told apart from noise, so the simpler Logistic Regression is the better practical choice: it is cheaper to train and easier to explain.