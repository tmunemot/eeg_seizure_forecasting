## Melbourne University AES/MathWorks/NIH Seizure Prediction (11th solution)

### Description

11th place solution for [Melbourne University AES/MathWorks/NIH Seizure Prediction](https://www.kaggle.com/c/melbourne-university-seizure-prediction) challenge.

### Dependency

Python 2.7

Python packages
* numpy 1.13.3
* scipy 1.0.0
* pandas 0.19.2
* sklearn 0.18.1
* xgboost 0.7

### Dataset

The temporal dynamics of brain activity can be categorized into four states - Interictal, Preictal, Ictal, and Post-ictal. Interictal is a state between seizures in which patients are in a normal condition. Preictal is a state prior to seizure onsets. This competition is to build a model capable of reliably predicting seizure onsets (Preictal). All EEG recordings are collected from three patients with 16 electrodes. Each of them is divided into 10 minutes segments and labeled as either 0 (Interictal) or 1 (Preictal).

### Features

Univariate Features
* spectral band power, edge frequency [1]
* statistics of the Morlet wavelet coefficients
* auto-correlation decay

Bivariate Features
* maximum linear cross-correlation [1]
* Pearson's correlation coefficients calculated over wavelet coefficients
* statistics of wavelet coherence [2]
* statistics of phase-locking synchrony [3]

### Model

I selected an XGBoost model for the final submission.

### Scores

AUC (area under the curve) is used as a metric for this challenge. My final score is 0.70214 and 0.78478 on public and private leader board respectively.

### Notes

* The dataset provided in this competition is highly inbalanced - the ratio of Interictal to Preictal is 12 to 1. I experimented with Exactly Balanced Bagging algorithm [4] to correct the bias introduced by the class inbalance. However, it didn't lead to any tangible improvements.
* As of writing this summary (1/27/2018), the original dataset is no longer available on the Kaggle competition page. I contacted the organizers, and they kindly informed me that it is moved to [a new cloud service](https://www.epilepsyecosystem.org/howitworks#data).

### References

[1] F Mormann, T Kreuz, C Rieke, RG Andrzejak, A Kraskov, P David, "On the predictability of epileptic seizures", Clin Neurophysiol , 2005 [\[pdf\]](http://www.dtic.upf.edu/~ralph/ClinNeurophysiol116569.pdf)

[2] R Saab, M.J. McKeown, L.J. Myers, R. Abu-Gharbieh, "A wavelet based approach for the detection of coupling in EEG signals", 2nd International IEEE EMBS Conference on Nueral Engineering, 2005 [\[pdf\]](http://www.math.ucsd.edu/~rsaab/publications/NER05_1.pdf)

[3] PW Mirowski, Y LeCun, D Madhavan, R Kuzniecky, "Comparing SVM and convolutional networks for epileptic seizure prediction from intracranial EEG", Machine Learning for Signal Processing, 2008 [\[pdf\]](http://yann.lecun.com/exdb/publis/pdf/mirowski-mlsp-08.pdf)

[4] M Galar, A Fernandez, E Barrenechea, H Bustince, F Herrera, "A review on ensembles for class imbalance problem: bagging, boosting and hybrid based approaches", IEEE Transactions on Systems, Man, and Cybernetics–Part C, 2011
[\[pdf\]](http://sci2s.ugr.es/keel/pdf/algorithm/articulo/2011-IEEE%20TSMC%20partC-%20GalarFdezBarrenecheaBustinceHerrera.pdf)
