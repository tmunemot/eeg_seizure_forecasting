## Melbourne University AES/MathWorks/NIH Seizure Prediction

This is a solution for [Melbourne University AES/MathWorks/NIH Seizure Prediction](https://www.kaggle.com/c/melbourne-university-seizure-prediction) challenge hosted at Kaggle. The goal of this competition is to forecast seizure onsets of patients diagnosed with Epilepsy given EEG signal recordings.

EEG signals used in this competition are collected from 3 patients who are diagnosed with Epilepsy, and are divided into negative (interictal) and positive (preictal) samples. Interictal is defined as a state between seizures and indicates a patient is in a normal condition whereas preictal is a state prior to seizure onsets. 16 electrodes at 400 Hz sampling frequency are used to record EEG signals. All EEG signals are divided into 10 minutes segments, and each segment is labeled either interictal or preictal in the training dataset.

Submissions are evaluated with AUC (area under curve). A final score of this solution is 0.79214 and 0.78478 on public and private leader board respectively (the 11th out of 478 teams).

Please note this repository is incomplete and only a part of scripts are currently uploaded.

### References
[1] PW Mirowski, Y LeCun, D Madhavan, R Kuzniecky, "Comparing SVM and convolutional networks for epileptic seizure prediction from intracranial EEG", Machine Learning for Signal Processing, 2008 [\[pdf\]](http://yann.lecun.com/exdb/publis/pdf/mirowski-mlsp-08.pdf)

[2] F Lotte, E Miranda, J Castet, "A Tutorial on EEG Signal Processing Techniques for Mental State Recognition in Brain-Computer Interfaces", Guide to Brain-Computer Music Interfacing, Springer, 2015 [\[pdf\]](https://hal.inria.fr/hal-01055103/file/lotte_EEGSignalProcessing.pdf)

[3] F Mormann, T Kreuz, C Rieke, RG Andrzejak, A Kraskov, P David, "On the predictability of epileptic seizures", Clin Neurophysiol , 2005 [\[pdf\]](http://www.dtic.upf.edu/~ralph/ClinNeurophysiol116569.pdf)

[4] R Saab, M.J. McKeown, L.J. Myers, R. Abu-Gharbieh, "A Wavelet Based Approach for the Detection of Coupling in EEG Signals", 2nd International IEEE EMBS Conference on Nueral Engineering, 2005 [\[pdf\]](http://www.math.ucsd.edu/~rsaab/publications/NER05_1.pdf)
