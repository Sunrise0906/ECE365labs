import numpy as np
import time
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# You may use this function as you like.
error = lambda y, yhat: np.mean(y!=yhat)

class Question1(object):
    # The sequence in this problem is different from the one you saw in the jupyter notebook. This makes it easier to grade. Apologies for any inconvenience.
    def BernoulliNB_classifier(self, traindata, trainlabels, valdata, vallabels):
        """ Train and evaluate a BernoulliNB classifier using the given data.

        Parameters:
        1. traindata    (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels  (Nt, ) numpy ndarray. The labels in the training set.
        3. valdata      (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels    (Nv, ) numpy ndarray. The labels in the validation set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. trainingError        Float. The reported training error. It should be less than 1.
        3. validationError      Float. The reported validation error. It should be less than 1.
        4. fittingTime          Float. The time it takes to fit the classifier (i.e. time to perform xxx.fit(X,y)). This is not evaluated.
        5. valPredictingTime    Float. The time it takes to run the classifier on the validation data (i.e. time to perform xxx.predict(X,y)). This is not evaluated.

        You can ignore all errors, if any.
        """
        # Put your code below
        classifier = BernoulliNB()
        fit_start = time.time()
        classifier.fit(traindata, trainlabels)
        fit_end = time.time()
        # compute training error
        pred_train = classifier.predict(traindata)
        trainingError = error(pred_train, trainlabels)
        # run the classifier ion the validation data
        run_start = time.time()
        pred_val = classifier.predict(valdata)
        run_end = time.time()
        # compute validation error
        validationError = error(pred_val, vallabels)

        fittingTime = fit_end - fit_start
        valPredictingTime = run_end - run_start

        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def MultinomialNB_classifier(self, traindata, trainlabels, valdata, vallabels):
        """ Train and evaluate a MultinomialNB classifier using the given data.

        Parameters:
        1. traindata    (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels  (Nt, ) numpy ndarray. The labels in the training set.
        3. valdata      (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels    (Nv, ) numpy ndarray. The labels in the validation set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. trainingError        Float. The reported training error. It should be less than 1.
        3. validationError      Float. The reported validation error. It should be less than 1.
        4. fittingTime          Float. The time it takes to fit the classifier (i.e. time to perform xxx.fit(X,y)). This is not evaluated.
        5. valPredictingTime    Float. The time it takes to run the classifier on the validation data (i.e. time to perform xxx.predict(X,y)). This is not evaluated.

        You can ignore all errors, if any.
        """
        # Put your code below
        classifier = MultinomialNB()
        fit_start = time.time()
        classifier.fit(traindata, trainlabels)
        fit_end = time.time()
        # compute training error
        pred_train = classifier.predict(traindata)
        trainingError = error(pred_train, trainlabels)
        # run the classifier ion the validation data
        run_start = time.time()
        pred_val = classifier.predict(valdata)
        run_end = time.time()
        # compute validation error
        validationError = error(pred_val, vallabels)

        fittingTime = fit_end - fit_start
        valPredictingTime = run_end - run_start

        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def LinearSVC_classifier(self, traindata, trainlabels, valdata, vallabels):
        """ Train and evaluate a LinearSVC classifier using the given data.

        Parameters:
        1. traindata    (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels  (Nt, ) numpy ndarray. The labels in the training set.
        3. valdata      (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels    (Nv, ) numpy ndarray. The labels in the validation set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. trainingError        Float. The reported training error. It should be less than 1.
        3. validationError      Float. The reported validation error. It should be less than 1.
        4. fittingTime          Float. The time it takes to fit the classifier (i.e. time to perform xxx.fit(X,y)). This is not evaluated.
        5. valPredictingTime    Float. The time it takes to run the classifier on the validation data (i.e. time to perform xxx.predict(X,y)). This is not evaluated.

        You can ignore all errors, if any.
        """
        # Put your code below
        classifier = LinearSVC()
        fit_start = time.time()
        classifier.fit(traindata, trainlabels)
        fit_end = time.time()
        # compute training error
        pred_train = classifier.predict(traindata)
        trainingError = error(pred_train, trainlabels)
        # run the classifier ion the validation data
        run_start = time.time()
        pred_val = classifier.predict(valdata)
        run_end = time.time()
        # compute validation error
        validationError = error(pred_val, vallabels)

        fittingTime = fit_end - fit_start
        valPredictingTime = run_end - run_start

        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def LogisticRegression_classifier(self, traindata, trainlabels, valdata, vallabels):
        """ Train and evaluate a LogisticRegression classifier using the given data.

        Parameters:
        1. traindata    (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels  (Nt, ) numpy ndarray. The labels in the training set.
        3. valdata      (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels    (Nv, ) numpy ndarray. The labels in the validation set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. trainingError        Float. The reported training error. It should be less than 1.
        3. validationError      Float. The reported validation error. It should be less than 1.
        4. fittingTime          Float. The time it takes to fit the classifier (i.e. time to perform xxx.fit(X,y)). This is not evaluated.
        5. valPredictingTime    Float. The time it takes to run the classifier on the validation data (i.e. time to perform xxx.predict(X,y)). This is not evaluated.

        You can ignore all errors, if any.
        """
        # Put your code below
        classifier = LogisticRegression()
        fit_start = time.time()
        classifier.fit(traindata, trainlabels)
        fit_end = time.time()
        # compute training error
        pred_train = classifier.predict(traindata)
        trainingError = error(pred_train, trainlabels)
        # run the classifier ion the validation data
        run_start = time.time()
        pred_val = classifier.predict(valdata)
        run_end = time.time()
        # compute validation error
        validationError = error(pred_val, vallabels)

        fittingTime = fit_end - fit_start
        valPredictingTime = run_end - run_start

        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def NN_classifier(self, traindata, trainlabels, valdata, vallabels):
        """ Train and evaluate a Nearest Neighbor classifier using the given data.

        Make sure to modify the default parameter.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. valdata              (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels            (Nv, ) numpy ndarray. The labels in the validation set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. trainingError        Float. The reported training error. It should be less than 1.
        3. validationError      Float. The reported validation error. It should be less than 1.
        4. fittingTime          Float. The time it takes to fit the classifier (i.e. time to perform xxx.fit(X,y)). This is not evaluated.
        5. valPredictingTime    Float. The time it takes to run the classifier on the validation data (i.e. time to perform xxx.predict(X,y)). This is not evaluated.

        You can ignore all errors, if any.
        """
        # Put your code below
        classifier = KNeighborsClassifier(n_neighbors=1, algorithm='brute')
        fit_start = time.time()
        classifier.fit(traindata, trainlabels)
        fit_end = time.time()
        # compute training error
        pred_train = classifier.predict(traindata)
        trainingError = error(pred_train, trainlabels)
        # run the classifier ion the validation data
        run_start = time.time()
        pred_val = classifier.predict(valdata)
        run_end = time.time()
        # compute validation error
        validationError = error(pred_val, vallabels)

        fittingTime = fit_end - fit_start
        valPredictingTime = run_end - run_start

        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def confMatrix(self,truelabels,estimatedlabels):
        """ Write a function that calculates the confusion matrix (cf. Fig. 2.1 in the notes).

        You may wish to read Section 2.1.1 in the notes -- it may be helpful, but is not necessary to complete this problem.

        Parameters:
        1. truelabels           (Nv, ) numpy ndarray. The ground truth labels.
        2. estimatedlabels      (Nv, ) numpy ndarray. The estimated labels from the output of some classifier.

        Outputs:
        1. cm                   (2,2) numpy ndarray. The calculated confusion matrix.
        """
        cm = np.zeros((2,2))
        # Put your code below
        cm[0, 0] = np.sum((estimatedlabels == 1) & (truelabels == 1))  # true positives
        cm[0, 1] = np.sum((estimatedlabels == 1) & (truelabels == -1))  # false positives
        cm[1, 0] = np.sum((estimatedlabels == -1) & (truelabels == 1))  # false negatives
        cm[1, 1] = np.sum((estimatedlabels == -1) & (truelabels == -1))  # true negatives

        return cm

    def classify(self, traindata, trainlabels, testdata, testlabels):
        """ Run the classifier you selected in the previous part of the problem on the test data.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. testdata             (Nte, d) numpy ndarray. The features in the test set.
        4. testlabels           (Nte, ) numpy ndarray. The labels in the test set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. testError            Float. The reported test error. It should be less than 1.
        3. confusionMatrix      (2,2) numpy ndarray. The resulting confusion matrix. This will not be graded.
        """

        # Put your code below
        # Use logistic regression
        classifier = LogisticRegression()
        classifier.fit(traindata, trainlabels)
        est_labels = classifier.predict(testdata)
        testError = error(est_labels, testlabels)

        # You can freely use the following line
        confusionMatrix = self.confMatrix(testlabels, est_labels)
        # Do not change this sequence!
        return (classifier, testError, confusionMatrix)

class Question2(object):
    def crossValidationkNN(self, traindata, trainlabels, k):
        """ Write a function which implements 5-fold cross-validation to estimate the error of a classifier with cross-validation with the 0,1-loss for k-Nearest Neighbors (kNN).

        For this problem, take your folds to be 0:N/5, N/5:2N/5, ..., 4N/5:N for cross-validation.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. k                    Integer. The cross-validated error estimates will be outputted for 1,...,k.

        Outputs:
        1. err                  (k+1,) numpy ndarray. err[i] is the cross-validated estimate of using i neighbors (the zero-th component of the vector will be meaningless).
        """
        # Put your code below
        err = np.zeros(k + 1)
        N = traindata.shape[0]

        for i in range(1, k+1):
            subErr = 0
            for j in range(5):
                # split data
                valData = traindata[j*N//5:(j+1)*N//5, :]
                valLabels = trainlabels[j*N//5:(j+1)*N//5]
                trainData = np.concatenate((traindata[:j*N//5, :], traindata[(j+1)*N//5:, :]))
                trainLabels = np.concatenate((trainlabels[:j*N//5], trainlabels[(j+1)*N//5:]))
                # train the model
                classifier = KNeighborsClassifier(n_neighbors=i, algorithm='brute')
                classifier.fit(trainData, trainLabels)
                # compute subError
                predictLabels = classifier.predict(valData)
                subErr += error(predictLabels, valLabels)
            subErr /= 5
            err[i] = subErr

        return err

    def minimizer_K(self, traindata, trainlabels, k):
        """ Write a function that calls the above function and returns 1) the output from the previous function, 2) the number of neighbors within  1,...,k  that minimizes the cross-validation error, and 3) the correponding minimum error.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. k                    Integer. The cross-validated error estimates will be outputted for 1,...,k.

        Outputs:
        1. err                  (k+1,) numpy ndarray. The output from crossValidationkNN().
        2. k_min                Integer (np.int64 or int). The number of neighbors within  1,...,k  that minimizes the cross-validation error.
        3. err_min              Float. The correponding minimum error.
        """
        err = self.crossValidationkNN(traindata, trainlabels, k)
        # Put your code below
        k_min = np.argmin(err[1:]) + 1
        err_min = err[k_min]

        # Do not change this sequence!
        return (err, k_min, err_min)

    def classify(self, traindata, trainlabels, testdata, testlabels):
        """ Train a kNN model on the whole training data using the number of neighbors you found in the previous part of the question, and apply it to the test data.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. testdata             (Nte, d) numpy ndarray. The features in the test set.
        4. testlabels           (Nte, ) numpy ndarray. The labels in the test set.

        Outputs:
        1. classifier           The classifier already trained on the training data. Use the best k value that you choose.
        2. testError            Float. The reported test error. It should be less than 1.
        """
        # Put your code below
        # k = 14
        classifier = KNeighborsClassifier(n_neighbors=14, algorithm='brute')
        classifier.fit(traindata, trainlabels)
        predictLabels = classifier.predict(testdata)
        testError = error(predictLabels, testlabels)

        # Do not change this sequence!
        return (classifier, testError)

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

class Question3(object):
    def LinearSVC_crossValidation(self, traindata, trainlabels):
        """ Use cross-validation to select a value of C for a linear SVM by varying C from 2^{-5},...,2^{15}.

        You should search by hand.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.

        Outputs:
        1. C_min                Float. The hyper-parameter C that minimizes the validation error.
        2. min_err              Float. The correponding minimum error.
        """
        # Put your code below
        C = [2 ** i for i in range(-5, 16)]
        err = np.zeros(len(C))
        for i in range(len(C)):
            classifier = LinearSVC(C=C[i])
            subErr = 1 - np.mean(cross_val_score(classifier, traindata, trainlabels, cv=10))
            err[i] = subErr

        index = np.argmin(err)
        min_err = err[index]
        C_min = C[index]

        # Do not change this sequence!
        return (C_min, min_err)

    def SVC_crossValidation(self, traindata, trainlabels):
        """ Use cross-validation to select a value of C for a linear SVM by varying C from 2^{-5},...,2^{15} and \gamma from 2^{-15},...,2^{3}.

        Use GridSearchCV to perform a grid search.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.

        Outputs:
        1. C_min                Float. The hyper-parameter C that minimizes the validation error.
        2. gamma_min            Float. The hyper-parameter \gamma that minimizes the validation error.
        3. min_err              Float. The correponding minimum error.
        """
        # Put your code below
        C = [2 ** i for i in range(-5, 16)]
        gamma = [2 ** i for i in range(-15, 4)]
        parameters = {'C': C, 'gamma': gamma}
        svc = SVC()
        classifier = GridSearchCV(svc, param_grid=parameters, cv=10)
        classifier.fit(traindata, trainlabels)

        C_min = classifier.best_params_['C']
        gamma_min = classifier.best_params_['gamma']
        min_err = 1 - classifier.best_score_


        # Do not change this sequence!
        return (C_min, gamma_min, min_err)

    def LogisticRegression_crossValidation(self, traindata, trainlabels):
        """ Use cross-validation to select a value of C for a linear SVM by varying C from 2^{-14},...,2^{14}.

        You may either use GridSearchCV or search by hand.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.

        Outputs:
        1. C_min                Float. The hyper-parameter C that minimizes the validation error.
        2. min_err              Float. The correponding minimum error.
        """
        # Put your code below
        C = [2 ** i for i in range(-14, 15)]
        parameters = {'C': C}
        lr = LogisticRegression()
        classifier = GridSearchCV(lr, param_grid=parameters, cv=10)
        classifier.fit(traindata, trainlabels)

        C_min = classifier.best_params_['C']
        min_err = 1 - classifier.best_score_

        # Do not change this sequence!
        return (C_min, min_err)

    def classify(self, traindata, trainlabels, testdata, testlabels):
        """ Train the best classifier selected above on the whole training set.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. testdata             (Nte, d) numpy ndarray. The features in the test set.
        4. testlabels           (Nte, ) numpy ndarray. The labels in the test set.

        Outputs:
        1. classifier           The classifier already trained on the training data. Use the best classifier that you choose.
        2. testError            Float. The reported test error. It should be less than 1.
        """
        # Put your code below
        # The best is SVC
        classifier = SVC(C=8.0, gamma=0.125)
        classifier.fit(traindata, trainlabels)
        predictLabels = classifier.predict(testdata)
        testError = error(predictLabels, testlabels)

        # Do not change this sequence!
        return (classifier, testError)
