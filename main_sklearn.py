if __name__ == '__main__':

    from sklearn import ensemble
    from sklearn import svm
    import numpy as np
    from sklearn.metrics import precision_score
    from Dataset.class_Dataset_FR import Dataset_FR_AEtest ,Dataset_FR_AEtrain

    num_dataset = 1
    # dataset_train = Dataset_Pic_train()
    # dataset_test = Dataset_Pic_test()
    dataset_train = Dataset_FR_AEtrain(num_dataset)
    dataset_test = Dataset_FR_AEtest(num_dataset)

    label_train = dataset_train.labels
    label_test = dataset_test.labels

    # data_train = np.array([np.array(i) for i in dataset_train.imgs]).reshape(len(label_train), -1)
    # data_test = np.array([np.array(i) for i in dataset_test.imgs]).reshape(len(label_test), -1)

    data_train = np.array([np.array(i) for i in dataset_train.samples]).reshape(len(label_train), -1)
    data_test = np.array([np.array(i) for i in dataset_test.samples]).reshape(len(label_test), -1)

    clf = svm.SVC(decision_function_shape='ovo', probability=True)
    rfc = ensemble.RandomForestClassifier()

    rfc.fit(data_train, label_train)
    clf.fit(data_train, label_train)

    y_hat_clf = clf.predict(data_test)
    y_hat_rfc = rfc.predict(data_test)

    print(precision_score(label_test, y_hat_clf, average='macro'))
    print(precision_score(label_test, y_hat_rfc, average='macro'))
