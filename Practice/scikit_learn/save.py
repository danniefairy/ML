    from sklearn import svm
    from sklearn import datasets

    iris=datasets.load_iris()
    X,y=iris.data,iris.target

    clf=svm.SVC()
    clf.fit(X,y)


    #方法一:pickle
    import pickle
    #save
    with open('save/clf.pickle','wb') as f:
        pickle.dump(clf,f)
    #store
    with open('save/clf.pickle','rb') as f:
        clf_load=pickle.load(f)


    #方法二:joblib 比pickle快
    from sklearn.externals import joblib
    #save
    joblib.dump(clf,'save/clf.pkl')
    #store
    clf3=joblib.load('save/clf.pkl')
