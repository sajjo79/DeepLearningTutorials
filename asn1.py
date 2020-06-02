import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def Task_1_1():
    """
    1.1: Create a list of randomly selected 200 integers between 0 and 1000
    and print in ascending order all the odd numbers that are divisible by 7.
    :return:
    """
    import random
    randomlist = []
    for i in range(200):
        n = random.randint(0, 1000)
        randomlist.append(n)

    randomlist.sort()
    print(randomlist)
    rlist=np.asarray(randomlist)
    idxs=np.where(rlist%7==0)
    vals=rlist[idxs]
    print(vals[vals%2==1])

def Task_1_2():
    """
    1.2: Write a python code to calculate Body Mass Index (the ratio of weight in kilograms (kg)
    divided by height in metres-squared (m2)).
    Write the initial input commands to enter "weight" and "height",
    round Body Mass Index to 1 decimal place and print.
    :return:
    """
    weight=float(input("Enter weight (in kg):"))
    height=float(input("Enter height(in meters):"))
    bmi=weight/(height*height)
    bmi=np.round(bmi,1)
    print("body mass index is:",bmi)

def Task_1_3():
    """
    1.3: Equilateral triangle all sides are equal.
    Scalene triangle none of the sides are equal.
    Isosceles triangle atleast two sides are equal.
    Set the vertices A, B and C as below.
    :return:
    """
    A = np.random.randint(low=2, high=100, size=2)
    B = np.random.randint(low=2, high=100, size=2)
    C = np.random.randint(low=2, high=100, size=2)
    print(A, B, C)

    a = np.sqrt(np.square(A[0] - B[0]) + np.square(A[1] - B[1]))
    b = np.sqrt(np.square(B[0] - B[0]) + np.square(B[1] - C[1]))
    c = np.sqrt(np.square(C[0] - A[0]) + np.square(C[1] - A[1]))
    if (a == b and b == c):          print("The triangle is Equilateral")
    if (a == b or b == c or c == a):   print("The triangle is Isosceles")
    if (a != b and b != c and c != a): print("The triangle is Scalene")

    # calculate the area of triangle
    s = (a + b + c) / 2
    s_a, s_b, s_c = np.abs(s - a), np.abs(s - b), np.abs(s - c)
    if (s_a == 0): s_a = 1
    if (s_b == 0): s_b = 1
    if (s_c == 0): s_c = 1
    area = np.sqrt(s * (s_a) * (s_b) * (s_c))
    print("Area of Triangle is:", area)

def Task_1_4():
    """
    1.4: In a Dataframe create an array in range(10-3000) in one column and a true or false column
    adjacent to this column which would be true if the number is a palindrome else false.
    Also print the total number of true values in this column.
    :return:
    """
    plaindromes=[]
    arr=np.asarray(range(10,3000))
    for i in arr:
        num = [int(x) for x in str(i)]
        num_rev=num.copy()
        num.reverse()
        if(num==num_rev): plaindromes.append(True)
        else: plaindromes.append(False)
    data=np.vstack([arr,plaindromes])
    data=data.transpose()
    df=pd.DataFrame(data=data,columns=["Values","Plaindromes"])
    print(df.shape)
    ps=df.loc[:,'Plaindromes']
    count=(ps==True).sum()
    print("total plaindromes are:",count)
    #filter = df["Plaindromes"]==1
    #df2=df.where(filter)
    idxs=df.values[:,1]
    vals=df.values[:,0]
    print(vals[np.where(idxs==1)])
#========================================================
def Task_2():
    from sklearn.datasets import load_breast_cancer
    combined_data = load_breast_cancer()
    print(combined_data.feature_names)
    print(combined_data.target_names)
    data=combined_data.data
    target=combined_data.target
    print(data.shape)

    mean_texture=data[:,1]
    worst_texture=data[:,21]
    plt.scatter(mean_texture,worst_texture)
    plt.xlabel('Mean texture', fontsize=12)
    plt.ylabel('worst texture', fontsize=12)
    plt.show()

    mean_radius = data[:, 0]
    radius_error = data[:, 10]
    plt.scatter(mean_radius, radius_error)
    plt.xlabel('Mean Radius', fontsize=12)
    plt.ylabel('Radius Error', fontsize=12)
    plt.show()
#========================================================
#   TASK - 3
def Task_3():
    """
    Your task this week is to create a new notebook called "[yourstudentID]_Task 3.ipynb". In this you need
    to create cells to load Task 3_data.csv and perform the following:
    1. Find and correct misprints/errors;
    2. Count the total number of missing values for each column;
    3. Remove columns with more than 10% missing values;
    4. Remove rows with more than 20% missing values;
    5. Where possible replace missing values for the remaining missing cells
    6. Plot histograms for each remaining column
    """
    df = pd.read_csv('Task 3_data.csv')#, sep=',',header=True)
    df = df.astype({"c1": float, "c2": float,"c3": float,"c4": float,"c5": float,"c5": float,"c6":float,"c7":float,"c8":str})

    # 1- Find correct and misprints/errors; true values in the output of following statement are misprints/errors
    print(df.isnull())          # list of missing and non-missing values
    # 2- Count the total number of missing values for each column;
    print(df.isnull().sum())    # column-wise missing values = (column,missing_count)(0,60),(1, 315),(2,0),(3,0),(4,145),(5,320),(6,0),(7,0)
    print(df.isnull().sum().sum()) # total missing values
    # 3- Remove columns with more than 10% missing values;
    a=df.isnull().sum()
    rows,cols=df.shape
    tenpc=np.floor(rows*0.1)
    df_data=[]
    for i in range(len(a)):
        print(a.values[i],df.shape)
        if(a.values[i]<=tenpc):
            #df=df.drop(i,axis=1)
            df_data.append(df.iloc[:,i])
    df=pd.DataFrame(df_data)
    df=df.transpose()
    # 4. Remove rows with more than 20% missing values;
    twentypc = np.floor(cols * 0.2)
    b=df.isnull().sum(axis=1)
    df_data=[]
    for r in range(rows):
        print(b.values[r], df.shape)
        if (b.values[r] <= twentypc):
            #df = df.drop(r, axis=0)
            df_data.append(df.iloc[r,:])
    df = pd.DataFrame(df_data)
    print("done")
    # 5. Where possible replace missing values for the remaining missing cells with zero
    c=df.isnull()
    print(c.sum().sum())
    df=df.fillna(0)
    c = df.isnull()
    print(c.sum().sum())
    print("done")

    # 6. Plot histograms for each remaining column
    for i in range(5):
        ax = df.iloc[:,i].plot.hist(bins=10, alpha=0.5)
        plt.show()
#===========================================================================================
def Task_4_1():
    #Generate an array of random integer numbers in range [10,1000] with size 100;
    import random
    randomlist = []
    for i in range(100):
        n = random.randint(10, 1000)
        randomlist.append(n)
    print(randomlist)
    #Discretize this array into k= 10 bins, such that each bin is of equal width using (delta = xmax - xmin/k);
    randomlist.sort()
    df=pd.DataFrame(randomlist,columns=["rlist"])
    delta=(np.max(randomlist)-np.min(randomlist))/100
    df['binned'] = pd.cut(x=df['rlist'], bins=[0, delta, delta*2, delta*3, delta*4,delta*5,delta*6,delta*7,delta*8,delta*9,delta*10])
    # Plot the histogram;
    ax = df.loc[:, 'rlist'].plot.hist(bins=10, alpha=0.5)
    plt.show()
    print("done")

    # iv) Reapet i)-iii) by generating an array of random numbers by normal distribution with (loc=0.0, scale=1.0, size = 100);
    rlist=np.random.normal(loc=0.0, scale=1.0, size=100)
    rlist.sort()
    df = pd.DataFrame(rlist, columns=["rlist"])
    delta = (np.max(rlist) - np.min(rlist)) / 100
    df['binned'] = pd.cut(x=df['rlist'],
                          bins=[0, delta, delta * 2, delta * 3, delta * 4, delta * 5, delta * 6, delta * 7, delta * 8,
                                delta * 9, delta * 10])
    # Plot the histogram;
    ax = df.loc[:, 'rlist'].plot.hist(bins=10, alpha=0.5)
    plt.show()
    print("done")

    # v) Generate an array of random numbers by lognormal distribution with (mean=2.0, sigma=2.0, size=100) and
    # plot the histogram by setting bins as [0, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80,90, 100].
    rlist = np.random.lognormal(mean=2.0, sigma=2.0, size=100)
    rlist.sort()
    df = pd.DataFrame(rlist, columns=["rlist"])
    delta = (np.max(rlist) - np.min(rlist)) / 100
    df['binned'] = pd.cut(x=df['rlist'],
                          bins=[0, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80,90, 100])
    # Plot the histogram;
    ax = df.loc[:, 'rlist'].plot.hist(bins=10, alpha=0.5)
    plt.show()
    print("done")

def Task_4_2():
    """
    Generate a data frame with 3 columns and 200 rows by selecting randomly selected numbers:
    use for the first column the "normal" distribution with (loc=0.0, scale=1.0, size=200) and
    for the last 2 columns the "lognormal" distribution with (mean=0.0, sigma=2.0, size=200) and
    (mean=1.0, sigma=3.0, size=200).
    Find any outlier with a z-value > 3 or z-value < -3 and replace that with the mean of the values,
    eg perform Mean substitution for outliers.
    Write a report about cells with outliers, thier z-scores and substituted values.
    """
    col1 = np.random.normal(loc=0.0, scale=1.0, size=200)
    col2 = np.random.lognormal(mean=0.0,sigma=2.0,size=200)
    col3 = np.random.lognormal(mean=1.0, sigma=3.0, size=200)
    data2=np.vstack([col1,col2,col3])
    data2=data2.transpose()
    df=pd.DataFrame(data2,columns=['col1','col2','col3'])
    print(df.shape)
    print(df.mean())
    # z=(x-mu)/sigma
    cols = list(df.columns)
    for col in cols:
        col_zscore = col + '_zscore'
        df[col_zscore] = (df[col] - df[col].mean()) / df[col].std(ddof=0)
        df[df[col_zscore]>3]=df[col].mean()
        df[df[col_zscore] < -3] = df[col].mean()
    print(df.shape)
    return df

def Task_4_3(df):
    """
    1) Apply PCA to the data - output of Exercise 4.2 (where all outliers are with substituted values)
    in order to reduce the dimension from 3 to 2 (use "PCA(n_components = 2)".
    2) Transform your data (3 dimensional) to a new data (2 dimensional) by applying "pca.transform",
    then print the shapes and the first 5 rows of your data and new/transformed data.
    :return:
    """
    from sklearn.decomposition import PCA
    cols=df.columns
    dfn=df[["col1_zscore","col2_zscore","col3_zscore"]]
    pca = PCA(n_components=2)
    pca.fit(dfn)
    ndata=pca.transform(dfn)
    dft=pd.DataFrame(ndata)
    print(ndata.shape)
    print(dfn.head(5))
    print(dft.head(5))
#====================================================================
def Task_5_1():
    from sklearn.datasets import load_breast_cancer
    from sklearn import svm
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import ShuffleSplit

    combined_data = load_breast_cancer()
    print(combined_data.feature_names)
    print(combined_data.target_names)
    data = combined_data.data
    target = combined_data.target
    print(data.shape)

    clf = svm.SVC(kernel='linear', C=1)
    scores = cross_val_score(clf, data, target, cv=10)
    print("Each fold accuracy (StratifiedKFold)",scores)
    print("Mean Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
    scores=cross_val_score(clf, data, target, cv=cv)
    print("Each fold accuracy (ShuffleSplit)", scores)
    print("Mean Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#====================================================================

def Task_6():
    from sklearn.datasets import load_breast_cancer
    from sklearn import svm
    from sklearn.model_selection import cross_val_score
    from sklearn.feature_selection import mutual_info_classif

    combined_data = load_breast_cancer()

    #print(combined_data.target_names)
    data = combined_data.data
    target = combined_data.target
    print(data.shape)

    mi=mutual_info_classif(data,target,discrete_features='auto',random_state=0)
    fransk=zip(combined_data.feature_names,mi)
    for feat,m in fransk:
        print(feat,'---',m)

    feature_ranks=np.vstack((combined_data.feature_names,mi))
    feature_ranks=feature_ranks.transpose()
    feature_ranks=feature_ranks[feature_ranks[:, 1].argsort()[::-1]]

    # select top 5 and top 3 features
    top_5=feature_ranks[0:5,:]
    top_3=feature_ranks[0:3,:]
    top_5_features=feature_ranks[0:5,0]
    top_3_features=feature_ranks[0:3,0]
    print("Top five features:",top_5)
    print("Top three features:",top_3)

    df=pd.DataFrame(data=data,columns=combined_data.feature_names)
    df_top_5=df[top_5_features]
    df_top_3 = df[top_3_features]

    classifiers=['SVM(kernel=linear)','SVM(kernel=rbf','KNeighbors','DecisionTree','MLP','GaussianNB','RandomForest','AdaBoost']
    df_results=pd.DataFrame(data=classifiers,columns=['Classifier'])
    mean_accuracies_all=[]
    mean_accuracies_top_5=[]
    mean_accuracies_top_3=[]

    # SVC(kernel='linear', C=1),
    clf = svm.SVC(kernel='linear', C=1)
    scores_all = cross_val_score(clf, data, target, cv=10)
    scores_top_5 = cross_val_score(clf, df_top_5, target, cv=10)
    scores_top_3 = cross_val_score(clf, df_top_3, target, cv=10)
    print("Results for SVM with linear kernel")
    print("==============================================================")
    print("All features - Each fold accuracy (StratifiedKFold)", scores_all)
    print("Mean Accuracy: %0.2f (+/- %0.2f)" % (scores_all.mean(), scores_all.std() * 2))
    print("Top 5 featuers - Each fold accuracy (StratifiedKFold)", scores_top_5)
    print("Mean Accuracy: %0.2f (+/- %0.2f)" % (scores_top_5.mean(), scores_top_5.std() * 2))
    print("Top 3 featuers - Each fold accuracy (StratifiedKFold)", scores_top_3)
    print("Mean Accuracy: %0.2f (+/- %0.2f)" % (scores_top_3.mean(), scores_top_3.std() * 2))
    mean_accuracies_all.append(scores_all.mean())
    mean_accuracies_top_5.append(scores_top_5.mean())
    mean_accuracies_top_3.append(scores_top_3.mean())
    print("---------------------------------------------------------------")

    # SVC(kernel='rbf', C=1, gamma = 'auto')
    clf = svm.SVC(kernel='rbf', C=1, gamma = 'auto')
    scores_all = cross_val_score(clf, data, target, cv=10)
    scores_top_5 = cross_val_score(clf, df_top_5, target, cv=10)
    scores_top_3 = cross_val_score(clf, df_top_3, target, cv=10)
    print("Results for SVM with RBF kernel")
    print("==============================================================")
    print("All features - Each fold accuracy (StratifiedKFold)", scores_all)
    print("Mean Accuracy: %0.2f (+/- %0.2f)" % (scores_all.mean(), scores_all.std() * 2))
    print("Top 5 featuers - Each fold accuracy (StratifiedKFold)", scores_top_5)
    print("Mean Accuracy: %0.2f (+/- %0.2f)" % (scores_top_5.mean(), scores_top_5.std() * 2))
    print("Top 3 featuers - Each fold accuracy (StratifiedKFold)", scores_top_3)
    print("Mean Accuracy: %0.2f (+/- %0.2f)" % (scores_top_3.mean(), scores_top_3.std() * 2))
    mean_accuracies_all.append(scores_all.mean())
    mean_accuracies_top_5.append(scores_top_5.mean())
    mean_accuracies_top_3.append(scores_top_3.mean())
    print("---------------------------------------------------------------")

    # KNeighborsClassifier(),
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=2)
    scores_all = cross_val_score(clf, data, target, cv=10)
    scores_top_5 = cross_val_score(clf, df_top_5, target, cv=10)
    scores_top_3 = cross_val_score(clf, df_top_3, target, cv=10)
    print("Results for KNN")
    print("==============================================================")
    print("All features - Each fold accuracy (StratifiedKFold)", scores_all)
    print("Mean Accuracy: %0.2f (+/- %0.2f)" % (scores_all.mean(), scores_all.std() * 2))
    print("Top 5 featuers - Each fold accuracy (StratifiedKFold)", scores_top_5)
    print("Mean Accuracy: %0.2f (+/- %0.2f)" % (scores_top_5.mean(), scores_top_5.std() * 2))
    print("Top 3 featuers - Each fold accuracy (StratifiedKFold)", scores_top_3)
    print("Mean Accuracy: %0.2f (+/- %0.2f)" % (scores_top_3.mean(), scores_top_3.std() * 2))
    mean_accuracies_all.append(scores_all.mean())
    mean_accuracies_top_5.append(scores_top_5.mean())
    mean_accuracies_top_3.append(scores_top_3.mean())
    print("---------------------------------------------------------------")

    # DecisionTreeClassifier(),
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(random_state=0)
    scores_all = cross_val_score(clf, data, target, cv=10)
    scores_top_5 = cross_val_score(clf, df_top_5, target, cv=10)
    scores_top_3 = cross_val_score(clf, df_top_3, target, cv=10)
    print("Results for Decision Tree")
    print("==============================================================")
    print("All features - Each fold accuracy (StratifiedKFold)", scores_all)
    print("Mean Accuracy: %0.2f (+/- %0.2f)" % (scores_all.mean(), scores_all.std() * 2))
    print("Top 5 featuers - Each fold accuracy (StratifiedKFold)", scores_top_5)
    print("Mean Accuracy: %0.2f (+/- %0.2f)" % (scores_top_5.mean(), scores_top_5.std() * 2))
    print("Top 3 featuers - Each fold accuracy (StratifiedKFold)", scores_top_3)
    print("Mean Accuracy: %0.2f (+/- %0.2f)" % (scores_top_3.mean(), scores_top_3.std() * 2))
    mean_accuracies_all.append(scores_all.mean())
    mean_accuracies_top_5.append(scores_top_5.mean())
    mean_accuracies_top_3.append(scores_top_3.mean())
    print("---------------------------------------------------------------")

    # MLPClassifier(max_iter=1000),
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(random_state=1, max_iter=500)
    scores_all = cross_val_score(clf, data, target, cv=10)
    scores_top_5 = cross_val_score(clf, df_top_5, target, cv=10)
    scores_top_3 = cross_val_score(clf, df_top_3, target, cv=10)
    print("Results for MLP Classifier")
    print("==============================================================")
    print("All features - Each fold accuracy (StratifiedKFold)", scores_all)
    print("Mean Accuracy: %0.2f (+/- %0.2f)" % (scores_all.mean(), scores_all.std() * 2))
    print("Top 5 featuers - Each fold accuracy (StratifiedKFold)", scores_top_5)
    print("Mean Accuracy: %0.2f (+/- %0.2f)" % (scores_top_5.mean(), scores_top_5.std() * 2))
    print("Top 3 featuers - Each fold accuracy (StratifiedKFold)", scores_top_3)
    print("Mean Accuracy: %0.2f (+/- %0.2f)" % (scores_top_3.mean(), scores_top_3.std() * 2))
    mean_accuracies_all.append(scores_all.mean())
    mean_accuracies_top_5.append(scores_top_5.mean())
    mean_accuracies_top_3.append(scores_top_3.mean())
    print("---------------------------------------------------------------")

    # GaussianNB()
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    scores_all = cross_val_score(clf, data, target, cv=10)
    scores_top_5 = cross_val_score(clf, df_top_5, target, cv=10)
    scores_top_3 = cross_val_score(clf, df_top_3, target, cv=10)
    print("Results for Gaussian NB")
    print("==============================================================")
    print("All features - Each fold accuracy (StratifiedKFold)", scores_all)
    print("Mean Accuracy: %0.2f (+/- %0.2f)" % (scores_all.mean(), scores_all.std() * 2))
    print("Top 5 featuers - Each fold accuracy (StratifiedKFold)", scores_top_5)
    print("Mean Accuracy: %0.2f (+/- %0.2f)" % (scores_top_5.mean(), scores_top_5.std() * 2))
    print("Top 3 featuers - Each fold accuracy (StratifiedKFold)", scores_top_3)
    print("Mean Accuracy: %0.2f (+/- %0.2f)" % (scores_top_3.mean(), scores_top_3.std() * 2))
    mean_accuracies_all.append(scores_all.mean())
    mean_accuracies_top_5.append(scores_top_5.mean())
    mean_accuracies_top_3.append(scores_top_3.mean())
    print("---------------------------------------------------------------")

    # RandomForestClassifier(n_estimators=10),
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    scores_all = cross_val_score(clf, data, target, cv=10)
    scores_top_5 = cross_val_score(clf, df_top_5, target, cv=10)
    scores_top_3 = cross_val_score(clf, df_top_3, target, cv=10)
    print("Results for Random Forest")
    print("==============================================================")
    print("All features - Each fold accuracy (StratifiedKFold)", scores_all)
    print("Mean Accuracy: %0.2f (+/- %0.2f)" % (scores_all.mean(), scores_all.std() * 2))
    print("Top 5 featuers - Each fold accuracy (StratifiedKFold)", scores_top_5)
    print("Mean Accuracy: %0.2f (+/- %0.2f)" % (scores_top_5.mean(), scores_top_5.std() * 2))
    print("Top 3 featuers - Each fold accuracy (StratifiedKFold)", scores_top_3)
    print("Mean Accuracy: %0.2f (+/- %0.2f)" % (scores_top_3.mean(), scores_top_3.std() * 2))
    mean_accuracies_all.append(scores_all.mean())
    mean_accuracies_top_5.append(scores_top_5.mean())
    mean_accuracies_top_3.append(scores_top_3.mean())
    print("---------------------------------------------------------------")

    # AdaBoostClassifier()
    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier(n_estimators=10, random_state=0)
    scores_all = cross_val_score(clf, data, target, cv=10)
    scores_top_5 = cross_val_score(clf, df_top_5, target, cv=10)
    scores_top_3 = cross_val_score(clf, df_top_3, target, cv=10)
    print("Results for AdaBoostClassifier")
    print("==============================================================")
    print("All features - Each fold accuracy (StratifiedKFold)", scores_all)
    print("Mean Accuracy: %0.2f (+/- %0.2f)" % (scores_all.mean(), scores_all.std() * 2))
    print("Top 5 featuers - Each fold accuracy (StratifiedKFold)", scores_top_5)
    print("Mean Accuracy: %0.2f (+/- %0.2f)" % (scores_top_5.mean(), scores_top_5.std() * 2))
    print("Top 3 featuers - Each fold accuracy (StratifiedKFold)", scores_top_3)
    print("Mean Accuracy: %0.2f (+/- %0.2f)" % (scores_top_3.mean(), scores_top_3.std() * 2))
    mean_accuracies_all.append(scores_all.mean())
    mean_accuracies_top_5.append(scores_top_5.mean())
    mean_accuracies_top_3.append(scores_top_3.mean())
    print("---------------------------------------------------------------")

    df_results['Accuracy_all_features'] = np.asarray(mean_accuracies_all)
    df_results['Accuracy_top5_features'] = np.asarray(mean_accuracies_top_5)
    df_results['Accuracy_top3_features'] = np.asarray(mean_accuracies_top_3)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df_results)





if __name__=='__main__':
    pass
    #Task_1_1()
    #Task_1_2()
    #Task_1_3()
    #Task_1_4()
    #Task_2()
    Task_3()
    #Task_4_1()
    #df=Task_4_2()
    #Task_4_3(df)
    #Task_5_1()
    #Task_6()
