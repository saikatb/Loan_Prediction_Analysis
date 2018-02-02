
**Loading python Libraries**

    ```python
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style('whitegrid')
        %matplotlib inline
    ```
**Loading the dataset**

The csv file has been loaded into the dataframe "Loan"

    ```python
        Loan = pd.read_csv('ML_Masterclass_dataset_Dec17.csv')
    ```

The dataset has total 13 columns and 614 rows

    ```python
        Loan.shape
        (614, 13)
    ```

**Types of Data**

From our initial observation we can say that the datframe is consisting of qualitative and quantitative data
There are all together 13 columns in the dataframe. The dataframe contains 7 columns which consists of categorical data i.e.

   1) Loan_ID
   2) Gender
   3) Married
   4) Education
   5) Self_Employed
   6) Property_Area
   7) Loan_Status
        
These 7 columns of categorical data are all nominal. The rest of the 6 columns out of 13 are quantitative and are as follow:

   1) Dependents
   2) ApplicantIncome
   3) CoapplicantIncome
   4) LoanAmount
   5) Loan_Amount_Term
   6) Credit_History
        
Out of these 6 quantitative columns ApplicantIncome is discrete and the rest 5 are continuous

For continuous Data we can illustrate the information though Histogram or Boxplot

For the sake of simplicity the value NaN has been filled with the categorical value which has occured most of the time in a specific column.As for example, since "Male" is the most frequently occured categorical value in the column "Gender" hence existing NaN value had been replaced with the catergorical value "Male". 
The same strategy has been applied for the column "Self Employed"

```python
Loan = Loan.fillna({"Gender":"Male", "Self_Employed":"No"})
Loan.drop('Loan_ID',axis=1,inplace=True)
```

**Univariate Analysis** 

```
python  
    Loan['Gender'].describe()
  
    count      614
    unique       2
    top       Male
    freq       502
     Name: Gender, dtype: object
  
```
1) Number of "Male" is more in the observation

```python
   
Loan['Education'].describe()

    count          614
    unique           2
    top       Graduate
    freq           480
    Name: Education, dtype: object
```
2) There are more number of Graduates present among those who have applied for the Loan 

```python
                 
 Loan['Dependents'].describe()
  
    count    599.000000
    mean       0.846411
    std        1.217123
    min        0.000000
    25%        0.000000
    50%        0.000000
    75%        2.000000
    max        4.000000
    Name: Dependents, dtype: float64
 
```
3) People with "0" dependent expects to get their loan approved

```python

Loan['Married'].describe()

    count     611
    unique      2
    top       Yes
    freq      398
    Name: Married, dtype: object
    
```
4) In the observation count of married people is more. The same clearly signifies that married people are inclined towards taking loan

```python
   
Loan['Self_Employed'].describe()

    count     614
    unique      2
    top        No
    freq      532
    Name: Self_Employed, dtype: object
    
```
5 ) Self employed people are not so comfortable in applying for the loan as they are not sure of paying it back. 

```python
    
Loan['Property_Area'].describe()

    count           614
    unique            3
    top       Semiurban
    freq            233
    Name: Property_Area, dtype: object
```
6) People who hail from semiurban are more inclined towards applying for loan as compare to those who are from Urban or Rural.

```python
Loan['Loan_Status'].describe()

    mean       0.687296
    std        0.463973
    min        0.000000
    25%        0.000000
    50%        1.000000
    75%        1.000000
    max        1.000000
    Name: Loan_Status, dtype: float64


```
7) Most of the loan applications have been approved.


In a nuttshell, the gist of the analysis is as below : 
        
   1)  Number of "Male" is more in the observation.
   2)  There are more number of Graduates present among those who have applied for the Loan. 
   3)  People with "0" dependent expects to get their loan approved.
   4)  In the observation Number of married people is more which clearly signifies that married people are inclined 
        towards taking loan.
   5)  People who are self employed are not so comfortable or sure in applying for the loan.
   6)  People who hail from semiurban are more inclined towards applying for loan as compare to those who are from Urban or Rural.
   7)  Most of the loan applications have been approved. 


**Converting categorical values to numerical values using Scikit-Learn Level Encoding for the main Loan dataset**

```python

from sklearn.preprocessing import LabelEncoder

    lbl_encoder = LabelEncoder()
    Loan["Gender"] = lbl_encoder.fit_transform(Loan["Gender"])
    Loan["Married"] = lbl_encoder.fit_transform(Loan["Married"])
    Loan["Education"] = lbl_encoder.fit_transform(Loan["Education"])
    Loan["Self_Employed"] = lbl_encoder.fit_transform(Loan["Self_Employed"])
    Loan["Property_Area"] = lbl_encoder.fit_transform(Loan["Property_Area"])
    Loan["Loan_Status"] = lbl_encoder.fit_transform(Loan["Loan_Status"])
```


Finding out the null values in the column

```python
Loan.isnull().sum()

    Gender                0
    Married               0
    Dependents           15
    Education             0
    Self_Employed         0
    ApplicantIncome       0
    CoapplicantIncome     0
    LoanAmount           22
    Loan_Amount_Term     14
    Credit_History       50
    Property_Area         0
    Loan_Status           0
    dtype: int64
```

Calculating the peercentage of Null values in the dataframe

```python
    Loan_null = pd.DataFrame((Loan.isnull().sum()),columns=['Null_Values'])
    Loan_null['%ofNullValeues'] = ((Loan_null['Null_Values'])/614*100).sort_values(ascending=True)
    Loan_null


                Null_Values   %ofNullValeues

    Loan_ID              0    0.000000
    Gender              13    2.117264
    Married              3    0.488599
    Dependents          15    2.442997
    Education            0    0.000000
    Self_Employed       32    5.211726
    ApplicantIncome      0    0.000000
    CoapplicantIncome    0    0.000000
    LoanAmount          22    3.583062
    Loan_Amount_Term    14    2.280130
    Credit_History      50    8.143322
    Property_Area        0    0.000000
    Loan_Status          0    0.000000
```

So, the maximum number of null values **(8.14 % )** present in the column **Credit_History**. Since the number of null values present is far below 15% and henceforth there is no need to drop any of the column from the dataframe.

Inspite of that, those null values can be replaced with the suitable values i.e. 0, mean, median or mode.

Number of null values present in the columns are as below :
    
   1) Dependents **(15)**
   2) LoanAmount **(22)**
   3) Loan_Amount_Term **(14)**
   4) Credit_History **(50)**

**Replacing NAN values with 0 and mean** 

```python
    Loan['Dependents'].fillna(value=0,axis=0,inplace=True)
    Loan['LoanAmount'].fillna(value=Loan['LoanAmount'].mean(),axis=0,inplace=True)
    Loan['Loan_Amount_Term'].fillna(value=Loan['Loan_Amount_Term'].mean(),axis=0,inplace=True)
    Loan['Credit_History'].fillna(value=Loan['Credit_History'].mean(),axis=0,inplace=True)
```
Fron the below info of dataframe it is found that all the NaN values have been replaced with sutitable value depending upon the types of the data

```python
Loan.info()
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 614 entries, 0 to 613
    Data columns (total 12 columns):
    Gender               614 non-null int64
    Married              614 non-null int64
    Dependents           614 non-null float64
    Education            614 non-null int64
    Self_Employed        614 non-null int64
    ApplicantIncome      614 non-null int64
    CoapplicantIncome    614 non-null float64
    LoanAmount           614 non-null float64
    Loan_Amount_Term     614 non-null float64
    Credit_History       614 non-null float64
    Property_Area        614 non-null int64
    Loan_Status          614 non-null int64
    dtypes: float64(5), int64(7)
    memory usage: 57.6 KB
```    
**Histogram**

Histogram is a technique to do extrapolatory data analysis of any dataset with the visual method.

    ```python
    import seaborn as sns
    sns.distplot(Loan['ApplicantIncome'],color='r')
    ```

![png](output_27_1.png)

From the above histogram of ***ApplicantIncome*** column we can see that the distribution is heavily positively skewed. 
Also the ***skewness*** and the ***kurtosis*** is really high as calculated below.

    ```python
    print("Skewness: %f" % Loan['ApplicantIncome'].skew())
    print("Kurtosis: %f" % Loan['ApplicantIncome'].kurt())

        Skewness: 6.539513
        Kurtosis: 60.540676
    ```

In order to achieve a more acceptable distribution we took the ***log*** of column ***ApplicantIncome*** and ploted the histogram.
From the below graph it can infered that heavily skewed distribution has been replaced by a reasonable normal distribution.

    ```python
    import seaborn as sns
    import numpy as np
    sns.distplot(np.log(Loan['ApplicantIncome']),color='b')
    ```

![png](output_29_1.png)

    ```python
    print("Skewness: %f" % np.log(Loan['ApplicantIncome']).skew())
    print("Kurtosis: %f" % np.log(Loan['ApplicantIncome']).kurt())

        Skewness: 0.479580
        Kurtosis: 3.686875
    ```

From the below plot of normal distribution of ***LoanAmount*** we can infer that the distribution is a little positively skewed with a moderately high peak ( high kurtosis )

    ```python
    import seaborn as sns
    sns.distplot(Loan['LoanAmount'],color='r')
    ```

![png](output_31_1.png)


    ```python
    print("Skewness: %f" % Loan['LoanAmount'].skew())
    print("Kurtosis: %f" % Loan['LoanAmount'].kurt())

        Skewness: 2.726601
        Kurtosis: 10.896456
    ```

So to make the distribution look more normal again the same procedure has been followed.

    ```python
    import seaborn as sns
    import numpy as np
    sns.distplot(np.log(Loan['LoanAmount']),color='b')
    ```

![png](output_33_1.png)

The distribution is slightly negatively skewed

    ```python
    print("Skewness: %f" % np.log(Loan['LoanAmount']).skew())
    print("Kurtosis: %f" % np.log(Loan['LoanAmount']).kurt())
    ```

    Skewness: -0.223227
    Kurtosis: 2.799973
    

    ```python
    import seaborn as sns
    sns.distplot(Loan['CoapplicantIncome'],color='r')
    ```

![png](output_36_1.png)


    ```python
    Loan['CoapplicantIncome'].loc[Loan['CoapplicantIncome'] == 0] = 1
    ```
The value 0 has been replaced with 1 as log(0) is undefined and we wont able to plot the histogram.

    ```python
    import seaborn as sns
    sns.distplot(np.log(Loan['CoapplicantIncome']),color='b')
    ```

![png](output_39_1.png)

Frome the above distribution of the log value of CoapplicantIncome it has been found that the range of the modified CoapplicantIncome varied from -3.00 to +13.00. For the rest of the columns, the range varied from 0 to 1. Henceforth, the idea of normalizing the data by taking the log of the column values will not work here. 


**Multivariate Analysis**

In multivariate analysis we tried to figure out the pearson correlation coefficient among different columns using a heatmap.

```python
    corr=Loan.corr()#["Loan_Status"]
    plt.figure(figsize=(12, 8))

    sns.heatmap(corr, 
                vmax=.8, 
                linewidths=0.01,
                square=True,
                annot=True,
                cmap='Blues',
                linecolor="lightblue")
    plt.title('Correlation between features');
```


![png](output_42_0.png)


From the above heatmap we can conclude following things:

  1) Customers ***Credit history*** has got a huge influence in getting the approval of the Loan (corr=0.54)
  2) The more is the ***applicant's income*** the more chance is there to get the loan (corr=0.57)
  3) The bigger is the ***loan amount*** the lesser is the chance of getting the loan to be approved(corr=-0.036)

Below is the pairplot which clearly shows the scatter plot in between different columns of loan dataset and how they are correlated or even they are related whatsoever



**Normalization**

Below is the dataset after normalizing the scale of all the columns from 0 to 1.
Below python code has been used in order to implement the normalization in the present dataset.

```python
    from sklearn import preprocessing
    import pandas as pd
    x = Loan.values 
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    Loan_Norm = pd.DataFrame(x_scaled)

    Loan_Norm.columns = ['Gender_Norm', 'Married_Norm', 'Dependents_Norm', 'Education_Norm', 'Self_Employed_Norm',           'ApplicantIncome_Norm', 'CoapplicantIncome_Norm', 'LoanAmount_Norm', 'Loan_Amount_Term_Norm', 'Credit_History_Norm', 'Property_Area_Norm', 'Loan_Status_Norm']

```
Heatmap plot after doing the normalizing the scale of all the columns in between range 0 and 1

```python
    corr=Loan_Norm.corr()
    plt.figure(figsize=(12, 8))

    sns.heatmap(corr, 
                vmax=.8, 
                linewidths=0.01,
                square=True,
                annot=True,
                cmap='Blues',
                linecolor="lightblue")
    plt.title('Correlation between features with Normalization');
```
![png](output_76_1.png)


Even though the scale has been normalized in a scale of 0 and 1, no significant differences have been found in the pearson correlation coefficients between multple independent variables. Henceforth, it can be concluded that normalization did not play any significant role in data preprocessing.


**Splittig the dataframe in training and test sample**

Before splitting the dataframe dataframe has been splitted into seperate dataframe. One datframe(Y) will have the target variable i.e. ***Loan_Status*** and the other (X) will contain the rest of the independent variables. 

```python
    Y = Loan['Loan_Status']
    X = pd.concat([Loan['Gender'], Loan['Married'], Loan['Dependents'], Loan['Education'], Loan['Self_Employed'], Loan['ApplicantIncome'], Loan['CoapplicantIncome'], Loan['LoanAmount'], Loan['Loan_Amount_Term'], Loan['Credit_History'], Loan['Property_Area']],axis=1) 
```

```python
    Y.shape, X.shape
```
    ((614L,), (614, 11))

Below are the models been used to predict the status of the loan. ( whether it is issued or not ? )

   1) Logistic Regression
   2) Support Vector Machine
   3) K Nearest Neighbour 
   4) Naive Bayes
   5) Random Forrest

An effort has been made to find out the accuracies for all the possible test size from 0.01 to 0.99 for a specific model. 
Graphs have been plotted against test sizes and accuracies for all the above-mentioned models and eventually been compared by plotting them in a single frame.

**Logistic Regression** 

```python

    import numpy as np
    #from array import array
    from sklearn.linear_model import LogisticRegression
    from sklearn.cross_validation import train_test_split
    from sklearn import metrics

    test_size = np.arange(0.01,1,0.01)
    accuracyl = []


    for t in test_size:

    logreg = LogisticRegression()
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=t, random_state = 3)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    Y_train = Y_train.reset_index(drop=True)
    Y_test = Y_test.reset_index(drop=True)
    logreg = LogisticRegression()
    logreg.fit(X_train,Y_train)
    Y_pred_logreg = logreg.predict(X_test)
    accuracyl.append(metrics.accuracy_score(Y_test,Y_pred_logreg))

    plt.figure(figsize=(10,5))
    plt.plot(test_size,accuracyl,color='b')
    plt.xlabel('Sample Test Size')
    plt.ylabel('Testing Accuracy')
    plt.title('Plot Accuracy Vs Testsize Using Logisitic Regression')

```

![png](output_61_2.png)


**Support Vector Machine** 

```python
    from sklearn.svm import SVC
    from sklearn import metrics

    accuracys = []

for t in test_size:

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=t, random_state = 3)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    Y_train = Y_train.reset_index(drop=True)
    Y_test = Y_test.reset_index(drop=True)
    model = SVC()
    model.fit(X_train,Y_train)
    Y_pred_SVC = model.predict(X_test)
    accuracys.append(metrics.accuracy_score(Y_test,Y_pred_SVC))
    
    plt.figure(figsize=(10,5))
    plt.plot(test_size,accuracys,color='g')
    plt.xlabel('Sample Test Size')
    plt.ylabel('Testing Accuracy')
    plt.title('Plot Accuracy Vs Testsize Using Support Vector Machine')
```

![png](output_62_1.png)

**K Nearest Neighbour** 

```python

    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier

    test_size = np.arange(0.01,1,0.01)

    accuracyk1 = []
    accuracyk2 = []
    accuracyk3 = []
    accuracyk4 = []
    accuracyk5 = []
    accuracyk6 = []

    for t in test_size:
    
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=t, random_state = 3)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    Y_train = Y_train.reset_index(drop=True)
    Y_test = Y_test.reset_index(drop=True)
    
    #for neighbours =1:
    knn1 = KNeighborsClassifier(n_neighbors = 1)
    knn1.fit(X_train,Y_train)
    Y_pred_knn1 = knn1.predict(X_test)
    accuracyk1.append(metrics.accuracy_score(Y_test,Y_pred_knn1))
        
    #for n_neighbors = 2 :
    knn2 = KNeighborsClassifier(n_neighbors = 2)
    knn2.fit(X_train,Y_train)
    Y_pred_knn2 = knn2.predict(X_test)
    accuracyk2.append(metrics.accuracy_score(Y_test,Y_pred_knn2))
    
    #for n_neighbors = 3 :
    knn3 = KNeighborsClassifier(n_neighbors = 3)
    knn3.fit(X_train,Y_train)
    Y_pred_knn3 = knn3.predict(X_test)
    accuracyk3.append(metrics.accuracy_score(Y_test,Y_pred_knn3))
    
    #for n_neighbors = 4 :
    knn4 = KNeighborsClassifier(n_neighbors = 4)
    knn4.fit(X_train,Y_train)
    Y_pred_knn4 = knn4.predict(X_test)
    accuracyk4.append(metrics.accuracy_score(Y_test,Y_pred_knn4))
    
    #for n_neighbors = 5 :
    knn5 = KNeighborsClassifier(n_neighbors = 5)
    knn5.fit(X_train,Y_train)
    Y_pred_knn5 = knn5.predict(X_test)
    accuracyk5.append(metrics.accuracy_score(Y_test,Y_pred_knn5))
    
    #for n_neighbors = 6 :
    knn6 = KNeighborsClassifier(n_neighbors = 6)
    knn6.fit(X_train,Y_train)
    Y_pred_knn6 = knn6.predict(X_test)
    accuracyk6.append(metrics.accuracy_score(Y_test,Y_pred_knn6))

    plt.figure(figsize=(15,7))
    plt.xlabel('Sample Test Size')
    plt.ylabel('Testing Accuracy')
    plt.title('Plot Accuracy Vs Testsize Using K Nearest Neighbour with neighbours = 1,2,3,4,5,6')

    ax1 = plt.plot(test_size,accuracyk1,color='red')
    ax2 = plt.plot(test_size,accuracyk2,color='blue')
    ax3 = plt.plot(test_size,accuracyk3,color='green')
    ax4 = plt.plot(test_size,accuracyk4,color='grey')
    ax5 = plt.plot(test_size,accuracyk5,color='brown')
    ax6 = plt.plot(test_size,accuracyk6,color='yellow')

    plt.figtext(0.6,0.8, "red, neighbour = 1")
    plt.figtext(0.6,0.77, "blue, neighbour = 2")
    plt.figtext(0.6,0.74, "green, neighbour = 3")
    plt.figtext(0.6,0.71, "grey, neighbour = 4")
    plt.figtext(0.6,0.68, "brown, neighbour = 5")
    plt.figtext(0.6,0.65, "yellow, neighbour = 6")
```

![png](output_63_1.png)

**Naive Bayes** 

```python

    import numpy as np
    from sklearn.naive_bayes import GaussianNB
    from sklearn.cross_validation import train_test_split

    test_size = np.arange(0.01,1,0.01)
    accuracyn = []

    for t in test_size:
    
    model = GaussianNB()
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=t, random_state = 3)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    Y_train = Y_train.reset_index(drop=True)
    Y_test = Y_test.reset_index(drop=True)
    
    model.fit(X_train,Y_train)
    Y_pred_NB = model.predict(X_test)
    accuracyn.append(metrics.accuracy_score(Y_test,Y_pred_NB))

    plt.figure(figsize=(10,5))
    plt.plot(test_size,accuracyn,color='purple')
    plt.xlabel('Sample Test Size')
    plt.ylabel('Testing Accuracy')
    plt.title('Plot Accuracy Vs Testsize Using Naive Bayes')
```

![png](output_64_1.png)

**Random Forrest Regression**

```python

    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.cross_validation import train_test_split

    test_size = np.arange(0.01,1,0.01)
    accuracyrf = []

    for t in test_size:
    
    rfr = RandomForestRegressor() 
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=t, random_state = 3)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    Y_train = Y_train.reset_index(drop=True)
    Y_test = Y_test.reset_index(drop=True)
    
    rfr.fit(X_train,Y_train)
    Y_pred_RFR = rfr.predict(X_test)
    accuracyrf.append(metrics.accuracy_score(Y_test,Y_pred_RFR.astype(int)))
    
    plt.figure(figsize=(10,5))
    plt.plot(test_size,accuracyrf,color='seagreen')
    plt.xlabel('Sample Test Size')
    plt.ylabel('Testing Accuracy')
    plt.title('Plot Accuracy Vs Testsize Using Random Forrest')
```


![png](output_65_1.png)


```python
    import numpy as np
    import pandas as pd

    t = test_size

    l = accuracyl
    s = accuracys
    k5 = accuracyk5
    n = accuracyn
    rf = accuracyrf

    df = pd.DataFrame({'test_size':t, 'Logistic Regression':l, 'Support Vector Machine':s, 'k Nearrest Neighbor=5':k5, 'Naive Bayes':n, 'Random Forest':rf})
    df.set_index('test_size',inplace=True)
    df.head()

```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Logistic Regression</th>
      <th>Naive Bayes</th>
      <th>Random Forest</th>
      <th>Support Vector Machine</th>
      <th>k Nearrest Neighbor=5</th>
    </tr>
    <tr>
      <th>test_size</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.01</th>
      <td>0.857143</td>
      <td>0.857143</td>
      <td>0.714286</td>
      <td>0.857143</td>
      <td>0.857143</td>
    </tr>
    <tr>
      <th>0.02</th>
      <td>0.923077</td>
      <td>0.923077</td>
      <td>0.461538</td>
      <td>0.923077</td>
      <td>0.846154</td>
    </tr>
    <tr>
      <th>0.03</th>
      <td>0.894737</td>
      <td>0.894737</td>
      <td>0.526316</td>
      <td>0.894737</td>
      <td>0.736842</td>
    </tr>
    <tr>
      <th>0.04</th>
      <td>0.880000</td>
      <td>0.880000</td>
      <td>0.440000</td>
      <td>0.880000</td>
      <td>0.760000</td>
    </tr>
    <tr>
      <th>0.05</th>
      <td>0.903226</td>
      <td>0.903226</td>
      <td>0.419355</td>
      <td>0.903226</td>
      <td>0.806452</td>
    </tr>
  </tbody>
</table>
</div>


```python
    import plotly.plotly as py
    import plotly.graph_objs as go
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    ax = df.plot(legend=True, figsize=(12,6),linestyle='--',marker='o', title='TestSize Vs Accuracy with differnt Algorithms')
    ax.set_xlabel("Test Size")
    ax.set_ylabel("Accuracy")
```


![png](output_67_1.png)



```python

    LG = [df['Logistic Regression'].max(),df['Logistic Regression'].min()]
    NB = [df['Naive Bayes'].max(),df['Naive Bayes'].min()]
    RF = [df['Random Forest'].max(),df['Random Forest'].min()]
    KNN = [df['k Nearrest Neighbor=5'].max(),df['k Nearrest Neighbor=5'].min()]
    SVM = [df['Support Vector Machine'].max(),df['Support Vector Machine'].min()]

    Measure = ['Max','Min']

    df_algo_max_min = pd.DataFrame({'Accuracy Value':Measure, 'Logistic Regression':LG, 'Naive Bayes':NB, 'Support Vector Machine':SVM, 'k Nearrest Neighbor=5':KNN, 'Random Forest':RF})
    df_algo_max_min.set_index('Accuracy Value',inplace=True)
    df_algo_max_min.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Logistic Regression</th>
      <th>Naive Bayes</th>
      <th>Random Forest</th>
      <th>Support Vector Machine</th>
      <th>k Nearrest Neighbor=5</th>
    </tr>
    <tr>
      <th>Accuracy Value</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Max</th>
      <td>0.923077</td>
      <td>0.923077</td>
      <td>0.714286</td>
      <td>0.923077</td>
      <td>0.857143</td>
    </tr>
    <tr>
      <th>Min</th>
      <td>0.644295</td>
      <td>0.381853</td>
      <td>0.339286</td>
      <td>0.676417</td>
      <td>0.632550</td>
    </tr>
  </tbody>
</table>
</div>


```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    ax = df_algo_max_min.plot(legend=True, figsize=(10,7),linestyle='--',marker='o', title='Accuracy with differnt Algorithms')
    ax.set_xlabel("Accuracy_Value")
    ax.set_ylabel("Algo")
```

![png](output_69_1.png)


```python
    import seaborn as sns
    sns.distplot(df['Logistic Regression'])
```
![png](output_70_1.png)

```python
    import seaborn as sns
    sns.distplot(df['Naive Bayes'])
```

![png](output_71_1.png)

```python
    import seaborn as sns
    sns.distplot(df['Random Forest'])
```

![png](output_72_1.png)

```python
    import seaborn as sns
    sns.distplot(df['Support Vector Machine'])
```

![png](output_73_1.png)

```python
    import seaborn as sns
    sns.distplot(df['k Nearrest Neighbor=5'])
```

![png](output_74_1.png)

```python
    sns.pairplot(df)
```
![png](output_75_1.png)


