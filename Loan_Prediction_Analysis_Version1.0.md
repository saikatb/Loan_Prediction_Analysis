

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline
```


```python
Loan = pd.read_csv('C:/Users/admin/Desktop/Misc_ML/ML_Masterclass_dataset_Dec17.csv')
```


```python
Loan.shape
```




    (614, 13)




```python
Loan.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 614 entries, 0 to 613
    Data columns (total 13 columns):
    Loan_ID              614 non-null object
    Gender               601 non-null object
    Married              611 non-null object
    Dependents           599 non-null float64
    Education            614 non-null object
    Self_Employed        582 non-null object
    ApplicantIncome      614 non-null int64
    CoapplicantIncome    614 non-null float64
    LoanAmount           592 non-null float64
    Loan_Amount_Term     600 non-null float64
    Credit_History       564 non-null float64
    Property_Area        614 non-null object
    Loan_Status          614 non-null object
    dtypes: float64(5), int64(1), object(7)
    memory usage: 62.4+ KB
    


```python
Loan.isnull().sum()
```




    Loan_ID               0
    Gender               13
    Married               3
    Dependents           15
    Education             0
    Self_Employed        32
    ApplicantIncome       0
    CoapplicantIncome     0
    LoanAmount           22
    Loan_Amount_Term     14
    Credit_History       50
    Property_Area         0
    Loan_Status           0
    dtype: int64




```python
# For the sake of simplicity filling in the value with the number which has occured most of the type. 
#As for example, in case of column "Gender" Male is the most frequently occured category hence we will fill in the null value
#with "Male" and the same with the column "Self_Employed"

Loan = Loan.fillna({"Gender":"Male", "Self_Employed":"No"})
```


```python
Loan.drop('Loan_ID',axis=1,inplace=True)
Loan.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Married</th>
      <th>Dependents</th>
      <th>Education</th>
      <th>Self_Employed</th>
      <th>ApplicantIncome</th>
      <th>CoapplicantIncome</th>
      <th>LoanAmount</th>
      <th>Loan_Amount_Term</th>
      <th>Credit_History</th>
      <th>Property_Area</th>
      <th>Loan_Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Male</td>
      <td>No</td>
      <td>0.0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>5849</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Male</td>
      <td>Yes</td>
      <td>1.0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>4583</td>
      <td>1508.0</td>
      <td>128.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Rural</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Male</td>
      <td>Yes</td>
      <td>0.0</td>
      <td>Graduate</td>
      <td>Yes</td>
      <td>3000</td>
      <td>0.0</td>
      <td>66.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Male</td>
      <td>Yes</td>
      <td>0.0</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>2583</td>
      <td>2358.0</td>
      <td>120.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Male</td>
      <td>No</td>
      <td>0.0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>6000</td>
      <td>0.0</td>
      <td>141.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Univariate Analysis 

Loan.isnull().sum()
```




    Gender                0
    Married               3
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




```python
Loan['Gender'].value_counts()
```




    Male      502
    Female    112
    Name: Gender, dtype: int64




```python
Loan['Education'].value_counts()
```




    Graduate        480
    Not Graduate    134
    Name: Education, dtype: int64




```python
Loan['Dependents'].value_counts()
```




    0.0    345
    1.0    102
    2.0    101
    4.0     50
    3.0      1
    Name: Dependents, dtype: int64




```python
Loan['Married'].value_counts()
```




    Yes    398
    No     213
    Name: Married, dtype: int64




```python
Loan['Dependents'].value_counts()
```




    0.0    345
    1.0    102
    2.0    101
    4.0     50
    3.0      1
    Name: Dependents, dtype: int64




```python
Loan['Self_Employed'].value_counts()
```




    No     532
    Yes     82
    Name: Self_Employed, dtype: int64




```python
Loan['Property_Area'].value_counts()
```




    Semiurban    233
    Urban        202
    Rural        179
    Name: Property_Area, dtype: int64




```python
Loan['Loan_Status'].value_counts()
```




    Y    422
    N    192
    Name: Loan_Status, dtype: int64




```python
# Taking the backup of Loan data into Loan_bkp
Loan_bkp = Loan
Loan_bkp.shape,Loan.shape
```




    ((614, 12), (614, 12))




```python
Loan_bkp.shape
```




    (614, 12)




```python
## Converting categorical values to numerical values using Scikit-Learn Level Encoding for the main Loan dataset ##

from sklearn.preprocessing import LabelEncoder

lbl_encoder = LabelEncoder()
Loan["Gender"] = lbl_encoder.fit_transform(Loan["Gender"])
Loan["Married"] = lbl_encoder.fit_transform(Loan["Married"])
Loan["Education"] = lbl_encoder.fit_transform(Loan["Education"])
Loan["Self_Employed"] = lbl_encoder.fit_transform(Loan["Self_Employed"])
Loan["Property_Area"] = lbl_encoder.fit_transform(Loan["Property_Area"])
Loan["Loan_Status"] = lbl_encoder.fit_transform(Loan["Loan_Status"])
```

    C:\Users\admin\Anaconda2\lib\site-packages\numpy\lib\arraysetops.py:216: FutureWarning: numpy not_equal will not check object identity in the future. The comparison did not return the same result as suggested by the identity (`is`)) and will change.
      flag = np.concatenate(([True], aux[1:] != aux[:-1]))
    


```python
Loan.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Married</th>
      <th>Dependents</th>
      <th>Education</th>
      <th>Self_Employed</th>
      <th>ApplicantIncome</th>
      <th>CoapplicantIncome</th>
      <th>LoanAmount</th>
      <th>Loan_Amount_Term</th>
      <th>Credit_History</th>
      <th>Property_Area</th>
      <th>Loan_Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>5849</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>4583</td>
      <td>1508.0</td>
      <td>128.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>3000</td>
      <td>0.0</td>
      <td>66.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>2583</td>
      <td>2358.0</td>
      <td>120.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>6000</td>
      <td>0.0</td>
      <td>141.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>2</td>
      <td>2.0</td>
      <td>0</td>
      <td>1</td>
      <td>5417</td>
      <td>4196.0</td>
      <td>267.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>2</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>2333</td>
      <td>1516.0</td>
      <td>95.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>2</td>
      <td>3.0</td>
      <td>0</td>
      <td>0</td>
      <td>3036</td>
      <td>2504.0</td>
      <td>158.0</td>
      <td>360.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>2</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>4006</td>
      <td>1526.0</td>
      <td>168.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>2</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>12841</td>
      <td>10968.0</td>
      <td>349.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Checking Null Values
Loan.isnull().sum()
```




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




```python
#descriptive statistics summary
Loan['LoanAmount'].describe()
#Loan['LoanAmount'].mean()
```




    count    592.000000
    mean     146.412162
    std       85.587325
    min        9.000000
    25%      100.000000
    50%      128.000000
    75%      168.000000
    max      700.000000
    Name: LoanAmount, dtype: float64




```python
Loan['Loan_Amount_Term'].describe()
```




    count    600.00000
    mean     342.00000
    std       65.12041
    min       12.00000
    25%      360.00000
    50%      360.00000
    75%      360.00000
    max      480.00000
    Name: Loan_Amount_Term, dtype: float64




```python
Loan['Credit_History'].describe()
```




    count    564.000000
    mean       0.842199
    std        0.364878
    min        0.000000
    25%        1.000000
    50%        1.000000
    75%        1.000000
    max        1.000000
    Name: Credit_History, dtype: float64




```python
Loan['Credit_History'].value_counts()
```




    1.0    475
    0.0     89
    Name: Credit_History, dtype: int64




```python
## Replacing NAN values with 0 and mean ## 

Loan['Dependents'].fillna(value=0,axis=0,inplace=True)
Loan['LoanAmount'].fillna(value=Loan['LoanAmount'].mean(),axis=0,inplace=True)
Loan['Loan_Amount_Term'].fillna(value=Loan['Loan_Amount_Term'].mean(),axis=0,inplace=True)
Loan['Credit_History'].fillna(value=Loan['Credit_History'].mean(),axis=0,inplace=True)
```


```python
Loan.info()
```

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
    


```python
## Histogram

import seaborn as sns
sns.distplot(Loan['ApplicantIncome'],color='r')
#sns.distplot(Loan['LoanAmount'],color='b')
```




    <matplotlib.axes._subplots.AxesSubplot at 0xc586278>




![png](output_27_1.png)



```python
print("Skewness: %f" % Loan['ApplicantIncome'].skew())
print("Kurtosis: %f" % Loan['ApplicantIncome'].kurt())
```

    Skewness: 6.539513
    Kurtosis: 60.540676
    


```python
import seaborn as sns
import numpy as np
sns.distplot(np.log(Loan['ApplicantIncome']),color='b')
```




    <matplotlib.axes._subplots.AxesSubplot at 0xc780f60>




![png](output_29_1.png)



```python
print("Skewness: %f" % np.log(Loan['ApplicantIncome']).skew())
print("Kurtosis: %f" % np.log(Loan['ApplicantIncome']).kurt())
```

    Skewness: 0.479580
    Kurtosis: 3.686875
    


```python
import seaborn as sns
sns.distplot(Loan['LoanAmount'],color='r')
```




    <matplotlib.axes._subplots.AxesSubplot at 0xd9304a8>




![png](output_31_1.png)



```python
print("Skewness: %f" % Loan['LoanAmount'].skew())
print("Kurtosis: %f" % Loan['LoanAmount'].kurt())
```

    Skewness: 2.726601
    Kurtosis: 10.896456
    


```python
import seaborn as sns
import numpy as np
sns.distplot(np.log(Loan['LoanAmount']),color='b')
```




    <matplotlib.axes._subplots.AxesSubplot at 0xc97a358>




![png](output_33_1.png)



```python
print("Skewness: %f" % np.log(Loan['LoanAmount']).skew())
print("Kurtosis: %f" % np.log(Loan['LoanAmount']).kurt())
```

    Skewness: -0.223227
    Kurtosis: 2.799973
    


```python
Loan['CoapplicantIncome'].describe()
```




    count      614.000000
    mean      1621.245798
    std       2926.248369
    min          0.000000
    25%          0.000000
    50%       1188.500000
    75%       2297.250000
    max      41667.000000
    Name: CoapplicantIncome, dtype: float64




```python
import seaborn as sns
sns.distplot(Loan['CoapplicantIncome'],color='r')
```




    <matplotlib.axes._subplots.AxesSubplot at 0xdda8cc0>




![png](output_36_1.png)



```python
Loan['CoapplicantIncome'].loc[Loan['CoapplicantIncome'] == 0] = 1
```

    C:\Users\admin\Anaconda2\lib\site-packages\pandas\core\indexing.py:179: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self._setitem_with_indexer(indexer, value)
    


```python
Loan.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Married</th>
      <th>Dependents</th>
      <th>Education</th>
      <th>Self_Employed</th>
      <th>ApplicantIncome</th>
      <th>CoapplicantIncome</th>
      <th>LoanAmount</th>
      <th>Loan_Amount_Term</th>
      <th>Credit_History</th>
      <th>Property_Area</th>
      <th>Loan_Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>5849</td>
      <td>1.0</td>
      <td>146.412162</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>4583</td>
      <td>1508.0</td>
      <td>128.000000</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>3000</td>
      <td>1.0</td>
      <td>66.000000</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>2583</td>
      <td>2358.0</td>
      <td>120.000000</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>6000</td>
      <td>1.0</td>
      <td>141.000000</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
import seaborn as sns
sns.distplot(np.log(Loan['CoapplicantIncome']),color='b')
```




    <matplotlib.axes._subplots.AxesSubplot at 0xe34ba58>




![png](output_39_1.png)



```python
Loan['CoapplicantIncome'].describe()
```




    count      614.000000
    mean      1621.690423
    std       2926.001661
    min          1.000000
    25%          1.000000
    50%       1188.500000
    75%       2297.250000
    max      41667.000000
    Name: CoapplicantIncome, dtype: float64




```python
# Multivariate Analysis
```


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



```python
import seaborn as sns
sns.pairplot(Loan,kind='scatter',size=1.5)
```




    <seaborn.axisgrid.PairGrid at 0xe596860>




![png](output_43_1.png)



```python
# Data binning
```


```python
def AppIncome(ApplicantIncome):
    if ApplicantIncome < 5000:
        return 1
    elif  5000 <= ApplicantIncome < 10000:
        return 2
    elif  10000 <= ApplicantIncome < 15000:
        return 3
    elif  15000 <= ApplicantIncome < 20000:
        return 4
    else:
        return 5
```


```python
Loan['ApplicantIncome'] = Loan['ApplicantIncome'].apply(AppIncome)
```


```python
def Loan_Amount(LoanAmount):
    if LoanAmount < 100:
        return 1
    elif  100 <= LoanAmount < 200:
        return 2
    elif  200 <= LoanAmount < 300:
        return 3
    else:
        return 4 
```


```python
Loan['LoanAmount'] = Loan['LoanAmount'].apply(Loan_Amount)
```


```python
def Coapplicant_Income(CoapplicantIncome):
    if CoapplicantIncome < 1000:
        return 1
    elif  1000 <= CoapplicantIncome < 2000:
        return 2
    elif  2000 <= CoapplicantIncome < 3000:
        return 3
    elif  3000 <= CoapplicantIncome < 4000:
        return 4
    elif  4000 <= CoapplicantIncome < 5000:
        return 5
    elif  5000 <= CoapplicantIncome < 6000:
        return 6
    elif  6000 <= CoapplicantIncome < 7000:
        return 7
    elif  7000 <= CoapplicantIncome < 8000:
        return 8
    elif  8000 <= CoapplicantIncome < 9000:
        return 9
    else:
        return 10
```


```python
Loan['CoapplicantIncome'] = Loan['CoapplicantIncome'].apply(Coapplicant_Income)
```


```python
def LoanAmount_Term(Loan_Amount_Term):
    if Loan_Amount_Term < 100:
        return 1
    elif  100 <= Loan_Amount_Term < 200:
        return 2
    elif  200 <= Loan_Amount_Term < 300:
        return 3
    elif  300 <= Loan_Amount_Term < 400:
        return 4
    else:
        return 5
```


```python
Loan['Loan_Amount_Term'] = Loan['Loan_Amount_Term'].apply(LoanAmount_Term)
```


```python
Loan.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Married</th>
      <th>Dependents</th>
      <th>Education</th>
      <th>Self_Employed</th>
      <th>ApplicantIncome</th>
      <th>CoapplicantIncome</th>
      <th>LoanAmount</th>
      <th>Loan_Amount_Term</th>
      <th>Credit_History</th>
      <th>Property_Area</th>
      <th>Loan_Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>1.0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1.0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>1.0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>1.0</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
Loan.isnull().sum()
```




    Gender               0
    Married              0
    Dependents           0
    Education            0
    Self_Employed        0
    ApplicantIncome      0
    CoapplicantIncome    0
    LoanAmount           0
    Loan_Amount_Term     0
    Credit_History       0
    Property_Area        0
    Loan_Status          0
    dtype: int64




```python
# Dataframe after data binning
Loan.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Married</th>
      <th>Dependents</th>
      <th>Education</th>
      <th>Self_Employed</th>
      <th>ApplicantIncome</th>
      <th>CoapplicantIncome</th>
      <th>LoanAmount</th>
      <th>Loan_Amount_Term</th>
      <th>Credit_History</th>
      <th>Property_Area</th>
      <th>Loan_Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>1.0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1.0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>1.0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>1.0</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#correlation matrix
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


![png](output_56_0.png)



```python
Loan.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 614 entries, 0 to 613
    Data columns (total 12 columns):
    Gender               614 non-null int64
    Married              614 non-null int64
    Dependents           614 non-null float64
    Education            614 non-null int64
    Self_Employed        614 non-null int64
    ApplicantIncome      614 non-null int64
    CoapplicantIncome    614 non-null int64
    LoanAmount           614 non-null int64
    Loan_Amount_Term     614 non-null int64
    Credit_History       614 non-null float64
    Property_Area        614 non-null int64
    Loan_Status          614 non-null int64
    dtypes: float64(2), int64(10)
    memory usage: 57.6 KB
    


```python
Loan.shape
```




    (614, 12)




```python
Y = Loan['Loan_Status']
X = pd.concat([Loan['Gender'], Loan['Married'], Loan['Dependents'], Loan['Education'], Loan['Self_Employed'], Loan['ApplicantIncome'], Loan['CoapplicantIncome'], Loan['LoanAmount'], Loan['Loan_Amount_Term'], Loan['Credit_History'], Loan['Property_Area']],axis=1) 
```


```python
Y.shape, X.shape
```




    ((614L,), (614, 11))




```python
#################### Logistic Regression ###############################
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

    C:\Users\admin\Anaconda2\lib\site-packages\sklearn\cross_validation.py:44: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)
    




    <matplotlib.text.Text at 0x1aa670f0>




![png](output_61_2.png)



```python
#################### Support Vector Machine ###############################

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




    <matplotlib.text.Text at 0x1ac8c198>




![png](output_62_1.png)



```python
#################### K Nearest Neighbour ###############################
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
plt.title('Plot Accuracy Vs Testsize Using K Nearest Neighbour with neighbours = 1,2,3,4,5,6,7')

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




    <matplotlib.text.Text at 0x1c139908>




![png](output_63_1.png)



```python
#################### Naive Bayes ###############################
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




    <matplotlib.text.Text at 0x1c230828>




![png](output_64_1.png)



```python
#################### Random Forrest Regression ###############################
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




    <matplotlib.text.Text at 0x1c319668>




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




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
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




    <matplotlib.text.Text at 0x1f79bd68>




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




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
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




    <matplotlib.text.Text at 0x1f6a9860>




![png](output_69_1.png)



```python
import seaborn as sns
sns.distplot(df['Logistic Regression'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x20154860>




![png](output_70_1.png)



```python
import seaborn as sns
sns.distplot(df['Naive Bayes'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x207df2b0>




![png](output_71_1.png)



```python
import seaborn as sns
sns.distplot(df['Random Forest'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2092c5f8>




![png](output_72_1.png)



```python
import seaborn as sns
sns.distplot(df['Support Vector Machine'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x20d3c160>




![png](output_73_1.png)



```python
import seaborn as sns
sns.distplot(df['k Nearrest Neighbor=5'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x20fb14a8>




![png](output_74_1.png)



```python
sns.pairplot(df)
```




    <seaborn.axisgrid.PairGrid at 0x212789e8>




![png](output_75_1.png)



```python

```