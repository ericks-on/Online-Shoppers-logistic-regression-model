# Business-Understanding

As a Data Scientist, your task is to develop a predictive model to identify purchase intent for a segment of our online users. The goal is to accurately predict which users are likely to make a purchase during their web session, enabling the business to target potential buyers more effectively.

# Data-Understanding

The data is from [Kaggle](https://www.kaggle.com/competitions/22122shop/data)

There are 18 variables with 10 quantitative and 7 categorical input features, 500K observations (one row or web session per online user) in a tabular format with the first 50K rows missing the Rev revenue flag (1 for purchase intent and 0 otherwise). The data was collected for the period of 1 year.

Features description (from the dataset authors):

1. `Administrative (Adm)`, `Informational (Inf)`,` Product Related (Prd)`: 
    - counts of different page types viewed by the user in that session. Values are derived from the URL information of the pages visited by the user and updated in real time when a user takes an action, e.g. moving from one page to another.
2. `Administrative Duration (AdmDur)`, `Informational Duration (InfDur)`, and `Product Related Duration (PrdDur)`: 
    - total time spent on the page of the specified type.
3. `Bounce Rate (BncRt):` 
    - %visitors who enter the site from that page and then leave ("bounce") without triggering any other requests to the analytics server during that session. Measured by Google Analytics
4. `Exit Rate (ExtRt)`: 
    - %visitors that were the last in the session. Calculated as for all pageviews to the page. Measured by Google Analytics.
5. `Page Value (PgVal)`: 
    - average value for a web page that the user visited before completing an e-commerce transaction.
6. `Special Day (SpclDay)`: 
    - closeness of the site visiting time to a specific special day (e.g. Mother’s Day, Valentine's Day) in which the sessions are more likely to be finalized with transaction. The value of this attribute is determined by considering the dynamics of e-commerce such as the duration between the order date and delivery date. For example, for Valentine's day, this value takes a nonzero value between Feb 2 and Feb 12, zero before and after this date unless it is close to another special day, and its maximum value of 1 on Feb 8.
7. `Operating system (OS)` of the user's PC
8. `Browser (Bsr)`: web user's web browser
9. `Region (Rgn) `of the web user
10. `Traffic type (TfcTp)`: TBD
11. `Visitor type (VstTp)`: Type of visitor
12. `Weekend (Wknd)`: whether the page view event took place on weekend
13. `Month of the year (Mo)`: the month of the page view event


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
sns.set_style('darkgrid')
```


```python
df = pd.read_csv('XY_Shop.csv')
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Adm</th>
      <th>AdmDur</th>
      <th>Inf</th>
      <th>InfDur</th>
      <th>Prd</th>
      <th>PrdDur</th>
      <th>BncRt</th>
      <th>ExtRt</th>
      <th>PgVal</th>
      <th>SpclDay</th>
      <th>Mo</th>
      <th>OS</th>
      <th>Bsr</th>
      <th>Rgn</th>
      <th>TfcTp</th>
      <th>VstTp</th>
      <th>Wkd</th>
      <th>Rev</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
      <td>18</td>
      <td>132.99</td>
      <td>0.038211</td>
      <td>0.054523</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
      <td>37</td>
      <td>1150.20</td>
      <td>0.001245</td>
      <td>0.030321</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>11</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>191.98</td>
      <td>0</td>
      <td>0.00</td>
      <td>38</td>
      <td>1266.78</td>
      <td>0.004742</td>
      <td>0.019551</td>
      <td>17.816864</td>
      <td>0.0</td>
      <td>10</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>263.68</td>
      <td>0</td>
      <td>0.00</td>
      <td>24</td>
      <td>749.14</td>
      <td>0.004474</td>
      <td>0.024079</td>
      <td>14.578547</td>
      <td>0.0</td>
      <td>7</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
      <td>3</td>
      <td>136.41</td>
      <td>0.000000</td>
      <td>0.066300</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>12</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>499995</th>
      <td>14</td>
      <td>262.46</td>
      <td>0</td>
      <td>0.00</td>
      <td>170</td>
      <td>3967.02</td>
      <td>0.003314</td>
      <td>0.015669</td>
      <td>2.904034</td>
      <td>0.0</td>
      <td>12</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>499996</th>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
      <td>7</td>
      <td>295.57</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>9</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>499997</th>
      <td>6</td>
      <td>134.64</td>
      <td>3</td>
      <td>92.28</td>
      <td>30</td>
      <td>888.50</td>
      <td>0.000000</td>
      <td>0.003452</td>
      <td>30.172020</td>
      <td>0.0</td>
      <td>11</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>499998</th>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
      <td>27</td>
      <td>1185.14</td>
      <td>0.000000</td>
      <td>0.001593</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>499999</th>
      <td>6</td>
      <td>51.36</td>
      <td>0</td>
      <td>0.00</td>
      <td>59</td>
      <td>1898.21</td>
      <td>0.000000</td>
      <td>0.003224</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>12</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>500000 rows × 18 columns</p>
</div>



# EDA

## Data-Cleaning

We first split the data to remove the rows with missing values in the rev column.


```python
# checking the missing values
df.isna().sum().sum()
```




    50000




```python
missing_rev = df.loc[df.Rev.isna()]
df.dropna(inplace=True)
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 450000 entries, 50000 to 499999
    Data columns (total 18 columns):
     #   Column   Non-Null Count   Dtype  
    ---  ------   --------------   -----  
     0   Adm      450000 non-null  int64  
     1   AdmDur   450000 non-null  float64
     2   Inf      450000 non-null  int64  
     3   InfDur   450000 non-null  float64
     4   Prd      450000 non-null  int64  
     5   PrdDur   450000 non-null  float64
     6   BncRt    450000 non-null  float64
     7   ExtRt    450000 non-null  float64
     8   PgVal    450000 non-null  float64
     9   SpclDay  450000 non-null  float64
     10  Mo       450000 non-null  int64  
     11  OS       450000 non-null  int64  
     12  Bsr      450000 non-null  int64  
     13  Rgn      450000 non-null  int64  
     14  TfcTp    450000 non-null  int64  
     15  VstTp    450000 non-null  int64  
     16  Wkd      450000 non-null  int64  
     17  Rev      450000 non-null  float64
    dtypes: float64(8), int64(10)
    memory usage: 65.2 MB
    

Now we check for duplicates


```python
df.duplicated().sum()
```




    87




```python
df.drop_duplicates(inplace=True)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Adm</th>
      <th>AdmDur</th>
      <th>Inf</th>
      <th>InfDur</th>
      <th>Prd</th>
      <th>PrdDur</th>
      <th>BncRt</th>
      <th>ExtRt</th>
      <th>PgVal</th>
      <th>SpclDay</th>
      <th>Mo</th>
      <th>OS</th>
      <th>Bsr</th>
      <th>Rgn</th>
      <th>TfcTp</th>
      <th>VstTp</th>
      <th>Wkd</th>
      <th>Rev</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>50000</th>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.157489</td>
      <td>0.115168</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>12</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>50001</th>
      <td>7</td>
      <td>116.19</td>
      <td>0</td>
      <td>0.00</td>
      <td>79</td>
      <td>2683.58</td>
      <td>0.000000</td>
      <td>0.001910</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>50002</th>
      <td>6</td>
      <td>233.07</td>
      <td>0</td>
      <td>0.00</td>
      <td>8</td>
      <td>171.08</td>
      <td>0.033427</td>
      <td>0.062316</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>12</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>50003</th>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
      <td>5</td>
      <td>0.00</td>
      <td>0.140943</td>
      <td>0.160706</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>6</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>50004</th>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
      <td>6</td>
      <td>112.57</td>
      <td>0.035324</td>
      <td>0.021440</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>10</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>499995</th>
      <td>14</td>
      <td>262.46</td>
      <td>0</td>
      <td>0.00</td>
      <td>170</td>
      <td>3967.02</td>
      <td>0.003314</td>
      <td>0.015669</td>
      <td>2.904034</td>
      <td>0.0</td>
      <td>12</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>499996</th>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
      <td>7</td>
      <td>295.57</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>9</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>499997</th>
      <td>6</td>
      <td>134.64</td>
      <td>3</td>
      <td>92.28</td>
      <td>30</td>
      <td>888.50</td>
      <td>0.000000</td>
      <td>0.003452</td>
      <td>30.172020</td>
      <td>0.0</td>
      <td>11</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>499998</th>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
      <td>27</td>
      <td>1185.14</td>
      <td>0.000000</td>
      <td>0.001593</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>499999</th>
      <td>6</td>
      <td>51.36</td>
      <td>0</td>
      <td>0.00</td>
      <td>59</td>
      <td>1898.21</td>
      <td>0.000000</td>
      <td>0.003224</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>12</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>449913 rows × 18 columns</p>
</div>



## Page Visits


```python
page_visits = df[['Adm', 'Inf', 'Prd']]

# plotting the average page visits per session
avg_visits = page_visits.mean().astype(int)
print(avg_visits)


full_names = {'Inf': 'Informational',
              'Adm': 'Administrative', 'Prd': 'Products'}
colors = sns.color_palette('pastel')  # Seaborn color palette

plt.figure(figsize=(8, 8))  # Optional: Set the figure size
plt.pie(avg_visits, labels=avg_visits.index.map(full_names),
        autopct='%1.1f%%', startangle=140, colors=colors)

# Add a custom legend
plt.legend(title="Categories", loc="best")

plt.title('Average Visits Distribution')
plt.show()
```

    Adm     2
    Inf     0
    Prd    30
    dtype: int32
    


    
![png](index_files/index_14_1.png)
    


- Products page averages the most visits per session with about 30 per session.
- There is a contrast in page visits for the Administrative pages with an average of 0 visits per session.
- Informational pages averages no visits per session.


```python
page_time = df[['AdmDur', 'InfDur','PrdDur']]
avg_page_time = page_time.mean()
print(avg_page_time)

# plotting
sns.barplot(x=avg_page_time.index, y=avg_page_time);
```

    AdmDur      76.870743
    InfDur      31.830116
    PrdDur    1165.230601
    dtype: float64
    


    
![png](index_files/index_16_1.png)
    


Same case for the duration, most time is spent on the product pages compared to the administrative and informational pages.

## Bounce and Exit Rates


```python
bounce_exit = df[['BncRt', 'ExtRt']]
bounce_exit.mean()
```




    BncRt    0.018359
    ExtRt    0.036841
    dtype: float64



- Bounce rate is relatively low at around 2%.
- Same case for Exit rate at around 4%

## Purchases on different times of the year


```python
monthly_rev = df.groupby('Mo')['Rev'].sum().reset_index()
monthly_rev
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mo</th>
      <th>Rev</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>252.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4388.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>4440.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>11889.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>8223.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>9561.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>3218.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>3813.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>4389.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>14331.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>17111.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>19243.0</td>
    </tr>
  </tbody>
</table>
</div>



we now convert the month numbers to the actual names of the months then plot.



```python
import calendar

# Transforming the month column
monthly_rev['Mo'] = monthly_rev.Mo.map(
    lambda x: calendar.month_name[int(x)]
)
```


```python
# plotting the distribution
sns.lineplot(
    monthly_rev,
    x='Mo', 
    y='Rev'
)
plt.xlabel('Month')
plt.ylabel('Purchases')
plt.title('Purchases Throgh the year')
plt.xticks(rotation=45);
```

    c:\Users\mutis\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    c:\Users\mutis\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    


    
![png](index_files/index_25_1.png)
    


- Purchases tend to peak in October and continue to rise steadily, reaching their highest levels by the end of the year. This can be contributed by special events such as holiday shopping and special promotions among others.
- Purchases tend to be lowest in January, gradually rise to a peak in April, though not as high as the peak in October, and then steadily decline afterward.
- This pattern reflects typical consumer behavior influenced by seasonal events, financial cycles, and marketing strategies.

## Purchases on special days

Investigating how closeness to the special days influences the purchases.


```python
special_purchases = df.loc[df.SpclDay > 0].groupby(
    'SpclDay'
)['Rev'].sum().reset_index()
special_purchases.columns = ['Closeness', 'Purchases']
```


```python
# Plotting the trend
sns.lineplot(
    special_purchases,
    x='Closeness',
    y='Purchases'
)

plt.title('Effect of closeness to Special days on purchases');
```

    c:\Users\mutis\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    c:\Users\mutis\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    


    
![png](index_files/index_30_1.png)
    



Purchases tend to increase when the closeness value is around the midpoint, indicating a balanced anticipation of the special day. As the date of the special event approaches, the likelihood of making a purchase gradually decreases. Conversely, when the event is farther away, purchase activity is relatively lower but starts to rise as the date nears, peaking at the midpoint before gradually declining again.

## Purchases on weekends

Effect of weekends on purchases


```python
df.Wkd.value_counts()
```




    Wkd
    0    321053
    1    128860
    Name: count, dtype: int64




```python
weekend_purchases = df.groupby('Wkd')['Rev'].mean().reset_index()

# plotting
sns.barplot(
    weekend_purchases,
    x='Wkd',
    y='Rev'
)
print(weekend_purchases)

plt.xticks([0, 1], labels=['Normal', 'Weekend'])
plt.xlabel('Day')
plt.ylabel('purchases %');
```

       Wkd       Rev
    0    0  0.171043
    1    1  0.356542
    


    
![png](index_files/index_35_1.png)
    


Purchases occur more on weekends than on normal days as indicated by the chart. About 35% of the visits in the weekends result to purchases compared to normal days at 17%.

## Page value

Investigating the effect of page value to the purchases.


```python
df[['PgVal']].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PgVal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>449913.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.500630</td>
    </tr>
    <tr>
      <th>std</th>
      <td>16.525529</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>360.740676</td>
    </tr>
  </tbody>
</table>
</div>



The page value ranges from 0 to 360. We bin the range to different values to get the  purchases as value increases.


```python
value_df = df.copy()
# binning the page value to 10 categories
value_df['val_cat'] = pd.cut(df['PgVal'], bins=10, labels=np.arange(1, 11))

# get the percentage of purchases
value_cat_purchases = value_df.groupby('val_cat')['Rev'].mean().reset_index()

# plotting
sns.barplot(
    value_cat_purchases,
    x='val_cat',
    y='Rev',
)

plt.title('Purchases According to page value')
plt.xlabel('Page Value')
plt.ylabel('Purchases');
```

    C:\Users\mutis\AppData\Local\Temp\ipykernel_18132\934730105.py:6: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      value_cat_purchases = value_df.groupby('val_cat')['Rev'].mean().reset_index()
    c:\Users\mutis\anaconda3\Lib\site-packages\seaborn\categorical.py:641: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped_vals = vals.groupby(grouper)
    


    
![png](index_files/index_41_1.png)
    


There is an upward trend as the page value increases, more purchases are made.

# Data preparation


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 449913 entries, 50000 to 499999
    Data columns (total 18 columns):
     #   Column   Non-Null Count   Dtype  
    ---  ------   --------------   -----  
     0   Adm      449913 non-null  int64  
     1   AdmDur   449913 non-null  float64
     2   Inf      449913 non-null  int64  
     3   InfDur   449913 non-null  float64
     4   Prd      449913 non-null  int64  
     5   PrdDur   449913 non-null  float64
     6   BncRt    449913 non-null  float64
     7   ExtRt    449913 non-null  float64
     8   PgVal    449913 non-null  float64
     9   SpclDay  449913 non-null  float64
     10  Mo       449913 non-null  int64  
     11  OS       449913 non-null  int64  
     12  Bsr      449913 non-null  int64  
     13  Rgn      449913 non-null  int64  
     14  TfcTp    449913 non-null  int64  
     15  VstTp    449913 non-null  int64  
     16  Wkd      449913 non-null  int64  
     17  Rev      449913 non-null  float64
    dtypes: float64(8), int64(10)
    memory usage: 65.2 MB
    

most columns are either in float64 format or int64 but some of them are categorical. So we convert the categorical columns to object.


```python
df[['Mo', 'OS', 'Bsr', 'Rgn', 'TfcTp', 'VstTp']] = (
    df[['Mo', 'OS', 'Bsr', 'Rgn', 'TfcTp', 'VstTp']].astype('object')
)
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 449913 entries, 50000 to 499999
    Data columns (total 18 columns):
     #   Column   Non-Null Count   Dtype  
    ---  ------   --------------   -----  
     0   Adm      449913 non-null  int64  
     1   AdmDur   449913 non-null  float64
     2   Inf      449913 non-null  int64  
     3   InfDur   449913 non-null  float64
     4   Prd      449913 non-null  int64  
     5   PrdDur   449913 non-null  float64
     6   BncRt    449913 non-null  float64
     7   ExtRt    449913 non-null  float64
     8   PgVal    449913 non-null  float64
     9   SpclDay  449913 non-null  float64
     10  Mo       449913 non-null  object 
     11  OS       449913 non-null  object 
     12  Bsr      449913 non-null  object 
     13  Rgn      449913 non-null  object 
     14  TfcTp    449913 non-null  object 
     15  VstTp    449913 non-null  object 
     16  Wkd      449913 non-null  int64  
     17  Rev      449913 non-null  float64
    dtypes: float64(8), int64(4), object(6)
    memory usage: 65.2+ MB
    

now our data is in the correct data types.

# Modelling

We first start by spitting the data to the train and test sets.

## Preprocessing


```python
# importing modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
```


```python
# the seed for random generators
random_state = 20
```


```python
X = df.drop('Rev', axis=1)
y = df['Rev']
# splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=random_state)
```

We now proceed to create dummy variables for the categorical columns. Remember wkd is already a binary variable so we include it after creating dummy variables for the other categorical features.


```python
# creating dummy variables for categorical
X_train_dummy = pd.get_dummies(X_train.select_dtypes('object'),
                               drop_first=True,
                               dtype=int)
# add weekend to the set of dummy variable
X_train_dummy['Wkd'] = df['Wkd']
```

## 1. Baseline model

For our case we first fit a Logistic regression model


```python
y_train.value_counts(normalize=True)
```




    Rev
    0.0    0.775805
    1.0    0.224195
    Name: proportion, dtype: float64



There is class imbalance in the target variable with about 78% of the sessions leading to no purchases and other 22% to purchases.

We first fit a baseline model with the data as is.



```python
scaler = StandardScaler()
numeric_training = X_train.select_dtypes(exclude='object').drop('Wkd',
                                                                axis=1)

# scaling X
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(numeric_training),
    columns=numeric_training.columns,
    index=numeric_training.index
)

X_train_scaled = pd.concat([X_train_scaled, X_train_dummy], axis=1)


baseline_model = LogisticRegression(random_state=random_state, solver='liblinear')
baseline_model.fit(X_train_scaled, y_train)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression(random_state=20, solver=&#x27;liblinear&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(random_state=20, solver=&#x27;liblinear&#x27;)</pre></div></div></div></div></div>



### Evaluation of the baseline model

We use the log loss to measure the performance of the model. We measure both log loss in the performance of the model on train data and also use cross validation and get the log loss.


```python
# importing the modules
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score


# Get the log loss for train and validation
baseline_train_loss = log_loss(y_train, baseline_model.decision_function(X_train_scaled) )

# use cross validation
val_scores = cross_val_score(baseline_model,
                             X_train_scaled,
                             y_train,
                             cv=10,
                             scoring='neg_log_loss')
baseline_validation_loss = -val_scores.mean()

print('Baseline model train log loss: ', baseline_train_loss)
print('Baseline_model validation log loss: ', baseline_validation_loss)
```

    Baseline model train log loss:  6.490913445799422
    Baseline_model validation log loss:  0.4416013217663197
    

This model performs better in the validation data, but this could mean underfitting due to the difference in the log loss between train and validation data performance.

## 2. Fix class imbalance

There is class imbalance in the targer with about 78% for one class and 22% for the other.

We use SMOTE oversampling to fix this but not on the whole train data. This is because of data leakage during cross validation. We therefore define afunction to carry out cross validation and include smote.


```python
# Creating a custom function for validation to include smote
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from imblearn.over_sampling import SMOTE


# function for validation
def get_train_val_loss(model, X_train, y_train, cont_cols, cat_cols):
    """calculates the validation log loss and the train log loss and includes 
    SMOTE oversampling to reduce class imbalance
    Args:
        model: the model
        X_train: the model features for train data where dummy variables are already
            created
        y_train: the target in the train data
        cont_cols: list of continuous columns
        cat_cols: list of categorical columns
    Returns:
        val_loss: validation log loss which is the mean of the test scores
    """
    # making a copy of the model not to change the original one
    modelcp = clone(model)
    
    sm = SMOTE(random_state=random_state)
    scaler = StandardScaler()
    
    # cross validation
    kfold = StratifiedKFold(n_splits=3)
    train_scores = np.ndarray(3)
    test_scores = np.ndarray(3)
    
    
    
    for i, (train_idx, test_idx) in enumerate(kfold.split(X_train, y_train)):
        X_t = X_train.iloc[train_idx]
        y_t = y_train.iloc[train_idx]
        X_v = X_train.iloc[test_idx]
        y_v = y_train.iloc[test_idx]
        
        # scaling
        # first separating continuous and categorical features
        # scaling the training set
        X_t_cat = X_t[cat_cols]
        X_t_cont = pd.DataFrame(
            scaler.fit_transform(X_t[cont_cols]),
            columns=cont_cols,
            index=X_t.index
        )
        # the test set
        X_v_cat = X_v[cat_cols]
        X_v_cont = pd.DataFrame(
            scaler.transform(X_v[cont_cols]),
            columns=cont_cols,
            index=X_v.index
        )
        
        # joinig categorical and continuous back together
        X_t_scaled = pd.concat(
            [X_t_cont, X_t_cat],
            axis=1
        )
        
        X_v_scaled = pd.concat(
            [X_v_cont, X_v_cat],
            axis=1
        )
        
        # Oversampling
        X_t_oversampled, y_t_oversampled = sm.fit_resample(X_t_scaled, y_t)
        
        # fitting model
        modelcp.fit(X_t_oversampled, y_t_oversampled)
        
        # training log loss 
        cross_train_score = log_loss(y_t_oversampled, modelcp.predict_proba(X_t_oversampled))
        
        # validation log loss
        cross_test_score = log_loss(y_v, modelcp.predict_proba(X_v_scaled))
        
        # append to list
        train_scores[i] =  cross_train_score
        test_scores[i] =  cross_test_score

    train_loss = train_scores.mean()
    val_loss = test_scores.mean()
    
    return train_loss, val_loss
```

First i convert the X_train to contain only dummy variables and create two lists to contain the names for categorical and continuous columns.


```python
# creating list for categorical and continuous columns
cat_cols = X_train_dummy.columns
cont_cols = list(X_train.select_dtypes(exclude='object'))
cont_cols.remove('Wkd')

# preprocessed X train
X_train_preprocessed = pd.concat(
    [numeric_training, X_train_dummy],
    axis=1
)
X_train_preprocessed
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Adm</th>
      <th>AdmDur</th>
      <th>Inf</th>
      <th>InfDur</th>
      <th>Prd</th>
      <th>PrdDur</th>
      <th>BncRt</th>
      <th>ExtRt</th>
      <th>PgVal</th>
      <th>SpclDay</th>
      <th>...</th>
      <th>TfcTp_7</th>
      <th>TfcTp_8</th>
      <th>TfcTp_9</th>
      <th>TfcTp_10</th>
      <th>TfcTp_11</th>
      <th>TfcTp_12</th>
      <th>TfcTp_13</th>
      <th>VstTp_1</th>
      <th>VstTp_2</th>
      <th>Wkd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>199738</th>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
      <td>22</td>
      <td>758.22</td>
      <td>0.000000</td>
      <td>0.017185</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>52555</th>
      <td>3</td>
      <td>35.97</td>
      <td>2</td>
      <td>15.98</td>
      <td>74</td>
      <td>4206.55</td>
      <td>0.019160</td>
      <td>0.026590</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>311137</th>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
      <td>3</td>
      <td>258.25</td>
      <td>0.000000</td>
      <td>0.031969</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>325494</th>
      <td>4</td>
      <td>98.46</td>
      <td>0</td>
      <td>0.00</td>
      <td>10</td>
      <td>63.57</td>
      <td>0.065401</td>
      <td>0.082387</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>379185</th>
      <td>3</td>
      <td>18.92</td>
      <td>0</td>
      <td>0.00</td>
      <td>8</td>
      <td>343.35</td>
      <td>0.000000</td>
      <td>0.007333</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>81962</th>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
      <td>16</td>
      <td>242.39</td>
      <td>0.000000</td>
      <td>0.004898</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>270085</th>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
      <td>4</td>
      <td>46.23</td>
      <td>0.000000</td>
      <td>0.084468</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>335966</th>
      <td>3</td>
      <td>93.33</td>
      <td>0</td>
      <td>0.00</td>
      <td>1</td>
      <td>4.00</td>
      <td>0.000000</td>
      <td>0.042272</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>87136</th>
      <td>6</td>
      <td>188.81</td>
      <td>0</td>
      <td>0.00</td>
      <td>7</td>
      <td>601.31</td>
      <td>0.020828</td>
      <td>0.029748</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>404838</th>
      <td>0</td>
      <td>0.00</td>
      <td>3</td>
      <td>447.54</td>
      <td>78</td>
      <td>3667.34</td>
      <td>0.004826</td>
      <td>0.018921</td>
      <td>10.024848</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>337434 rows × 52 columns</p>
</div>




```python
oversampled_train_loss, oversampled_val_loss = get_train_val_loss(
    baseline_model,
    X_train_preprocessed,
    y_train,
    cont_cols,
    cat_cols
)
print('Oversampled model train log loss:', oversampled_train_loss)
print('Oversampled model val log loss:', oversampled_val_loss)
```

    Oversampled model train log loss: 6.920449322542662
    Oversampled model val log loss: 4.653150186266458
    

This model performs worse than the baseline model indicatied by the higher log loss, but the variance bias tradeoff is better in this model. 

## 3. Less regularization

We try to reduce the regularization to improve the performance.


```python
c_vals = [10, 100, 1e3, 1e6]
for c in c_vals:
    less_reg_model = LogisticRegression(solver='liblinear',
                                        random_state=random_state,
                                        C=c)

    # getting train and validation scores for the  model 
    less_reg_train_loss, less_reg_val_loss = get_train_val_loss(less_reg_model,
                                                                X_train_preprocessed,
                                                                y_train,
                                                                cont_cols,
                                                                cat_cols)


    print(f'Less regularization({c}) model log loss on train data:', less_reg_train_loss)
    print(f'Less Regularization({c}) model log loss on validation:', less_reg_val_loss)
    print()
```

    Less regularization(10) model log loss on train data: 6.924038233818931
    Less Regularization(10) model log loss on validation: 4.655176337979004
    
    Less regularization(100) model log loss on train data: 6.924221889515656
    Less Regularization(100) model log loss on validation: 4.655790669341802
    
    Less regularization(1000.0) model log loss on train data: 6.924181771152384
    Less Regularization(1000.0) model log loss on validation: 4.65594174372054
    
    Less regularization(1000000.0) model log loss on train data: 6.9243141564783235
    Less Regularization(1000000.0) model log loss on validation: 4.655940369179685
    
    

Performance gets worse with less regularization. Lets try increasing the  regularization.


```python
c_vals = [.1, .01, .001, 1e-4, .1e-5, 1e-6, 1e-7, 1e-8]
l2_log_loss_items = []
for c in c_vals:
    item = {
        'c': c
    }
    less_reg_model = LogisticRegression(solver='liblinear',
                                  random_state=random_state,
                                  C=c)
    l2_train_loss, l2_val_loss = get_train_val_loss(less_reg_model,
                                                    X_train_preprocessed,
                                                    y_train,
                                                    cont_cols,
                                                    cat_cols)

    item['train_loss'] = l2_train_loss
    item['val_loss'] = l2_val_loss
    l2_log_loss_items.append(item)
    

```


```python
l2_loss_df = pd.DataFrame(l2_log_loss_items)
l2_loss_df['Train val difference'] = l2_loss_df['train_loss'] - \
    l2_loss_df['val_loss']

l2_loss_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>c</th>
      <th>train_loss</th>
      <th>val_loss</th>
      <th>Train val difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.000000e-01</td>
      <td>6.889687</td>
      <td>4.623700</td>
      <td>2.265987</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.000000e-02</td>
      <td>6.735191</td>
      <td>4.438825</td>
      <td>2.296366</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.000000e-03</td>
      <td>6.796242</td>
      <td>4.224101</td>
      <td>2.572141</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.000000e-04</td>
      <td>7.092018</td>
      <td>3.981004</td>
      <td>3.111014</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.000000e-06</td>
      <td>6.584323</td>
      <td>2.946785</td>
      <td>3.637538</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.000000e-06</td>
      <td>6.584323</td>
      <td>2.946785</td>
      <td>3.637538</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.000000e-07</td>
      <td>7.051304</td>
      <td>3.141002</td>
      <td>3.910302</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.000000e-08</td>
      <td>7.819918</td>
      <td>3.485243</td>
      <td>4.334676</td>
    </tr>
  </tbody>
</table>
</div>



### Plotting log loss at different regularization for L2


```python
from matplotlib.ticker import FuncFormatter

# Plotting


def scientific_format(x, _):
    return f'{x:.0e}'


# Plotting
fig, axes = plt.subplots(ncols=3, figsize=(15, 5))

# c vs train loss
sns.lineplot(
    data=l2_loss_df,
    x='c',
    y='train_loss',
    ax=axes[0]
)
axes[0].set_xlabel('C')
axes[0].set_ylabel('Train log loss')
axes[0].set_xscale('log')
axes[0].set_xticks([0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8])
axes[0].xaxis.set_major_formatter(FuncFormatter(scientific_format))

# c vs val loss
sns.lineplot(
    data=l2_loss_df,
    x='c',
    y='val_loss',
    ax=axes[1]
)
axes[1].set_xlabel('C')
axes[1].set_ylabel('Validation log loss')
axes[1].set_xscale('log')
axes[1].set_xticks([0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8])
axes[1].xaxis.set_major_formatter(FuncFormatter(scientific_format))

# c vs difference
sns.lineplot(
    data=l2_loss_df,
    x='c',
    y='Train val difference',
    ax=axes[2]
)
axes[2].set_xlabel('C')
axes[2].set_ylabel('Train-Validation Difference')
axes[2].set_xscale('log')
axes[2].set_xticks([0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8])
axes[2].xaxis.set_major_formatter(FuncFormatter(scientific_format))

fig.suptitle('L2 regularization')
plt.tight_layout()
plt.show()
```

    c:\Users\mutis\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    c:\Users\mutis\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    c:\Users\mutis\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    c:\Users\mutis\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    c:\Users\mutis\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    c:\Users\mutis\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    


    
![png](index_files/index_79_1.png)
    


For l2 regularization, As the regularization increases(lower C value), the difference between log loss in train data and validation data increases indicating possibility of underfitting.

Even when the log loss is low in train and validation sets at around C = 1e06, there is underfitting.

## 4. L1 regularization

We start with the l1 Regularization(Lasso)


```python
c_vals = [1, 10, 100, 1e3, 1e6]
for c in c_vals:
    l1_model = LogisticRegression(solver='saga',
                                random_state=random_state,
                                penalty='l1',
                                C=c)
    l1_train_loss, l1_val_loss = get_train_val_loss(l1_model,
                                                    X_train_preprocessed,
                                                    y_train,
                                                    cont_cols,
                                                    cat_cols)

    print(f'L1 regularization({c}) model log loss on train data:', l1_train_loss)
    print(f'L1 Regularization({c}) model log loss on validation:', l1_val_loss)
    print()
```

    L1 regularization(1) model log loss on train data: 6.922744438179024
    L1 Regularization(1) model log loss on validation: 4.654101704215641
    
    L1 regularization(10) model log loss on train data: 6.923966945276252
    L1 Regularization(10) model log loss on validation: 4.655553510318879
    
    L1 regularization(100) model log loss on train data: 6.924042418229262
    L1 Regularization(100) model log loss on validation: 4.655453570349502
    
    L1 regularization(1000.0) model log loss on train data: 6.924046941811287
    L1 Regularization(1000.0) model log loss on validation: 4.655460839683827
    
    L1 regularization(1000000.0) model log loss on train data: 6.9240480871271375
    L1 Regularization(1000000.0) model log loss on validation: 4.655462966680484
    
    


```python
c_vals = [.1, .01, .001, 1e-4, .1e-5, 1e-6, 1e-7,1e-8]
l1_log_loss_items = []
for c in c_vals:
    item = {
        'c': c
    }
    l1_model = LogisticRegression(solver='saga',
                                  random_state=random_state,
                                  penalty='l1',
                                  C=c)
    l1_train_loss, l1_val_loss = get_train_val_loss(l1_model,
                                                    X_train_preprocessed,
                                                    y_train,
                                                    cont_cols,
                                                    cat_cols)

    item['train_loss'] = l1_train_loss
    item['val_loss'] = l1_val_loss
    l1_log_loss_items.append(item)
    
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>c</th>
      <th>train_loss</th>
      <th>val_loss</th>
      <th>Train val difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.000000e-01</td>
      <td>6.909002</td>
      <td>4.636216</td>
      <td>2.272786</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.000000e-02</td>
      <td>6.764112</td>
      <td>4.422848</td>
      <td>2.341265</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.000000e-03</td>
      <td>6.621629</td>
      <td>3.992579</td>
      <td>2.629050</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.000000e-04</td>
      <td>8.727381</td>
      <td>4.462992</td>
      <td>4.264389</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.000000e-06</td>
      <td>2.256407</td>
      <td>1.018138</td>
      <td>1.238269</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.000000e-06</td>
      <td>2.256407</td>
      <td>1.018138</td>
      <td>1.238269</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.000000e-07</td>
      <td>2.256577</td>
      <td>1.018213</td>
      <td>1.238364</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.000000e-08</td>
      <td>2.256577</td>
      <td>1.018213</td>
      <td>1.238364</td>
    </tr>
  </tbody>
</table>
</div>




```python
l1_loss_df = pd.DataFrame(l1_log_loss_items)
l1_loss_df['Train val difference'] = l1_loss_df['train_loss'] - \
    l1_loss_df['val_loss']
l1_loss_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>c</th>
      <th>train_loss</th>
      <th>val_loss</th>
      <th>Train val difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.000000e-01</td>
      <td>6.909002</td>
      <td>4.636216</td>
      <td>2.272786</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.000000e-02</td>
      <td>6.764112</td>
      <td>4.422848</td>
      <td>2.341265</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.000000e-03</td>
      <td>6.621629</td>
      <td>3.992579</td>
      <td>2.629050</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.000000e-04</td>
      <td>8.727381</td>
      <td>4.462992</td>
      <td>4.264389</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.000000e-06</td>
      <td>2.256407</td>
      <td>1.018138</td>
      <td>1.238269</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.000000e-06</td>
      <td>2.256407</td>
      <td>1.018138</td>
      <td>1.238269</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.000000e-07</td>
      <td>2.256577</td>
      <td>1.018213</td>
      <td>1.238364</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.000000e-08</td>
      <td>2.256577</td>
      <td>1.018213</td>
      <td>1.238364</td>
    </tr>
  </tbody>
</table>
</div>



### Comparing log loss at different L1 regularizations


```python

# Plotting
fig, axes = plt.subplots(ncols=3, figsize=(15, 5))

# c vs train loss
sns.lineplot(
    data=l1_loss_df,
    x='c',
    y='train_loss',
    ax=axes[0]
)
axes[0].set_xlabel('C')
axes[0].set_ylabel('Train log loss')
axes[0].set_xscale('log')
axes[0].set_xticks([0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8])
axes[0].xaxis.set_major_formatter(FuncFormatter(scientific_format))

# c vs val loss
sns.lineplot(
    data=l1_loss_df,
    x='c',
    y='val_loss',
    ax=axes[1]
)
axes[1].set_xlabel('C')
axes[1].set_ylabel('Validation log loss')
axes[1].set_xscale('log')
axes[1].set_xticks([0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8])
axes[1].xaxis.set_major_formatter(FuncFormatter(scientific_format))

# c vs difference
sns.lineplot(
    data=l1_loss_df,
    x='c',
    y='Train val difference',
    ax=axes[2]
)
axes[2].set_xlabel('C')
axes[2].set_ylabel('Train-Validation Difference')
axes[2].set_xscale('log')
axes[2].set_xticks([0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8])
axes[2].xaxis.set_major_formatter(FuncFormatter(scientific_format))
fig.suptitle('L1 regularization')
plt.tight_layout()
plt.show()
```

    c:\Users\mutis\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    c:\Users\mutis\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    c:\Users\mutis\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    c:\Users\mutis\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    c:\Users\mutis\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    c:\Users\mutis\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    


    
![png](index_files/index_87_1.png)
    


## 5. Elasticnet regularization


```python
for c in c_vals:
    elasticnet_model = LogisticRegression(solver='saga',
                                        random_state=random_state,
                                        penalty='elasticnet',
                                        l1_ratio=.5,
                                        C=10)
    elasticnet_train_loss, elasticnet_val_loss = get_train_val_loss(elasticnet_model,
                                                                    X_train_preprocessed,
                                                                    y_train,
                                                                    cont_cols,
                                                                    cat_cols)

    print(f'Elasticnet regularization({c}) model log loss on train data:', 
          elasticnet_train_loss)
    print(f'Elasticnet Regularization({c}) model log loss on validation:', 
          elasticnet_val_loss)
    print()
```

    Elasticnet regularization(1) model log loss on train data: 6.9238287176437
    Elasticnet Regularization(1) model log loss on validation: 4.655310505298025
    
    Elasticnet regularization(10) model log loss on train data: 6.9238287176437
    Elasticnet Regularization(10) model log loss on validation: 4.655310505298025
    
    Elasticnet regularization(100) model log loss on train data: 6.9238287176437
    Elasticnet Regularization(100) model log loss on validation: 4.655310505298025
    
    Elasticnet regularization(1000.0) model log loss on train data: 6.9238287176437
    Elasticnet Regularization(1000.0) model log loss on validation: 4.655310505298025
    
    Elasticnet regularization(1000000.0) model log loss on train data: 6.9238287176437
    Elasticnet Regularization(1000000.0) model log loss on validation: 4.655310505298025
    
    

## Decision tree


```python
# decision tree
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(random_state=random_state)
dt_train_loss, dt_val_loss = get_train_val_loss(dt_model,
                                                X_train_preprocessed,
                                                y_train,
                                                cont_cols,
                                                cat_cols)
print('Decision tree model train log loss:', dt_train_loss)
print('Decision tree model val log loss:', dt_val_loss)
```

    Decision tree model train log loss: 2.2204460492503136e-16
    Decision tree model val log loss: 1.2855413756113048
    

Comparing all the models, the baseline model performs better on the validation set but the difference in the log loss in the training and the validation could imply underfitting.


On further investigation, the l1 model with more regularization is our best model. It has the lowest log loss with a good balance in bias and variance this reducing chances of overfitting or underfitting.

# Evaluation of the final model

Our best performing model is the l1 model with higher regularization. We now preprocess the full training and test data and evaluate the model.


```python
final_model = LogisticRegression(random_state=random_state,
                                 penalty='l1',
                                 C=1e06,
                                 solver='saga')

final_scaler = StandardScaler()
final_sm = SMOTE(random_state=random_state)

X_train_cont = X_train_preprocessed[cont_cols]
X_train_cat = X_train_preprocessed[cat_cols]

# scaling X for train data
X_train_scaled = pd.DataFrame(
    final_scaler.fit_transform(X_train_cont),
    columns=X_train_cont.columns,
    index=X_train_cont.index
)

X_train_scaled = pd.concat([X_train_scaled, X_train_cat], axis=1)

# scaling X for test data
X_test_dummy = pd.get_dummies(X_test.select_dtypes('object'),
                              drop_first=True,
                              dtype=int)
X_test_dummy['Wkd'] = X_test['Wkd']

numeric_test = X_test.select_dtypes(exclude='object').drop('Wkd',
                                                           axis=1)

X_test_scaled = pd.DataFrame(
    final_scaler.transform(numeric_test),
    columns=numeric_test.columns,
    index=numeric_test.index,
)

X_test_scaled = pd.concat(
    [X_test_scaled, X_test_dummy],
    axis=1
)

# oversampling to reduce class imbalance
X_train_oversampled, y_train_oversampled = final_sm.fit_resample(X_train_scaled,
                                                           y_train)
# fitting the model
final_model.fit(X_train_oversampled, y_train_oversampled)
```




<style>#sk-container-id-11 {color: black;background-color: white;}#sk-container-id-11 pre{padding: 0;}#sk-container-id-11 div.sk-toggleable {background-color: white;}#sk-container-id-11 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-11 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-11 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-11 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-11 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-11 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-11 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-11 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-11 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-11 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-11 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-11 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-11 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-11 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-11 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-11 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-11 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-11 div.sk-item {position: relative;z-index: 1;}#sk-container-id-11 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-11 div.sk-item::before, #sk-container-id-11 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-11 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-11 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-11 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-11 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-11 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-11 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-11 div.sk-label-container {text-align: center;}#sk-container-id-11 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-11 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-11" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression(C=1000000.0, penalty=&#x27;l1&#x27;, random_state=20, solver=&#x27;saga&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-11" type="checkbox" checked><label for="sk-estimator-id-11" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(C=1000000.0, penalty=&#x27;l1&#x27;, random_state=20, solver=&#x27;saga&#x27;)</pre></div></div></div></div></div>




```python
# importing modules
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
```

## Accuracy


```python
y_pred = final_model.predict(X_test_scaled)
accuracy_score(y_test, y_pred)
```




    0.7406004676428489



This means that out of all the predictions our model made, 74% were correct.

## Recall


```python
recall_score(y_test, y_pred)
```




    0.6516047129765541



This means that out of all the customers that had purchase intent, our model was able to predict about 65%.

## Precision


```python
precision_score(y_test, y_pred)
```




    0.44608908202064096



Out of all the customers that our model predicted as having purchase intent, about 45% actually had purchase intent.

## F1- score


```python
f1_score(y_test, y_pred)
```




    0.5296080739033002



# Conclusion

The predictive model successfully identifies potential buyers with a recall rate of 65%, capturing a significant portion of users likely to make a purchase. Although the precision is 45%, indicating that not all predicted buyers convert, the model still provides valuable insights for targeting marketing efforts. By focusing on the users with the highest likelihood of purchasing and continuously refining the model, we can enhance the efficiency of our marketing strategies and drive increased sales. Implementing these insights will enable us to optimize resource allocation and improve overall conversion rates.

## predicting rev for missing set


```python
missing_rev.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 50000 entries, 0 to 49999
    Data columns (total 18 columns):
     #   Column   Non-Null Count  Dtype  
    ---  ------   --------------  -----  
     0   Adm      50000 non-null  int64  
     1   AdmDur   50000 non-null  float64
     2   Inf      50000 non-null  int64  
     3   InfDur   50000 non-null  float64
     4   Prd      50000 non-null  int64  
     5   PrdDur   50000 non-null  float64
     6   BncRt    50000 non-null  float64
     7   ExtRt    50000 non-null  float64
     8   PgVal    50000 non-null  float64
     9   SpclDay  50000 non-null  float64
     10  Mo       50000 non-null  int64  
     11  OS       50000 non-null  int64  
     12  Bsr      50000 non-null  int64  
     13  Rgn      50000 non-null  int64  
     14  TfcTp    50000 non-null  int64  
     15  VstTp    50000 non-null  int64  
     16  Wkd      50000 non-null  int64  
     17  Rev      0 non-null      float64
    dtypes: float64(8), int64(10)
    memory usage: 7.2 MB
    


```python
missing_rev[['Mo', 'OS', 'Bsr', 'Rgn', 'TfcTp', 'VstTp']] = (
    missing_rev[['Mo', 'OS', 'Bsr', 'Rgn', 'TfcTp', 'VstTp']].astype('object')
)
missing_rev.drop('Rev', axis=1, inplace=True)
```

    C:\Users\mutis\AppData\Local\Temp\ipykernel_17580\2206965402.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      missing_rev[['Mo', 'OS', 'Bsr', 'Rgn', 'TfcTp', 'VstTp']] = (
    C:\Users\mutis\AppData\Local\Temp\ipykernel_17580\2206965402.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      missing_rev.drop('Rev', axis=1, inplace=True)
    


```python
missing_numeric = missing_rev.select_dtypes(exclude='object').drop('Wkd',
                                                                   axis=1)
missing_rev_dummy = pd.get_dummies(missing_rev.select_dtypes('object'),
                                   drop_first=True,
                                   dtype=int)
missing_rev_dummy['Wkd'] = missing_rev['Wkd']

missing_rev_scaled = pd.DataFrame(
    final_scaler.transform(missing_numeric),
    columns=missing_numeric.columns,
    index=missing_numeric.index,
)

missing_rev_scaled = pd.concat(
    [missing_rev_scaled, missing_rev_dummy],
    axis=1
)

missing_pred = final_model.predict(missing_rev_scaled)
```


```python
predicted = pd.DataFrame(missing_pred, columns=['Rev'])
predicted['id'] = missing_rev.index + 1
predicted = predicted[['id', 'Rev']].astype('Int64').set_index('id')
predicted
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rev</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>49996</th>
      <td>1</td>
    </tr>
    <tr>
      <th>49997</th>
      <td>0</td>
    </tr>
    <tr>
      <th>49998</th>
      <td>1</td>
    </tr>
    <tr>
      <th>49999</th>
      <td>1</td>
    </tr>
    <tr>
      <th>50000</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>50000 rows × 1 columns</p>
</div>




```python
predicted.to_csv('predicted.csv')
```
