# SalaryClassPrediction 预测工资水平项目
背景： 根据一个人口普查的数据，判断这个人的工资收入水平是属于大于50K, 还是小于等于50k。原始数据文件存在某些数值丢失，数据拼写问题，错误数据问题。
- 任务1：对数据进行探索、清洗、可视化展现

        1.  读取数据
        2.  探索数据，如最大值，最小值，平均值，std，数据量等
        3.  处理丢失数据，数值错误数据，拼写错误的数据
        4.  使用matplotlib绘制可视化直方图
        
- 任务2：数据预处理、使用多种机器学习分类算法对数据建模，比较各种模型的性能，选取最好的一个模型应用到新数据上做出分类预测

        1.  将分类变量转换为数值变量或者进行离散编码get_dummies
        2.  标准化数值 normalize
        3.  使用DecisionTreeClassfier、KNeighborsClassifier、GaussianNB、MLPClassifier、SVC分类算法建模
        4.  比较模型性能
        5.  提高模型性能
        6.  对新数据预测
        
成果： **项目整体预测率约82%，为班级第一，受到教授表扬奖励**

### 文件介绍
1. data_dirty.csv 有脏数据的原始人口普查数据，3万+条数据
2. data_preprocessing.ipynb 数据清洗代码文件
3. data_preprocessing.pdf 数据清洗英文报告
4. clssification.ipynb 分类算法代码文件
5. test_data.csv 待预测人口普查数据，1.5万+条数据
6. class_prediction.csv 预测结果

## 任务1： 数据清洗
### 1. *读取 csv数据文件，保存成为pandas的dataframe*
```python
# define the name of headers
col_name = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','salary']
# read data from csv file
data = pd.read_csv('dataset1_dirty.csv',header=None,names = col_name)
```

### 2. *探索数据*
- 查看数据格式

![Image of Data Format](https://github.com/Pam1024/SalaryClassPrediction/blob/main/image/z_data_format.PNG)

- 查看各feature的统计数值，对于数值特征查看其统计数值：如最大值、最小值、平均值、标准差。对于类别特征，查看其类别数量、各类别有多少数据。查看结果按照格式化打印
```python
# better way to print
print("rows: {}".format(len(data.index)))
i = 0
while i < len(data.columns):
    if(type(data.loc[0][i]) == np.int64):
        print("{n}: min:{min}, mean:{mean}, max:{max}, std:{std}".format(n=data.columns[i],min=np.min(data.iloc[:,i]),mean=np.mean(data.iloc[:,i]),max=np.max(data.iloc[:,i]),std=np.std(data.iloc[:,i])))

    else:
        print("{n} : {count}".format(n=data.columns[i],count=data.iloc[:,i].nunique()))
        # refer to https://stackoverflow.com/questions/34178751/extract-unique-values-and-number-of-occurrences-of-each-value-from-dataframe-col
        uni_list = Counter(data.iloc[:,i])
        for uni in uni_list:
            print("  {}: {}".format(uni,uni_list[uni]))
        
    i = i+1
```
  部分打印结果如下图所示，可清楚了解到个feature的情况：
  
  ![Image of features](https://github.com/Pam1024/SalaryClassPrediction/blob/main/image/z_print_result.PNG)
  
- 通过上述数据探索，可以发现以下问题：
  1. 丢失数据： Workclass，Occupation特征存在‘？’的值
  2. 错误数据： Age特征存在负数的值
  3. 拼写错误： Workclass，Occupation特征存在人为拼写错误，如Local-gov这个类别存在了多个人为拼写值，如local-gov，local gov，Localgov，Local gov，localgov等多种拼写方式。其实这几个值都是代表同一个东西，但是却出现了几个不同值，如果不清理，会被机器误认为是其他类别，就会造成模型不准确
  
### 3. *数据清理*

1. 清理值为‘？’的数据

   由于值‘？’的数据来自职业和工作类别特征，这些人口普查信息没有渠道再次核对获取，并且只有1000条左右数据含有丢失数据，对于3万多条原始数据不算特别多，所以将含有‘？’值的数据舍弃。
   
```python
   # delete the row with the value ?
data_clean1 = data[data['workclass'] != '?']
data_clean2 = data_clean1[data_clean1['occupation'] != '?']
```

2. 清理年龄为负数的数据

   年龄是数值数据，采用数据的平均值去替代负数的数据
```python
"""     the column age has bad data
        change the value which less than 0 to the mean value of the dataset
"""
data_clean2['age'].values[data_clean2['age'].values <=0] = np.mean(data_clean2['age'])
```

3. 清理拼写偏差的数据
   本文只论述对workclass的具体清理方法，education的清理类同，可参考代码。
   - 首先获取workclass的标准类别，参考人口普查给出的定义可得到workclass的标准值为：
   
     `['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked ']`
     
   - 再利用Levenshtein这个package，去计算具体某个值离上述标准类别的相似程度，选出最相似的标准类别代替现在的值。例如localgov与上面8个标准类别对比后，与Local-gov最相似，就用Local-gov去代替localgov的值。
   
```python
#clean the bad data for column workclass
#use the previous unique values to match the right one
work_list = data_clean2['workclass'].unique().tolist()
#print(len(work_list))
i=0
work_dict = {}
#print(len(work_list))
while i < len(work_list) :
    j = 0
    s = workclass[j]
    k = 0
    while j < len(workclass)-1:
   
        #print(Levenshtein.ratio(s,str))
        #print(Levenshtein.ratio(workclass[j+1],str))
        #print('++++')
        #print(list[i])
        if Levenshtein.ratio(workclass[j+1],work_list[i])>Levenshtein.ratio(s,work_list[i]):
            s =  workclass[j+1]
            #print(s)
            #print('------------')
            #print(s)
        j = j+1
   
    #print(list[i])
    #print(s)
    work_dict[work_list[i]] = s
    #print(i)
    i = i+1
def match(x,dict):
    return dict[x]
data_clean2['workclass']=data_clean2['workclass'].apply(lambda x: match(x,work_dict))
```
  - 清洗后结果
  
   ![Image of cleaning](https://github.com/Pam1024/SalaryClassPrediction/blob/main/image/z_clean_workclass.PNG)

### 3. *数据可视化*

- 使用matplotlib绘制清理后特征的直方图

``` python
fig, ax = plt.subplots(figsize=(10,5))
N, bins, patches =ax.hist(data_clean2['age'],20,color='c',histtype='bar',ec='black')
ax.set_xlabel('Age')
ax.set_ylabel('Frequency')
ax.set_title("Age Histogram")
```

  ![Image of age histogram](https://github.com/Pam1024/SalaryClassPrediction/blob/main/image/z_age_histogram.PNG)
     

## 任务2： 预测工资分类

### 1. *加载上次已经清洗好的数据*

```python
# define the name of headers
col_name = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','salary']
# read data from csv file
persons = pd.read_csv('dataset1_processed.csv',header=None,names = col_name)

```

![Image of clean data](https://github.com/Pam1024/SalaryClassPrediction/blob/main/image/z_clean_data.PNG)


### 2. *convert categorical columns into multiple binary or numerical columns 将分类变量转换为数值变量或者进行离散编码get_dummies*

  - 因为education这个特征的值是连续有序的，可以将其转化成数值
      
```pyhton
#refer to https://towardsdatascience.com/preprocessing-with-sklearn-a-complete-and-comprehensive-guide-670cb98fcfb9
edu = pd.Categorical(persons['education'],categories=['Preschool','1st-4th','5th-6th','7th-8th','9th','10th','11th','12th','HS-grad','Prof-school','Assoc-acdm','Assoc-voc','Some-college','Bachelors','Masters','Doctorate'],ordered=True)
labels, unique = pd.factorize(edu, sort=True)
persons['education'] = labels
```

   - 其他分类变量的值是离散的，如'workclass','marital-status','occupation','relationship','race','sex','native-country'，可以 用get_dummies将其转化成multiple binary values
 
 ```python
 #get all the data columns except target value 'salary' for get_dummies process
data = persons.loc[:, persons.columns != 'salary']
# use get_dummies method to convert other categorial features to binary values
# refer to https://towardsdatascience.com/encoding-categorical-features-21a2651a065c
data_dummies = pd.get_dummies(data, prefix_sep='_', drop_first=True)
```
   转化后的数据：
   ![Image of data convert](https://github.com/Pam1024/SalaryClassPrediction/blob/main/image/z_get_dummy.PNG)
       
       
### 2. *Normalize Data数据标准化*

```python
# retrive all the numerical columns
data_numerical = data_dummies[['age','fnlwgt','education','education-num','capital-gain','capital-loss','hours-per-week']]
data_normalize = data_dummies
# normalize every numerical columns in for loop using normalize() method
for col in data_numerical.columns:
    x = np.array(data_numerical[col])
    y = normalize([x]) 
    data_normalize[col] = y[0].tolist()   
```
标准化后的数据：
       ![Image of data normalize](https://github.com/Pam1024/SalaryClassPrediction/blob/main/image/z_data_normalize.PNG)
 
 
### 3. *使用分类算法建模*
      
   - 将数据分为traning set和 test set      
```python
X = data_normalize
y = persons['salary']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
```

   - 使用DecisionTreeClassfier、KNeighborsClassifier、GaussianNB、MLPClassifier、SVC这几个分类算法对数据建模，具体算法代码使用请参考clssification.ipynb
```
#KNeighborsClassifier
knc = KNeighborsClassifier()
knc.fit(X_train,y_train)
#accuracy on train data 
train_pre = knc.predict(X_train)
train_accuracy = accuracy_score(y_train, train_pre)
train_average_class_accuracy = average_class_accuracy(y_train, train_pre)
#accuracy on test data
predictions = knc.predict(X_test)
knc_accuracy = accuracy_score(y_test, predictions)
test_average_class_accuracy = average_class_accuracy(y_test, predictions)
```
### 4. *比较模型性能*   
   - 5个分类算法的性能用matplotlib显示如下：
    
     由图可知，对test data性能最好的是MLPClassifier，对training data性能最好的是DecisionTreeClassfier。
      
  
 ### 5. *提高模型性能*   
   - 深入分析发现，预测 '<=50K'工资类别的准确率是0.92,  '>50K'工资类别准确率是 0.58。 查看两个工资类别的数据数量发现， '<=50K'的数据数量是21764, '>50K' 的数据数量只有7194。两个类别数据数量不平衡，于是补充'>50K' 的数据数量。
     
```python
#adding more samples with salary '>50K' to the dataset
data_complete = data_normalize
data_complete['salary'] = persons['salary']
data_complete.head()
data_greater50K = data_complete[data_complete['salary'] == '>50K']
data1 = data_complete.append(data_greater50K)
data_new = data1.append(data_greater50K)
data_new['salary'].value_counts()
```
   补充后各类别数据如下图所示：
             
     图片
       
   - 对补充后的数据重新建模
```python
# calculate the accuracy of the model prediction after renew the dataset 
MLPC = MLPClassifier()
MLPC.fit(X_train,y_train)
prediction = MLPC.predict(X_test)
cm = confusion_matrix(y_test, prediction, labels=['<=50K','>50K'])
tp = cm[0][0]
tn = cm[1][1]
less50class = np.sum(cm[0])
greater50class = np.sum(cm[1])
less50class_accuracy = tp/less50class
greater50class_accuracy = tn/greater50class
average_class_accuracy = (less50class_accuracy+greater50class_accuracy)/2
print('less50class_accuracy: ',less50class_accuracy)
print('greater50class_accuracy: ',greater50class_accuracy)
print('average_class_accuracy: ',average_class_accuracy)
```
   -性能比之前提高，类别平均准确率从0.77提高到0.83
   
   图片
   
   
### 6. *对新数据预测*
   — 从文件中读取新数据，进行categorical变量转化，离散编码get_dummies,再利用上面的MLPClassifier模型进行预测。
     
```python
# read the test data from csv file
test_data = pd.read_csv('dataset1_test.csv',header=None,names = col_name)
```
  - 数据处理的代码请参考clssification.ipynb，不再重复描述。
  - 利用选好的模型对新数据预测，并把结果保存到文档里。
    
 ```python
#use the best model to predict the value
prediction = MLPC.predict(data_normalize)
# change the result array to dataframe in order to save to the file 'B00809814_prediction.csv'
predict_data = pd.DataFrame(prediction)
predict_data.to_csv('prediction.csv',index=False,header=False)
```

    

     








