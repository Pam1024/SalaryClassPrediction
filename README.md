# SalaryClassPrediction 预测工资水平项目
背景： 根据一个人口普查的数据，判断这个人的工资收入水平是属于大于50K, 还是小于等于50k。原始数据文件已损坏，部分数据丢失，部分数据没有按照格式填写。
- 任务1：对数据进行清洗、数据转换、预准备
- 任务2：使用多种机器学习分类算法对数据建模，比较各种模型的性能，选取最好的一个模型应用到这个问题上
成果： **项目整体预测率约80%，为班级第一，受到教授表扬奖励**
### 文件介绍
1. data_dirty.csv 有脏数据的原始人口普查数据，约3万条数据
2. data_preprocessing.ipynb 数据清洗代码文件
3. data_preprocessing.pdf 数据清洗英文报告
4. clssification.ipynb 分类算法代码文件
5. test_data.csv 待预测人口普查数据，1.5万条数据
6. class_prediction.csv 预测结果

## 任务1： 数据预处理
### 1. *读取csv文件，保存成为pandas的dataframe*
```python
# define the name of headers
col_name = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','salary']
# read data from csv file
data = pd.read_csv('dataset1_dirty.csv',header=None,names = col_name)
```

### 2. *探索数据*
- 查看数据格式

![Image of Data Format](https://github.com/Pam1024/SalaryClassPrediction/blob/main/Z_data_format.PNG)

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
  部分打印结果如下图所示，可清洗了解到个feature的情况：
  ![Image of features](https://github.com/Pam1024/SalaryClassPrediction/blob/main/z_print_result.PNG)
  
- 通过上述数据探索，可以发现以下问题：
  1. 丢失数据： Workclass，Occupation特征存在‘？’的值
  2. 错误数据： Age特征存在负数的值
  3. 拼写错误： Workclass，Occupation特征存在人为拼写错误，如Local-gov这个类别存在了多个人为拼写值，如local-gov，local gov，Localgov，Local gov，localgov等多种拼写方式。其实这几个值都是代表同一个东西，但是却出现了几个不同值，如果不清理，会被机器误认为是其他类别，就会造成模型不准确
  
  








