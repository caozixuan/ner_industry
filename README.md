# ner_industry
## Abstract
> industry named entity recognition with LSTM. 
> 
> the code references [ner_lstm](https://github.com/monikkinom/ner-lstm)
> 
> the code is written in Django framework. You can excute the code using Django console.

In this code, we use some keywords to locate the industry entity, then we use word embedding to encode the entity and the the context. Finally, we use LSTM to do the classification.

## Version of important used packages

  + jieba 0.39
  + numpy 1.13.1
  + tensorflow 1.2.0

## Steps

### 1. build word2vec

In this code, we use a word2vec model 'word2vec_from_weixin 'trained by other programmers. You can also train word2vec by yourself. We provide the code. Remember to change the file path.

	from industry.views import train_data_build, train_data
	train_data_build()
	train_data() 

### 2. build train and test data

In directory data_example, we share 3 reports. We don't provide all the data. You can get the reports of companies from some financial web sites.

how to build data?

1. In the report, we find some keywords, then we choose the left context, right context to build the database. Both left context and right context are five words. If the sentence can not find five words, replace the positions with '#'. For example:

    ['#', '#', '#', '#', '#', '工程施工', '业务', '上游', '产业', '包括', '建材', '民爆', '产品', '生产', '供应']

    We can tag it as:

    [0,0,0,0,0,1,0,0,0,0,1,0,0,0,0]

    We save the data to database.

2. But not all the words can be changed to vector. So we should replace those words with '#'.

3. In order to improve the rate of industry entity, you can delete the sentences that do not have any entities. This step is optional.

4. the code to build the data

	    from industry.views import collect_data
	    collect_data(True)
	    collect_data(False)
	    from industry.views import delete_word, delete_data
	    delete_word()
	    delete_data()

### Train model

1. In this model, we both try 2 classes classification and 3 classes classification. In 2 classes classification, we encode non-industry words as [1,0], industry words as [0,1]. In 3 classes classfication, we encode non-industry words as [1,0,0], industry words but not in the end of entity as [0,1,0], industry words in the end of entity as [0,0,1]. You can change the variable class_size to change the mode.

2. the code to train model

		from industry.views import train
		train()

## Result
the result is not very ideal.

For 2 classification, the F1 of two classes:

[0.95779816513761473,0.72941176470588232]

and the F1 of three classes:

[0.9595588235294118, 0.40000000000000002, 0.71604938271604945]

the prediction probability example:

['#', '#', '#', '#', '公司', '房地产', '行业', '上游', '产业', '主要', '包括', '建筑业', '建材业', '包括', '机械']

[
[  9.99966979e-01   1.55737087e-06   3.14216340e-05]
 
[  9.99990582e-01   4.55379535e-07   8.90888259e-06]
 
[  9.99941945e-01   4.67149493e-06   5.33625825e-05]

 [  9.97278512e-01   6.51383481e-04   2.07014778e-03]

 [  9.94468451e-01   4.50271787e-03   1.02872960e-03]

 [  6.28789887e-03   5.39945886e-02   9.39717472e-01]

 [  9.98816729e-01   5.72303361e-05   1.12605561e-03]

 [  9.99713361e-01   1.25709839e-05   2.74072343e-04]

 [  9.98275757e-01   3.54322663e-04   1.36994233e-03]

 [  1.90898255e-02   2.09030160e-03   9.78819907e-01]

 [  9.98887837e-01   1.78836053e-05   1.09432684e-03]

 [  9.99973178e-01   1.20756727e-06   2.55898740e-05]

 [  9.99550045e-01   2.70814635e-04   1.79059527e-04]

 [  8.58513359e-03   8.34852904e-02   9.07929540e-01]

 [  6.72985683e-04   7.27007631e-03   9.92056906e-01]
] 

 
        
