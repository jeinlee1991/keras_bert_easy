# keras_bert_easy
## 做最好用的keras bert接口


 
### 已更新以下几点：  
1、构建Bert模型时，可以不用传入config.json  

2、构建tokenizer时，可以不用传入vocab.txt  

3、优化Bert预训练参数加载的速度，比keras_bert快4倍，比bert4keras快20%
     
4、提供支持分层学习率的优化器实现，包括“两段式分层学习率优化器”、
“三段式分层学习率优化器”、“逐层衰减学习率优化器”，
此外还支持weight decay参数设置，还支持梯度累积优化器（在内存不足时，增大batch size的方法）
