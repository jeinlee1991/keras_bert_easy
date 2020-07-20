# keras_bert_easy
## 做最好用的keras bert接口


 
### 已更新以下几点：  
1、构建bert模型时，可以不用传入config.json  

2、构建tokenizer时，可以不用传入vocab.txt  

3、可以自由传入seq_len参数，支持训练和推理的seq_len不同（keras_bert不支持）  

4、将build_model_from_config, load_trained_model_from_checkpoint整合成一个函数build_pretrained_model：     
finetune训练时，传入checkpoint_file即可，推理时则不必传入（checkpoint_file默认为None）。

5、当模型构建函数build_pretrained_model的training参数为False时，会自动将bert model中的所有dropout层全部丢弃。（keras_bert不支持）    
     

### 即将更新以下几点：   
1、可以传入input_mask，支持attention_mask（keras_bert和bert4keras均不支持）  
