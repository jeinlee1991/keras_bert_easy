from keras_bert_easy.bert import get_model

model = get_model(token_num=20000, seq_len=None)
model.summary()