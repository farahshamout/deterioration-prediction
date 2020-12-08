from keras.layers import Input, merge, LSTM, Dense, SimpleRNN, Masking, Bidirectional, Dropout, concatenate, Embedding, TimeDistributed, multiply, add, dot, Conv2D
from keras.optimizers import Adam, Adagrad, SGD
from keras import regularizers, callbacks
from keras.layers.core import *
from keras.models import *



# Attention blocks used in DEWS 
def attention_block(inputs_1, num):
    # num is used to label the attention_blocks used in the model 
    
    # Compute eij i.e. scoring function (aka similarity function) using a feed forward neural network 
    v1 = Dense(10, use_bias=True)(inputs_1)
    v1_tanh = Activation('relu')(v1)
    e = Dense(1)(v1_tanh)
    e_exp = Lambda(lambda x: K.exp(x))(e)
    sum_a_probs = Lambda(lambda x: 1/ K.cast(K.sum(x, axis=1, keepdims=True) + K.epsilon(), K.floatx()))(e_exp)
    a_probs = multiply([e_exp, sum_a_probs], name='attention_vec_'+str(num))
     
    context = multiply([inputs_1,a_probs])
    context = Lambda(lambda x: K.sum(x, axis=1))(context)  
    
    return context



# DEWS Architecture 
def dews(NUM_VIT, TIME_STEPS):
    i1 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc1 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'), 'ave')(i1)    
    dec1 = attention_block(enc1,1)
    
    i2 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc2 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'), 'ave')(i2)    
    dec2 = attention_block(enc2,2)
    
    
    i3 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc3 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'), 'ave')(i3)    
    dec3 = attention_block(enc3,3)
    
    i4 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc4 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'), 'ave')(i4)    
    dec4 = attention_block(enc4,4)
    
    i5 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc5 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'), 'ave')(i5)    
    dec5 = attention_block(enc5,5)
    
    i6 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc6 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'), 'ave')(i6)    
    dec6 = attention_block(enc6,6)
    
    i7 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc7 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'), 'ave')(i7)    
    dec7 = attention_block(enc7,7)
    
    c_agg_1 = add([dec1, dec2, dec3, dec4, dec5, dec6, dec7])
    c_m = Dense(5)(c_agg_1)
    
    i12 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc12 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'), 'ave')(i12)    
    dec12 = attention_block(enc12,8)
    
    i22= Input(shape=(TIME_STEPS,1), dtype='float32')
    enc22 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'), 'ave')(i22)    
    dec22 = attention_block(enc22,9)
    
    
    i32 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc32 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'), 'ave')(i32)    
    dec32 = attention_block(enc32,10)
    
    i42 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc42 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'), 'ave')(i42)    
    dec42 = attention_block(enc42,11)
    
    
    i52 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc52 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'), 'ave')(i52)    
    dec52 = attention_block(enc52,12)
    

    c_agg_2 = add([dec12, dec22, dec32, dec42, dec52])
    c_v = Dense(5)(c_agg_2)
    
    c = add([c_m, c_v])
    
    dec = Dense(5, activation='relu')(c)
    dec2 = Dropout(0.2)(dec)
    output = Dense(1, activation='sigmoid')(dec2)
    model = Model(input=[i1, i2, i3, i4, i5, i6, i7, i12, i22, i32, i42, i52], output=output)
    return model