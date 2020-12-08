# -*- coding: utf-8 -*-
"""
Created on Wed May 27 13:38:38 2020

@author: ball4624
"""

def ua_lstm_attention_2(INPUT_DIM, TIME_STEPS):
    i1 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc1 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'), 'ave')(i1)    
    dec1 = attention_3d_block_7(enc1,1)
    
    i2 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc2 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'), 'ave')(i2)    
    dec2 = attention_3d_block_7(enc2,2)
    
    
    i3 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc3 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'), 'ave')(i3)    
    dec3 = attention_3d_block_7(enc3,3)
    
    i4 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc4 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'), 'ave')(i4)    
    dec4 = attention_3d_block_7(enc4,4)
    
    i5 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc5 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'), 'ave')(i5)    
    dec5 = attention_3d_block_7(enc5,5)
    
    i6 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc6 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'), 'ave')(i6)    
    dec6 = attention_3d_block_7(enc6,6)
    
    i7 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc7 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'), 'ave')(i7)    
    dec7 = attention_3d_block_7(enc7,7)
    
    c_agg_1 = add([dec1, dec2, dec3, dec4, dec5, dec6, dec7])
    c_m = Dense(5)(c_agg_1)
    
    i12 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc12 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'), 'ave')(i12)    
    dec12 = attention_3d_block_7(enc12,8)
    
    i22= Input(shape=(TIME_STEPS,1), dtype='float32')
    enc22 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'), 'ave')(i22)    
    dec22 = attention_3d_block_7(enc22,9)
    
    
    i32 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc32 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'), 'ave')(i32)    
    dec32 = attention_3d_block_7(enc32,10)
    
    i42 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc42 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'), 'ave')(i42)    
    dec42 = attention_3d_block_7(enc42,11)
    
    
    i52 = Input(shape=(TIME_STEPS,1), dtype='float32')
    enc52 = Bidirectional(LSTM(TIME_STEPS, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'), 'ave')(i52)    
    dec52 = attention_3d_block_7(enc52,12)
    

    c_agg_2 = add([dec12, dec22, dec32, dec42, dec52])
    c_v = Dense(5)(c_agg_2)
    
    c = add([c_m, c_v])
    
    #dec = LSTM(12, return_sequences=True)(context)
    #dec = Dense(5, activation='relu', kernel_regularizer=regularizers.l2(0.01))(c)
    #dec2= BatchNormalization()(dec)
    dec = Dense(5, activation='relu')(c)
    dec2 = Dropout(0.2)(dec)
    output = Dense(1, activation='sigmoid')(dec2)
    model = Model(input=[i1, i2, i3, i4, i5, i6, i7, i12, i22, i32, i42, i52], output=output)
    model.summary()
    return model