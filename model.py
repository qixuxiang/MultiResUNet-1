import  tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D

def mlti_res_block(inputs,filter_size1,filter_size2,filter_size3,filter_size4):
    cnn1 = Conv2D(filter_size1,(3,3),padding = 'same',activation="relu")(inputs)
    cnn2 = Conv2D(filter_size2,(3,3),padding = 'same',activation="relu")(cnn1)
    cnn3 = Conv2D(filter_size3,(3,3),padding = 'same',activation="relu")(cnn2)

    cnn = Conv2D(filter_size4,(1,1),padding = 'same',activation="relu")(inputs)

    concat = layers.Concatenate()([cnn1,cnn2,cnn3])
    add = layers.Add()([concat,cnn])

    return add

def res_path(inputs,filter_size,path_number):
    def block(x,fl):
        cnn1 = Conv2D(filter_size,(3,3),padding = 'same',activation="relu")(inputs)
        cnn2 = Conv2D(filter_size,(1,1),padding = 'same',activation="relu")(inputs)

        add = layers.Add()([cnn1,cnn2])

        return add
    
    cnn = block(inputs, filter_size)
    if path_number <= 3:
        cnn = block(cnn,filter_size)
        if path_number <= 2:
            cnn = block(cnn,filter_size)
            if path_number <= 1:
                cnn = block(cnn,filter_size)

    return cnn

def multi_res_u_net(pretrained_weights = None,input_size = (256,256,1),lr=0.001):
    inputs = layers.Input(input_size)

    res_block1 = mlti_res_block(inputs,8,17,26,51)
    pool1 = layers.MaxPool2D()(res_block1)

    res_block2 = mlti_res_block(pool1,17,35,53,105)
    pool2 = layers.MaxPool2D()(res_block2)

    res_block3 = mlti_res_block(pool2,31,72,106,209)
    pool3 = layers.MaxPool2D()(res_block3)

    res_block4 = mlti_res_block(pool3,71,142,213,426)
    pool4 = layers.MaxPool2D()(res_block4)

    res_block5 = mlti_res_block(pool4,142,284,427,853)
    upsample = layers.Upsampling2D()(res_block5)

    res_path4 = res_path(res_block4,256,4)
    concat = layers.Concatenate()([upsample,res_path4])

    res_block6 = mlti_res_block(concat,71,142,213,426)
    upsample = layers.Upsampling2D()(res_block6)

    res_path3 = res_path(res_block3,128,3)
    concat = layers.Concatenate()([upsample,res_path3])

    res_block7 = mlti_res_block(concat,31,72,106,212)
    upsample = layers.Upsampling2D()(res_block7)

    res_path2 = res_path(res_block2,64,2)
    concat = layers.Concatenate()([upsample,res_path2])

    res_block8 = mlti_res_block(concat,17,35,53,105)
    upsample = layers.Upsampling2D()(res_block8)

    res_path1 = res_path(res_block1,32,1)
    concat = layers.Concatenate()([upsample,res_path1])
    
    res_block9 = mlti_res_block(concat,8,17,26,51)
    sigmoid = Conv2D(1,(1,1),padding = 'same',activation="sigmoid")(res_block9)

    model = tf.keras.Modedl(inputs,sigmoid)
    modle.compile(tf.keras.optimizer.Adam(lr),loss = 'binary_crossentropy', metrics = ['accuracy'])

    if(pretrained_weights):
        	model.load_weights(pretrained_weights)
            
    return model
     
