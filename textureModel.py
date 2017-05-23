import keras
from keras.callbacks import ModelCheckpoint
from keras.applications import VGG16
from keras.layers import Dense,Lambda,Input
from keras.engine.topology import Layer
from keras.models import Model
from keras import optimizers
from keras.preprocessing import image as kpi
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
import matplotlib.pyplot as plt
import keras.backend as K
import scipy.misc

def scipy_image_to_vgg(scipy_image):
    data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if data_format == 'channels_first':
        # 'RGB'->'BGR'
        x = np.expand_dims(scipy_image.astype(float),0)
        x = np.transpose(x,(0,3,1,2))
        x = x[:, ::-1, :, :]
        # Zero-center by mean pixel
        x[:, 0, :, :] -= 103.939
        x[:, 1, :, :] -= 116.779
        x[:, 2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = np.expand_dims(scipy_image.astype(float),0)
        x = x[:, :, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, :, 0] -= 103.939
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 123.68
    return x
    
def vgg_to_scipy_image(vgg_image):
    data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}
    x = np.copy(vgg_image)
    if data_format == 'channels_first':
        # 'RGB'->'BGR'
        # Zero-center by mean pixel
        x[:, 0, :, :] += 103.939
        x[:, 1, :, :] += 116.779
        x[:, 2, :, :] += 123.68
        x = x[:, ::-1, :, :]
        x = np.transpose(x,(0,2,3,1))
    else:
        x[:, :, :, 0] += 103.939
        x[:, :, :, 1] += 116.779
        x[:, :, :, 2] += 123.68

                        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
        # Zero-center by mean pixel
    x = np.squeeze(x)
    x = np.clip(x,0,255)
    x = x.astype('uint8')

    return x    



class OffsetLayer(Layer):

    def __init__(self, output_shape=(227,227,3), **kwargs):
        self.initial_output_shape = output_shape
        self.output_dim = output_shape[-1]
        super(OffsetLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(shape=(input_shape[1:]),
                                      initializer='zeros',
                                      trainable=True)
        super(OffsetLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return (x + self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape)

def gram_matrix(x):
    layer_shape = K.shape(x)
    batch_size = layer_shape[0]
    if K.image_data_format() == 'channels_last':
        num_filters = layer_shape[3]
        num_pixels  = layer_shape[1]*layer_shape[2]
        vectorized_layer = K.reshape(K.permute_dimensions(x,(0,3,1,2)),[batch_size,num_filters,-1])
    else:
        num_filters = layer_shape[1]
        num_pixels = layer_shape[2]*layer_shape[3]
        vectorized_layer = K.reshape(x,[batch_size,num_filters,-1])
    vectorized_layer_transpose = K.permute_dimensions(vectorized_layer,(0,2,1))
    y = K.batch_dot(vectorized_layer,vectorized_layer_transpose)
    y = y/K.cast(2*num_filters*num_pixels,K.floatx())
    return y
        
def getModel( texture_dims=12,input_dims=[]):
    ''' 
        * output_dim: the number of classes (int)
        
        * return: compiled model (keras.engine.training.Model)
    '''
    if type(texture_dims)==int:
        assert(texture_dims <= 18)
        texture_dims = texture_dims * [1.0]
    if type(input_dims) == int:
        assert(input_dims <= 18)
        input_dims = input_dims * [1.0]
    print "Loading VGG.."
    vgg_model = VGG16( weights='imagenet', include_top=False )

    print "done"
    print "Creating outputs..."
    output_list = []
    loss_weights = []
    for index,texture_weight in enumerate(texture_dims):
        if texture_weight != 0:
            current_vgg_layer = vgg_model.layers[index]
            output_list.append(Lambda(gram_matrix,name = '{}_gram'.format(current_vgg_layer.name))(current_vgg_layer.output))
            loss_weights.append(texture_weight)
    for index,input_weight in enumerate(input_dims):
        if input_weight != 0:
            output_list.append((vgg_model.layers[index].output))
            loss_weights.append(input_weight)
    #Create new transfer learning model
    tl_model = Model( inputs=vgg_model.input,outputs = output_list )
    print "done"
    print "Freezing  VGG Layers..."
    #Freeze all layers of VGG16 and Compile the model
    for index,layer in enumerate(vgg_model.layers):
        layer.trainable = False
        current_weights_list = layer.get_weights()
        if len(current_weights_list) == 2:
            current_weights = current_weights_list[0]
            current_biases = current_weights_list[1]
            num_inputs = current_weights.shape[-2]
            scale_factors = np.abs(np.mean(np.sum(current_weights,axis=(0,1,2))+current_biases))
            new_weights = current_weights/scale_factors
            new_biases = current_biases/scale_factors
            layer.set_weights([new_weights,new_biases])
            print "Normalizing weights in layer {} by {}".format(index,scale_factors)
    print "done"
    
    print "Compiling Model..."
    tl_model.compile(optimizer='rmsprop', loss='mean_squared_error')
    print "Done"
    #Confirm the model is appropriate

  
    
    return tl_model

def getTextureModel(input_texture,output_shape,match_image=None,texture_dims=[0,1,0,1,0,0,1,0,1,0,1,0,1,0,1],input_dims=[]):
    tl_model = getModel(texture_dims,input_dims)
    targets = tl_model.predict(input_texture);
    num_input_dims = len([ i for i in input_dims if i > 0])
    if num_input_dims > 0:
        targets2 = tl_model.predict(match_image)
        targets[-num_input_dims:] = targets2[-num_input_dims:]
    my_input  = Input(shape= output_shape)
    my_offset = OffsetLayer(output_shape=output_shape)(my_input)
    my_outputs = tl_model(my_offset)
    my_model = Model(inputs = my_input,outputs=my_outputs)
#    optimizer = optimizers.RMSprop(lr=1.0,decay = 1.0/100)
    optimizer = optimizers.Adam(lr=2.0)
    checkpointer = ModelCheckpoint(filepath="weights_{epoch:04d}.hdf5", verbose=1, save_weights_only=True,period=50)
    my_model.compile(loss='mean_squared_error',loss_weights = tl_model.loss_weights,optimizer=optimizer)
    my_model.summary()
    return my_model,targets
    



if __name__ == '__main__':
    #Output dim for your dataset
    output_dim = 11#For UrbanTribes
    fancy_training=False
    
    num_epochs = 500
    input_image = scipy.misc.imread('daisies_2.jpg',mode='RGB');
    input_image_small = scipy.misc.imresize(input_image,.3)
    if(True):
        transfer_image = scipy.misc.imread('kitten.jpg')
        output_size = transfer_image.shape;
        noise_image = np.copy(transfer_image)

    else:
        output_size = (200,500,3)
        noise_image = np.random.uniform(0,255,output_size).astype('uint8')
    match_image = scipy_image_to_vgg(noise_image);
    model,targets = getTextureModel(scipy_image_to_vgg(input_image_small),output_size,match_image,texture_dims=[0,1,1,1,0,0,1,0,1,0,1,0,1,0,1],input_dims=[0,1e-9])
    batch_input = scipy_image_to_vgg(noise_image);
    batch_output = targets
    start_epoch = 0
    epoch_skip = 5
    plt.figure(1);plt.imshow(input_image_small)
    plt.figure(2);plt.imshow(noise_image)
    plt.show()
    for start_epoch,current_num_epochs in zip(range(0,num_epochs,epoch_skip),range(epoch_skip,num_epochs+epoch_skip,epoch_skip)):
        history = model.fit(x= batch_input,y=batch_output,initial_epoch=start_epoch,epochs=current_num_epochs)
        
        vgg_image_output =batch_input + model.get_weights()[0]
        scipy_image_output = vgg_to_scipy_image(vgg_image_output)

        plt.figure(3);plt.imshow(scipy_image_output)
        plt.show()
    

    
        
