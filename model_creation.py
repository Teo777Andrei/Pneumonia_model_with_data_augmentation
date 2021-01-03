import os 
import numpy as np
from tensorflow.keras.layers import (Dense , Dropout ,Flatten ,
                                     MaxPool2D ,Conv2D ,Activation )
from tensorflow.keras.models import Sequential

from tensorflow.keras.activations import relu ,sigmoid
from tensorflow.keras.preprocessing.image import load_img , img_to_array  ,ImageDataGenerator


process_dir ="C:\\Datasets\\chest_xray\\chest_xray"

test_dir = process_dir + "\\test"
train_dir = process_dir + "\\train"
validation_dir = process_dir + "\\validation"
 
image_gen = ImageDataGenerator(rotation_range= 20,
                               width_shift_range = 0.1 ,
                               height_shift_range = 0.1 ,
                               shear_range = 0.1 ,
                               zoom_range = 0.1,
                               horizontal_flip =True ,
                               rescale = 1/255 ,
                               fill_mode = "nearest")
batch_size = 32

#augment images  in NORMAL training  directory 

"""
to_augment_dataset = []
for image_id in os.listdir(train_dir  + "\\NORMAL"):
    try:    
        image = load_img(train_dir + "\\NORMAL\\" + image_id ,target_size = (130 ,130) ,color_mode ="grayscale")
        image=  img_to_array(image)
        to_augment_dataset.append(image)
    except:
        pass
    
to_augment_dataset = np.array(to_augment_dataset)

batch_iter = 0
for batch in  ImageDataGenerator(zoom_range = 0.1 , horizontal_flip = True ).flow(to_augment_dataset , 
                                                                                       batch_size = 1 ,
                                                                                       save_to_dir  = train_dir + "\\NORMAL" ,
                                                                                       save_prefix = "augmented" ,
                                                                                       save_format ="jpeg"):
    batch_iter+=1
    if batch_iter  == 2534:
        break
    
    #verbosity
    print(batch_iter)
                                                                                                                                   )
"""
training_samples = 7751
test_samples = 640

width ,height = 64 ,64

x_generator_train  = image_gen.flow_from_directory( directory = train_dir ,
                                                       batch_size = batch_size ,
                                                       target_size= (width  ,height) ,
                                                       color_mode ="grayscale" ,
                                                       class_mode = "binary" ,
                                                       )

x_generator_test = ImageDataGenerator(rescale = 1/255)
x_generator_test = x_generator_test.flow_from_directory(directory = test_dir ,
                                                        batch_size = batch_size ,
                                                        target_size=(width ,height) ,
                                                        color_mode = "grayscale" ,
                                                        class_mode = "binary" ,
                                                        shuffle= False)

classes =  x_generator_train.class_indices
    
    
    
model = Sequential()
model.add(Conv2D(filters =  64 ,kernel_size  = (4 ,4) ,
                 input_shape = (width ,height ,1) ,strides  =(1 ,1) ,
                 padding ="same"))


model.add(Activation(relu))

model.add(MaxPool2D(pool_size = (2 ,2) ,strides = (1,1) ,
                    padding ="same"))

model.add(Conv2D(filters =  32 ,kernel_size  = (4 ,4) ,
                 strides  =(1 ,1) ,padding ="same"))


model.add(Activation(relu))

model.add(MaxPool2D(pool_size = (2 ,2) ,strides = (1,1) ,
                    padding ="same"))

model.add(Conv2D(filters =  32 ,kernel_size  = (2 ,2) ,
                 strides  =(1 ,1) ,
                 padding ="same"))


model.add(Activation(relu))

model.add(MaxPool2D(pool_size = (2 ,2) ,strides = (1,1) ,
                    padding ="same"))
    

model.add(Conv2D(filters = 16 ,kernel_size= (2 ,2) ,strides = (1 ,1) ,padding ="same"))
model.add(Activation(relu))
model.add(MaxPool2D(pool_size = (2 ,2) ,strides = (1 ,1) ,padding = "same"))
model.add(Flatten())


model.add(Dense(256))
model.add(Activation(relu))
model.add(Dropout(0.3))

model.add(Dense(128))
model.add(Activation(relu))
model.add(Dropout(0.3))



model.add(Dense(1))
model.add(Activation(sigmoid))


model.compile(optimizer  ="adam" ,loss = "binary_crossentropy" ,metrics =["accuracy"])

score =model.fit_generator(x_generator_train   ,
                           steps_per_epoch = 1.0 * (training_samples // batch_size) ,
                           validation_data = x_generator_test ,
                           validation_steps =1.0 * (test_samples // batch_size) ,
                           epochs = 10,
                           callbacks = None
                           )

model.save("pneumonia_model_augmented5.h5")

