# Pneumonia_model_with_data_augmentation

model trained on grayscale data set of images (no_imgages , width ,height ,1)  ,where no_images is 7750.

Train_set samples : 7750

Test_set samples =  Validation_set samples :  624

all the images initially have 3 channels (RGB) . 

Before training  , first data augmentation procedure  is done on first class (NORMAL) ,since   in the second class (PNEUMONIA) ,there are 3 times more  images than in the first class.

after augmentation both classes have almost the same number of images.

second augmentation procedure  is done during training (ImageDataGen( .. ).flow_from_directory( .. )) . The images are generated in  grayscale ,during data generation while training  . 


--------- MODEL summary ----------

Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_3 (Conv2D)            (None, 64, 64, 64)        1088      
_________________________________________________________________
activation_3 (Activation)    (None, 64, 64, 64)        0         
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 64, 64, 64)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 64, 64, 32)        32800     
_________________________________________________________________
activation_4 (Activation)    (None, 64, 64, 32)        0         
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 64, 64, 32)        0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 64, 64, 32)        4128      
_________________________________________________________________
activation_5 (Activation)    (None, 64, 64, 32)        0         
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 64, 64, 32)        0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 64, 64, 16)        2064      
_________________________________________________________________
activation_6 (Activation)    (None, 64, 64, 16)        0         
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 64, 64, 16)        0         
_________________________________________________________________
flatten (Flatten)            (None, 65536)             0         
_________________________________________________________________
dense (Dense)                (None, 256)               16777472  
_________________________________________________________________
activation_7 (Activation)    (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               32896     
_________________________________________________________________
activation_8 (Activation)    (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 129       
_________________________________________________________________
activation_9 (Activation)    (None, 1)                 0         
=================================================================
Total params: 16,850,577
Trainable params: 16,850,577
Non-trainable params: 0


 accuracy  --- 0.9038
 loss      --- 0.3243
