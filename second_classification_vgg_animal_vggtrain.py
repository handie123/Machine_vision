from keras.applications.vgg16 import VGG16 
from keras.models import Sequential 
from keras.layers import Conv2D,MaxPool2D,Activation,Dropout,Flatten,Dense 
from keras.optimizers import SGD 
from keras.optimizers import adam
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img 
import numpy as np

vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(513, 45, 3))
# 搭建全连接层 
top_model = Sequential() 
top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:])) 
top_model.add(Dense(256,activation='relu')) 
top_model.add(Dropout(0.5)) 
top_model.add(Dense(2,activation='softmax'))  

model = Sequential() 
model.add(vgg16_model) 
model.add(top_model)

train_datagen = ImageDataGenerator(
        width_shift_range=0.2,  # 随机水平平移
        height_shift_range=0.2,  # 随机竖直平移
        rescale=1/255,         # 数据归一化
        horizontal_flip=True,  # 水平翻转
        )  
test_datagen = ImageDataGenerator(     
        rescale=1/255,          # 数据归一化
)

batch_size = 32

# 生成训练数据 
train_generator = train_datagen.flow_from_directory(  
        'images/train',
        target_size=(513,45),     
        batch_size=batch_size,
)

# 验证数据 
validation_generator = test_datagen.flow_from_directory(     
        'images/val',
        target_size=(513, 45),
        batch_size=batch_size,    
 )
# 测试数据 
# test_generator = test_datagen.flow_from_directory(
#        'img/test',     
#        target_size=(513,45),     
#        batch_size=batch_size,    
# )

train_generator.class_indices

# 定义优化器，代价函数，训练过程中计算准确率 
model.compile(optimizer=SGD(lr=1e-3, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_generator, steps_per_epoch=50, epochs=1, validation_data=validation_generator, validation_steps=50)
model.save('model_vgg161.h5')


