from keras.models import load_model 
import numpy as np  
from keras.applications.vgg16 import VGG16 
from keras.models import Sequential 
from keras.layers import Conv2D,MaxPool2D,Activation,Dropout,Flatten,Dense 
from keras.optimizers import SGD 
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img 
import glob

label = np.array(['sunflowers', 'tulips'])
# 载入模型
model = load_model('model_vgg161.h5')  
# model = load_model('model_cnn1.h5')
# 导入图片
num = 0

list = glob.glob('img/test/sunflowers/*.jpg')
for i in(list):
    image = load_img(i)  # 45 513
    image = image.resize((45, 513))
    image = img_to_array(image)  # 513 45 3
    image = image/255 
    image = np.expand_dims(image, 0)
    labelp = label[model.predict_classes(image)]
    # print(labelp)
    if labelp == 'sunflowers':
        num = num+1
    


list1 = glob.glob('img/test/tulips/*.jpg')
for j in(list1):
    image1 = load_img(j)
    image1 = image1.resize((45, 513))
    image1 = img_to_array(image1) 
    image1 = image1/255 
    image1 = np.expand_dims(image1, 0)
    labelp = label[model.predict_classes(image1)]
    if labelp == 'tulips':
        num = num+1
# print(num)
# print(len(list))
print(str(len(list) + len(list1))+"张图片，预测成功"+str(num)+"张")



