

import glob as gb
import os
from PIL import Image
from tqdm import tqdm
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from operator import itemgetter, attrgetter


def image_merged(img_full_dir,img_blur_dir,i,img_nums): #图像合并
    image_full = Image.open(img_full_dir)
    image_blur = Image.open(img_blur_dir)

    merged_size =image_full.size
    merged_img =Image.new("RGB",(merged_size[0]*2,merged_size[1]),(0,0,0))
    
    merged_img.paste(image_full,(0,0))
    merged_img.paste(image_blur,(merged_size[0],0))
    merged_img.save ('/2tb/imagedeblurring/imagdeblurring/data/small/generated_train/'+img_nums+i+".jpg")
    return merged_img

def file_name(file_dir):  #文件名获取
    L=[]   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == '.jpg':  
               L.append(os.path.join(root, file))  
    return L  

def data_generator (image_dir,generate_volume):#数据生成
	datagen = ImageDataGenerator(
		rotation_range=.5,
		width_shift_range=0,
		height_shift_range=0,
		shear_range=0.08,
		zoom_range=0.08,
		horizontal_flip=True,
		vertical_flip=True,
                channel_shift_range=10,
		fill_mode='nearest')#图像数据生成器配置

	seed = 1#对模糊图像和清晰图像选用相同的seed以保证随机变换相同
	image = load_img(image_dir)#图像加载，切割，转化数组
	merged_size = image.size
	image_full = image.crop((0, 0, image.size[0] / 2, image.size[1]))
	    
	image_blur = image.crop((image.size[0] / 2, 0, image.size[0], image.size[1]))

	array_full = img_to_array(image_full)  
	array_blur = img_to_array(image_blur)  
	array_full = array_full.reshape((1,) + array_full.shape) 
	array_blur = array_blur.reshape((1,) + array_blur.shape) 




	i = 0
	for batch in datagen.flow(array_full,
		                  batch_size=1,
		                  save_to_dir='/2tb/imagedeblurring/imagdeblurring/data/small/generated_train_full',#生成后的图像保存路径
		                  save_prefix='full',
		                  save_format='jpg',
		                  seed=seed):
	    i += 1
	    if i > generate_volume: #这个generate_volume指出要扩增多少个数据
                  break


	i = 0
	for batch in datagen.flow(array_blur,
		                  batch_size=1,
		                  save_to_dir='/2tb/imagedeblurring/imagdeblurring/data/small/generated_train_blur',
		                  save_prefix='blur',
		                  save_format='jpg',
		                  seed=seed):
	    i += 1
	    if i > generate_volume:
                  break


def image_joint(img_num):
     fullimg_name_list = file_name('/2tb/imagedeblurring/imagdeblurring/data/small/generated_train_full')
     blurimg_name_list = file_name('/2tb/imagedeblurring/imagdeblurring/data/small/generated_train_blur')

     fullimg_name_list.sort()#对两个名字数组重新排序以对称
     blurimg_name_list.sort()

     
     gen_img_length = (len(fullimg_name_list))

     for j in range(gen_img_length):
           image_merged(fullimg_name_list[j],blurimg_name_list[j],str(j),str(img_num))#将两个数组安装顺序一个个拼接
if __name__ == '__main__':
   train_name_list = file_name('/2tb/imagedeblurring/imagdeblurring/data/small/train')
   train_name_list.sort()
   train_img_length = (len(train_name_list))
   for k in range(train_img_length):
     data_generator (train_name_list[k],8)
     image_joint(k)







