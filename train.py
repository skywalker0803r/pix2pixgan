# example of pix2pix gan for satellite to map image-to-image translation
# example of pix2pix gan for satellite to map image-to-image translation
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
#from tensorflow import keras 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model,  load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D, UpSampling1D
# from keras.layers import Conv2D
# from keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU,ZeroPadding1D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Add

#--- 20210820:存檔錄加上日期 ---#
import datetime
from time import time
import os
import sys
#----- the dir of the py -----#
# file_dir = os.path.abspath('')
# --- __file__ applies to modules and Python scripts, not to notebooks. ---# 
file_dir = os.path.abspath('')
parent_path = os.path.abspath(os.pardir)
pathname = parent_path+"/Functions"
sys.path.append(pathname)

#----- import functions -----#
from MkDir import mkdir
from Promopt_and_Check_Input import Promopt_and_Check_Input
from ListAllFile import ListAllFile

#--- select GPU #
strPromote = "select GPU  [0]GPU 0: TITAN RTX [1]GPU 1: RTX 2080Ti:\n>>"
GPUSettings = Promopt_and_Check_Input(1, strPromote, 0)
if GPUSettings == 0:
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

strPromote = "請輸入訓練週期數目(trainEpochs，建議:1 ~ 600):\n>>"
trainEpochs = Promopt_and_Check_Input(600, strPromote, 0)

strPromote = "請選擇運作模式:\n"
strPromote += "[0]Train [1]Load and ReTrain:\n>>"
OperationMode = Promopt_and_Check_Input(1, strPromote, 0)

#--- 20210729:存檔錄加上日期 ---#
#--- date time ---#
data_name = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d_%H%M%S')
model_path = "./models/" + data_name
predict_img_path =  "./predict_img/" + data_name
history_path ="./results_baseline/" + data_name

#--- Weights_Path ---#

WeightPath = file_dir + "/WeightPath.txt"

strWeightPath=""

if OperationMode == 0:
    previous_epoch = 0
    catalog_num = 1
    model_parent_path = "./models/" + data_name
    predict_parent_img_path =  "./predict_img/" + data_name
    history_parent_path ="./results_baseline/" + data_name
elif OperationMode == 1:  
    #--- Load and ReTrain ---#
    with open(WeightPath) as fweight:
        fweight.readline() # ignore one line
        pre_model_path = fweight.readline().replace('\n', '')
        # fweight.readline() # ignore one line
        # pre_predict_img_path = fweight.readline().replace('\n', '')
        fweight.readline() # ignore one line
        model_parent_path = fweight.readline().replace('\n', '')
        fweight.readline() # ignore one line
        predict_parent_img_path = fweight.readline().replace('\n', '')
        fweight.readline() # ignore one line
        history_parent_path = fweight.readline().replace('\n', '')
        fweight.readline() # ignore one line
        pre_epoch_num = fweight.readline().replace('\n', '')
        fweight.readline() # ignore one line
        catalog = fweight.readline().replace('\n', '')
        fweight.close()
        
    previous_epoch = int(pre_epoch_num)
    catalog_num = int(catalog) + 1
    #--- List all Weight files ---//
    
    strModelList, Weight_Matrix, FinalIndex = ListAllFile(pre_model_path, 7)
#     print(Weight_Matrix)
    strPromote = "Models 存放路徑:{0}\n".format(pre_model_path)
    strPromote += "Total {0} Weight Files\n".format(FinalIndex+1)
    strPromote += "[0].{0} to [{1}].{2} \n".format(Weight_Matrix[0], FinalIndex, Weight_Matrix[-1])
    strPromote += "請選擇 Weight File:\n>>"
    
    Weightindex = Promopt_and_Check_Input(FinalIndex, strPromote, 0)
    weight_file = pre_model_path + "/" + Weight_Matrix[Weightindex]
    weight_num = weight_file.split("/")[4].split("_")[2]

model_path = model_parent_path + "/%04d" % (catalog_num)
predict_img_path =  predict_parent_img_path + "/%04d" % (catalog_num)
history_path = history_parent_path + "/%04d" % (catalog_num)
LocalWeightPath = model_parent_path + "/WeightPath.txt"

mkdir(model_path)
mkdir(predict_img_path)
mkdir(history_path)

#--- update the path ---#
strWeightPath = 'model_path:\n{0}\n'.format(model_path)
strWeightPath += 'model_parent_path:\n{0}\n'.format(model_parent_path)
strWeightPath += 'predict_parent_img_path:\n{0}\n'.format(predict_parent_img_path)
strWeightPath += 'history_parent_path:\n{0}\n'.format(history_parent_path)


# #--- save weight path ---#
# f_WeightPath = open(WeightPath, 'w')
# f_WeightPath.write(strWeightPath)
# f_WeightPath.close()

# f_WeightPath01 = open(LocalWeightPath, 'w')
# f_WeightPath01.write(strWeightPath)
# f_WeightPath01.close()

#--- 限制 GPU 使用所有的 Memory ---#
import tensorflow as tf
from tensorflow.python.keras import backend as K
gpu_mem_fraction = 0.9
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_fraction, allow_growth=True)
#--- The name tf.ConfigProto is deprecated. Please use tf.compat.v1.ConfigProto instead. ---#
#run_config = tf.ConfigProto(
run_config = tf.compat.v1.ConfigProto(
    log_device_placement=True,
    allow_soft_placement=True,
    gpu_options=gpu_options
)

sess = tf.compat.v1.Session(config=run_config)
K.set_session(sess)


# define the discriminator model# 定義判別器模型
def define_discriminator(image_shape):
    # weight initialization權重初始化
    init = RandomNormal(stddev=0.02)
    # source image input 源圖像輸入
    in_src_image = Input(shape=image_shape)
    # target image input 目標圖像輸入
    in_target_image = Input(shape=image_shape)
    # concatenate images channel-wise 按通道連接圖像
    merged = Concatenate()([in_src_image, in_target_image]) #merged 合併 
    # C64
    d = Conv1D(64, 4, strides=2, padding='same', kernel_initializer=init)(merged)
    # d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv1D(128, 4, strides=2, padding='same', kernel_initializer=init)(d)
    # d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv1D(256, 4, strides=2, padding='same', kernel_initializer=init)(d)
    # d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv1D(512, 4, strides=2, padding='same', kernel_initializer=init)(d)
    # d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv1D(512, 4, padding='same', kernel_initializer=init)(d)
    # d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv1D(1, 4, padding='same', kernel_initializer=init)(d)
    # d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_target_image], patch_out)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model
"""
if __name__ == '__main__':
    d_model = define_discriminator((image_shape))
print(d_model.summary())
"""
def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same'):#-------------------------------------------------------------------
    init = RandomNormal(stddev=0.02)
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x
# define an encoder block 編碼器塊 U-NET編碼器裡面幫助函式建立用於編碼器的層塊
def define_encoder_block(layer_in, n_filters, batchnorm=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer 添加下採樣層
    g = Conv1D(n_filters, 4, strides=2, padding='same', kernel_initializer=init)(layer_in)
    # g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    # conditionally add batch normalization 有條件地添加批量正歸化
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g

# define a decoder block  U-NET編碼器裡面函式建立用於解碼器的層塊
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv1D(n_filters, 4, strides=1, padding='same', kernel_initializer=init)(layer_in)
    g = UpSampling1D(2)(g)
    # g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    # add batch normalization
    g = BatchNormalization()(g, training=True)#training=false會使預測出來的值分佈更加離散，差別相對較大；training=true預測出來的數值差別比較的小，甚至無差別。
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # merge with skip connection
    g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)
    return g
def define_encoder_block_plus(layer_in, n_filters, batchnorm=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer 添加下採樣層
    g = Conv1D(n_filters, 4, strides=1, padding='same', kernel_initializer=init)(layer_in)
    # g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    # conditionally add batch normalization 有條件地添加批量正歸化
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g

def decoder_block_plus(layer_in, skip_in, n_filters, dropout=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv1D(n_filters, 4, strides=1, padding='same', kernel_initializer=init)(layer_in)
    #g = UpSampling1D(2)(g)
    # g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    # add batch normalization
    g = BatchNormalization()(g, training=True)#training=false會使預測出來的值分佈更加離散，差別相對較大；training=true預測出來的數值差別比較的小，甚至無差別。
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # merge with skip connection
    g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)
    
    return g
def define_res_encoder_block(layer_in, n_filters,dropout=True,batch=True ):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    dro_x=0.5
    # add downsampling layer 添加下採樣層
    #g = Conv1D(n_filters, 1, strides=1, padding='same', kernel_initializer=init)(layer_in)
    
    #y = LeakyReLU(alpha=0.2)(layer_in)
    
    g = Conv1D(n_filters, 4, strides=1, padding='same', kernel_initializer=init)(layer_in)
    g = BatchNormalization()(g, training=True)
    #g = LeakyReLU(alpha=0.2)(g)
    g = Activation('relu')(g)
    
    g = Conv1D(n_filters, 4, strides=1, padding='same', kernel_initializer=init)(g)
    if batch:
        g = BatchNormalization()(g, training=True)
    
    #g = LeakyReLU(alpha=0.2)(g)
    g = Activation('relu')(g)
    #if dropout:
        #g = Dropout(dro_x)(g, training=True)
    
    g=Add()([g,layer_in])
    return g


def decoder_res_block(layer_in, skip_in, n_filters, dropout=True,batch=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    dro_x=0.5
    # add upsampling layer
    g = Conv1D(n_filters, 4, strides=1, padding='same', kernel_initializer=init)(layer_in)
    g = UpSampling1D(2)(g)
    # g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    # add batch normalization
    g = BatchNormalization()(g, training=True)#training=false會使預測出來的值分佈更加離散，差別相對較大；training=true預測出來的數值差別比較的小，甚至無差別。
    # conditionally add dropout
    #if dropout:
    #    g = Dropout(dro_x)(g, training=True)
    # merge with skip connection
    g = Concatenate()([g, skip_in])
    # relu activation
    
    g = Activation('relu')(g)
    
    g = Conv1D(n_filters, 4, strides=1, padding='same', kernel_initializer=init)(g)
    y =  Activation('relu')(g)
    
    g = Conv1D(n_filters, 4, strides=1, padding='same', kernel_initializer=init)(y)
    g = BatchNormalization()(g, training=True)
    g =  Activation('relu')(g)
    
    g = Conv1D(n_filters, 4, strides=1, padding='same', kernel_initializer=init)(g)
    #g = BatchNormalization()(g, training=True)
    if batch:
       g = BatchNormalization()(g, training=True) 
    g =  Activation('relu')(g)
    if dropout:
        g = Dropout(dro_x)(g, training=True)
    
    g=Add()([g,y])


    return g
def define_generator(image_shape=(256,1)):
    # weight initialization  中間=8塊20230202_215535
    init = RandomNormal(stddev=0.02)
    # image input  kerner size filter=2
    in_image = Input(shape=image_shape)
    # encoder model: C64-C128-C256-C512-C512-C512-C512-C512
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)
    e7 = define_encoder_block(e6, 512)
    e8 =define_encoder_block(e7, 512)
    # bottleneck, no batch norm and relu
    b = Conv1D(512, 4, strides=2, padding='same', kernel_initializer=init)(e8)
    # b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
    b = Activation('relu')(b)
    
    b=define_res_encoder_block(b,512)
    b = Activation('relu')(b)
    
    b=define_res_encoder_block(b,512)
    b = Activation('relu')(b)
    
    b=define_res_encoder_block(b,512)
    b = Activation('relu')(b)
    
    b=define_res_encoder_block(b,512)
    #########################################
    
    
    
    

    # decoder model: CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
    d0 = decoder_block(b, e8, 512)
    d1 = decoder_block(d0, e7, 512)
    d2 = decoder_block(d1, e6, 512)
    d3 = decoder_block(d2, e5, 512,dropout=False)
    d4 = decoder_block(d3, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)
    # output
    g = Conv1D(1, 4, strides=1, padding='same', kernel_initializer=init)(d7)
    g = UpSampling1D(2)(g)
    # g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation('tanh')(g)
    # define model
    model = Model(in_image, out_image)
    return model

"""
###編碼是RESNET34 解碼是RESNET+UNET
def define_generator(image_shape=(256,1)):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # encoder model: C64-C128-C256-C512-C512-C512-C512-C512
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 256)#512->256
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)
    e7 = define_encoder_block(e6, 512)
    # bottleneck, no batch norm and relu
    b = Conv1D(512, 4, strides=2, padding='same', kernel_initializer=init)(e7)
    # b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
    b = Activation('relu')(b)
    # decoder model: CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
    d1 = decoder_block(b, e7, 512)
    d2 = decoder_block(d1, e6, 512)
    d3 = decoder_block(d2, e5, 512)
    d4 = decoder_block(d3, e4, 256, dropout=False)#512->256
    d5 = decoder_block(d4, e3, 256, dropout=False)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)
    # output
    g = Conv1D(1, 4, strides=1, padding='same', kernel_initializer=init)(d7)
    g = UpSampling1D(2)(g)
    # g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation('tanh')(g)
    # define model
    model = Model(in_image, out_image)
    return model
    

    return model
"""



#-------------------------------------------
def load_real_samples(filename): #加載生成真實樣本
    # load compressed arrays
    data = load(filename)
    # unpack arrays
    X1, X2 = data['arr_0'], data['arr_1']
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    print('data',data)
    return [X1, X2]
""" 
datsset_name = 'OneD'
dataset = load_real_samples('./data/OCT/' + datsset_name + '/HyBride_BtoAfter_Training_1024.npz')
image_shape = dataset[0].shape[1:]
print('image_shape',image_shape)
#--------------------------------------------


if __name__ == '__main__':
	g_model = define_generator((image_shape))
	print(g_model.summary())

if __name__ == '__main__':
111	g_model = 
	print(g_model.summary())
    """
# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
    # make weights in the discriminator not trainable 使鑑別器中的權重不可訓練
    d_model.trainable = False
    # define the source image定義源圖像
    in_src = Input(shape=image_shape)
    # connect the source image to the generator input # 將源圖像連接到生成器輸入
    gen_out = g_model(in_src)
    # connect the source input and generator output to the discriminator input# 將源輸入和生成器輸出連接到鑑別器輸入
    dis_out = d_model([in_src, gen_out])
    # identity element # 標識元素
    input_id = Input(shape=image_shape)
    output_id = g_model(input_id)
    # src image as input, generated image and classification output src 圖像作為輸入，生成圖像和分類輸出
    model = Model([in_src, input_id], [dis_out, gen_out, output_id])
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae', 'mae'], optimizer=opt, loss_weights=[1, 100, 100])
    return model
"""
if __name__ == '__main__':
	d_model = define_discriminator((image_shape))
	g_model = define_generator((image_shape))
	gan_model = define_gan(g_model, d_model, image_shape)
	print(g_model.summary())
"""
# load and prepare training images# 加載並準備訓練圖像
def load_real_samples(filename): #加載真實樣本
    # load compressed arrays
    data = load(filename)  #HyBride_BtoAfter_Training_1024.npz
    # unpack arrays
    X1, X2 = data['arr_0'], data['arr_1']
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    print('data',data)
    return [X1, X2]

# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape): #加載生成真實樣本
    # unpack dataset
    trainA, trainB = dataset
    # choose random instances
    # print(n_samples.shape)
    ix = randint(0, trainA.shape[0], n_samples)
    # retrieve selected images
    X1, X2 = trainA[ix], trainB[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, patch_shape, 1))
    return [X1, X2], y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance生成假實例
    X = g_model.predict(samples)
    # create 'fake' class labels (0)創建“假”類標籤 (0)
    y = zeros((len(X), patch_shape, 1))
    return X, y

# generate samples and save as a plot and save the model
#--- 改用 epoch 做紀錄 ---#
def summarize_performance(epoch_num, g_model, d_model, dataset, n_samples=3):
    # select a sample of input images# 選擇輸入圖像的樣本
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
    # generate a batch of fake samples 生成一批假樣本
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    # scale all pixels from [-1,1] to [0,1]
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0
    
    # plot real source images
    x_num = np.arange(X_realA.shape[1])
    fig, axs = plt.subplots(3, n_samples)
    # plot real source images
    for i in range(n_samples):
        axs[0, i].plot(x_num, X_realA[i])
        axs[0, i].set_title('Input')
        axs[0, i].axis('off')
    # plot generated target image
    for i in range(n_samples):
        axs[1, i].plot(x_num, X_realB[i])
        axs[1, i].set_title('Target')
        axs[1, i].axis('off')
    # plot real target image
    for i in range(n_samples):
        axs[2, i].plot(x_num, X_fakeB[i])
        axs[2, i].set_title('Translated')
        axs[2, i].axis('off')
    # save plot to file
    filename1 = predict_img_path + '/plot_%06d.png' % (epoch_num)
    plt.savefig(filename1)
    plt.close()
    # save the generator model
    filename2 = model_path + '/g_model_%06d.h5' % (epoch_num)
    g_model.save(filename2)
    # save the discriminator model
    filename3 = model_path + '/d_model_%06d.h5' % (epoch_num)
    d_model.save(filename3)
    print('>Saved: %s, %s and %s' % (filename1, filename2, filename3))

# create a line plot of loss for the gan and save to file
def plot_history(d1_hist, d2_hist, g_hist):
    # plot Discriminator loss
    plt.subplot(2, 1, 1)
    plt.plot(d1_hist, label='d-real')
    plt.plot(d2_hist, label='d-fake')
    plt.legend()
    # save plot to file
    plt.savefig(history_path + '/plot_Discriminator_loss.png')
    plt.close()
    # plot Generator loss
    plt.subplot(2, 1, 1)
    plt.plot(g_hist, label='gen')
    plt.legend()
    # save plot to file
    plt.savefig(history_path + '/plot_Generator_loss.png')
    plt.close()
    
# train pix2pix models
def train(d_model, g_model, gan_model, dataset, previous_epoch=0, n_epochs=100, n_batch=1):
    # determine the output square shape of the discriminator #確定鑑別器的輸出方形
    n_patch = d_model.output_shape[1]
    # unpack dataset 打開數據集
    trainA, trainB = dataset
    # calculate the number of batches per training epoch 計算每個訓練時期的批次數
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations 計算訓練迭代次數
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs 手動枚舉epoch
    d1_hist, d2_hist, g_hist= list(), list(), list()
    epoch_num = previous_epoch #0
    for i in range(n_steps):
        # select a batch of real samples# 選擇一批真實樣本 
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        #print('[X_realA, X_realB], y_real',[X_realA, X_realB], y_real)
        # generate a batch of fake samples生成一批假樣本
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        #print('X_fakeB, y_fake',X_fakeB, y_fake)
        # update discriminator for real samples 更新真實樣本的鑑別器
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
       # print('d_loss1',d_loss1)
        # update discriminator for generated samples # 更新生成樣本的鑑別器
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        #print('d_loss2',d_loss2)
        # update the generator# 更新生成器
        g_loss, _, _, _ = gan_model.train_on_batch([X_realA, X_realB], [y_real, X_realB, X_realB])
        #print('g_loss, _, _, _',g_loss, _, _, _)
        # summarize performance         # summarize performance

        print('>%d, d1[%.3e] d2[%.3e] g[%.3e]' % (i+1, d_loss1, d_loss2, g_loss))
        # summarize model performance
        # record history
        d1_hist.append(d_loss1)
        d2_hist.append(d_loss2)
        g_hist.append(g_loss)
        if (i+1) % (bat_per_epo * 10) == 0:
            epoch_num = epoch_num + 10
            summarize_performance(epoch_num, g_model, d_model, dataset)
    plot_history(d1_hist, d2_hist, g_hist)
    
# load image data
datsset_name = ''
# dataset = load_real_samples('./data/OCT/' + datsset_name + '/BeforePiShitToAfterPiShift_Training_256.npz')
# dataset = load_real_samples('./data/OCT/' + datsset_name + '/SpceToAfterPiShift_Training_256.npz')
#dataset = load_real_samples('./data/OCT/' + datsset_name + '/HyBride_BtoAfter_Training_1024.npz')
###原本使用ㄉ 
dataset = load_real_samples('./data/' + datsset_name + '/HyBride_BtoAfter_Training_1024.npz')

#dataset = load_real_samples('./data/OCT/' + datsset_name + '/SingleLayer/SL_BtoAfter_Training_1024.npz')

#dataset = load_real_samples('./data/OCT/' + datsset_name + '/MultiLayer/ML_BtoAfter_Training_1024.npz') 
# dataset = load_real_samples('./data/OCT/Training_256.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
print('image_shape',image_shape)
"""
#---------------------生成
import imgaug.augmenters as iaa

hflip = iaa.Fliplr(1)
hflip_aug = hflip(images=dataset)
"""
"""
vflip = iaa.Flipud(1)
vflip_aug = vflip(images=X)
"""
"""
X_aug = np.concatenate((dataset, hflip_aug)) 
np.random.shuffle(X_aug)
aaa=X_aug
print('Augemented Data set sanity check:增強數據集健全性檢查', aaa.shape)
print('image_shape',image_shape)

#import matplotlib.pyplot as plt
#plt.imshow(dataset[0])
#------------------------
"""
if OperationMode == 0:
    # define the models
    d_model = define_discriminator(image_shape)
    g_model = define_generator(image_shape)
else:
    # load the models
    d_model = load_model(pre_model_path + '/d_model_' + weight_num + '.h5')
    g_model = load_model(pre_model_path + '/g_model_' + weight_num + '.h5')
    
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)

# train model
train(d_model, g_model, gan_model, dataset, previous_epoch, trainEpochs)

previous_epoch = previous_epoch + trainEpochs
strWeightPath += 'Epoch_Num:\n{0}\n'.format(previous_epoch)
strWeightPath += 'Catalog_Num:\n{0}\n'.format(catalog_num)


#--- save weight path ---#
f_WeightPath = open(WeightPath, 'w')
f_WeightPath.write(strWeightPath)
f_WeightPath.close()

f_WeightPath01 = open(LocalWeightPath, 'w')
f_WeightPath01.write(strWeightPath)
f_WeightPath01.close()