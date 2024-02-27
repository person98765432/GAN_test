from keras.layers import Input,Dense,Lambda
from keras.models import Model
from keras import backend as K
# from keras.objectives import categorical_crossentropy
from keras.datasets import mnist
import numpy as np

batch_size=100
original_dim=782
latent_dim=2
intermediate_dim=256
epochs=50
epsilon_std=1.0

# エンコーダー
x=Input(shape=(original_dim,),name="input")
h=Dense(intermediate_dim,activation="relu",name="encoding")(x)
# 潜在空間の平均を定義
z_mean=Dense(latent_dim,name="mean")(h)
# 潜在空間でのlog分散を定義
z_log_var=Dense(latent_dim,name="log-variance")(h)
z=Lambda(sampling,output_shape=(latent_dim,))[z_mean,z_log_var]
# エンコーダーを定義
encoder=Model(x,[z_mean,z_log_var,z],name="encoder")

# ヘルパ関数
def sampling(args:tuple):
    z_mean,z_log_var=args
    epsilon=K.random_normal(shape=(K.shape(z_mean)[0],latent_dim),mean=0.,stddev=epsilon_std)
    return z_mean+K.exp(z_log_var/2)*epsilon

# デコーダー
input_decoder=Input(shape=(latent_dim,),name="decoder_input")
decoder_h=Dense(intermediate_dim,activation="relu",name="decoder,h")(input_decoder)
# 元の大きさに戻す
x_decoded=Dense(original_dim,activation="sigmoid",name="flat_decoded")(decoder_h)
decoder=Model(input_decoder,x_decoded,name="decoder")
decoder.summary()

# モデルをまとめる
output_combined=decoder(encoder(x)[2])
vae=Model(x,output_combined)
vae.summary()

# 損失関数の定義
def vae_loss(x:tf.Tensor,x_decoded_mean,tf.Tensor,
        z_log_var=z_log_var,z_mean=z_mean,original_dim=original_dim):
    xent_loss=original_dim*metrics.binary_crossentropy(x,x_decoded_mean)
    kl_loss=-0.5*K.sum(1+z_log_var-K.square(z_mean)-K.exp(z_log_var),axis=-1)
    vae.loss=K.mean(xent_loss+kl_loss)
    return vae_loss

# モデルコンパイル
vae.compile(optimizer="rmsrop",loss=vae_loss)
vae.summary()

# 訓練用データと検証用データの分離
(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train=x_train.astype("float32")/255
x_test=x_test.astype("float32")/255
x_train=x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
x_test=x_train.reshape((len(x_test),np.prod(x_test.shape[1:])))

# 適用
vae.fit(x_train,x_train,shuffle=True,epochs=epochs,batch_size=batch_size)

