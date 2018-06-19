from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras import backend
import numpy as np
from utils import *
from scipy.optimize import fmin_l_bfgs_b
from PIL import Image

style_path = 'p3.jpg' # PATH TO STYLE IMAGE
content_path = 'm2.jpg' # PATH TO CONTENT IMAGE

height = 512
width = 512

style_image = image.load_img(style_path, target_size=(height, width))
content_image = image.load_img(content_path, target_size=(height, width))

content_array=np.asarray(content_image,dtype='float32')
content_array=np.expand_dims(content_array,axis=0)
style_array=np.asarray(style_image,dtype='float32')
style_array=np.expand_dims(style_array,axis=0)

content_array[:, :, :, 0] -= 103.939
content_array[:, :, :, 1] -= 116.779
content_array[:, :, :, 2] -= 123.68
content_array=content_array[:, :, :, ::-1]
style_array[:, :, :, 0] -= 103.939
style_array[:, :, :, 1] -= 116.779
style_array[:, :, :, 2] -= 123.68
style_array=style_array[:, :, :, ::-1]

print(style_array.shape)
print(content_array.shape)

content_image=backend.variable(content_array)
style_image=backend.variable(style_array)

combination_image=backend.placeholder((1,height,width,3))
input_tensor=backend.concatenate([content_image,style_image,combination_image],axis=0)

model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)

content_weight = 0.05
style_weight = 10.0
total_variation_weight = 1.0

layers=dict([(layer.name, layer.output) for layer in model.layers])
loss=backend.variable(0.)


layer_features=layers['block2_conv2']
content_image_features=layer_features[0,:,:,:]
combination_features=layer_features[2,:,:,:]
loss+=content_weight*content_loss(content_image_features,combination_features)

feature_layers = ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2',
                  'block3_conv1', 'block3_conv2', 'block3_conv3', 
                  'block4_conv1', 'block4_conv2', 'block4_conv3',
                  'block5_conv1', 'block5_conv2', 'block5_conv3']

for layer_name in feature_layers:
    layer_features=layers[layer_name]
    style_features=layer_features[1,:,:,:]
    combination_features=layer_features[2,:,:,:]
    sl=style_loss(style_features,combination_features, height, width)
    loss+=(style_weight/len(feature_layers))*sl

loss += total_variation_weight * total_variation_loss(combination_image, height, width)

grads = backend.gradients(loss, combination_image)
outputs=[loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)
f_outputs = backend.function([combination_image], outputs)


class Evaluator(object):
    def __init__(self):
        self.loss_value=None
        self.grads_values=None
    
    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x, height, width, f_outputs)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


evaluator=Evaluator()

x=np.random.uniform(0,255,(1,height,width,3))-128.0

iterations = 10
import time
for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                           fprime=evaluator.grads, maxfun=20)
    print(min_val)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))

print()
x = x.reshape((height, width, 3))
x = x[:, :, ::-1]
x[:, :, 0] += 103.939
x[:, :, 1] += 116.779
x[:, :, 2] += 123.68
x = np.clip(x, 0, 255).astype('uint8')
res = Image.fromarray(x)
res.save(str(content_path).split('.')[0] + str(style_path).split('.')[0] + '.jpg')
