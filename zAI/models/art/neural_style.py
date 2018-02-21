# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np 
import scipy.io  
import struct
import errno
import time                       
import cv2
import os

class NeuralStyle(object):
    def __init__(self,args):
        self.args = args
        
        # normalize weights
        self.args['style_layer_weights']   = self.normalize(self.args['style_layer_weights'])
        self.args['content_layer_weights'] = self.normalize(self.args['content_layer_weights'])
        self.args['style_imgs_weights']    = self.normalize(self.args['style_imgs_weights'])
        
#        # create directories for output
#        if self.args['video']:
#          self.maybe_make_directory(self.args['video_output_dir'])
#        else:
#          self.maybe_make_directory(self.args['img_output_dir'])

    '''
      pre-trained vgg19 convolutional neural network
      remark: layers are manually initialized for clarity.
    '''
    
    def build_model(self,input_img):
      if self.args['verbose']: print('\nBUILDING VGG-19 NETWORK')
      net = {}
      _, h, w, d     = input_img.shape
      
      if self.args['verbose']: print('loading model weights...')
      vgg_rawnet     = scipy.io.loadmat(self.args['model_weights'])
      vgg_layers     = vgg_rawnet['layers'][0]
      if self.args['verbose']: print('constructing layers...')
      net['input']   = tf.Variable(np.zeros((1, h, w, d), dtype=np.float32))
    
      if self.args['verbose']: print('LAYER GROUP 1')
      net['conv1_1'] = self.conv_layer('conv1_1', net['input'], W=self.get_weights(vgg_layers, 0))
      net['relu1_1'] = self.relu_layer('relu1_1', net['conv1_1'], b=self.get_bias(vgg_layers, 0))
    
      net['conv1_2'] = self.conv_layer('conv1_2', net['relu1_1'], W=self.get_weights(vgg_layers, 2))
      net['relu1_2'] = self.relu_layer('relu1_2', net['conv1_2'], b=self.get_bias(vgg_layers, 2))
      
      net['pool1']   = self.pool_layer('pool1', net['relu1_2'])
    
      if self.args['verbose']: print('LAYER GROUP 2')  
      net['conv2_1'] = self.conv_layer('conv2_1', net['pool1'], W=self.get_weights(vgg_layers, 5))
      net['relu2_1'] = self.relu_layer('relu2_1', net['conv2_1'], b=self.get_bias(vgg_layers, 5))
      
      net['conv2_2'] = self.conv_layer('conv2_2', net['relu2_1'], W=self.get_weights(vgg_layers, 7))
      net['relu2_2'] = self.relu_layer('relu2_2', net['conv2_2'], b=self.get_bias(vgg_layers, 7))
      
      net['pool2']   = self.pool_layer('pool2', net['relu2_2'])
      
      if self.args['verbose']: print('LAYER GROUP 3')
      net['conv3_1'] = self.conv_layer('conv3_1', net['pool2'], W=self.get_weights(vgg_layers, 10))
      net['relu3_1'] = self.relu_layer('relu3_1', net['conv3_1'], b=self.get_bias(vgg_layers, 10))
    
      net['conv3_2'] = self.conv_layer('conv3_2', net['relu3_1'], W=self.get_weights(vgg_layers, 12))
      net['relu3_2'] = self.relu_layer('relu3_2', net['conv3_2'], b=self.get_bias(vgg_layers, 12))
    
      net['conv3_3'] = self.conv_layer('conv3_3', net['relu3_2'], W=self.get_weights(vgg_layers, 14))
      net['relu3_3'] = self.relu_layer('relu3_3', net['conv3_3'], b=self.get_bias(vgg_layers, 14))
    
      net['conv3_4'] = self.conv_layer('conv3_4', net['relu3_3'], W=self.get_weights(vgg_layers, 16))
      net['relu3_4'] = self.relu_layer('relu3_4', net['conv3_4'], b=self.get_bias(vgg_layers, 16))
    
      net['pool3']   = self.pool_layer('pool3', net['relu3_4'])
    
      if self.args['verbose']: print('LAYER GROUP 4')
      net['conv4_1'] = self.conv_layer('conv4_1', net['pool3'], W=self.get_weights(vgg_layers, 19))
      net['relu4_1'] = self.relu_layer('relu4_1', net['conv4_1'], b=self.get_bias(vgg_layers, 19))
    
      net['conv4_2'] = self.conv_layer('conv4_2', net['relu4_1'], W=self.get_weights(vgg_layers, 21))
      net['relu4_2'] = self.relu_layer('relu4_2', net['conv4_2'], b=self.get_bias(vgg_layers, 21))
    
      net['conv4_3'] = self.conv_layer('conv4_3', net['relu4_2'], W=self.get_weights(vgg_layers, 23))
      net['relu4_3'] = self.relu_layer('relu4_3', net['conv4_3'], b=self.get_bias(vgg_layers, 23))
    
      net['conv4_4'] = self.conv_layer('conv4_4', net['relu4_3'], W=self.get_weights(vgg_layers, 25))
      net['relu4_4'] = self.relu_layer('relu4_4', net['conv4_4'], b=self.get_bias(vgg_layers, 25))
    
      net['pool4']   = self.pool_layer('pool4', net['relu4_4'])
    
      if self.args['verbose']: print('LAYER GROUP 5')
      net['conv5_1'] = self.conv_layer('conv5_1', net['pool4'], W=self.get_weights(vgg_layers, 28))
      net['relu5_1'] = self.relu_layer('relu5_1', net['conv5_1'], b=self.get_bias(vgg_layers, 28))
    
      net['conv5_2'] = self.conv_layer('conv5_2', net['relu5_1'], W=self.get_weights(vgg_layers, 30))
      net['relu5_2'] = self.relu_layer('relu5_2', net['conv5_2'], b=self.get_bias(vgg_layers, 30))
    
      net['conv5_3'] = self.conv_layer('conv5_3', net['relu5_2'], W=self.get_weights(vgg_layers, 32))
      net['relu5_3'] = self.relu_layer('relu5_3', net['conv5_3'], b=self.get_bias(vgg_layers, 32))
    
      net['conv5_4'] = self.conv_layer('conv5_4', net['relu5_3'], W=self.get_weights(vgg_layers, 34))
      net['relu5_4'] = self.relu_layer('relu5_4', net['conv5_4'], b=self.get_bias(vgg_layers, 34))
    
      net['pool5']   = self.pool_layer('pool5', net['relu5_4'])
    
      return net
    
    def conv_layer(self,layer_name, layer_input, W):
      conv = tf.nn.conv2d(layer_input, W, strides=[1, 1, 1, 1], padding='SAME')
      if self.args['verbose']: print('--{} | shape={} | weights_shape={}'.format(layer_name, 
        conv.get_shape(), W.get_shape()))
      return conv
    
    def relu_layer(self,layer_name, layer_input, b):
      relu = tf.nn.relu(layer_input + b)
      if self.args['verbose']: 
        print('--{} | shape={} | bias_shape={}'.format(layer_name, relu.get_shape(), 
          b.get_shape()))
      return relu
    
    def pool_layer(self,layer_name, layer_input):
      if self.args['pooling_type'] == 'avg':
        pool = tf.nn.avg_pool(layer_input, ksize=[1, 2, 2, 1], 
          strides=[1, 2, 2, 1], padding='SAME')
      elif self.args['pooling_type'] == 'max':
        pool = tf.nn.max_pool(layer_input, ksize=[1, 2, 2, 1], 
          strides=[1, 2, 2, 1], padding='SAME')
      if self.args['verbose']: 
        print('--{}   | shape={}'.format(layer_name, pool.get_shape()))
      return pool
    
    def get_weights(self, vgg_layers, i):
      weights = vgg_layers[i][0][0][2][0][0]
      W = tf.constant(weights)
      return W
    
    def get_bias(self, vgg_layers, i):
      bias = vgg_layers[i][0][0][2][0][1]
      b = tf.constant(np.reshape(bias, (bias.size)))
      return b
    
    '''
      'a neural algorithm for artistic style' loss functions
    '''
    def content_layer_loss(self, p, x):
      _, h, w, d = p.get_shape()
      M = h.value * w.value
      N = d.value
      if self.args['content_loss_function']   == 1:
        K = 1. / (2. * N**0.5 * M**0.5)
      elif self.args['content_loss_function'] == 2:
        K = 1. / (N * M)
      elif self.args['content_loss_function'] == 3:  
        K = 1. / 2.
      loss = K * tf.reduce_sum(tf.pow((x - p), 2))
      return loss
    
    def style_layer_loss(self,a, x):
      _, h, w, d = a.get_shape()
      M = h.value * w.value
      N = d.value
      A = self.gram_matrix(a, M, N)
      G = self.gram_matrix(x, M, N)
      loss = (1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A), 2))
      return loss
    
    def gram_matrix(self, x, area, depth):
      F = tf.reshape(x, (area, depth))
      G = tf.matmul(tf.transpose(F), F)
      return G
    
    def mask_style_layer(self, a, x, mask_img):
      _, h, w, d = a.get_shape()
      mask = self.get_mask_image(mask_img, w.value, h.value)
      mask = tf.convert_to_tensor(mask)
      tensors = []
      for _ in range(d.value): 
        tensors.append(mask)
      mask = tf.stack(tensors, axis=2)
      mask = tf.stack(mask, axis=0)
      mask = tf.expand_dims(mask, 0)
      a = tf.multiply(a, mask)
      x = tf.multiply(x, mask)
      return a, x
    
    def sum_masked_style_losses(self, sess, net, style_imgs):
      total_style_loss = 0.
      weights = self.args['style_imgs_weights']
      masks = self.args['style_mask_imgs']
      for img, img_weight, img_mask in zip(style_imgs, weights, masks):
        sess.run(net['input'].assign(img))
        style_loss = 0.
        for layer, weight in zip(self.args['style_layers'], self.args['style_layer_weights']):
          a = sess.run(net[layer])
          x = net[layer]
          a = tf.convert_to_tensor(a)
          a, x = self.mask_style_layer(a, x, img_mask)
          style_loss += self.style_layer_loss(a, x) * weight
        style_loss /= float(len(self.args['style_layers']))
        total_style_loss += (style_loss * img_weight)
      total_style_loss /= float(len(style_imgs))
      return total_style_loss
    
    def sum_style_losses(self, sess, net, style_imgs):
      total_style_loss = 0.
      weights = self.args['style_imgs_weights']
      for img, img_weight in zip(style_imgs, weights):
        sess.run(net['input'].assign(img))
        style_loss = 0.
        for layer, weight in zip(self.args['style_layers'], self.args['style_layer_weights']):
          a = sess.run(net[layer])
          x = net[layer]
          a = tf.convert_to_tensor(a)
          style_loss += self.style_layer_loss(a, x) * weight
        style_loss /= float(len(self.args['style_layers']))
        total_style_loss += (style_loss * img_weight)
      total_style_loss /= float(len(style_imgs))
      return total_style_loss
    
    def sum_content_losses(self, sess, net, content_img):
      sess.run(net['input'].assign(content_img))
      content_loss = 0.
      for layer, weight in zip(self.args['content_layers'], self.args['content_layer_weights']):
        p = sess.run(net[layer])
        x = net[layer]
        p = tf.convert_to_tensor(p)
        content_loss += self.content_layer_loss(p, x) * weight
      content_loss /= float(len(self.args['content_layers']))
      return content_loss
    
    '''
      'artistic style transfer for videos' loss functions
    '''
    def temporal_loss(self, x, w, c):
      c = c[np.newaxis,:,:,:]
      D = float(x.size)
      loss = (1. / D) * tf.reduce_sum(c * tf.nn.l2_loss(x - w))
      loss = tf.cast(loss, tf.float32)
      return loss
    
    def get_longterm_weights(self, i, j):
      c_sum = 0.
      for k in range(self.args['prev_frame_indices']):
        if i - k > i - j:
          c_sum += self.get_content_weights(i, i - k)
      c = self.get_content_weights(i, i - j)
      c_max = tf.maximum(c - c_sum, 0.)
      return c_max
    
    def sum_longterm_temporal_losses(self, sess, net, frame, input_img):
      x = sess.run(net['input'].assign(input_img))
      loss = 0.
      for j in range(self.args['prev_frame_indices']):
        prev_frame = frame - j
        w = self.get_prev_warped_frame(frame)
        c = self.get_longterm_weights(frame, prev_frame)
        loss += self.temporal_loss(x, w, c)
      return loss
    
    def sum_shortterm_temporal_losses(self, sess, net, frame, input_img):
      x = sess.run(net['input'].assign(input_img))
      prev_frame = frame - 1
      w = self.get_prev_warped_frame(frame)
      c = self.get_content_weights(frame, prev_frame)
      loss = self.temporal_loss(x, w, c)
      return loss
    
    '''
      utilities and i/o
    '''
    def read_image(self, path):
      # bgr image
      img = cv2.imread(path, cv2.IMREAD_COLOR)
      self.check_image(img, path)
      img = img.astype(np.float32)
      img = self.preprocess(img)
      return img
    
    def write_image(self, path, img):
      img = self.postprocess(img)
      cv2.imwrite(path, img)
    
    def preprocess(self, img):
      # bgr to rgb
      img = img[...,::-1]
      # shape (h, w, d) to (1, h, w, d)
      img = img[np.newaxis,:,:,:]
      img -= np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
      return img
    
    def postprocess(self, img):
      img += np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
      # shape (1, h, w, d) to (h, w, d)
      img = img[0]
      img = np.clip(img, 0, 255).astype('uint8')
      # rgb to bgr
      img = img[...,::-1]
      return img
    
    def read_flow_file(self, path):
      with open(path, 'rb') as f:
        # 4 bytes header
        header = struct.unpack('4s', f.read(4))[0]
        # 4 bytes width, height    
        w = struct.unpack('i', f.read(4))[0]
        h = struct.unpack('i', f.read(4))[0]   
        flow = np.ndarray((2, h, w), dtype=np.float32)
        for y in range(h):
          for x in range(w):
            flow[0,y,x] = struct.unpack('f', f.read(4))[0]
            flow[1,y,x] = struct.unpack('f', f.read(4))[0]
      return flow
    
    def read_weights_file(self, path):
      lines = open(path).readlines()
      header = list(map(int, lines[0].split(' ')))
      w = header[0]
      h = header[1]
      vals = np.zeros((h, w), dtype=np.float32)
      for i in range(1, len(lines)):
        line = lines[i].rstrip().split(' ')
        vals[i-1] = np.array(list(map(np.float32, line)))
        vals[i-1] = list(map(lambda x: 0. if x < 255. else 1., vals[i-1]))
      # expand to 3 channels
      weights = np.dstack([vals.astype(np.float32)] * 3)
      return weights
    
    def normalize(self, weights):
      denom = sum(weights)
      if denom > 0.:
        return [float(i) / denom for i in weights]
      else: return [0.] * len(weights)
    
#    def maybe_make_directory(self, dir_path):
#      if not os.path.exists(dir_path):  
#        os.makedirs(dir_path)
    
    def check_image(self, img, path):
      if img is None:
        raise OSError(errno.ENOENT, "No such file", path)
    
    '''
      rendering -- where the magic happens
    '''
    def stylize(self, content_img, style_imgs, init_img, frame=None):
      with tf.device(self.args['device']), tf.Session() as sess:
        # setup network
        net = self.build_model(content_img)
        
        # style loss
        if self.args['style_mask']:
          L_style = self.sum_masked_style_losses(sess, net, style_imgs)
        else:
          L_style = self.sum_style_losses(sess, net, style_imgs)
        
        # content loss
        L_content = self.sum_content_losses(sess, net, content_img)
        
        # denoising loss
        L_tv = tf.image.total_variation(net['input'])
        
        # loss weights
        alpha = self.args['content_weight']
        beta  = self.args['style_weight']
        theta = self.args['tv_weight']
        
        # total loss
        L_total  = alpha * L_content
        L_total += beta  * L_style
        L_total += theta * L_tv
        
        # video temporal loss
        if self.args['video'] and frame > 1:
          gamma      = self.args['temporal_weight']
          L_temporal = self.sum_shortterm_temporal_losses(sess, net, frame, init_img)
          L_total   += gamma * L_temporal
    
        # optimization algorithm
        optimizer = self.get_optimizer(L_total)
    
        if self.args['optimizer'] == 'adam':
          self.minimize_with_adam(sess, net, optimizer, init_img, L_total)
        elif self.args['optimizer'] == 'lbfgs':
          self.minimize_with_lbfgs(sess, net, optimizer, init_img)
        
        output_img = sess.run(net['input'])
        
        if self.args['original_colors']:
          output_img = self.convert_to_original_colors(np.copy(content_img), output_img)
    
        if self.args['video']:
          self.write_video_output(frame, output_img)
        else:
#          self.write_image_output(output_img, content_img, style_imgs, init_img)
          return output_img
    
    def minimize_with_lbfgs(self, sess, net, optimizer, init_img):
      if self.args['verbose']: print('\nMINIMIZING LOSS USING: L-BFGS OPTIMIZER')
      init_op = tf.global_variables_initializer()
      sess.run(init_op)
      sess.run(net['input'].assign(init_img))
      optimizer.minimize(sess)
    
    def minimize_with_adam(self, sess, net, optimizer, init_img, loss):
      if self.args['verbose']: print('\nMINIMIZING LOSS USING: ADAM OPTIMIZER')
      train_op = optimizer.minimize(loss)
      init_op = tf.global_variables_initializer()
      sess.run(init_op)
      sess.run(net['input'].assign(init_img))
      iterations = 0
      while (iterations < self.args['max_iterations']):
        sess.run(train_op)
        if iterations % self.args['print_iterations'] == 0 and self.args['verbose']:
          curr_loss = loss.eval()
          print("At iterate {}\tf=  {:.5E}".format(iterations, curr_loss))
        iterations += 1
    
    def get_optimizer(self, loss):
      print_iterations = self.args['print_iterations'] if self.args['verbose'] else 0
      if self.args['optimizer'] == 'lbfgs':
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
          loss, method='L-BFGS-B',
          options={'maxiter': self.args['max_iterations'],
                      'disp': print_iterations})
      elif self.args['optimizer'] == 'adam':
        optimizer = tf.train.AdamOptimizer(self.args['learning_rate'])
      return optimizer
    
    def write_video_output(self, frame, output_img):
      fn = self.args['content_frame_frmt'].format(str(frame).zfill(4))
      path = os.path.join(self.args['video_output_dir'], fn)
      self.write_image(path, output_img)
    
#    def write_image_output(self, output_img, content_img, style_imgs, init_img):
#      out_dir = os.path.join(self.args['img_output_dir'], self.args['img_name'])
#      self.maybe_make_directory(out_dir)
#      img_path = os.path.join(out_dir, self.args['img_name']+'.png')
#      content_path = os.path.join(out_dir, 'content.png')
#      init_path = os.path.join(out_dir, 'init.png')
#    
#      self.write_image(img_path, output_img)
#      self.write_image(content_path, content_img)
#      self.write_image(init_path, init_img)
#      index = 0
#      for style_img in style_imgs:
#        path = os.path.join(out_dir, 'style_'+str(index)+'.png')
#        self.write_image(path, style_img)
#        index += 1
#      
#      # save the configuration settings
#      out_file = os.path.join(out_dir, 'meta_data.txt')
#      f = open(out_file, 'w')
#      f.write('image_name: {}\n'.format(self.args['img_name']))
#      f.write('content: {}\n'.format(self.args['content_img']))
#      index = 0
#      for style_img, weight in zip(self.args['style_imgs'], self.args['style_imgs_weights']):
#        f.write('styles['+str(index)+']: {} * {}\n'.format(weight, style_img))
#        index += 1
#      index = 0
#      if self.args['style_mask_imgs'] is not None:
#        for mask in self.args['style_mask_imgs']:
#          f.write('style_masks['+str(index)+']: {}\n'.format(mask))
#          index += 1
#      f.write('init_type: {}\n'.format(self.args['init_img_type']))
#      f.write('content_weight: {}\n'.format(self.args['content_weight']))
#      f.write('style_weight: {}\n'.format(self.args['style_weight']))
#      f.write('tv_weight: {}\n'.format(self.args['tv_weight']))
#      f.write('content_layers: {}\n'.format(self.args['content_layers']))
#      f.write('style_layers: {}\n'.format(self.args['style_layers']))
#      f.write('optimizer_type: {}\n'.format(self.args['optimizer']))
#      f.write('max_iterations: {}\n'.format(self.args['max_iterations']))
#      f.write('max_image_size: {}\n'.format(self.args['max_size']))
#      f.close()
    
    '''
      image loading and processing
    '''
    def get_init_image(self, init_type, content_img, style_imgs, frame=None):
      if init_type == 'content':
        return content_img
      elif init_type == 'style':
        return style_imgs[0]
      elif init_type == 'random':
        init_img = self.get_noise_image(self.args['noise_ratio'], content_img)
        return init_img
      # only for video frames
      elif init_type == 'prev':
        init_img = self.get_prev_frame(frame)
        return init_img
      elif init_type == 'prev_warped':
        init_img = self.get_prev_warped_frame(frame)
        return init_img
    
    def get_content_frame(self, frame):
      fn = self.args['content_frame_frmt'].format(str(frame).zfill(4))
      path = os.path.join(self.args['video_input_dir'], fn)
      img = self.read_image(path)
      return img
    
#    def get_content_image(self, content_img):
#      path = os.path.join(self.args['content_img_dir'], content_img)
#       # bgr image
#      img = cv2.imread(path, cv2.IMREAD_COLOR)
#      self.check_image(img, path)
#      img = img.astype(np.float32)
#      h, w, d = img.shape
#      mx = self.args['max_size']
#      # resize if > max size
#      if h > w and h > mx:
#        w = (float(mx) / float(h)) * w
#        img = cv2.resize(img, dsize=(int(w), mx), interpolation=cv2.INTER_AREA)
#      if w > mx:
#        h = (float(mx) / float(w)) * h
#        img = cv2.resize(img, dsize=(mx, int(h)), interpolation=cv2.INTER_AREA)
#      img = self.preprocess(img)
#      return img
      
    def get_content_image(self, img):
      img = img.astype(np.float32)
      h, w, d = img.shape
      mx = self.args['max_size']
      # resize if > max size
      if h > w and h > mx:
        w = (float(mx) / float(h)) * w
        img = cv2.resize(img, dsize=(int(w), mx), interpolation=cv2.INTER_AREA)
      if w > mx:
        h = (float(mx) / float(w)) * h
        img = cv2.resize(img, dsize=(mx, int(h)), interpolation=cv2.INTER_AREA)
      img = self.preprocess(img)
      return img
    
#    def get_style_images(self, content_img):
#      _, ch, cw, cd = content_img.shape
#      style_imgs = []
#      for style_fn in self.args['style_imgs']:
#        path = os.path.join(self.args['style_imgs_dir'], style_fn)
#        # bgr image
#        img = cv2.imread(path, cv2.IMREAD_COLOR)
#        self.check_image(img, path)
#        img = img.astype(np.float32)
#        img = cv2.resize(img, dsize=(cw, ch), interpolation=cv2.INTER_AREA)
#        img = self.preprocess(img)
#        style_imgs.append(img)
#      return style_imgs
      
    def get_style_images(self, content_img):
      _, ch, cw, cd = content_img.shape
      style_imgs = []
      for img in self.args['style_imgs']:
        img = img.astype(np.float32)
        img = cv2.resize(img, dsize=(cw, ch), interpolation=cv2.INTER_AREA)
        img = self.preprocess(img)
        style_imgs.append(img)
      return style_imgs
    
    def get_noise_image(self, noise_ratio, content_img):
      np.random.seed(self.args['seed'])
      noise_img = np.random.uniform(-20., 20., content_img.shape).astype(np.float32)
      img = noise_ratio * noise_img + (1.-noise_ratio) * content_img
      return img
    
    def get_mask_image(self, mask_img, width, height):
      path = os.path.join(self.args['content_img_dir'], mask_img)
      img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
      self.check_image(img, path)
      img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_AREA)
      img = img.astype(np.float32)
      mx = np.amax(img)
      img /= mx
      return img
    
    def get_prev_frame(self, frame):
      # previously stylized frame
      prev_frame = frame - 1
      fn = self.args['content_frame_frmt'].format(str(prev_frame).zfill(4))
      path = os.path.join(self.args['video_output_dir'], fn)
      img = cv2.imread(path, cv2.IMREAD_COLOR)
      self.check_image(img, path)
      return img
    
    def get_prev_warped_frame(self, frame):
      prev_img = self.get_prev_frame(frame)
      prev_frame = frame - 1
      # backwards flow: current frame -> previous frame
      fn = self.args['backward_optical_flow_frmt'].format(str(frame), str(prev_frame))
      path = os.path.join(self.args['video_input_dir'], fn)
      flow = self.read_flow_file(path)
      warped_img = self.warp_image(prev_img, flow).astype(np.float32)
      img = self.preprocess(warped_img)
      return img
    
    def get_content_weights(self, frame, prev_frame):
      forward_fn = self.args['content_weights_frmt'].format(str(prev_frame), str(frame))
      backward_fn = self.args['content_weights_frmt'].format(str(frame), str(prev_frame))
      forward_path = os.path.join(self.args['video_input_dir'], forward_fn)
      backward_path = os.path.join(self.args['video_input_dir'], backward_fn)
      forward_weights = self.read_weights_file(forward_path)
      backward_weights = self.read_weights_file(backward_path)
      return forward_weights #, backward_weights
    
    def warp_image(self, src, flow):
      _, h, w = flow.shape
      flow_map = np.zeros(flow.shape, dtype=np.float32)
      for y in range(h):
        flow_map[1,y,:] = float(y) + flow[1,y,:]
      for x in range(w):
        flow_map[0,:,x] = float(x) + flow[0,:,x]
      # remap pixels to optical flow
      dst = cv2.remap(
        src, flow_map[0], flow_map[1], 
        interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
      return dst
    
    def convert_to_original_colors(self, content_img, stylized_img):
      content_img  = self.postprocess(content_img)
      stylized_img = self.postprocess(stylized_img)
      if self.args['color_convert_type'] == 'yuv':
        cvt_type = cv2.COLOR_BGR2YUV
        inv_cvt_type = cv2.COLOR_YUV2BGR
      elif self.args['color_convert_type'] == 'ycrcb':
        cvt_type = cv2.COLOR_BGR2YCR_CB
        inv_cvt_type = cv2.COLOR_YCR_CB2BGR
      elif self.args['color_convert_type'] == 'luv':
        cvt_type = cv2.COLOR_BGR2LUV
        inv_cvt_type = cv2.COLOR_LUV2BGR
      elif self.args['color_convert_type'] == 'lab':
        cvt_type = cv2.COLOR_BGR2LAB
        inv_cvt_type = cv2.COLOR_LAB2BGR
      content_cvt = cv2.cvtColor(content_img, cvt_type)
      stylized_cvt = cv2.cvtColor(stylized_img, cvt_type)
      c1, _, _ = cv2.split(stylized_cvt)
      _, c2, c3 = cv2.split(content_cvt)
      merged = cv2.merge((c1, c2, c3))
      dst = cv2.cvtColor(merged, inv_cvt_type).astype(np.float32)
      dst = self.preprocess(dst)
      return dst
    
    def render_single_image(self):
      content_img = self.get_content_image(self.args['content_img'])
      style_imgs = self.get_style_images(content_img)
      with tf.Graph().as_default():
        print('\n---- RENDERING SINGLE IMAGE ----\n')
        init_img = self.get_init_image(self.args['init_img_type'], content_img, style_imgs)
        stylized_img = self.stylize(content_img, style_imgs, init_img)
        stylized_img = self.postprocess(stylized_img)
        return stylized_img
    
    def render_video(self):
      for frame in range(self.args['start_frame'], self.args['end_frame']+1):
        with tf.Graph().as_default():
          print('\n---- RENDERING VIDEO FRAME: {}/{} ----\n'.format(frame, self.args['end_frame']))
          if frame == 1:
            content_frame = self.get_content_frame(frame)
            style_imgs = self.get_style_images(content_frame)
            init_img = self.get_init_image(self.args['first_frame_type'], content_frame, style_imgs, frame)
            self.args['max_iterations'] = self.args['first_frame_iterations']
            tick = time.time()
            self.stylize(content_frame, style_imgs, init_img, frame)
            tock = time.time()
            print('Frame {} elapsed time: {}'.format(frame, tock - tick))
          else:
            content_frame = self.get_content_frame(frame)
            style_imgs = self.get_style_images(content_frame)
            init_img = self.get_init_image(self.args['init_frame_type'], content_frame, style_imgs, frame)
            self.args['max_iterations'] = self.args['frame_iterations']
            tick = time.time()
            self.stylize(content_frame, style_imgs, init_img, frame)
            tock = time.time()
            print('Frame {} elapsed time: {}'.format(frame, tock - tick))