# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
import cv2
import os

from zAI import zImage

class zFace(zImage):
    ''' Class to work with face images, usually created by using the 'extract_face' method of a zImage. 
    
    Parameters:
    -----------
        data: image path, URL or a 3-dimensional numpy array
    
    Atributes:
    ----------
        im: image data as 3-dimensional ndarray. Shape is Height * Width * 3. Channels use BGR (Open-cv) convention
        
        data: image binary data
        
        faces: encodes information (if available) about the face detected on the image. If the zFace is created from scratch, it is initialized as an empty list.
            When the zFace is created by the 'extract_face' method, it is a list with a single element which contains a dictionary
            with the following information about the corresponding detected face:
                - rectangle: coordinates of the rectangle enclosing the face region. Dictionary with the keys:
                    * BOTTOM_RIGHT: list with the x - y coordinates of the bottom right corner
                    * TOP_LEFT: list with the x - y coordinates of the top left corner
                - landmarks: a dictionary with the location of different facial landmarks (the specific landmarks vary with backend selection).
                    Each key in the dictionary represents a facial landmark and contains a list with the x-y coordinates of that particular landmark
    '''        
    
    def __init__(self,data):
        zImage.__init__(self,data)
        
    def compare(self,ref_face):
        '''
        Compares two face images to find if the belong to the same person, computing a distance score.
        The lower this distance, the most likely the two images belong to the same person
        
        Parameters:
        -----------
        ref_face: a zFace object to compare against
        
        Returns:
        --------
        dist: distance between both face embeddings. The lower this distance, the most likely the two images belong to the same person.
        
        This method uses the code from https://github.com/davidsandberg/facenet
        See copyright notice in zAI/models/facenet
        '''
            
        #TO-DO: check that ref_face is a valid zFace
        #TO-DO: check what happens with grayscale images
        import tensorflow as tf
        from .models.facenet import facenet
        from zAI.utils import downloads

        target_folder = 'facenet'
        model_dir = downloads.maybe_download_and_unzip('20170512-110547.zip',target_folder,'20170512-110547','https://www.dropbox.com/s/h8nxpvjpon12g9c/20170512-110547.zip?dl=1')

        image_size = 160
        
        im1 = cv2.cvtColor(self.im, cv2.COLOR_BGR2RGB)
        im2 = cv2.cvtColor(ref_face.im, cv2.COLOR_BGR2RGB)
        
        im1 = cv2.resize(im1, (image_size, image_size)) 
        im1 = facenet.prewhiten(im1)
        im2 = cv2.resize(im2, (image_size, image_size)) 
        im2 = facenet.prewhiten(im2)
        
        images = np.stack((im1,im2))
        
        with tf.Graph().as_default():

            with tf.Session() as sess:
              
                # Load the model
                facenet.load_model(model_dir+'/20170512-110547/20170512-110547.pb')
                
                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        
                # Run forward pass to calculate embeddings
                feed_dict = { images_placeholder: images, phase_train_placeholder:False }
                emb = sess.run(embeddings, feed_dict=feed_dict)
                
                dist = np.sqrt(np.sum(np.square(np.subtract(emb[0,:], emb[1,:]))))
                
        return dist
    
