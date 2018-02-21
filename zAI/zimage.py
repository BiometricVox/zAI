# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
import cv2
from matplotlib import pyplot as plt
import base64
import json
try:
  from http.client import HTTPSConnection
except ImportError:
  from httplib import HTTPSConnection
from six.moves.urllib.parse import urlencode
import os

from ._config import zAI_BACKEND, GOOGLE_CLOUD_API_KEY, MICROSOFT_AZURE_URL, MICROSOFT_AZURE_VISION_API_KEY, MICROSOFT_AZURE_FACE_API_KEY

class zImage:
    ''' Class to work with image data. 
    
    Parameters:
    -----------
        data: image path, URL or a 3-dimensional numpy array
    
    Atributes:
    ----------
        im: image data as 3-dimensional ndarray. Shape is Height * Width * 3. Channels use BGR (Open-cv) convention
        
        data: image binary data
        
        faces: encodes information (if available) about the faces detected on the image. It is initialized as an empty list.
            When faces are detected, 'faces' is a list with length equal to the number of faces successfully detected.
            Each element on the list is a dictionary with the following information about the corresponding detected face:
                - rectangle: coordinates of the rectangle enclosing the face region. Dictionary with the keys:
                    * BOTTOM_RIGHT: list with the x - y coordinates of the bottom right corner
                    * TOP_LEFT: list with the x - y coordinates of the top left corner
                - landmarks: a dictionary with the location of different facial landmarks (the specific landmarks vary with backend selection).
                    Each key in the dictionary represents a facial landmark and contains a list with the x-y coordinates of that particular landmark
    '''        
    def __init__(self,data):
        if type(data) is str:
            self.im = cv2.imread(data)
            with open(data, 'rb') as image_file:
                    self.data = image_file.read()
#            if os.path.isfile(data):
#                self.im = cv2.imread(data)
#                with open(data, 'rb') as image_file:
#                    self.data = image_file.read()
#            else:
#                # TO-DO: No parece que este try - except esté haciendo nada
#                try:
#                    url = urllib.urlopen(data)
#                    self.data = url.read()
#                    url.close()
#                    self.im = cv2.imdecode(np.fromstring(self.data, np.uint8), cv2.CV_LOAD_IMAGE_COLOR)
#                    if self.im is None:
#                        raise Exception('URL not found.')
#                except:
#                    raise Exception('URL not found.')
            
        elif isinstance(data,np.ndarray) and len(data.shape)==3:
            self.im = data
            self.data = cv2.imencode('.png',self.im)[1].tostring()
        else:
            raise Exception('Wrong input. Expected a path direction or a 3-dim numpy array.')
        self.faces = []
        
    # @classmethod
    # def fromURL(cls, url):
        # url = urllib.urlopen(url)
        # data = url.read()
        # url.close()
        # im = cv2.imdecode(np.fromstring(data, np.uint8), cv2.CV_LOAD_IMAGE_COLOR)
        # if im is None:
            # raise Exception('URL not found.')
        # return cls(im,data)
        
    def display(self):
        '''
        Display image
        '''
        # Convert to RGB space
        cv_rgb = cv2.cvtColor(self.im, cv2.COLOR_BGR2RGB)
        
        # Plot with matplotlib
        plt.imshow(cv_rgb)
        plt.show()
        
    def display_face_detection_results(self,destPath=None):
        '''
        Display the information about faces present on the image. Faces are marked with a white rectangle and landmarks with red dots.
        
        Parameters:
        -----------
        destPath: destination path for resulting image. If destPath = None (default value), image is displayed on screen.
            If it is desired path and file name, image is saved to that location
        '''
        # TO-DO: make marker and rectangle line width dependent on image size
        if len(self.faces):
            draw = np.copy(self.im)
            for face in self.faces:
                
                # Draw rectangle around face
                cv2.rectangle(draw, (face['rectangle']['TOP_LEFT'][0], face['rectangle']['TOP_LEFT'][1]), (face['rectangle']['BOTTOM_RIGHT'][0], face['rectangle']['BOTTOM_RIGHT'][1]), (255, 255, 255))
                
                # Draw a red circle for each facial landmark
                for key in face['landmarks']:
                    cv2.circle(draw, (int(face['landmarks'][key][0]), int(face['landmarks'][key][1])), 1, (0, 0, 255), 2)
                    
            # Show on screen or save to desired destination
            if destPath is not None:
                cv2.imwrite(destPath,draw)
            else:
                cv_rgb = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
                plt.imshow(cv_rgb)
                plt.show()
        else:
            raise Exception('No face information present')
        
    def save(self,destPath):
        '''
        Save image to disk
        
        Parameters:
        -----------
        destPath: desired path
        '''
        cv2.imwrite(destPath,self.im)
            
    def __request_google_vision__(self,action='LABEL_DETECTION',n=1):
        '''
        Request Google Cloud Vision REST API to perform provided action
        '''
        # Format header and data for the request
        b64 = base64.b64encode(self.data).decode('UTF-8')
        content_json_obj = {'content': b64}
        feature_json_obj = {'type': action,'maxResults': n}
    
        request = {'features': feature_json_obj,'image': content_json_obj}
        data = json.dumps({'requests': request})

        # Perform request and extract response
        try:
            conn = HTTPSConnection('vision.googleapis.com')
            conn.request('POST','/v1/images:annotate?key=%s' % GOOGLE_CLOUD_API_KEY, data, {'Content-Type': 'application/json'})
            response = conn.getresponse()
            
            if response.status == 200:
                data = response.read()
                data = json.loads(data)
                conn.close()
                return data
            else:
                raise Exception('Request failed')
        except Exception as e:
            print('Error:')
            print(e)
            
    def __request_azure__(self,endpoint,headers,params):
        '''
        Request Microsoft Azure
        '''
        
        # Perform request and extract response
        try:
            conn = HTTPSConnection(MICROSOFT_AZURE_URL)
            conn.request("POST", endpoint+"?%s" % params, self.data, headers)
            response = conn.getresponse()
            
            if response.status == 200:
                data = response.read()
                data = json.loads(data)
                conn.close()
                return data
            else:
                raise Exception('Request failed')
        
        except Exception as e:
            print('Error:')
            print(e)
        
    def label(self,n=1,backend=zAI_BACKEND):
        '''
        Label the content of the image
        
        Parameters:
        -----------
        n: maximum number of labels to generate
        
        backend: desired backend (Google | Microsot | local)
        
        Returns:
        --------
        labels: list containing one label per entry
        '''
        
        # Check that the keys for the selected backend are available
        if backend == 'Google':
            if GOOGLE_CLOUD_API_KEY == '':
                print('GOOGLE_CLOUD_API_KEY is empty. Using local.')
                backend = 'local'
        elif backend == 'Microsoft':
            if MICROSOFT_AZURE_URL == '':
                print('MICROSOFT_AZURE_URL is empty. Using local.')
                backend = 'local'
            elif MICROSOFT_AZURE_VISION_API_KEY == '':
                print('MICROSOFT_AZURE_VISION_API_KEY is empty. Using local.')
                backend = 'local'
        
        if backend == 'Google':
            
            # Request Google Vision and extract results
            data = self.__request_google_vision__('LABEL_DETECTION',n)
            responses = data['responses'][0]
            labelAnnotations = responses['labelAnnotations']
    
            # Generate list with predicted labels
            labels = []
            for lbl in range(len(labelAnnotations)):
                labels.append(str(labelAnnotations[lbl]['description']))
                
        elif backend == 'Microsoft':
            
            # Set up request to Azure
            headers = {
                'Content-Type': 'application/octet-stream',
                'Ocp-Apim-Subscription-Key': MICROSOFT_AZURE_VISION_API_KEY,
            }
            params = urlencode({
                'visualFeatures': "Tags",
                'language': 'en',
            })
            
            # Perform request and extract information
            data = self.__request_azure__("/vision/v1.0/analyze",headers,params)
            tags = data['tags']
            
            # Generate list with predicted labels
            labels = []
            for lbl in range(len(tags)):
                labels.append(str(tags[lbl]['name']))
                
        elif backend == 'local':
            raise NotImplementedError("label method is currently not available with local backend")
        else:
            raise Exception('invalid backend selection. Valid values are currently "Google", "Microsoft" or "local".')
            
        return labels
        
    def ocr(self,backend=zAI_BACKEND):
        '''
        Perform Optical Character Recognition extracting text from image
        
        Parameters:
        -----------       
        backend: desired backend (Google | Microsot | local)
        
        Returns:
        --------
        text: zText object with detected text and language
        '''
        from zAI import zText
        
        # Check that the keys for the selected backend are available
        if backend == 'Google':
            if GOOGLE_CLOUD_API_KEY == '':
                print('GOOGLE_CLOUD_API_KEY is empty. Using local.')
                backend = 'local'
        elif backend == 'Microsoft':
            if MICROSOFT_AZURE_URL == '':
                print('MICROSOFT_AZURE_URL is empty. Using local.')
                backend = 'local'
            elif MICROSOFT_AZURE_VISION_API_KEY == '':
                print('MICROSOFT_AZURE_VISION_API_KEY is empty. Using local.')
                backend = 'local'
                
        if backend == 'Google':
            
            # Perform request and extact text and language information
            data = self.__request_google_vision__('TEXT_DETECTION')
            responses = data['responses'][0]
            fullTextAnnotation = responses['fullTextAnnotation']
            text = fullTextAnnotation['text']
            metadata = fullTextAnnotation['pages'][0]
            lang = metadata['property']['detectedLanguages'][0]['languageCode']

        elif backend == 'Microsoft':
            
            # Set up request
            headers = {
                'Content-Type': 'application/octet-stream',
                'Ocp-Apim-Subscription-Key': MICROSOFT_AZURE_VISION_API_KEY,
            }
            params = urlencode({
                'language': 'unk',
                'detectOrientation': 'true',
            })
            
            # Perform request and extact text and language information
            data = self.__request_azure__("/vision/v1.0/ocr",headers,params)
            lang = data['language']
            text=''
            regions = data['regions']
            for region in regions:
                for line in region['lines']:
                    for word in line['words']:
                        text+=' ' + word['text']

        elif backend == 'local':
            raise NotImplementedError('ocr method is currently not available with local backend')
        else:
            raise Exception('invalid backend selection. Valid values are currently "Google", "Microsoft" or "local".')
            
        # Postprocess text and generate resulting zText
        text = text.replace('\n',' ') #Remove line breaks
        text = zText(text,lang)
        return text
        
    def find_faces(self,backend=zAI_BACKEND, max_faces=20):
        '''
        Locate human faces on the image.
        The results are stored on the atribute 'faces'. See details above.
        
        Parameters:
        -----------
         backend: desired backend (Google | Microsot | local)
         
         max_faces: maximum number of faces to return. Detected faces are sorted by confidence.
        '''
        
        # Check that the keys for the selected backend are available
        if backend == 'Google':
            if GOOGLE_CLOUD_API_KEY == '':
                print('GOOGLE_CLOUD_API_KEY is empty. Using local.')
                backend = 'local'
        elif backend == 'Microsoft':
            if MICROSOFT_AZURE_URL == '':
                print('MICROSOFT_AZURE_URL is empty. Using local.')
                backend = 'local'
            elif MICROSOFT_AZURE_FACE_API_KEY == '':
                print('MICROSOFT_AZURE_FACE_API_KEY is empty. Using local.')
                backend = 'local'
        
        
        if backend == 'Google':
            
            # Perform request and extract results
            data = self.__request_google_vision__('FACE_DETECTION',max_faces)
            data = data['responses'][0]['faceAnnotations']
            
            # Store information for each detected face
            num_faces = len(data)
            self.faces = []
            for fc in range(num_faces):
                face_data = data[fc]
                confidence = face_data['detectionConfidence']
                vertices = face_data['fdBoundingPoly']['vertices']
                rectangle = {'TOP_LEFT': [vertices[0]['x'],vertices[0]['y']], 'BOTTOM_RIGHT': [vertices[2]['x'],vertices[2]['y']]}
                emotions = {'anger': str(face_data['angerLikelihood']), 'joy': str(face_data['joyLikelihood']), 'sorrow': str(face_data['sorrowLikelihood']), 'surprise': str(face_data['surpriseLikelihood'])}
                points = face_data['landmarks']
                landmarks = {}
                for point in range(len(points)):
                    landmarks[points[point]['type']] = [points[point]['position']['x'],points[point]['position']['y']]
                
                self.faces.append({'confidence': confidence, 'rectangle': rectangle, 'landmarks': landmarks, 'emotions': emotions})
                
        elif backend == 'Microsoft':
            
            # Set up request
            headers = {
                'Content-Type': 'application/octet-stream',
                'Ocp-Apim-Subscription-Key': MICROSOFT_AZURE_FACE_API_KEY,
            }
            params = urlencode({
                'returnFaceId': 'true',
                'returnFaceLandmarks': 'true',
                'returnFaceAttributes': 'age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure,noise',
            })
            
            data = self.__request_azure__("/face/v1.0/detect",headers,params)
            
            # Store information for each detected face
            num_faces = len(data)
            self.faces = []
            for fc in range(num_faces):
                top_left = [data[fc]['faceRectangle']['left'],data[fc]['faceRectangle']['top']]
                bottom_right = [data[fc]['faceRectangle']['left']+data[fc]['faceRectangle']['width'],
                                data[fc]['faceRectangle']['top']+data[fc]['faceRectangle']['height']]
                rectangle = {'TOP_LEFT': top_left, 'BOTTOM_RIGHT': bottom_right}
                points = data[fc]['faceLandmarks']
                landmarks = {}
                for key, value in points.items():
                    landmarks[key] = [value['x'],value['y']]
            
                self.faces.append({'rectangle': rectangle, 'landmarks': landmarks})
                
        elif backend == 'local':
            '''
            This method uses the code from https://github.com/davidsandberg/facenet
            See copyright notice in zAI/models/face_detection
            '''
            
            from zAI.models.face_detection import detect_face
            import tensorflow as tf
   
            # Set up Tensorflow session and create neural network
            with tf.Graph().as_default():
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
                sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
                with sess.as_default():
                    pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
            
            minsize = 20 # minimum size of face
            threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
            factor = 0.709 # scale factor
            margin = 0
            
            # Convert image to desired format
            img = cv2.cvtColor(self.im, cv2.COLOR_BGR2RGB)
            img = img[:,:,0:3]
            
            # Perform face detection
            bounding_boxes, points, scores = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            
            # Store information for each detected face
            nrof_faces = bounding_boxes.shape[0]
            if nrof_faces>0:
              det = bounding_boxes[:,0:4]
              img_size = np.asarray(img.shape)[0:2]
              self.faces = []
              for i in range(nrof_faces):
                ddet = det[i,:]
                ddet = np.squeeze(ddet)
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(ddet[0]-margin/2, 0)
                bb[1] = np.maximum(ddet[1]-margin/2, 0)
                bb[2] = np.minimum(ddet[2]+margin/2, img_size[1])
                bb[3] = np.minimum(ddet[3]+margin/2, img_size[0])
                                
                rectangle = {'TOP_LEFT': [int(bb[0]), int(bb[1])], 'BOTTOM_RIGHT': [int(bb[2]), int(bb[3])]}
                face_points = points.T[i]                    
                landmarks = {'RIGHT_EYE': [face_points[0], face_points[5]], 'LEFT_EYE': [face_points[1], face_points[6]],
                             'NOSE_TIP': [face_points[2], face_points[7]], 'MOUTH_RIGHT': [face_points[3], face_points[8]],
                             'MOUTH_LEFT': [face_points[4], face_points[9]]}
                    
                self.faces.append({'rectangle': rectangle, 'landmarks': landmarks})
        else:
            raise Exception('Invalid backend selection. Valid values are currently "Google", "Microsoft" or "local".')
            
    def extract_face(self, n = 0, margin = 5):
        '''
        Extract a detected face to a zFace object
        
        Parameters:
        -----------
        n: Extract nth detected face
        
        margin: Number of pixels to expand face rectangle before extraction. Allows the extraction of a bigger region than the tight face rectangle
        
        Returns:
        --------
        outimg: zface object containing the required face
        '''
    
        # TO-DO: Let margin be a relative rather than absolute value
        num_faces = len(self.faces)
        if num_faces < n+1:
            raise Exception('Requested %dth face (0-indexing), but only %d face/s were detected' % (n+1,num_faces))
        else:
            from zAI import zFace
            top = self.faces[n]['rectangle']['TOP_LEFT']
            bottom = self.faces[n]['rectangle']['BOTTOM_RIGHT']
            im = self.im[np.maximum(top[1]-margin,0):np.minimum(bottom[1]+margin,self.im.shape[0]),np.maximum(top[0]-margin,0):np.minimum(bottom[0]+margin,self.im.shape[1])]
            
            outimg = zFace(im)
            rectangle = {'TOP_LEFT': [np.minimum(margin,top[0]), np.minimum(top[1],margin)],
                          'BOTTOM_RIGHT': [bottom[0]+np.minimum(margin,top[0])-top[0], bottom[1]+np.minimum(top[1],margin)-top[1]]}
            landmarks = dict()
            for key, value in self.faces[n]['landmarks'].items():
                   landmarks[key] = [value[0]+np.minimum(margin,top[0])-top[0],value[1]+np.minimum(top[1],margin)-top[1]]
                            
            outimg.faces.append({'rectangle': rectangle, 'landmarks': landmarks})
            return outimg

    def style(self, style_im, iterations=100, style_weight=1e3, preserve_color = False):
        '''
        Combines source image with the style of another image/s.
        The resulting image will have the content of source image and the style of style image/s
        
        Parameters:
        -----------
        style_im: zImage or list of zImages. Style image/s to combine with source (content) image.
        
        style_weight: weight of style image. The bigger the value, the more weight the style will have in the resulting combination
        
        preserve_color: logical. If True, resulting image will maintain the colors of the original image.
            If false, style will affect the resulting colors.
            
        Returns:
        --------
        out_im: zImage containing combined result
        
        This method uses the code from https://github.com/cysmith/neural-style-tf
        '''
        from zAI.models.art import neural_style
        from zAI.utils import downloads
        
        model_file = downloads.maybe_download('imagenet-vgg-verydeep-19.mat',os.path.join(os.path.dirname(__file__),'models/art/models'),
                            'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat')
                
        
        # TO-DO: What if it is a zFace?
        if isinstance(style_im,zImage):
            style_imgs = [style_im.im]
            style_imgs_weights = [1.0]
        elif isinstance(style_im,list):
            style_imgs = []
            for si in range(len(style_im)):
                if isinstance(style_im[si],zImage):
                    style_imgs.append(style_im[si].im)
                else:
                    raise Exception('One or more of the provided style images were not a valid zImage')
            style_imgs_weights = [1.0/len(style_im)]*len(style_im)
        else:
            raise Exception('style images must be either a zImage or a list of zImages')
        
        # Set up algorithm options
        args = {'verbose': False,
        'img_name': 'result',
        'style_imgs': style_imgs,
        'style_imgs_weights': style_imgs_weights,
        'content_img': self.im,
        'init_img_type': 'content',
        'max_size': 512,
        'content_weight': 5e0,
        'style_weight': style_weight,
        'tv_weight': 1e-3,
        'temporal_weight': 2e2,
        'content_loss_function': 1,
        'content_layers': ['conv4_2'],
        'style_layers': ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'],
        'content_layer_weights': [1.0],
        'style_layer_weights': [0.2, 0.2, 0.2, 0.2, 0.2],
        'original_colors': preserve_color,
        'color_convert_type': 'yuv',
        'color_convert_time': 'after',
        'style_mask': False,
        'style_mask_imgs': None,
        'noise_ratio': 1.0,
        'seed': 0,
        'model_weights': model_file,
        'pooling_type': 'avg',
        'device': '/cpu:0',
        'optimizer': 'lbfgs',
        'learning_rate': 1e0,
        'max_iterations': iterations,
        'print_iterations': 50,
        'video': False,
        'start_frame': 1,
        'end_frame': 1,
        'first_frame_type': 'content',
        'init_frame_type': 'prev_warped',
        'video_input_dir': './video_input',
        'video_output_dir': './video_output',
        'content_frame_frmt': 'frame_{}.ppm',
        'backward_optical_flow_frmt': 'backward_{}_{}.flo',
        'forward_optical_flow_frmt': 'forward_{}_{}.flo',
        'content_weights_frmt': 'reliable_{}_{}.txt',
        'prev_frame_indices': [1],
        'first_frame_iterations': 2000,
        'frame_iterations': 800}
 
        # Style image
        styler = neural_style.NeuralStyle(args)
  
        out_im = styler.render_single_image()
        
        return zImage(out_im)
    
    @staticmethod
    def set_backend_key(key_name, new_key):
        from zAI.utils import keys
        available_keys = keys.get_available_keys()
        if key_name in available_keys:
            exec('%s = "%s"' %(key_name, new_key), globals())
        else:
            raise Exception('%s is not available.' % key_name)
