# -*- coding: utf-8 -*-
import os
from six.moves.urllib.request import urlretrieve

def get_zai_dir():
    '''
    Work out where to store configuration and model files locally
    '''
    _zAI_base_dir = os.path.expanduser('~')

    if not os.access(_zAI_base_dir, os.W_OK):
        _zAI_base_dir = '/temp'

    _zAI_dir = os.path.join(_zAI_base_dir,'.zAI')
    
    return _zAI_dir

def maybe_download(filename, target_folder, source_url):
  """Download the data from source url, unless it's already here.
  Args:
      filename: string, name of the file in the directory.
      target_folder: string, path to output folder relative to _zAI_dir folder.
      source_url: url to download from if file doesn't exist.
  Returns:
      Path to resulting file.
  """
     
  _zAI_dir = get_zai_dir()
  
  model_dir = os.path.join(_zAI_dir,target_folder)
  filepath = os.path.join(_zAI_dir,target_folder, filename)
  
  if not os.path.isfile(filepath):
      print('Downloading file. Please wait...')
      if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
      temp_file_name, _ = urlretrieve(source_url, filepath)
      os.rename(temp_file_name, filepath)

  return model_dir

def maybe_download_and_unzip(filename, target_folder, destination_directory, source_url):
    
    _zAI_dir = get_zai_dir()
    
    if not os.path.isdir(os.path.join(_zAI_dir,target_folder,destination_directory)):
        model_dir = maybe_download(filename, target_folder, source_url)
        filepath = os.path.join(model_dir, filename)
        
        import zipfile
        zip_file = zipfile.ZipFile(filepath, 'r')
        zip_file.extractall(model_dir)
        zip_file.close()
        
        os.remove(filepath)
    else:
        model_dir = os.path.join(_zAI_dir,target_folder)
        
    return model_dir
