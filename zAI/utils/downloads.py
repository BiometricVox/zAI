# -*- coding: utf-8 -*-
import os
from six.moves.urllib.request import urlretrieve

def maybe_download(filename, work_directory, source_url):
  """Download the data from source url, unless it's already here.
  Args:
      filename: string, name of the file in the directory.
      work_directory: string, path to working directory.
      source_url: url to download from if file doesn't exist.
  Returns:
      Path to resulting file.
  """
  
  if not os.path.isdir(work_directory):
      os.mkdir(work_directory)
  
  filepath = os.path.join(work_directory, filename)
  
  if not os.path.isfile(filepath):
      print('Downloading file. Please wait...')
      temp_file_name, _ = urlretrieve(source_url, filepath)
      os.rename(temp_file_name, filepath)

  return filepath

def maybe_download_and_unzip(filename, work_directory, destination_directory, source_url):
    
    if not os.path.isdir(os.path.join(work_directory,destination_directory)):
        filepath = maybe_download(filename, work_directory, source_url)
        
        import zipfile
        zip_file = zipfile.ZipFile(filepath, 'r')
        zip_file.extractall(work_directory)
        zip_file.close()
        
        os.remove(filepath)
