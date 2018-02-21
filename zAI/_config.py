from __future__ import absolute_import

import os, json

_zAI_base_dir = os.path.expanduser('~')

if not os.access(_zAI_base_dir, os.W_OK):
    _zAI_base_dir = '/temp'
    
_zAI_dir = os.path.join(_zAI_base_dir,'.zAI')
if not os.path.exists(_zAI_dir):
    try:
        os.makedirs(_zAI_dir)
    except OSError:
        pass

config_file = os.path.join(_zAI_dir, "config.json")

if os.path.isfile(config_file):
    with open(config_file) as json_config_file:
        config = json.load(json_config_file)

    if 'zAI_BACKEND' in config:
        zAI_BACKEND = config['zAI_BACKEND']
        if zAI_BACKEND not in ['Google','Microsoft','local','']:
            raise Exception('Invalid backend selection. Valid values are currently "Google", "Microsoft" or "local".')
        if zAI_BACKEND is '':
            print('Assuming zAI_BACKEND as local.')
            zAI_BACKEND = 'local'
    else:
        print('Assuming zAI_BACKEND as local.')
        zAI_BACKEND = 'local'
    
    if 'GOOGLE_CLOUD_API_KEY' in config:
        GOOGLE_CLOUD_API_KEY = config['GOOGLE_CLOUD_API_KEY']
    else:
        GOOGLE_CLOUD_API_KEY = ''
        config['GOOGLE_CLOUD_API_KEY'] = ''
        
    if 'MICROSOFT_AZURE_VISION_API_KEY' in config:
       MICROSOFT_AZURE_VISION_API_KEY = config['MICROSOFT_AZURE_VISION_API_KEY']
    else:
        MICROSOFT_AZURE_VISION_API_KEY = ''
        config['MICROSOFT_AZURE_VISION_API_KEY'] = ''
        
    if 'MICROSOFT_AZURE_FACE_API_KEY' in config:
        MICROSOFT_AZURE_FACE_API_KEY = config['MICROSOFT_AZURE_FACE_API_KEY']
    else:
        MICROSOFT_AZURE_FACE_API_KEY = ''
        config['MICROSOFT_AZURE_FACE_API_KEY'] = ''
        
    if 'MICROSOFT_AZURE_URL' in config:
        MICROSOFT_AZURE_URL = config['MICROSOFT_AZURE_URL']
    else:
        MICROSOFT_AZURE_URL = ''
        config['MICROSOFT_AZURE_URL'] = ''
        
    if 'MICROSOFT_AZURE_TEXT_TRANSLATION_API_KEY' in config:
        MICROSOFT_AZURE_TEXT_TRANSLATION_API_KEY = config['MICROSOFT_AZURE_TEXT_TRANSLATION_API_KEY']
    else:
        MICROSOFT_AZURE_TEXT_TRANSLATION_API_KEY = ''
        config['MICROSOFT_AZURE_TEXT_TRANSLATION_API_KEY'] = ''
        
    if 'MICROSOFT_AZURE_BING_VOICE_API_KEY' in config:
        MICROSOFT_AZURE_BING_VOICE_API_KEY = config['MICROSOFT_AZURE_BING_VOICE_API_KEY']
    else:
        MICROSOFT_AZURE_BING_VOICE_API_KEY = ''
        config['MICROSOFT_AZURE_BING_VOICE_API_KEY'] = ''
else:
    config = {}
    zAI_BACKEND = 'local'
    config['zAI_BACKEND'] ='local'
    GOOGLE_CLOUD_API_KEY = ''
    config['GOOGLE_CLOUD_API_KEY'] = ''
    MICROSOFT_AZURE_VISION_API_KEY = ''
    config['MICROSOFT_AZURE_VISION_API_KEY'] = ''
    MICROSOFT_AZURE_FACE_API_KEY = ''
    config['MICROSOFT_AZURE_FACE_API_KEY'] = ''
    MICROSOFT_AZURE_URL = ''
    config['MICROSOFT_AZURE_URL'] = ''
    MICROSOFT_AZURE_TEXT_TRANSLATION_API_KEY = ''
    config['MICROSOFT_AZURE_TEXT_TRANSLATION_API_KEY'] = ''
    MICROSOFT_AZURE_BING_VOICE_API_KEY = ''
    config['MICROSOFT_AZURE_BING_VOICE_API_KEY'] = ''

with open(config_file,'w') as json_config_file:
    config = json.dump(config, json_config_file,indent=2)