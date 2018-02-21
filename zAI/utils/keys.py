# -*- coding: utf-8 -*-
import os
import json

        
def get_available_keys():
    _zAI_base_dir = os.path.expanduser('~')
    _zAI_dir = os.path.join(_zAI_base_dir,'.zAI')
    config_file = os.path.join(_zAI_dir, "config.json")

    if not os.path.exists(config_file):
        raise Exception('%s does not exist' % config_file)
        
    with open(config_file, 'r') as jsonFile:
        data = json.load(jsonFile)
    del data['zAI_BACKEND']
    return data.keys()

def display_available_keys():
    available_keys = get_available_keys()
    for x in available_keys:
        print('%s' %x)

def save_backend_key(key_name, key_value):
    _zAI_base_dir = os.path.expanduser('~')
    _zAI_dir = os.path.join(_zAI_base_dir,'.zAI')
    config_file = os.path.join(_zAI_dir, "config.json")

    if not os.path.exists(config_file):
        raise Exception('%s does not exist' % config_file)
        
    if key_name not in get_available_keys():
        raise Exception('%s is not currently available in zAI.' % key_name)
        
    with open(config_file, 'r') as jsonFile:
        data = json.load(jsonFile)
        
    data[key_name] = key_value
    
    with open(config_file, 'w') as jsonFile:
        json.dump(data, jsonFile, indent=2)

def set_backend_key(key_name, key_value, save=False):
    from zAI import zImage, zText
    zImage.set_backend_key(key_name, key_value)
    zText.set_backend_key(key_name, key_value)
    if save:
        save_backend_key(key_name, key_value)