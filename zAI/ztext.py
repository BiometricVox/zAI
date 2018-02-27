# -*- coding: utf-8 -*-
from __future__ import absolute_import

import json
try:
  from http.client import HTTPSConnection
except ImportError:
  from httplib import HTTPSConnection
from six.moves.urllib.parse import urlencode
from xml.etree import ElementTree


from ._config import zAI_BACKEND, GOOGLE_CLOUD_API_KEY, MICROSOFT_AZURE_TEXT_TRANSLATION_API_KEY, MICROSOFT_AZURE_BING_VOICE_API_KEY

ISO639_1_CODES = {
	'ab': 'Abkhaz',
	'aa': 'Afar',
	'af': 'Afrikaans',
	'ak': 'Akan',
	'sq': 'Albanian',
	'am': 'Amharic',
	'ar': 'Arabic',
	'an': 'Aragonese',
	'hy': 'Armenian',
	'as': 'Assamese',
	'av': 'Avaric',
	'ae': 'Avestan',
	'ay': 'Aymara',
	'az': 'Azerbaijani',
	'bm': 'Bambara',
	'ba': 'Bashkir',
	'eu': 'Basque',
	'be': 'Belarusian',
	'bn': 'Bengali',
	'bh': 'Bihari',
	'bi': 'Bislama',
	'bs': 'Bosnian',
	'br': 'Breton',
	'bg': 'Bulgarian',
	'my': 'Burmese',
	'ca': 'Catalan; Valencian',
	'ch': 'Chamorro',
	'ce': 'Chechen',
	'ny': 'Chichewa; Chewa; Nyanja',
	'zh': 'Chinese',
	'cv': 'Chuvash',
	'kw': 'Cornish',
	'co': 'Corsican',
	'cr': 'Cree',
	'hr': 'Croatian',
	'cs': 'Czech',
	'da': 'Danish',
	'dv': 'Divehi; Maldivian;',
	'nl': 'Dutch',
	'dz': 'Dzongkha',
	'en': 'English',
	'eo': 'Esperanto',
	'et': 'Estonian',
	'ee': 'Ewe',
	'fo': 'Faroese',
	'fj': 'Fijian',
	'fi': 'Finnish',
	'fr': 'French',
	'ff': 'Fula',
	'gl': 'Galician',
	'ka': 'Georgian',
	'de': 'German',
	'el': 'Greek, Modern',
	'gn': 'Guaraní',
	'gu': 'Gujarati',
	'ht': 'Haitian',
	'ha': 'Hausa',
	'he': 'Hebrew (modern)',
	'hz': 'Herero',
	'hi': 'Hindi',
	'ho': 'Hiri Motu',
	'hu': 'Hungarian',
	'ia': 'Interlingua',
	'id': 'Indonesian',
	'ie': 'Interlingue',
	'ga': 'Irish',
	'ig': 'Igbo',
	'ik': 'Inupiaq',
	'io': 'Ido',
	'is': 'Icelandic',
	'it': 'Italian',
	'iu': 'Inuktitut',
	'ja': 'Japanese',
	'jv': 'Javanese',
	'kl': 'Kalaallisut',
	'kn': 'Kannada',
	'kr': 'Kanuri',
	'ks': 'Kashmiri',
	'kk': 'Kazakh',
	'km': 'Khmer',
	'ki': 'Kikuyu, Gikuyu',
	'rw': 'Kinyarwanda',
	'ky': 'Kirghiz, Kyrgyz',
	'kv': 'Komi',
	'kg': 'Kongo',
	'ko': 'Korean',
	'ku': 'Kurdish',
	'kj': 'Kwanyama, Kuanyama',
	'la': 'Latin',
	'lb': 'Luxembourgish',
	'lg': 'Luganda',
	'li': 'Limburgish',
	'ln': 'Lingala',
	'lo': 'Lao',
	'lt': 'Lithuanian',
	'lu': 'Luba-Katanga',
	'lv': 'Latvian',
	'gv': 'Manx',
	'mk': 'Macedonian',
	'mg': 'Malagasy',
	'ms': 'Malay',
	'ml': 'Malayalam',
	'mt': 'Maltese',
	'mi': 'Māori',
	'mr': 'Marathi (Marāṭhī)',
	'mh': 'Marshallese',
	'mn': 'Mongolian',
	'na': 'Nauru',
	'nv': 'Navajo, Navaho',
	'nb': 'Norwegian Bokmål',
	'nd': 'North Ndebele',
	'ne': 'Nepali',
	'ng': 'Ndonga',
	'nn': 'Norwegian Nynorsk',
	'no': 'Norwegian',
	'ii': 'Nuosu',
	'nr': 'South Ndebele',
	'oc': 'Occitan',
	'oj': 'Ojibwe, Ojibwa',
	'cu': 'Old Church Slavonic',
	'om': 'Oromo',
	'or': 'Oriya',
	'os': 'Ossetian, Ossetic',
	'pa': 'Panjabi, Punjabi',
	'pi': 'Pāli',
	'fa': 'Persian',
	'pl': 'Polish',
	'ps': 'Pashto, Pushto',
	'pt': 'Portuguese',
	'qu': 'Quechua',
	'rm': 'Romansh',
	'rn': 'Kirundi',
	'ro': 'Romanian, Moldavan',
	'ru': 'Russian',
	'sa': 'Sanskrit (Saṁskṛta)',
	'sc': 'Sardinian',
	'sd': 'Sindhi',
	'se': 'Northern Sami',
	'sm': 'Samoan',
	'sg': 'Sango',
	'sr': 'Serbian',
	'gd': 'Scottish Gaelic',
	'sn': 'Shona',
	'si': 'Sinhala, Sinhalese',
	'sk': 'Slovak',
	'sl': 'Slovene',
	'so': 'Somali',
	'st': 'Southern Sotho',
	'es': 'Spanish; Castilian',
	'su': 'Sundanese',
	'sw': 'Swahili',
	'ss': 'Swati',
	'sv': 'Swedish',
	'ta': 'Tamil',
	'te': 'Telugu',
	'tg': 'Tajik',
	'th': 'Thai',
	'ti': 'Tigrinya',
	'bo': 'Tibetan',
	'tk': 'Turkmen',
	'tl': 'Tagalog',
	'tn': 'Tswana',
	'to': 'Tonga',
	'tr': 'Turkish',
	'ts': 'Tsonga',
	'tt': 'Tatar',
	'tw': 'Twi',
	'ty': 'Tahitian',
	'ug': 'Uighur, Uyghur',
	'uk': 'Ukrainian',
	'ur': 'Urdu',
	'uz': 'Uzbek',
	've': 'Venda',
	'vi': 'Vietnamese',
	'vo': 'Volapük',
	'wa': 'Walloon',
	'cy': 'Welsh',
	'wo': 'Wolof',
	'fy': 'Western Frisian',
	'xh': 'Xhosa',
	'yi': 'Yiddish',
	'yo': 'Yoruba',
	'za': 'Zhuang, Chuang',
	'zu': 'Zulu',
}

AZURE_TTS_DATA = {
        'ar': {'female': {'locale': 'ar-EG', 'gender': 'Female', 'name': "Microsoft Server Speech Text to Speech Voice (ar-EG, Hoda)"},
               'male': {'locale': 'ar-SA', 'gender': 'Male', 'name': "Microsoft Server Speech Text to Speech Voice (ar-SA, Naayf)"}},
        'ca': {'female': {'locale': 'ca-ES', 'gender':	'Female', 'name': "Microsoft Server Speech Text to Speech Voice (ca-ES, HerenaRUS)"}},
        'cs': {'male': {'locale': 'cs-CZ', 'gender': 'Male', 'name': "Microsoft Server Speech Text to Speech Voice (cs-CZ, Vit)"}},
        'da': {'female': {'locale': 'da-DK', 'gender': 'Female', 'name': "Microsoft Server Speech Text to Speech Voice (da-DK, HelleRUS)"}},
        'de': {'female': {'locale': 'de-DE', 'gender': 'Female', 'name': "Microsoft Server Speech Text to Speech Voice (de-DE, HeddaRUS)"},
               'male': {'locale': 'de-DE', 'gender': 'Male', 'name':	"Microsoft Server Speech Text to Speech Voice (de-DE, Stefan, Apollo)"}},
        'el': {'male': {'locale': 'el-GR', 'gender': 'Male', 'name': "Microsoft Server Speech Text to Speech Voice (el-GR, Stefanos)"}},
        'en': {'female': {'locale': 'en-GB', 'gender': 'Female', 'name': "Microsoft Server Speech Text to Speech Voice (en-GB, Susan, Apollo)"},
               'male': {'locale': 'en-GB', 'gender': 'Male', 'name': "Microsoft Server Speech Text to Speech Voice (en-GB, George, Apollo)"}},
        'es': {'female': {'locale': 'es-ES', 'gender': 'Female',	'name': "Microsoft Server Speech Text to Speech Voice (es-ES, Laura, Apollo)"},
               'male': {'locale': 'es-ES', 'gender': 'Male', 'name': "Microsoft Server Speech Text to Speech Voice (es-ES, Pablo, Apollo)"}},
        'fi': {'female': {'locale': 'fi-FI', 'gender': 'Female', 'name': "Microsoft Server Speech Text to Speech Voice (fi-FI, HeidiRUS)"}},
        'fr': {'female':{'locale': 'fr-FR', 'gender': 'Female', 'name': "Microsoft Server Speech Text to Speech Voice (fr-FR, Julie, Apollo)"},
               'male': {'locale': 'fr-FR', 'gender': 'Male', 'name': "Microsoft Server Speech Text to Speech Voice (fr-FR, Paul, Apollo)"}},
        'he': {'male': {'locale': 'he-IL', 'gender': 'Male', 'name': "Microsoft Server Speech Text to Speech Voice (he-IL, Asaf)"}},
        'hi': {'female': {'locale': 'hi-IN', 'gender': 'Female', 'name': "Microsoft Server Speech Text to Speech Voice (hi-IN, Kalpana, Apollo)"},
               'male': {'locale': 'hi-IN', 'gender': 'Male', 'name': "Microsoft Server Speech Text to Speech Voice (hi-IN, Hemant)"}},
        'hu': {'male': {'locale': 'hu-HU', 'gender': 'Male', 'name': "Microsoft Server Speech Text to Speech Voice (hu-HU, Szabolcs)"}},
        'id': {'male': {'locale': 'id-ID', 'gender': 'Male', 'name': "Microsoft Server Speech Text to Speech Voice (id-ID, Andika)"}},
        'it': {'male': {'locale': 'it-IT', 'gender': 'Male', 'name': "Microsoft Server Speech Text to Speech Voice (it-IT, Cosimo, Apollo)"}},
        'ja': {'female': {'locale': 'ja-JP', 'gender': 'Female', 'name': "Microsoft Server Speech Text to Speech Voice (ja-JP, Ayumi, Apollo)"},
               'male': {'locale': 'ja-JP', 'gender': 'Male', 'name': "Microsoft Server Speech Text to Speech Voice (ja-JP, Ichiro, Apollo)"}},
        'ko': {'female': {'locale': 'ko-KR', 'gender': 'Female', 'name': "Microsoft Server Speech Text to Speech Voice (ko-KR, HeamiRUS)"}},
        'nb': {'female': {'locale': 'nb-NO', 'gender': 'Female', 'name': "Microsoft Server Speech Text to Speech Voice (nb-NO, HuldaRUS)"}},
        'nl': {'female': {'locale': 'nl-NL', 'gender': 'Female', 'name': "Microsoft Server Speech Text to Speech Voice (nl-NL, HannaRUS)"}},
        'pt': {'female': {'locale': 'pt-BR', 'gender': 'Female', 'name': "Microsoft Server Speech Text to Speech Voice (pt-BR, HeloisaRUS)"},
               'male': {'locale': 'pt-BR', 'gender': 'Male', 'name': "Microsoft Server Speech Text to Speech Voice (pt-BR, Daniel, Apollo)"}},
        'ro': {'male': {'locale': 'ro-RO', 'gender': 'Male', 'name': "Microsoft Server Speech Text to Speech Voice (ro-RO, Andrei)"}},
        'ru': {'female': {'locale': 'ru-RU', 'gender': 'Female', 'name': "Microsoft Server Speech Text to Speech Voice (ru-RU, Irina, Apollo)"},
               'male': {'locale': 'ru-RU', 'gender': 'Male', 'name': "Microsoft Server Speech Text to Speech Voice (ru-RU, Pavel, Apollo)"}},
        'sk': {'male': {'locale': 'sk-SK', 'gender': 'Male', 'name': "Microsoft Server Speech Text to Speech Voice (sk-SK, Filip)"}},
        'sv': {'female': {'locale': 'sv-SE', 'gender': 'Female', 'name': "Microsoft Server Speech Text to Speech Voice (sv-SE, HedvigRUS)"}},
        'th': {'male': {'locale': 'th-TH', 'gender': 'Male', 'name': "Microsoft Server Speech Text to Speech Voice (th-TH, Pattara)"}},
        'tr': {'female': {'locale': 'tr-TR', 'gender': 'Female', 'name': "Microsoft Server Speech Text to Speech Voice (tr-TR, SedaRUS)"}},
        'zh': {'female': {'locale': 'zh-HK', 'gender': 'Female', 'name': "Microsoft Server Speech Text to Speech Voice (zh-TW, Yating, Apollo)"},
               'male': {'locale': 'zh-TW', 'gender': 'Male', 'name': "Microsoft Server Speech Text to Speech Voice (zh-TW, Zhiwei, Apollo)"}}
}

class zText:
    ''' Class to work with text data.
    
    Thera are two possible constructors:
        
    zText(data,lang) creates a zText object from a text string
    
    Parameters:
    -----------
        data: text string
        
        lang: language ISO639-1 code (https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)
        
    zText.fromfile(filename,lang) creates a zText object from a text file
    
    Parameters:
    -----------
        filename: path to target text file
        
        lang: language ISO639-1 code (https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)
    
    Atributes:
    ----------
        text: text data a a single string
        
        lang: language ISO639-1 code (https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)
    '''            
    
    def __init__(self,data='',lang=None):
        self.text = data
        self.__check_lang__(lang)
        self.lang = lang
        
    @classmethod
    def fromfile(cls, filename, lang=None):
        text = open(filename).read()
        return cls(text,lang)
        
        
    def display(self):
        '''
        Print text
        '''
        print(self.text)
        
    def display_language(self):
        '''
        Prints language name
        '''
        if self.lang is not None:
            lang_name = ISO639_1_CODES[self.lang]
            print(self.lang + ': ' + lang_name)
        else:
            print('No language set.')
        
    def __check_lang__(self,lang):
        '''
        Check that lang is either None or a valid ISO639-1 code
        '''
        if lang is not None:
            if lang not in ISO639_1_CODES:
                raise ValueError('Invalid language provided. lang must be an ISO639-1 code. See https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes for a complete list')
    
    def __request_google_translate__(self,targetLang='es'):
        '''
        Request Google Translate REST API
        '''

        sourceText = self.text
        sourceLang = self.lang
        
        request = {'q': sourceText, 'target': targetLang,  'format': 'text'}
        if sourceLang is not None:
            request['source'] = sourceLang
        
        data=json.dumps(request)
        
        conn= HTTPSConnection('translation.googleapis.com')
        conn.request('POST','/language/translate/v2?key=%s' % GOOGLE_CLOUD_API_KEY, data, {'Content-Type': 'application/json'})
        response = conn.getresponse()
        
        data = response.read()
        data = json.loads(data)
        conn.close()
            
        if response.status == 200:
            data = data['data']['translations'][0]
            return data
        else:
            raise Exception('Request failed. Message: ' + data['error']['message'])

            
    def __request_microsoft_translator__(self,translateUrl,headers,params):
        '''
        Request Microsoft Translator REST API
        '''
        
        import requests
        
        # Perform request and extract response
        translateUrl = translateUrl + "?%s" % params
        response = requests.get(translateUrl, headers = headers)
        
        if response.status_code == 200:
            data = ElementTree.fromstring(response.text.encode('utf-8'))
            return data.text
        else:
            raise Exception('Request failed')
        
        
    def translate(self,targetLang='es',backend=zAI_BACKEND):
        '''
        Translate text to a target language
        
        Parameters:
        -----------
        targetLang: desired target language ISO639-1 code (https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)
        
        backend: desired backend (Google | Microsot | local)
        
        Returns:
        --------
        zText object with the translation
        
        If source language is not defined, it is updated with the automatically detected value
        '''
        
        sourceLang = self.lang
        
        # Check that targetLang is a valid language code
        self.__check_lang__(targetLang)
        
        # Check that the keys for the selected backend are available
        if backend == 'Google':
            if GOOGLE_CLOUD_API_KEY == '':
                print('GOOGLE_CLOUD_API_KEY is empty. Using local backend.')
                backend = 'local'
        elif backend == 'Microsoft':
            if MICROSOFT_AZURE_TEXT_TRANSLATION_API_KEY == '':
                print('MICROSOFT_AZURE_TEXT_TRANSLATION_API_KEY is empty. Using local backend.')
                backend = 'local'
                
        
        if backend=='Google':
            
            # Request Google translate
            data = self.__request_google_translate__(targetLang)
            translation = data['translatedText']
            
            # Update source language if not already defined
            if sourceLang is None:
                sourceLang = data['detectedSourceLanguage']
                self.lang = sourceLang
                
            return zText(translation,targetLang)
        
        elif backend == 'Microsoft':
            
            # If source language is not defined, perform language detection and update original zText
            if sourceLang is None:
                self.detect_language(backend=backend)
                sourceLang = self.lang
                
            # Set up request
            headers = {'Ocp-Apim-Subscription-Key': MICROSOFT_AZURE_TEXT_TRANSLATION_API_KEY}

            params = urlencode({
                'text': self.text.encode('utf-8'),
                'from': sourceLang,
                'to': targetLang
            })
            
            translateUrl = 'https://api.microsofttranslator.com/V2/Http.svc/Translate'
            
            # Request Microsoft Translator
            translation = self.__request_microsoft_translator__(translateUrl,headers,params)
            return zText(translation,targetLang)
            
            
        elif backend == 'local':
            raise NotImplementedError("tanslate method is currently not available with local backend")
        else:
            raise ValueError('invalid backend selection. Valid values are currently "Google", "Microsoft" or "local".')
        
    
    def detect_language(self,backend=zAI_BACKEND):
        '''
        Detect text language
        
        Parameters:
        -----------
        backend: desired backend (Google | Microsot | local)
        
        Returns:
        --------
        self.lang is updated with detected language
        
        '''
        
        # Check that the keys for the selected backend are available
        if backend == 'Google':
            if GOOGLE_CLOUD_API_KEY == '':
                print('GOOGLE_CLOUD_API_KEY is empty. Using local backend.')
                backend = 'local'
        elif backend == 'Microsoft':
            if MICROSOFT_AZURE_TEXT_TRANSLATION_API_KEY == '':
                print('MICROSOFT_AZURE_TEXT_TRANSLATION_API_KEY is empty. Using local backend.')
                backend = 'local'
                
        if backend=='Google':
            
            # Request Google translate
            data = self.__request_google_translate__('en')
            
            # Update lang
            self.lang = data['detectedSourceLanguage']
        
        elif backend == 'Microsoft':
            
            # Set up request
            headers = {'Ocp-Apim-Subscription-Key': MICROSOFT_AZURE_TEXT_TRANSLATION_API_KEY}
            
            params = urlencode({
                'text': self.text
            })
            
            translateUrl = 'https://api.microsofttranslator.com/V2/Http.svc/Detect'
            
            # Request Microsoft Translator
            lang = self.__request_microsoft_translator__(translateUrl,headers,params)
            
            # Update lang
            self.lang = lang
            
            
        elif backend == 'local':
            raise NotImplementedError("tanslate method is currently not available with local backend")
        else:
            raise ValueError('invalid backend selection. Valid values are currently "Google", "Microsoft" or "local".')
            
    def to_voice(self,outputFile,gender=None,backend=zAI_BACKEND):
        '''
        Perform Text-to-speech to synthesize audio.
        While we don't have a proper zAudio class, result is saved to disk as audio file.
        
        Parameters:
        -----------
        outputFile: path to save output audio file
        
        gender: desired gender of the voice used to generate audio. When set to None, any avalaible gender is used.
                If a gender is specified, an error will be thrown if no voice with thta gender is available for the corresponding backend and locale combination.
        
        backend: desired backend (Google | Microsot | local)
        
        Returns:
        --------
        saves a 16kHz 16bit mono PCM wav file to desired location
        
        '''
        
        if self.lang == None:
            raise Exception("Your zText object has no 'lang' defined. Please, set the 'lang' property or perform automatic language detection using 'detect_language()'")
        
        if gender is not None:
            gender = gender.lower()
            if  gender != 'female' and gender != 'male':
                raise Exception("Gender must be either 'female' or 'male', to force a gender, or None to use any available voice gender")
    
        # Check that the keys for the selected backend are available
        if backend == 'Microsoft':
            if MICROSOFT_AZURE_BING_VOICE_API_KEY == '':
                print('MICROSOFT_AZURE_BING_VOICE_API_KEY is empty. Using local backend.')
                backend = 'local'
        
        
        if backend=='Google':
            
            raise NotImplementedError("tanslate method is currently not available with Google backend")
        
        elif backend == 'Microsoft':
            
            if self.lang not in AZURE_TTS_DATA:
                raise ValueError('Conversion to speech is not supported for this backend and language: %s' % ISO639_1_CODES[self.lang])
                
            inputData = AZURE_TTS_DATA[self.lang]
            
            if gender == None:
                if 'females' not in inputData:
                    inputData = inputData['female']
                else:
                    inputData = inputData['male']
            else:
                if gender not in inputData:
                    raise ValueError('This combination of backend and language does not support a %s voice' % gender)
                
                inputData = inputData[gender]
                
            
            params = ""
            headers = {"Ocp-Apim-Subscription-Key": MICROSOFT_AZURE_BING_VOICE_API_KEY}
        
            AccessTokenHost = "api.cognitive.microsoft.com"
            path = "/sts/v1.0/issueToken"
        
            conn = HTTPSConnection(AccessTokenHost)
            conn.request("POST", path, params, headers)
            response = conn.getresponse()
            data = response.read()
            conn.close()
            accesstoken = data.decode("UTF-8")
        
            body = ElementTree.Element('speak', version='1.0')
            body.set('{http://www.w3.org/XML/1998/namespace}lang', inputData['locale'].lower())
            voice = ElementTree.SubElement(body, 'voice')
            voice.set('{http://www.w3.org/XML/1998/namespace}lang', inputData['locale'])
            voice.set('{http://www.w3.org/XML/1998/namespace}gender', inputData['gender'])
            voice.set('name', inputData['name'])
            voice.text = self.text
            
            headers = {"Content-type": "application/ssml+xml", 
            			"X-Microsoft-OutputFormat": "riff-16khz-16bit-mono-pcm", 
            			"Authorization": "Bearer " + accesstoken, 
            			"User-Agent": "zAI"}
            
            conn = HTTPSConnection("speech.platform.bing.com")
            conn.request("POST", "/synthesize", ElementTree.tostring(body), headers)
            response = conn.getresponse()
        
            data = response.read()
            conn.close()
            if response.status == 200:
                with open(outputFile, "wb") as wavfile:
                    wavfile.write(bytes(data))
            else:
                raise Exception('Request failed: %s' % response.reason)
                               
            
        elif backend == 'local':
            raise NotImplementedError("tanslate method is currently not available with local backend")
        else:
            raise ValueError('invalid backend selection. Valid values are currently "Google", "Microsoft" or "local".')
            
    @staticmethod
    def set_backend_key(key_name, new_key):
        from zAI.utils import keys
        available_keys = keys.get_available_keys()
        if key_name in available_keys:
            exec('%s = "%s"' %(key_name, new_key), globals())
        else:
            raise ValueError('%s is not a valid key name.' % key_name)
        
        
