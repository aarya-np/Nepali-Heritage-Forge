import streamlit as st
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
import requests
import base64
from io import BytesIO
import torch
import os
import pytesseract
from PIL import Image
import os
import cv2
from pytesseract import Output
from gtts import gTTS
import tempfile
from pydub import AudioSegment

AudioSegment.converter = "D:/virtual environment/nepalihertiageforge/ffmpeg/bin/ffmpeg.exe"


checkpoint_path = r'D:\virtual environment\nepalihertiageforge\checkpoint-90858'
model = AutoModelForSeq2SeqLM.from_pretrained(r'D:\virtual environment\nepalihertiageforge\M2M100')
tokenizer = AutoTokenizer.from_pretrained(r'D:\virtual environment\nepalihertiageforge\M2M100')
model_1 = PeftModel.from_pretrained(model, checkpoint_path)


def adjust_gamma(image,gamma=0.7):
  invGamma= 1.0/gamma
  table= np.array([((i/255)**invGamma)*255 for i in np.arange(0,256)])
  gamma_corrected= cv2.LUT(image.astype(np.uint8),table.astype(np.uint8))
  return gamma_corrected

def thick_font(image):
  image= cv2.bitwise_not(image)
  kernel= np.ones((3,3), np.uint8)
  image=cv2.dilate(image,kernel,iterations=1)
  image= cv2.bitwise_not(image)
  return(image)

def noise_removal(image):
  kernel= np.ones((1,1),np.uint8)
  image= cv2.dilate(image, kernel,iterations=1)
  kernel= np.ones((1,1),np.uint8)
  image= cv2.erode(image, kernel, iterations=1)
  image= cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
  image= cv2.medianBlur(image, 3)
  return(image)

def blur(image):
  return cv2.GaussianBlur(image, (5,5),0)

def sharp(image):
  f= image
  f= f/255
  f_blur= cv2.GaussianBlur(src=f,ksize=(5,5),sigmaX=0, sigmaY=0)

  g_mask= f - f_blur

  k=5
  g= f+ k*g_mask
  g= np.clip(g,0,1)
  g = (g * 255).astype(np.uint8)
  return g


def image_cleaning(image):
   #clahe_image= histo(image)
   gamma_corrected= adjust_gamma(image)
   dilated_image= thick_font(gamma_corrected)
   no_noise= noise_removal(dilated_image)
   blurred= blur(no_noise)
   sharpened_image = sharp(blurred)
   return sharpened_image


def ocr(image):
    #clahe_image= histo(image)
    pytesseract.pytesseract.tesseract_cmd = r'D:\virtual environment\nepalihertiageforge\Tesseract-OCR\tesseract.exe'
    os.environ['TESSDATA_PREFIX'] = r'D:\virtual environment\nepalihertiageforge\Tesseract-OCR\tessdata'
    config = '--oem 1 --psm 6'
    d = pytesseract.image_to_string(image, output_type=Output.DICT, lang='san', config=config)
    return d['text']

def text_translation_prediction_M2M100(text_list):
    translations = []
    for text in text_list:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = model_1.generate(**inputs, max_length=1024)
        english_translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.append(english_translation)
    return translations


def bhashini_translate(text, user_id = "c0b5cfa7bd9a4e32832c45d3ecdadd8f", api_key = "12ab84400e-816c-4872-a639-85df185d6a3a", from_code: str = "sa", to_code: str = "en"):
    translations =[]
    url = 'https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline'
    headers = {
            "Content-Type": "application/json",
            "userID": user_id,
            "ulcaApiKey": api_key
    }
    payload = {
            "pipelineTasks": [{"taskType": "translation", "config": {"language": {"sourceLanguage": from_code, "targetLanguage": to_code}}}],
            "pipelineRequestConfig": {"pipelineId": "64392f96daac500b55c543cd"}
        }
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        return {"status_code": response.status_code, "message": "Error in translation request", "translated_content": None}

        # Process the response to setup the translation execution
    response_data = response.json()
    service_id = response_data["pipelineResponseConfig"][0]["config"][0]["serviceId"]
    callback_url = response_data["pipelineInferenceAPIEndPoint"]["callbackUrl"]
    headers2 = {
            "Content-Type": "application/json",
            response_data["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["name"]: response_data["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["value"]
    }
    
    compute_payload = {
                "pipelineTasks": [{"taskType": "translation", "config": {"language": {"sourceLanguage": from_code, "targetLanguage": to_code}, "serviceId": service_id}}],
                "inputData": {"input": [{"source": text}], "audio": [{"audioContent": None}]}
            }

            # Execute the translation
    compute_response = requests.post(callback_url, json=compute_payload, headers=headers2)
    if compute_response.status_code != 200:
        return {"status_code": compute_response.status_code, "message": "Error in translation", "translated_content": None}
    compute_response_data = compute_response.json()
    translated_content = compute_response_data["pipelineResponse"][0]["output"][0]["target"]

    return translated_content


def bhashini_audio(encoded_audio,  user_id = "c0b5cfa7bd9a4e32832c45d3ecdadd8f", api_key = "12ab84400e-816c-4872-a639-85df185d6a3a", from_code: str = "sa"):
   url = 'https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline'
   headers = {
            "Content-Type": "application/json",
            "userID": user_id,
            "ulcaApiKey": api_key
    }
   payload = {
    'pipelineTasks': [
        {
            'taskType':'asr',
        "config": {
                "language": {
                    "sourceLanguage": from_code
                }
        }
        }
        ],
    'pipelineRequestConfig':
     {
         #'pipelineId':'643930aa521a4b1ba0f4c41d',
         'pipelineId':'64392f96daac500b55c543cd'
         }
        }
   response = requests.post(url, json = payload, headers = headers)
   response_data = response.json()
   service_id = response_data['pipelineResponseConfig'][0]['config'][0]['serviceId']
   callback_url = response_data['pipelineInferenceAPIEndPoint']['callbackUrl']
   headers2 = {
    'Content-Type':'application/json',
    response_data['pipelineInferenceAPIEndPoint']['inferenceApiKey']['name']:response_data['pipelineInferenceAPIEndPoint']['inferenceApiKey']['value']
    }
   compute_payload = {
    'pipelineTasks':[
        {
        'taskType':'asr',
        'config':{'language':{'sourceLanguage':'sa'},
        'serviceId':service_id,
        'audioFormat':'wav',
        "samplingRate": 16000
        }
        }
    ],
    'inputData':{
                 'audio':[
                     {'audioContent':encoded_audio}
                 ]}
        }
   response_2 = requests.post(callback_url, headers = headers2, json=compute_payload)
   source_value = response_2.json()['pipelineResponse'][0]['output'][0]['source']
   return source_value

def make_list(text):
    delimiter =["॥","।","|"]
    for delimiters in delimiter:
      text = text.replace(delimiters,'\n')
    #print(text)
    new_list = text.split('\n')
    #print(new_list)
    filtered_list = [item for item in new_list if item.strip()]
    #print(filtered_list)
    new_filtered_list =[item.replace('।','') for item in filtered_list]
    return new_filtered_list


def text_to_speech(text, lang_code):
    tts = gTTS(text, lang=lang_code)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tts.save(tmp_file.name)
        return tmp_file.name

def play_audio(file_path):
    audio_file = open(file_path, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/mp3')

def main():

    st.title('Nepali Heritage Forge')

    text = st.text_input('Sanskrit Text')
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
    uploaded_audio = st.file_uploader('Choose an audio', type =['wav'])
    source_value = ''

    translation_1 = ''
    translation_2 = ''
    result =""
    translation_4 = ''
    translation_5 = ''

    translation_6 =''
    translation_7=''
    audio_file_path =''
    given_audio =''
    if st.button('Translate'):
        if text:
            new_list=make_list(text)
            translation_1 = text_translation_prediction_M2M100([new_list])
            translation_2 = bhashini_translate(text)
            result = f"Translation by M2M100 Model: {translation_1} \n Translation by Bhashini model: {translation_2}\n"
            audio_file_1 = text_to_speech(text,'hi')

            flattened_translation = [str(item) for sublist in translation_1 for item in sublist]

            # Join the flattened list into a single string
            output = ' '.join(flattened_translation)
            audio_file_2 = text_to_speech(output, 'en')
            audio_file_3 = text_to_speech(translation_2, 'en')
            if result:
              st.success(result)
              st.write('Audio for Sanskrit text')
              play_audio(audio_file_1)
              st.write('Translation by M2M100')
              play_audio(audio_file_2)
              st.write('Translation by Bhashini Model')
              play_audio(audio_file_3)

                
            

        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            
            # Decode the image as a NumPy array using OpenCV
            image = cv2.imdecode(file_bytes, 1)
            
            # Convert the image from BGR to RGB (Streamlit uses RGB format)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            image_b = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            new_image = image_cleaning(image_b)

            # Display the image in Streamlit
            st.image(new_image, caption='Uploaded Image in blurred', use_column_width=True)

            source_value = ocr(new_image)
            
            new_list = make_list(source_value)
            translation_4 = text_translation_prediction_M2M100([new_list])
            translation_5 = bhashini_translate(source_value)

            flattened_translation = [str(item) for sublist in translation_4 for item in sublist]

            # Join the flattened list into a single string
            output = ' '.join(flattened_translation)


            result = f'Digitized form : {source_value}\n Translation by M2M100 Model: {translation_4}\n Translation by Bhashini Model: {translation_5} '
            audio_file_1 = text_to_speech(source_value,'hi')
            audio_file_2 = text_to_speech(output, 'en')
            audio_file_3 = text_to_speech(translation_5, 'en')
            if result:
              st.success(result)
              st.write('Audio for Sanskrit text')
              play_audio(audio_file_1)
              st.write('Translation by M2M100')
              play_audio(audio_file_2)
              st.write('Translation by Bhashini Model')
              play_audio(audio_file_3)


    # if result:
    #       st.success(result)
        if uploaded_audio:
          with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_audio.read())
            audio_file_path = tmp_file.name
          given_audio = AudioSegment.from_wav(audio_file_path)
          given_audio = given_audio.set_frame_rate(16000)
          with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as processed_tmp_file:
              given_audio.export(processed_tmp_file.name, format="wav", codec="pcm_s16le")
              processed_audio_path = processed_tmp_file.name
          with open(processed_audio_path, "rb") as wav_file:
              encoded_string = base64.b64encode(wav_file.read())
          encoded_string = encoded_string.decode('ascii', 'ignore')
          source_value = bhashini_audio(encoded_string)
          translation_6 = text_translation_prediction_M2M100([source_value])
          translation_7 = bhashini_translate(source_value)
          flattened_translation = [str(item) for sublist in translation_6 for item in sublist]
          output = ' '.join(flattened_translation)
          result = f"Audio to Text : {source_value} \n Translation by M2M100 Model: {output} \n Translation by Bhashini model: {translation_7}\n"


          # Join the flattened list into a single string
          audio_file_1 = text_to_speech(source_value,'hi')
          audio_file_2 = text_to_speech(output, 'en')
          audio_file_3 = text_to_speech(translation_7, 'en')
          if result:
            st.success(result)
            st.write('Audio for Sanskrit text')
            play_audio(audio_file_1)
            st.write('Translation by M2M100')
            play_audio(audio_file_2)
            st.write('Translation by Bhashini Model')
            play_audio(audio_file_3)

if __name__ == '__main__':
    main()