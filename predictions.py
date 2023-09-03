#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import cv2
import pytesseract
from glob import glob
import spacy
import re
import string
import warnings
warnings.filterwarnings('ignore')




#Load NER model
model_ner = spacy.load('./output/model-best/')


def cleanText(txt):
    whitespace = string.whitespace
    punctuation = '!#$%&\'()*+:;<=>?[\\]^`{|}~'
    tabeWhiteSpace = str.maketrans('','',whitespace)
    tablePunctuation = str.maketrans('','',punctuation)
    text=str(txt)
    # text = text.lower()
    removeWhiteSpace = text.translate(tabeWhiteSpace)
    removePunctuation = removeWhiteSpace.translate(tablePunctuation)
    return str(removePunctuation)    

# group the label
class groupgen():
    def __init__(self):
        self.id = 0
        self.text = ''
        
    def getgroup(self,text):
        if self.text == text:
            return self.id
        else:
            self.id +=1
            self.text = text
            return self.id
        
def parser(text, label):
    if label == 'DATE':
        text = text.lower()

        
    elif label == 'CARDINAL':
        text = re.sub(r'[^0-9]', '', text)
        
    elif label == 'MONEY':
        text = text.lower()
        text = re.sub(r'[^0-9$€£,.]', '', text)
        
    elif label == 'ORG':
        text = text.title()
        
    return text      


grp_gen = groupgen()

def getPredictions(image):

    tessData = pytesseract.image_to_data(image)
    #tessData
    # convert into dataframe
    tessList = list(map(lambda x:x.split('\t'), tessData.split('\n')))
    df = pd.DataFrame(tessList[1:],columns=tessList[0])
    df.dropna(inplace=True) # drop missing values
    df['text'] = df['text'].apply(cleanText)

    # convet data into content
    df_clean = df.query('text != "" ')
    content = " ".join([w for w in df_clean['text']])
    print(content)
    # get prediction from NER model
    doc = model_ner(content)


    # converting doc in json
    docjson = doc.to_json()
    docjson.keys()

    doc_text = docjson['text']
    doc_text
    
    # creating tokens
    datafram_tokens = pd.DataFrame(docjson['tokens'])
    datafram_tokens['token'] = datafram_tokens[['start','end']].apply(
        lambda x:doc_text[x[0]:x[1]] , axis = 1)
   



    right_table = pd.DataFrame(docjson['ents'])[['start','label']]
    datafram_tokens = pd.merge(datafram_tokens,right_table,how='left',on='start')
    datafram_tokens.fillna('O',inplace=True)


    # join lable to df_clean dataframe
    df_clean['end'] = df_clean['text'].apply(lambda x: len(x)+1).cumsum() - 1 
    df_clean['start'] = df_clean[['text','end']].apply(lambda x: x[1] - len(x[0]),axis=1)


    # inner join with start 
    dataframe_info = pd.merge(df_clean,datafram_tokens[['start','token','label']],how='inner',on='start')




    # ### Bounding Box


    bb_df = dataframe_info.query("label != 'O' ")
    bb_df['label'] = bb_df['label'].apply(lambda x: x[2:])        
    bb_df['group'] = bb_df['label'].apply(grp_gen.getgroup)




    # right and bottom of bounding box
    bb_df[['left','top','width','height']] = bb_df[['left','top','width','height']].astype(int)
    bb_df['right'] = bb_df['left'] + bb_df['width']
    bb_df['bottom'] = bb_df['top'] + bb_df['height']


    # tagging: groupby group
    col_group = ['left','top','right','bottom','label','token','group']
    group_tag_img = bb_df[col_group].groupby(by='group')

    img_tagging = group_tag_img.agg({
        
        'left':min,
        'right':max,
        'top':min,
        'bottom':max,
        'label':np.unique,
        'token':lambda x: " ".join(x)
        
    })




    img_bb = image.copy()
    for l, r, t, b, label, token in img_tagging.values:
        cv2.rectangle(img_bb, (l, t), (r, b), (0, 255, 0), 2)
        
        # Convert label to a string if it's not already
        label_str = str(label)
        cv2.putText(img_bb, label_str, (l, t), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)


    # #### Entities


    info_array = dataframe_info[['token','label']].values
    entities = dict(ORG=[],DATE=[],MONEY=[],CARDINAL=[])
    previous = 'O'

    for token, label in info_array:
        bio_tag = label[0]
        label_tag = label[2:]
        
        # step -1 parse the token
        text = parser(token,label_tag)
        
        if bio_tag in ('B','I'):
            
            if previous != label_tag:
                entities[label_tag].append(text)
                
            else:
                if bio_tag == "B":
                    entities[label_tag].append(text)
                    
                else:
                    if label_tag in ("ORG",'DATE','MONEY','CARDINAL'):
                        entities[label_tag][-1] = entities[label_tag][-1] + " " + text
                        
                    else:
                        entities[label_tag][-1] = entities[label_tag][-1] + text
                        
        
    
    previous = label_tag

    return img_bb, entities



