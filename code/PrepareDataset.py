#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import datetime
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk import sent_tokenize
from pathlib import Path
from dateutil.parser import parse as parse_date
from lxml import etree
from tqdm.auto import tqdm
from tehut.normalisation import cab
from flair.models import SequenceTagger, MultiTagger
from flair.data import Sentence
from nltk import sent_tokenize


# Steps:
# 0. Remove some authors and add others (see list)
# 1. Gather additional metadata
#     * Date of creation
#     * Author nationality
#     * Genre
# 2. Down-Sample 'Märchen'
# 3. Concatenate Peter Altenbergs works to one big doc

# ### Load dataset

# In[2]:


beta_df = pd.read_csv('../data/dataset_beta.csv')


# In[3]:


author_list = pd.read_excel('AutorenlistePandas.xlsx')


# In[4]:


author_list


# ### Load textgrid src data

# In[5]:


author_files = list(sorted(Path('/mnt/data/users/keller/textgrid_data/xml/').glob('*.xml')))


# In[6]:


def normalize_whitespace(string):
    whitespaces = re.compile(r'[\t ]+')
    return re.sub(whitespaces, ' ', string)


# In[7]:


romantik_start_date = datetime.datetime(1775, 1, 1, 0, 0)
romantik_end_date = datetime.datetime(1805, 12, 31, 23, 59)

realismus_start_date = datetime.datetime(1830, 1, 1, 0, 0)
realismus_end_date = datetime.datetime(1870, 12, 31, 23, 59)

pbar = tqdm(author_files)
data = []
MODELS = ['de-pos', 'de-ner-large',]
tagger = MultiTagger.load(MODELS)
for file in pbar:
    author_name = file.stem.replace('Literatur-', '')
    
    query_authorlist = author_list[author_list.author.str.contains(author_name, regex=False)]
    #print(author_name, len(query_authorlist))
    if len(query_authorlist) == 0:
        continue
    
    authorlist_entry = query_authorlist.iloc[0].to_dict()
    
    if '?' in authorlist_entry['Passend']:
        continue
    
    epoch = authorlist_entry['Epoch']
    
    
    doc = etree.parse(str(file))
    author_pnd = doc.xpath('//tei:sourceDesc[1]/tei:bibl/tei:author/@key', namespaces={'tei': 'http://www.tei-c.org/ns/1.0'})
    if len(set(author_pnd)) != 1:
        continue
        
    author_pnd = author_pnd[0].strip('pnd:')
    author_metadata = requests.get(f'http://lobid.org/gnd/{author_pnd}')
    if author_metadata.status_code != 200:
        continue
        
    city_of_birth = author_metadata.json().get('placeOfBirth', [None])[0]
    if city_of_birth is None:
        continue
    
    country_of_birth = requests.get(f'https://hub.culturegraph.org/entityfacts/{city_of_birth["id"].strip("https://d-nb.info/gnd/")}').json()['associatedCountry'][0]['preferredName']

    
    # Get year of birth
    date_of_birth = author_metadata.json().get('dateOfBirth', None)
    
    if date_of_birth is None:
        continue
    
    date_of_birth = parse_date(date_of_birth[0])
    
    
    ## Filter by epoch and assign epochs
    #if date_of_birth >= romantik_start_date and date_of_birth <= romantik_end_date:
    #    epoch = "romantik"
    #elif date_of_birth >= realismus_start_date and date_of_birth <= realismus_end_date:
    #    epoch = "realismus"
    #else:
    #    continue
    
    
    # Get gender
    author_gender = author_metadata.json()['gender'][0]['label']

    
    texts_subtrees = [
        etree.ElementTree(st) for st in doc.xpath(
            '//tei:TEI[.//tei:textClass/tei:keywords[@scheme="http://textgrid.info/namespaces/metadata/core/2010#genre"]/tei:term = "prose"]',
            namespaces={'tei': 'http://www.tei-c.org/ns/1.0'}
        )
    ]
    for text_subtree in texts_subtrees:
        
        text = normalize_whitespace(''.join(text_subtree.xpath('//tei:body//text()', namespaces={'tei': 'http://www.tei-c.org/ns/1.0'})))
        title = text_subtree.xpath('//tei:teiHeader//tei:titleStmt//tei:title//text()', namespaces={'tei': 'http://www.tei-c.org/ns/1.0'})[0]
        
        creation_year_elem = text_subtree.xpath('//tei:teiHeader//tei:profileDesc/tei:creation/tei:date', namespaces={'tei': 'http://www.tei-c.org/ns/1.0'})[0]
        #print(creation_year_elem.xpath('tei:date/@when', namespaces={'tei': 'http://www.tei-c.org/ns/1.0'}))
        if not creation_year_elem.text:
            creation_year = "-".join(creation_year_elem.attrib.values())
        else:
            creation_year = creation_year_elem.text

        
        pub_year = text_subtree.xpath('//tei:teiHeader//tei:sourceDesc/tei:biblFull/tei:publicationStmt/tei:date/@when', namespaces={'tei': 'http://www.tei-c.org/ns/1.0'})
        if pub_year :
            pub_year = pub_year[0]
        else:
            pub_year = np.nan
        
        pub_place = text_subtree.xpath('//tei:teiHeader//tei:sourceDesc/tei:biblFull/tei:publicationStmt/tei:pubPlace/text()', namespaces={'tei': 'http://www.tei-c.org/ns/1.0'})
        if pub_place :
            pub_place = pub_place[0]
        else:
            pub_place = np.nan
            
        entry = {
            'author': file.stem.replace('Literatur-', ''),
            'author_pnd': author_pnd,
            'author_gender': author_gender,
            'author_city_of_birth': city_of_birth['label'],
            'author_country_of_birth': country_of_birth,
            'author_date_of_birth': str(date_of_birth),
            'creation_year': creation_year,
            'pub_year': pub_year,
            'pub_place': pub_place,
            'epoch': epoch,
            'title': title,
            #'text': text.strip(),
            #'normed_text': '',
            #'tagged_text': '',
        }
        
        beta_entry = beta_df[(beta_df.author == file.stem.strip('Literatur-')) & (beta_df.title == title)]   # strip because of legacy bug
        if len(beta_entry) > 0:
            # Merge new data and old dataset
            beta_entry = beta_entry.iloc[0].to_dict()
            for key in beta_entry:
                if key not in entry:
                    entry[key] = beta_entry[key]
        else:
            # Orig text
            text = text.strip()
            # normed text
            normed_text = cab(text)
            # tagged text
            sentences = [Sentence(s) for s in sent_tokenize(normed_text, language='german')]
            tagged_sentences = []
            for sentence in tqdm(sentences):
                tagger.predict(sentence)
                tagged_sentences.append(sentence.to_tagged_string())
            tagged_text = ' '.join(tagged_sentences)
            
            entry['text'] = text
            entry['normed_text'] = normed_text
            entry['tagged_text'] = tagged_text
        
        
        
        data.append(entry)
    
    pbar.set_description(f'Found {len(data)} texts...')


# In[ ]:


new_df = pd.DataFrame.from_records(data)


# In[ ]:


new_df = new_df.drop('author_data_of_birth', axis=1)


# In[ ]:


new_df = new_df.rename({'tagged_texts': 'tagged_text', 'author_county_birth': 'author_country_of_birth'}, axis=1)


# ### Cutoff authors => Not more than 15 texts of each author

# In[ ]:


new_df.value_counts('author_pnd').mean()


# In[ ]:


author_parts = []
for pnd in new_df.author_pnd.unique():
    author_df = new_df.query('author_pnd == @pnd')
    if len(author_df) > 15:
        author_df = author_df.sample(15)
    author_parts.append(author_df)
sampled_df = pd.concat(author_parts, axis=0).reset_index(drop=True)


# In[ ]:


new_df.to_csv('../data/rom_real_dataset_full.csv', index=False)
sampled_df.to_csv('../data/rom_real_dataset.csv', index=False)

