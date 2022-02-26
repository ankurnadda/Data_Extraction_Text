#!/usr/bin/env python
# coding: utf-8

# In[90]:


import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[138]:


def open_file (str):
    document = open(str, "r").read()
    return document;


# In[139]:


GS_2001 = open_file("GS 2001-10-k.txt")
GS_2003 = open_file("GS 2003-10-k.txt")


# In[140]:


all_words_2001=[]
all_words_2003=[]


# In[141]:


GS_2001_words = word_tokenize(GS_2001)
GS_2003_words = word_tokenize(GS_2003)


# In[142]:


for word in GS_2001_words:
    all_words_2001.append(word.lower())
print(len(all_words_2001))
for word in GS_2003_words:
    all_words_2003.append(word.lower())
print(len(all_words_2003))


# In[95]:


all_words_freq_2001 = nltk.FreqDist(all_words_2001)
all_words_freq_2003 = nltk.FreqDist(all_words_2003)


# In[96]:


print(all_words_freq_2001)
print(all_words_freq_2003)


# In[97]:


from nltk.corpus import stopwords
stop_words = stopwords.words('english')
punctuations = ['(',')',';',':','[',']',',','.','"','*','$','%','_', '-', '—', '’', '•', '“', '”', "''", '&', "'s"]
digits = ['1', '2', '3', '4', '5', '6','7', '8', '9', '0']
goldman_sachs = ['goldman', 'sachs']
alphabets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
for i in punctuations:
    stop_words.append(i)
for i in alphabets:
    stop_words.append(i)
for i in digits:
    stop_words.append(i)
for i in goldman_sachs:
    stop_words.append(i)       


# In[98]:


filtered_words_2001 = []
filtered_words_2003 = []


# In[99]:


for w in all_words_2001: 
    if w not in stop_words: 
        filtered_words_2001.append(w) 


# In[100]:


for w in all_words_2003: 
    if w not in stop_words: 
        filtered_words_2003.append(w) 


# In[101]:


filtered_words_freq_2001 = nltk.FreqDist(filtered_words_2001)
filtered_words_freq_2003 = nltk.FreqDist(filtered_words_2003)


# In[102]:


most_common_2001 = filtered_words_freq_2001.most_common(20)
most_common_2003 = filtered_words_freq_2003.most_common(20)


# In[103]:


df_2001 = pd.DataFrame(most_common_2001, columns = ['Word', 'Count'])
df_2003 = pd.DataFrame(most_common_2003, columns = ['Word', 'Count'])
print(df_2001)
print("************************************************************************************")
print(df_2003)


# In[104]:


from wordcloud import WordCloud, STOPWORDS
stoplist = set(STOPWORDS)


# In[105]:


wordcloud_2001 = WordCloud(width=800, height=600, background_color='white', stopwords = stoplist).generate_from_frequencies(filtered_words_freq_2001) 


# In[106]:


plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud_2001) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 


# In[107]:


wordcloud_2003 = WordCloud(width=800, height=600, background_color='white', stopwords = stoplist).generate_from_frequencies(filtered_words_freq_2003)


# In[108]:


plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud_2003) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 


# In[109]:


plt.figure(figsize=(16,5))
topics = ['financial','assets', 'shareholders', 'securities', 'trading', 'stock','management','risk', 'government', 'capital']
mytext_2001 = nltk.Text(GS_2001_words)
mytext_2001.dispersion_plot(topics)


# In[110]:


plt.figure(figsize=(16,5))
topics = ['financial','assets', 'shareholders', 'securities', 'trading', 'stock','management','risk', 'government', 'capital']
mytext_2003 = nltk.Text(GS_2003_words)
mytext_2003.dispersion_plot(topics)


# In[111]:


df_2001.plot.bar(x='Word',y='Count')


# In[115]:


df_2003.plot.bar(x='Word',y='Count')


# In[ ]:





# In[ ]:




