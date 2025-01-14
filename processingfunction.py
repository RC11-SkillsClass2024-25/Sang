#!/usr/bin/env python
# coding: utf-8

# In[2]:


# 使用 gensim 模块对文本进行预处理
import gensim


# In[9]:


# 导入自然语言处理工具包
import nltk
from nltk.stem import PorterStemmer  # 导入词干提取器
from nltk.stem import WordNetLemmatizer  # 导入词形还原工具
from nltk.corpus import words, stopwords, names  # 导入单词、停用词和名称库

# 下载相关数据集（如未安装）
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')


# In[10]:


def preprocess_paragraphs(paragraphs):
    lemmatizer = WordNetLemmatizer()  #lemmatizer：使用 WordNet 的词形还原工具，将单词还原为原形（如动词的基本形式）。
    stemmer = PorterStemmer()  #stemmer：使用 Porter 词干提取器，将单词还原为词干（如将“running”变为“run”）。
    ENGLISH_WORDS = set(words.words())  #ENGLISH_WORDS：获取英语单词的集合，用于过滤非英语单词。
    STOP_WORDS = set(stopwords.words("english"))  #STOP_WORDS：定义英语中的停用词集合（如 "the"、"is"），这些词通常在自然语言处理中被忽略。
    processed_paragraphs = []  #processed_paragraphs：初始化一个空列表，用于存储处理后的段落。

    for i, paragraph in enumerate(paragraphs):
        words_in_paragraph = gensim.utils.simple_preprocess(paragraph, min_len=3, deacc=True)  
        #使用 gensim 提取段落中的单词，移除标点符号。min_len=3：只保留长度至少为 3 的单词。 deacc=True：移除单词中的标点符号。
        
        lemmatized_words = [
            lemmatizer.lemmatize(word) for word in words_in_paragraph if word.lower() in ENGLISH_WORDS
        ]
        filtered_words = [word for word in lemmatized_words if word not in STOP_WORDS]
        stemmed_words = [stemmer.stem(word) for word in filtered_words]
        processed_doc = " ".join(stemmed_words) #将提取的词干拼接成一个字符串，用空格分隔。

        # 保存关键信息到字典
        processed_paragraphs.append({
            "index": i + 1,
            "lemmatized_words": lemmatized_words[:100], 
            "stemmed_words": stemmed_words[:100],  # 前10个词干
            "processed_text": processed_doc
        })
    return processed_paragraphs

