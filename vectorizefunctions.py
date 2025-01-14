#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install scikit-learn')


# In[4]:


import pickle


# In[5]:


# 验证保存
with open('processed_text.pkl', 'rb') as file:
    processed_docs = pickle.load(file)


# In[6]:


print("processed_docs:", processed_docs[:6])


# In[21]:


from sklearn.feature_extraction.text import TfidfVectorizer

def generate_tfidf_matrix(processed_docs, min_df=2):
    """
    根据输入的文档列表生成 TF-IDF 矩阵，并返回矩阵和特征名称。

    参数:
        processed_docs (list of str): 预处理后的文档列表。
        min_df (int): 最小文档频率，用于过滤低频词汇，默认为 2。

    返回:
        tfidf_matrix (sparse matrix): TF-IDF 矩阵，表示文档的特征向量。
        feature_names (list of str): 特征名称列表（词汇表）。
    """
    # 初始化 TF-IDF 向量化器
    vectorizer = TfidfVectorizer(min_df=min_df)
    
    # 将文档列表转换为 TF-IDF 矩阵
    tfidf_matrix = vectorizer.fit_transform(processed_docs)
    
    # 获取特征名称
    feature_names = vectorizer.get_feature_names_out()
    
    return tfidf_matrix, feature_names


# In[22]:


# 调用函数生成 TF-IDF 矩阵和特征名称
tfidf_matrix, feature_names = generate_tfidf_matrix(processed_docs, min_df=2)

# 输出矩阵形状
print("TF-IDF Matrix Shape:", tfidf_matrix.shape)

# 输出特征名称
print("Feature Names:", feature_names)


# In[13]:


get_ipython().system('pip install nltk')


# In[8]:


# 导入自然语言处理工具包
import nltk
from nltk.stem import PorterStemmer  # 导入词干提取器
from nltk.stem import WordNetLemmatizer  # 导入词形还原工具
from nltk.corpus import words, stopwords, names  # 导入单词、停用词和名称库

# 下载相关数据集（如未安装）
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')


# In[9]:


from processingfunction import preprocess_paragraphs


# In[10]:


import pickle

# 从文件中加载列表 Paras
with open("paras.pkl", "rb") as file:  # "rb" 表示以二进制读取模式打开文件
    loaded_paras = pickle.load(file)


# In[26]:


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_top_similar_texts(query, vectorizer, tfidf_matrix, loaded_paras, top_n=20, top_terms_n=10):
    """
    根据查询文本计算与已知文档的相似度，返回最相似文档列表及其相关词汇 (top terms)。

    参数:
        query (str): 用户的查询文本。
        vectorizer (TfidfVectorizer): 已训练的 TF-IDF 向量化器。
        tfidf_matrix (sparse matrix): 所有已知文档的 TF-IDF 矩阵。
        loaded_paras (list): 包含所有原始文档的列表。
        top_n (int): 返回最相似文档的数量（默认 20）。
        top_terms_n (int): 返回与查询最相关的特征词汇数量（默认 10）。

    返回:
        results (list of dict): 包含排名、文档索引、相似度分数和文档内容的列表。
        top_terms (list of str): 与查询文本最相关的 top terms。
    """
    # 1. 预处理查询文本并生成 TF-IDF 向量
    processed_query = preprocess_paragraphs([query])  # 调用预处理函数
    query_texts = [doc["processed_text"] for doc in processed_query]  # 提取处理后的文本
    query_vector = vectorizer.transform(query_texts)  # 转换为 TF-IDF 向量

    # 2. 计算查询与所有文档的余弦相似度
    similarities = cosine_similarity(query_vector, tfidf_matrix)

    # 3. 找到最相似的文档索引和分数
    nearest_neighbor_index = similarities.argmax()  # 最相似文档的索引
    nearest_neighbor = tfidf_matrix[nearest_neighbor_index]  # 最相似文档的 TF-IDF 向量
    similarity_score = similarities[0, nearest_neighbor_index]  # 最相似文档的相似度分数
    document_content = loaded_paras[nearest_neighbor_index]  # 最相似文档的内容

    # 4. 从最相似文档中提取特征权重，获取 top terms
    top_indices = nearest_neighbor.toarray().flatten().argsort()[::-1][:top_terms_n]  # 从高到低取 top_terms_n 个特征索引
    top_terms = [vectorizer.get_feature_names_out()[idx] for idx in top_indices]  # 提取特征词汇

    # 5. 找到与查询最相似的 top_n 文档
    top_indices = np.argsort(similarities[0])[-top_n:][::-1]  # 获取相似度前 top_n 的文档索引，按降序排列
    top_scores = similarities[0, top_indices]  # 提取相应的相似度分数
    top_texts = [loaded_paras[i] for i in top_indices]  # 获取对应的文档内容

    # 6. 构建结果列表
    results = [
        {"rank": rank, "index": index, "score": score, "text": text}
        for rank, (index, score, text) in enumerate(zip(top_indices, top_scores, top_texts), start=1)
    ]

    # 返回相似文档结果和 top terms
    return results, top_terms


# In[34]:


# 调用函数
query = "second-hand fashion"
results, top_terms = get_top_similar_texts(query, vectorizer, tfidf_matrix, loaded_paras, top_n=10, top_terms_n=20)

# 打印结果
print("Top Terms:")
print(", ".join(top_terms))  # 打印 top terms 列表

print("\nTop Similar Documents:")
for result in results:
    print(f"Rank {result['rank']}: Document Index: {result['index']}, Similarity Score: {result['score']}")
    print(f"Text: {result['text']}")
    print("-" * 50)


# In[ ]:




