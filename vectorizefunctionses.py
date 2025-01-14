#!/usr/bin/env python
# coding: utf-8

# In[39]:


#!pip install scikit-learn


# In[1]:


import pickle


# In[2]:


# 验证保存
with open('processed_text.pkl', 'rb') as file:
    processed_docs = pickle.load(file)


# In[3]:


#print("processed_docs:", processed_docs[:6])


# In[4]:


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


# In[5]:


# 调用函数生成 TF-IDF 矩阵和特征名称
tfidf_matrix, feature_names = generate_tfidf_matrix(processed_docs, min_df=2)

# 输出矩阵形状
print("TF-IDF Matrix Shape:", tfidf_matrix.shape)

# 输出特征名称
print("Feature Names:", feature_names)


# In[6]:


#!pip install nltk


# In[7]:


# 导入自然语言处理工具包
import nltk
from nltk.stem import PorterStemmer  # 导入词干提取器
from nltk.stem import WordNetLemmatizer  # 导入词形还原工具
from nltk.corpus import words, stopwords, names  # 导入单词、停用词和名称库

# 下载相关数据集（如未安装）
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')


# In[8]:


from processingfunction import preprocess_paragraphs


# In[26]:


import pickle

# 从文件中加载列表 Paras
with open("paras.pkl", "rb") as file:  # "rb" 表示以二进制读取模式打开文件
    loaded_paras = pickle.load(file)


# In[10]:


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
    return results, top_terms, query_vector


# In[17]:


# 初始化 TF-IDF 向量化器
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([doc["paragraph"].lower() for doc in loaded_paras])

# 调用函数
query = "second-hand fashion"
results, top_terms, query_vector = get_top_similar_texts(query, vectorizer, tfidf_matrix, loaded_paras, top_n=10, top_terms_n=20)

# 打印结果
print("Top Terms:")
print(", ".join(top_terms))  # 打印 top terms 列表

print("\nTop Similar Documents:")
for result in results:
    print(f"Rank {result['rank']}: Document Index: {result['index']}, Similarity Score: {result['score']}")
    print(f"Text: {result['text']}")
    print("-" * 50)


# In[18]:


import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

def find_most_similar_document_with_reduction(tfidf_matrix, query_vector, loaded_paras, n_components=100):
    """
    使用 Truncated SVD 对 TF-IDF 矩阵和查询向量降维，并找到最相似的文档。

    参数:
        tfidf_matrix (sparse matrix): 文档的 TF-IDF 矩阵。
        query_vector (sparse matrix): 查询文本的 TF-IDF 向量。
        loaded_paras (list): 原始文档列表。
        n_components (int): 降维的目标维度数，默认值为 100。

    返回:
        nearest_neighbor_index (int): 最相似文档的索引。
        similarity_score (float): 最相似文档的余弦相似度分数。
        nearest_document (str): 最相似文档的内容。
    """
    # 初始化 SVD 降维器
    svd = TruncatedSVD(n_components=n_components, algorithm='randomized')
    
    # 对 TF-IDF 矩阵进行降维
    reduced_matrix = svd.fit_transform(tfidf_matrix)
    
    # 对查询向量进行降维
    reduced_query_vec = svd.transform(query_vector)
    
    # 计算降维后的余弦相似度
    similarities = cosine_similarity(reduced_query_vec, reduced_matrix)
    
    # 找到降维后最相似的文档索引
    nearest_neighbor_index = similarities.argmax()
    similarity_score = similarities[0, nearest_neighbor_index]
    
    # 获取最相似文档的内容
    nearest_document = loaded_paras[nearest_neighbor_index]
    
    return nearest_neighbor_index, similarity_score, nearest_document


# In[19]:


# 调用函数
nearest_neighbor_index, similarity_score, nearest_document = find_most_similar_document_with_reduction(
    tfidf_matrix=tfidf_matrix,
    query_vector=query_vector,
    loaded_paras=loaded_paras,
    n_components=100
)

# 打印结果
print(f"Most similar document index: {nearest_neighbor_index}, Similarity: {similarity_score}")
print("Most similar document content:")
print(nearest_document)


# In[20]:


import nltk
nltk.download('punkt')


# In[27]:


from gensim.models import Word2Vec  # 导入 Word2Vec 模型
from nltk.tokenize import word_tokenize  # 导入分词工具

def process_word2vec_model(loaded_paras, target_word, vector_size=50, window=3, min_count=1, topn=3):
    """
    使用 Word2Vec 模型处理文本并获取目标词的向量和相似词。

    参数:
        loaded_paras (list): 文本数据列表，每个元素包含一个段落。
        target_word (str): 目标词。
        vector_size (int): 向量的维度大小，默认值为 50。
        window (int): 窗口大小，默认值为 3。
        min_count (int): 词出现的最小次数，默认值为 1。
        topn (int): 与目标词最相似的词个数，默认值为 3。

    返回:
        dict: 包含目标词向量和相似词信息的字典。
    """
    # 对段落列表进行分词
    tokenized_sentences = [
        word_tokenize(sentence['paragraph'].lower()) for sentence in loaded_paras
    ]
    
    # 初始化 Word2Vec 模型
    w2v_model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4
    )
    
    result = {}
    try:
        # 获取目标词的向量
        word_vector = w2v_model.wv[target_word]
        result['word_vector'] = word_vector
        
        # 获取与目标词最相似的词
        similar_words = w2v_model.wv.most_similar(target_word, topn=topn)
        result['similar_words'] = similar_words
    except KeyError:
        result['error'] = f"{target_word} is not in the vocabulary"
    
    return result


# In[28]:


# 目标词
target_word = 'fashion'

# 调用函数
result = process_word2vec_model(loaded_paras, target_word)

# 打印结果
if 'error' in result:
    print(result['error'])
else:
    print(f"Vector of '{target_word}': {result['word_vector']}")
    print(f"Most similar to '{target_word}': {result['similar_words']}")


# In[38]:


#!pip install -U FlagEmbedding


# In[39]:


#!pip install ipywidgets


# In[40]:


from FlagEmbedding import FlagModel

def find_most_relevant_paragraph(query, loaded_paras, model_name='BAAI/bge-large-en-v1.5', device='cuda'):
    """
    使用指定的FlagEmbedding模型从段落中找到与查询最相关的段落。

    参数：
        query (str): 查询语句。
        loaded_paras (list): 包含段落字典的列表，每个字典包含键 'paragraph'。
        model_name (str): FlagEmbedding模型名称，默认 'BAAI/bge-large-en-v1.5'。
        device (str): 使用的设备，例如 'cuda' 或 'cpu'，默认 'cuda'。

    返回：
        dict: 与查询最相关的段落内容和索引。
    """
    # 加载模型
    model = FlagModel(
        model_name,
        device=device,
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages:"
    )
    
    # 对段落生成嵌入向量
    embeddings = model.encode([p['paragraph'] for p in loaded_paras])
    
    # 对查询生成嵌入向量
    query_embedding = model.encode_queries([query])
    
    # 计算相似度并找到最相关的段落索引
    similarities = embeddings @ query_embedding.T
    top_index = similarities.argmax()
    
    # 返回结果
    most_relevant_paragraph = loaded_paras[top_index]
    return {
        'index': top_index,
        'paragraph': most_relevant_paragraph
    }


# In[41]:


# 示例使用
query = 'second-hand fashion'
result = find_most_relevant_paragraph(query, loaded_paras)

print(f"Most relevant paragraph index: {result['index']}")
print(f"Most relevant paragraph: {result['paragraph']}")


# In[ ]:




