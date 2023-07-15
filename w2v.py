from string import punctuation
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import re
from gensim.models import word2vec
from gensim.models import Word2Vec

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

with open('data/allitems_processed.csv', 'r', encoding='utf-8') as f:
    content = f.read()

# 去除标点符号、停用词、数字，并进行词形还原
transtab = str.maketrans({key: ' ' for key in punctuation})
data = content.translate(transtab).replace('\n', ' ')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

tokens = word_tokenize(data.lower())
data = re.sub(r'\d+', '', data)

pos_tags = pos_tag(tokens)
lemmatized_tokens = []
for word, tag in pos_tags:
    if word not in stop_words:
        if tag.startswith('V'):
            lemmatized_tokens.append(lemmatizer.lemmatize(word, pos='v'))
        else:
            lemmatized_tokens.append(lemmatizer.lemmatize(word))

with open('data/allitems_processed.txt', 'w', encoding='utf-8') as f:
    f.write(' '.join(lemmatized_tokens))

# 训练词向量模型
num_features = 100    # 词向量维度
min_word_count = 10   # 最小词频
num_workers = 16      # 并行线程数
context = 5           # 上下文窗口大小
downsampling = 1e-3   # 高频词下采样阈值

sentences = word2vec.Text8Corpus('data/allitems_processed.txt')
model = word2vec.Word2Vec(sentences, workers=num_workers, vector_size=num_features, \
                          min_count = min_word_count, window = context, sg = 1, \
                          sample = downsampling)
model.save('model/cve_w2v')