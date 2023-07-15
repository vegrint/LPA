import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
from sklearn.semi_supervised import LabelPropagation
import keras
from sklearn.metrics import log_loss, f1_score
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import time
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# 加载数据集,划分训练集和测试集
labelled_df = pd.read_csv('data/labelled.csv')
# labelled_df = pd.read_csv('data/labelled_half.csv')
unlabelled_df = pd.read_csv('data/unlabelled.csv')

X_labelled = labelled_df['cve']
y_labelled = labelled_df['att_type']

att_types = ['Remote attacker', 'Local attacker', 'Authenticated user', 'Physically proximate attacker']

# 将y_labelled中的字符串转换为对应的漏洞类型序号
y_labelled_codes = pd.factorize(y_labelled, sort=True)[0]

X_unlabelled = unlabelled_df['cve']

model_w2v = Word2Vec.load('model/cve_w2v')
model_w2v_size = 100

X_train_labelled, X_test_labelled, y_train_labelled, y_test_labelled = train_test_split(
    X_labelled, y_labelled_codes, test_size=0.2, stratify=y_labelled, random_state=1710
)
X_train_labelled = X_train_labelled.reset_index(drop=True)
X_test_labelled = X_test_labelled.reset_index(drop=True)
X_unlabelled_1, X_unlabelled_2 = train_test_split(
    X_unlabelled, test_size=0.01, random_state=702
)
X_unlabelled_1 = X_unlabelled_1.reset_index(drop=True)
X_unlabelled_2 = X_unlabelled_2.reset_index(drop=True)

def word2vec_transform(text, model, size):
    text = text.apply(lambda x: [word.lower() for word in re.findall(r'\w+', x) if word.lower() in model.wv.key_to_index])
    text = text.apply(lambda x: np.mean([model.wv.get_vector(word) for word in x], axis=0) if len(x) > 0 else np.zeros(size))
    text = np.vstack(text)
    text = text / np.linalg.norm(text, axis=1)[:, np.newaxis]
    return text

X_train_labelled_trans = word2vec_transform(X_train_labelled, model_w2v, model_w2v_size)
X_test_labelled_trans = word2vec_transform(X_test_labelled, model_w2v, model_w2v_size)
X_unlabelled_1_trans = word2vec_transform(X_unlabelled_1, model_w2v, model_w2v_size)
X_unlabelled_2_trans = word2vec_transform(X_unlabelled_2, model_w2v, model_w2v_size)
X_total_trans = np.concatenate((X_train_labelled_trans, X_unlabelled_2_trans), axis=0)

y_train_labelled_con = np.concatenate((y_train_labelled, [-1 for i in range(len(X_unlabelled_2))]))

start_time = time.time()

lp_att_type = LabelPropagation(max_iter=500, kernel="rbf", n_neighbors=4, gamma=100)
lp_att_type.fit(X_total_trans, y_train_labelled_con)


# 基于相似度去噪
# 将数据分到不同的桶中
n_buckets = int(len(X_total_trans)/3)
bucket_indices = KMeans(n_clusters=n_buckets, random_state=2).fit_predict(X_total_trans)

# 判定噪声数据
for bucket_idx in range(n_buckets):
    indices = np.where(bucket_indices == bucket_idx)[0]
    if len(indices) == 0:
        continue
    cluster_counts = {}
    for i in indices:
        cluster = lp_att_type.predict([X_total_trans[i]])[0]
        if cluster not in cluster_counts:
            cluster_counts[cluster] = 0
        cluster_counts[cluster] += 1
    if len(cluster_counts) == 1:
        continue
    counts = list(cluster_counts.values())
    sorted_counts = sorted(counts, reverse=True)
    num = 0
    top_types = []
    while num < len(indices)/1.5:
        len_orig = len(sorted_counts)
        largest_count = sorted_counts[0]
        for k, v in cluster_counts.items():
            if v == largest_count:
                top_types.append(k)
        sorted_counts = [x for x in sorted_counts if x != largest_count]
        num += largest_count * (len_orig - len(sorted_counts))
    for i in indices:
        if lp_att_type.predict([X_total_trans[i]])[0] not in top_types:
            lp_att_type.transduction_[i] = -1

# 找到并删除噪声样本
noise_idx = np.where(lp_att_type.transduction_ == -1)[0]
X_total_trans_cleaned = np.delete(X_total_trans, noise_idx, axis=0)
y_train_labelled_cleaned = np.delete(y_train_labelled_con, noise_idx)

# 重新训练标签传播模型
lp_att_type = LabelPropagation(max_iter=100, kernel="rbf", n_neighbors=4, gamma=100)
lp_att_type.fit(X_total_trans_cleaned, y_train_labelled_cleaned)


# 第二部分
threshold = 0.5
similarity_matrix = X_unlabelled_1_trans.dot(X_total_trans_cleaned.T)
most_similar_indices = np.argsort(similarity_matrix, axis=1)[:, -5:]
drop_indices = []
for i, indices in enumerate(most_similar_indices):
    # 基于阈值去噪
    similar_indices = indices[similarity_matrix[i, indices] > threshold]
    # similar_indices = indices[similarity_matrix[i, indices]]
    label_counts = np.bincount(lp_att_type.transduction_[similar_indices])
    if len(label_counts) == 0 :
        drop_indices.append(i+len(X_total_trans))
        most_probable_label = -1
    else:
        most_probable_label = np.argmax(label_counts)
    if len(X_total_trans) + i >= len(lp_att_type.transduction_):
        lp_att_type.transduction_ = np.concatenate(
            (lp_att_type.transduction_, [-1 for i in range(len(X_unlabelled_1))]))
    lp_att_type.transduction_[len(X_total_trans) + i] = most_probable_label

# 从特征矩阵中删除
X_unlabelled_1_trans = np.delete(X_unlabelled_1_trans, drop_indices, axis=0)
lp_att_type.transduction_ = np.delete(lp_att_type.transduction_, drop_indices)

y_test_pred = lp_att_type.predict(X_test_labelled_trans)

print(f1_score(y_test_labelled, y_test_pred, average='macro'))
print(f1_score(y_test_labelled, y_test_pred, average='micro'))

end_time = time.time()
print(end_time-start_time)