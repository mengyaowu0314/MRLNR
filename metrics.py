import pickle

import numpy as np

def hit_at_k(ans, predicted, k):
    assert k >= 1
    for answer in ans:
        if answer in predicted[:k]:
            return 1
    return 0

def dcg_at_k(ans, method=1):
    if len(ans) == 0:
        return 0
    r = np.array(ans)
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(ans, predicted, k, method=1):
    rel = [int(pred in ans) for pred in predicted[:k]]
    dcg = dcg_at_k(rel, method)

    rel.sort(reverse=True)
    dcg_max = dcg_at_k(rel, method)
    if not dcg_max:
        return 0.
    return dcg / dcg_max


def MRR(ans, predicted,k):
    if len(ans) == 0 or len(predicted) == 0:
        return 0
    idxList = []
    for answer in ans:
        if answer not in predicted:
            firstIdx = 1000000000
        else:
            firstIdx = predicted.index(answer)
        idxList.append(firstIdx)
    idxList.sort()
    if not idxList or idxList[0] > k:
        return 0
    # print(idxList)
    return 1/(idxList[0] + 1)

# 复制文件
# path1 = "data/latest/P/pwa-api2tag.pkl"
path1 = "data/latest/H/huawei-api2tag.pkl"
f1 = open(path1, 'rb')
apiTag = pickle.load(f1)

# def SD(li, predicted,k):
#     cur = 0
#     # sub_answer = []
#     all = len(li) * k
#     for v1 in li:
#         for v2 in predicted[:k]:
#             if apiTag[v1] == apiTag[v2]:
#                 # sub_answer.append(v2)
#                 cur += 1
#     return cur / all
    # return cur / all,sub_answer

# HRF
def HRF(ans, predicted, k):
    ding = 0
    weak_answer = set()
    for answer in ans:
        for rr in predicted[:k]:
          if apiTag[answer] == apiTag[rr] and rr not in weak_answer:
              weak_answer.add(rr)
              ding += 1
    return ding / k
    # return ding / k , weak_answer




# def dcg_at_label(r, k, method=1):
#     r = r[:k]
#     r = np.array(r)
#
#     if len(r):
#         if method == 0:
#             return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
#         elif method == 1:
#             return np.sum(r / np.log2(np.arange(2, r.size + 2)))
#         else:
#             raise ValueError('method must be 0 or 1.')
#     return 0.


# def ndcg_at_label(r, k, method=1):
#     dcg_max = dcg_at_label(sorted(r[:k], reverse=True), k, method)
#     print ("dcg max: {}".format(dcg_max))
#     if not dcg_max:
#         return 0.
#     return dcg_at_label(r, k, method) / dcg_max


