#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import pickle

# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from tqdm import tqdm
from metrics import hit_at_k, ndcg_at_k, MRR, HRF
from util import parse_time


class BetaIntersection(nn.Module):
    def __init__(self, dim):
        super(BetaIntersection, self).__init__()
        self.dim = dim  #实体嵌入的维度
        # 定义两层神经网络，用于计算注意力权重
        self.layer1 = nn.Linear(2 * self.dim, 2 * self.dim)  #第一层：输入是α+β（2*dim），输出2*dim
        self.layer2 = nn.Linear(2 * self.dim, self.dim)  #第二层：输入2*dim，输出dim（与α/β维度一致）
        # 初始化权重（确保网络各层的输入和输出方差一致，避免训练时梯度异常。）
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    # 实现了“多路径分布→注意力权重→融合分布”的完整逻辑
    def forward(self, alpha_embeddings, beta_embeddings):
        all_embeddings = torch.cat([alpha_embeddings, beta_embeddings], dim=-1)  #a/b形状：[num_paths, batch_size, dim]。最后一层all_embeddings，形状为 [num_paths, batch_size, 2*dim]
        #第一层：对拼接后的嵌入做线性变换，再用ReLU激活，提取路径的特征
        layer1_act = F.relu(self.layer1(all_embeddings)) #形状：[num_paths, batch_size, 2*dim]
        #第二层：神经网络+Softmax，计算权重：所有路径的权重之和为1，得到每条路径的注意力权重
        attention = F.softmax(self.layer2(layer1_act), dim=0) #形状：[num_paths, batch_size, dim]
        #将每条路径的α与对应的注意力权重相乘，再在路径维度上求和
        alpha_embedding = torch.sum(attention * alpha_embeddings, dim=0) #[batch_size, dim]
        beta_embedding = torch.sum(attention * beta_embeddings, dim=0)
        #最终输出融合后的α和β，组成一个新的Beta分布，代表多条路径的交集结果
        return alpha_embedding, beta_embedding

class BetaProjection(nn.Module):
    def __init__(self, entity_dim, relation_dim, hidden_dim, projection_regularizer, num_layers):
        super(BetaProjection, self).__init__()
        self.entity_dim = entity_dim  #用于编码实体的固有特征
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim  #用于神经网络中间层的特征转换，决定了实体与关系交互时的特征提取能力，最终会映射回entity_dim
        self.num_layers = num_layers
        self.layer1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim) #接收“实体嵌入+关系嵌入”的拼接向量，输出到隐藏层
        self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim)  #将隐藏层特征映射回实体嵌入维度
        for nl in range(2, num_layers + 1):  #动态添加中间层（输入和输出都是hidden_dim）提取复杂特征
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight) #初始化所有层的权重(Xavier初始化：避免训练时梯度消失或爆炸)
        self.projection_regularizer = projection_regularizer  #正则化器（确保输出为正）

    def forward(self, e_embedding, r_embedding):
        x = torch.cat([e_embedding, r_embedding], dim=-1)  #[batch_size, entity_dim + relation_dim]

        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))  #每层先做线性变换,再用ReLU激活函数  ReLU的作用：引入非线性，让网络能学习实体与关系之间的复杂交互模式（如多步推理规则）
        x = self.layer0(x)
        x = self.projection_regularizer(x) #调用Regularizer类，通过“偏移+截断”确保输出值为正，满足Beta分布的数学要求（α>0，β>0）

        return x #新实体嵌入[batch_size, entity_dim]

class Regularizer():  #用torch.clamp（截断函数）把参数限制在[min_val, max_val]范围内，同时加一个基础值base_add确保参数为正。
    def __init__(self, base_add, min_val, max_val):
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, entity_embedding):
        return torch.clamp(entity_embedding + self.base_add, self.min_val, self.max_val)

class KGReasoning(nn.Module):
    # 初始化实体/关系嵌入、模型参数和核心组件
    def __init__(self, nentity, nrelation, hidden_dim, gamma,
                 test_batch_size=1, use_cuda=False,
                 query_name_dict=None, beta_mode=None):
        super(KGReasoning, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.use_cuda = use_cuda
        self.batch_entity_range = torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1).cuda() if self.use_cuda else torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1) # used in test_step
        self.query_name_dict = query_name_dict

        self.gamma = nn.Parameter( #初始化gamma参数 不可训练
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter( #计算嵌入的初始化范围：确保嵌入值在合理区间，避免梯度异常
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim
        #实体嵌入初始化（Beta分布需要α和β，因此维度是2*hidden_dim）
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim * 2))
        #实体正则化器：确保α和β>0
        self.entity_regularizer = Regularizer(1, 0.05, 1e9)
        #投影模块的正则化器（与实体正则化器一致）
        self.projection_regularizer = Regularizer(1, 0.05, 1e9)

        # 均匀初始化实体嵌入：范围 [-embedding_range, embedding_range]
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        #关系嵌入初始化（关系用固定向量表示，无需Beta分布，维度=hidden_dim）
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))  # [1, 128]
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        #加载核心组件（BetaIntersection和BetaProjection）
        hidden_dim, num_layers = beta_mode
        self.center_net = BetaIntersection(self.entity_dim)
        self.projection_net = BetaProjection(self.entity_dim * 2, #实体嵌入是α+β，维度=2*hidden_dim
                                             self.relation_dim, #关系嵌入维度
                                             hidden_dim,
                                             self.projection_regularizer,
                                             num_layers)


    # 接收批量查询和正负样本，调用embed_query生成查询分布，再调用cal_logit计算正负样本的得分，最终输出结果供训练/测试使用。
    def forward(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_idxs, all_alpha_embeddings, all_beta_embeddings = [], [], []
        for query_structure in batch_queries_dict:
            #按查询结构分组生成查询分布
            alpha_embedding, beta_embedding, _ = self.embed_query(batch_queries_dict[query_structure], #调用embed_query生成当前结构的查询分布
                                                                        query_structure, #当前结构的所有查询
                                                                        0)
            all_idxs.extend(batch_idxs_dict[query_structure])
            all_alpha_embeddings.append(alpha_embedding)  #收集所有查询的α
            all_beta_embeddings.append(beta_embedding)  #收集所有查询的β
        # 拼接查询分布，构建Beta分布对象
        if len(all_alpha_embeddings) > 0:
            all_alpha_embeddings = torch.cat(all_alpha_embeddings, dim=0).unsqueeze(1)
            all_beta_embeddings = torch.cat(all_beta_embeddings, dim=0).unsqueeze(1)
            all_dists = torch.distributions.beta.Beta(all_alpha_embeddings, all_beta_embeddings)

        # 处理采样权重（平衡不同查询的贡献，避免答案多的查询权重过高）
        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs]
        # 计算正样本得分（正确答案的得分）
        if type(positive_sample) != type(None):
            if len(all_alpha_embeddings) > 0:
                #按原始索引对齐正样本，获取正样本嵌入并正则化
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1))
                #计算正样本得分
                positive_logit = self.cal_logit(positive_embedding, all_dists)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)
        else:
            positive_logit = None
        # 计算负样本得分（错误答案的得分）
        if type(negative_sample) != type(None):
            if len(all_alpha_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]  # [batch,neg_size]

                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=negative_sample_regular.view(-1)).view(batch_size, negative_size, -1))

                negative_logit = self.cal_logit(negative_embedding, all_dists)

            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs

    def embed_query(self, queries, query_structure, idx):  #将查询结构转换为模型能处理的概率分布
        '''
        Iterative embed a batch of queries with same structure using BetaE
        queries: a flattened batch of queries
        '''
        all_relation_flag = True   # 判断是否为“实体+关系”传递结构
        for ele in query_structure[-1]:
            if ele not in ['r', 'n', 'h','s','w','sub']:
                all_relation_flag = False
                break
        if all_relation_flag:  #处理“实体+关系”传递结构
            if query_structure[0] == 'e': #起始实体是单个实体
                #从查询中取实体ID，获取实体嵌入并正则化（确保α, β>0）
                embedding = self.entity_regularizer(
                    torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])) #根据实体id提取预训练的实体嵌入
                idx += 1  #索引后移，准备处理下一个元素（关系或操作符）
            else:#起始非单个实体
                if query_structure[-1] == ('n',): #操作符是交集（'n'），需要融合多个子查询的结果
                    alpha_embedding_list = []
                    beta_embedding_list = []
                    for i in range(len(query_structure) - 1): #排除最后一个操作符
                        alpha_embedding, beta_embedding, idx = self.embed_query(queries, query_structure[i], idx)
                        alpha_embedding_list.append(alpha_embedding)
                        beta_embedding_list.append(beta_embedding)
                    #调用BetaIntersection融合所有子查询的分布（取交集）
                    alpha_embedding, beta_embedding = self.center_net(torch.stack(alpha_embedding_list),
                                                                      torch.stack(beta_embedding_list))
                    embedding = torch.cat([alpha_embedding, beta_embedding], dim=-1)

                else: #操作符不是交集，直接处理单个子查询
                    alpha_embedding, beta_embedding, idx = self.embed_query(queries, query_structure[0], idx)
                    embedding = torch.cat([alpha_embedding, beta_embedding], dim=-1)
            for i in range(len(query_structure[-1])): #处理操作符
                if query_structure[-1][i] == 'n':
                    assert (queries[:, idx] == -2).all() #-2为补集的标记id
                    embedding = 1. / embedding
                elif query_structure[-1][i] == 'h': #占位符，仅用于结构对齐
                    assert (queries[:, idx] == -3).all()
                else:  #关系传递（op=='r'）：调用投影模块
                    #提取关系嵌入
                    r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                    #实体嵌入+关系嵌入→新实体嵌入（通过BetaProjection）
                    embedding = self.projection_net(embedding, r_embedding)
                idx += 1
            #经过上述处理后，embedding是包含α和β的拼接向量（维度2*hidden_dim）
            #用torch.chunk按最后一维拆分，得到查询分布的α和β参数
            alpha_embedding, beta_embedding = torch.chunk(embedding, 2, dim=-1)

        else: #处理纯交集结构

            alpha_embedding_list = []
            beta_embedding_list = []
            for i in range(len(query_structure)): #递归处理每个子结构（如(e1, e2, e3)的交集）
                alpha_embedding, beta_embedding, idx = self.embed_query(queries, query_structure[i], idx)
                alpha_embedding_list.append(alpha_embedding)
                beta_embedding_list.append(beta_embedding)
            #融合所有子结构的分布（取交集）
            alpha_embedding, beta_embedding = self.center_net(torch.stack(alpha_embedding_list),
                                                              torch.stack(beta_embedding_list))
        #最终返回查询对应的Beta分布参数（α,β）和更新后的索引idx（供外层递归使用）
        return alpha_embedding, beta_embedding, idx

    def cal_logit(self, entity_embedding, query_dist): #计算“候选实体分布”与“查询分布”的相似度，得到推理得分

        alpha_embedding, beta_embedding = torch.chunk(entity_embedding, 2, dim=-1)
        entity_dist = torch.distributions.beta.Beta(alpha_embedding, beta_embedding)
        # 计算最终得分：gamma - L1范数(KL散度)（确保得分在[0, gamma]之间）
        logit = self.gamma - torch.norm(torch.distributions.kl.kl_divergence(entity_dist, query_dist), p=1, dim=-1)
        return logit



    @staticmethod
    # 通过优化模型参数，让模型学会区分“符合查询的实体”（正样本）和“不符合查询的实体”（负样本）
        #从训练数据中获取一批样本；
        #让模型对这批样本进行预测；
        #计算预测结果与真实标签的差异（损失）；
        #通过反向传播更新模型参数，减少损失。
    def train_step(model, optimizer, train_iterator, args):
        model.train()  #将模型设置为训练模式
        optimizer.zero_grad()  #清空优化器中所有参数的梯度
        #从训练迭代器中获取一批数据
        positive_sample, negative_sample, subsampling_weight, batch_queries, query_structures = next(train_iterator)

        #用字典按查询结构分组（key=结构，value=该结构的所有查询） 不同结构的查询需要用不同逻辑处理，分组后可提高效率
        batch_queries_dict = collections.defaultdict(list)
        batch_idxs_dict = collections.defaultdict(list)  #记录原始索引，用于对齐样本
        for i, query in enumerate(batch_queries):
            # print(i,query)
            batch_queries_dict[query_structures[i]].append(query)  #按结构分组查询
            batch_idxs_dict[query_structures[i]].append(i)   #记录每个查询的原始索引

        # 将分组后的查询转换为LongTensor（PyTorch中处理索引的常用类型）
        for query_structure in batch_queries_dict:
            if args.cuda: #若使用GPU，将数据移动到显存
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
            else:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
        if args.cuda: #移动正负样本和采样权重到GPU
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()
        # 调用模型的forward方法，得到正样本和负样本的推理得分
        positive_logit, negative_logit, subsampling_weight, _ = model(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)


        negative_score = F.logsigmoid(-negative_logit).mean(dim=1)  #负样本损失：鼓励负样本得分尽可能低
        positive_score = F.logsigmoid(positive_logit).squeeze(dim=1) #正样本损失：鼓励正样本得分尽可能高
        # 加权计算总损失（用subsampling_weight平衡不同查询的权重）
        positive_sample_loss = - (subsampling_weight * positive_score).sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()
        # 除以权重总和，归一化损失（避免样本数量影响损失大小）
        positive_sample_loss /= subsampling_weight.sum()
        negative_sample_loss /= subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2 #总损失为正负样本损失的平均
        loss.backward() #反向传播，计算所有可训练参数的梯度
        optimizer.step() #根据梯度更新模型参数
        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
        }
        return log

    #在训练完成后（或训练过程中）验证模型性能
        #无梯度推理：避免测试时更新模型参数，仅用训练好的参数做预测；
        #结果排序：对模型输出的实体得分排序，生成推荐列表；
        #指标计算：用行业标准指标（HIT@20、NDCG@20等）量化模型性能；
        #结果记录：关联“查询实体→推荐列表→真实答案”，方便后续分析。
    @staticmethod
    def test_step(model, answers, args, test_dataloader, query_name_dict, save_result=False, save_str="", save_empty=False):
        model.eval() #将模型设为测试模式,（关闭dropout、固定批归一化参数）

        step = 0
        total_steps = len(test_dataloader)
        logs = collections.defaultdict(list)  #按查询结构存储评估指标
        dict = {}

        rank_list_all = [] # 存放推荐结果
        hard_answer_all = [] # 存放真实结果
        li_all = []  # 存放查询集
        score_all = []  # 存放分数


        with torch.no_grad(): #禁用梯度计算，减少内存占用并加速推理
            for positive_sample,negative_sample,subsampling_weight, queries, queries_unflatten, query_structures in tqdm(test_dataloader, disable=not args.print_on_screen):
                # print(type(negative_sample))

                # 按查询结构分组
                batch_queries_dict = collections.defaultdict(list)
                batch_idxs_dict = collections.defaultdict(list)
                for i, query in enumerate(queries):
                    batch_queries_dict[query_structures[i]].append(query)
                    batch_idxs_dict[query_structures[i]].append(i)
                # 将查询转换为LongTensor并移动到GPU
                for query_structure in batch_queries_dict:
                    if args.cuda:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
                    else:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
                # 移动正负样本、采样权重到GPU
                if args.cuda:
                    negative_sample = negative_sample.cuda()
                    positive_sample = positive_sample.cuda()
                    subsampling_weight = subsampling_weight.cuda()
                positive_logit, negative_logit, subsampling_weight, idxs = model(positive_sample, negative_sample,
                                                                              subsampling_weight, batch_queries_dict,
                                                                              batch_idxs_dict)
                # 计算负样本损失（测试阶段仅用于监控，不反向传播）
                negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
                negative_sample_loss = - (subsampling_weight * negative_score).sum()
                negative_sample_loss /= subsampling_weight.sum()
                loss = negative_sample_loss

                # 按原始索引对齐未展平的查询和查询结构（确保后续处理的查询与得分对应）
                queries_unflatten = [queries_unflatten[i] for i in idxs]
                query_structures = [query_structures[i] for i in idxs]
                # 对负样本得分按“降序”排序（得分越高，模型越认为该实体符合查询）
                argsort = torch.argsort(negative_logit, dim=1, descending=True)
                ranking_list = argsort[0].tolist() #提取第一个查询的推荐实体ID
                score = negative_logit[0].tolist()  #提取第一个查询的实体得分，并按降序排序（与推荐列表对应）
                score = sorted(score, reverse=True)

                # 遍历每个查询，处理“查询实体→推荐列表→真实答案”的关联
                for idx, (i, query, query_structure) in enumerate(zip(argsort[:, 0], queries_unflatten, query_structures)):
                    # 获取当前查询的真实答案（从answers字典中根据查询key获取）
                    hard_answer = answers[query]
                    num_hard = len(hard_answer)

                    # 提取查询实体（根据查询结构的嵌套层数，提取输入实体ID）
                    if len(query) == 2:
                        if len(query[0]) == 3:
                            li = [query[0][0][0]] #嵌套3层的查询，取最内层实体ID
                        else:
                            li = [query[0][0][0][0], query[1][0][0][0]] #嵌套4层的查询，取两个实体ID
                    else:
                        li = [query[0][0][0][0], query[1][0][0][0], query[2][0][0][0]] #3个输入实体的查询

                    # 强+弱
                    # if len(query) == 2:
                    #     li = [query[0][0][0], query[1][0][0]]
                    # elif len(query) == 3:
                    #     if isinstance(query[0][0], int):
                    #         li = [query[0][0]]
                    #     else:
                    #         li = [query[0][0][0], query[1][0][0], query[2][0][0]]

                    # 强，弱
                    # if len(query) == 2:
                    #     if isinstance(query[0] , int):
                    #         li = [query[0]]
                    #     else:
                    #         li = [query[0][0] , query[1][0]]
                    # else:
                    #     li = [query[0][0], query[1][0], query[2][0]]

                    # 强/弱+非替补
                    # if len(query) == 2:
                    #     if isinstance(query[0][0] , int):
                    #         li = [query[0][0]]
                    #     else:
                    #         li = [query[0][0][0], query[1][0][0]]
                    # else:
                    #     li = [query[0][0][0], query[1][0][0], query[2][0][0]]

                    li_all.append(li)
                    new_ranking_list = []
                    new_hard_answer = []
                    new_score = []

                    for v in ranking_list:
                        if v not in li:
                            new_ranking_list.append(v)

                    for v in hard_answer:
                        if v not in li:
                            new_hard_answer.append(v)

                    for i , v in enumerate(ranking_list):
                        if v not in li:
                            new_score.append(score[i])

                    dict[query] = new_ranking_list  # 存储当前查询的推荐列表
                    h20 = hit_at_k(new_hard_answer, new_ranking_list, 20)
                    ndcg20 = ndcg_at_k(new_hard_answer, new_ranking_list, 20)
                    mrr20 = MRR(new_hard_answer, new_ranking_list, 20)
                    # sd20 = SD(li, new_ranking_list, 20)
                    hrf20 = HRF(new_hard_answer, new_ranking_list, 20)
                    # sd20, sub_list = SD(li, new_ranking_list, 20)
                    # hrf20, weak_list = HRF(new_hard_answer, new_ranking_list, 20)

                    # print("******************************")
                    # print("查询集", li)
                    # print("推荐答案", new_ranking_list[: 20])
                    # print("真实答案", new_hard_answer)
                    # print("弱互补", weak_list)
                    # print("替补", sub_list)
                    # print(f"HIT@20: {h20}")
                    # print(f"NDCG@20: {ndcg20}")
                    # print(f"MRR@20: {mrr20}")
                    # print(f"SD@20: {sd20}")
                    # print(f"HRF@20: {hrf20}")

                    # 按查询结构存储指标（同一结构的查询指标累加，后续求平均）
                    if query_structure not in logs:
                        logs[query_structure].append({
                            'HIT@20': h20,
                            'NDCG@20': ndcg20,
                            'MRR@20': mrr20,
                            # 'SD@20': sd20,
                            'HRF@20': hrf20,
                            'loss': loss,
                            'num_queries':1,
                            'num_hard_answer': num_hard,
                        })
                    else:
                        # 累加同结构查询的指标
                        logs[query_structure][0]['HIT@20'] += h20
                        logs[query_structure][0]['NDCG@20'] += ndcg20
                        logs[query_structure][0]['MRR@20'] += mrr20
                        # logs[query_structure][0]['SD@20'] += sd20
                        logs[query_structure][0]['HRF@20'] += hrf20

                        logs[query_structure][0]['loss'] += loss
                        logs[query_structure][0]['num_queries'] += 1

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1

        # 计算最终指标（按查询结构求平均）
        metrics = collections.defaultdict(lambda: collections.defaultdict(int)) #初始化最终指标字典
        for query_structure in logs:
            metrics[query_structure]['num_queries'] = logs[query_structure][0]['num_queries'] #提取当前结构的总查询数
            # 对每个指标求平均
            for metric in logs[query_structure][0].keys():
                if metric in ['num_hard_answer','num_queries']:
                    continue
                metrics[query_structure][metric] = logs[query_structure][0][metric]/logs[query_structure][0]['num_queries']


        return metrics
