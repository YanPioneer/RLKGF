import ast

import torch
import os


class KGADJ:

    def __init__(self, device, dis_num, sym_num, data_path):

        self.disease_pre = []
        self.sys_pre = []
        self.dis_sys_pre = {}
        self.device = device
        self.dis_num = dis_num
        self.sym_num = sym_num
        self.kg_matrix = torch.zeros(dis_num + sym_num, dis_num + sym_num).to(self.device)  # 构建邻接矩阵
        # print(self.kg_matrix)
        # print(self.kg_matrix.size())

        # self.initialize_adj()

        with open(os.path.join(data_path, 'med_d_disease.txt'), 'r', encoding='utf-8') as d:
            content = d.readlines()
            for line in content:
                info = line.strip().split('\t')
                self.disease_pre.append(info[0])
        # print(self.disease_pre)

        with open(os.path.join(data_path, 'med_d_symptoms.txt'), 'r', encoding='utf-8') as d:
            content = d.readlines()
            for line in content:
                info = line.strip().split('\t')
                self.sys_pre.append(info[0])
        # print(self.sys_pre)

        # with open(os.path.join(data_path, 'dise_sym_num_dict.txt'), 'r', encoding='utf-8') as f:
        with open(os.path.join(data_path, 'dise_sym_num_high_dict(≥3).txt'), 'r', encoding='utf-8') as f:
            content = f.readlines()
            self.dis_sys_pre = ast.literal_eval(content[0])
            # print(self.dis_sys_pre)

        with open(os.path.join(data_path, 'dise_sym_num_dict.txt'), 'r', encoding='utf-8') as f:
            content = f.readlines()
            self.dis_sys_all_pre = ast.literal_eval(content[0])
            # print(self.dis_sys_all_pre)

        self.disease_ = self.disease_pre
        self.sys_ = self.sys_pre
        self.dis_sys_ = self.dis_sys_pre

    def initialize_adj(self):
        self.kg_matrix = torch.zeros(self.dis_num + self.sym_num, self.dis_num + self.sym_num).to(self.device)  # 构建邻接矩阵
        # 构建上三角、下三角
        for i in range(len(self.disease_pre)):
            for j in range(len(self.sys_pre)):
                if self.sys_pre[j] in list(self.dis_sys_pre[self.disease_pre[i]].keys()):
                    # print(self.sys_pre[j], self.disease_pre[i])
                    self.kg_matrix[i][j + len(self.disease_pre)] = 1
                    self.kg_matrix[j + len(self.disease_pre)][i] = 1

        # print(self.kg_matrix.size()[0])
        # for i in range(self.kg_matrix.size()[0]):
        #     print(self.kg_matrix[i])

    def update_adj(self, confirm_symptoms):  # 根据患者存在的症状进行疾病排除
        del_dis = []
        if len(confirm_symptoms) > 0:  # 确定要删除的疾病
            for sym in confirm_symptoms:
                for d in list(self.dis_sys_all_pre.keys()):
                    if sym not in list(self.dis_sys_all_pre[d].keys()):
                        del_dis.append(d)

        for d_ in del_dis:
            # 更改索引
            self.kg_matrix[self.disease_pre.index(d_), :] = 0
            self.kg_matrix[:, self.disease_pre.index(d_)] = 0
            # self.kg_matrix[self.disease_pre.index(d_), self.disease_pre.index(d_)] = 1


