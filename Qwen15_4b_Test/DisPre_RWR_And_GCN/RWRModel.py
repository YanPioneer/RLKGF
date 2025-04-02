import copy

import torch.nn as nn
import torch
import networkx as nx

from utils import dialog_config


class TemperatureSoftmax(nn.Module):
    def __init__(self, temperature=1.):
        super(TemperatureSoftmax, self).__init__()
        self.temperature = temperature

    def forward(self, x):
        x = x / self.temperature
        x = torch.softmax(x, dim=-1)
        return x


class RWR:
    def __init__(self, dis_shape, sym_shape, disease, symptoms, slot_set, dis_sym_num_to_graph, kggraph, device=dialog_config.device, alpha=0.6, temperature=0.1):
        '''
        dis_shape: 疾病数目，int
        sym_shape: 症状数目，int
        disease: 疾病列表，list
        symptoms: 症状列表，list
        slot_set: 全体疾病和症状字典，dict  {'疾病1':0, '疾病2':1, ...}
        dis_sym_num_to_graph: 疾病和对应症状次数图，dict  {'疾病名称1': {'症状1': 次数,'症状2': 次数,...},...}
        kggraph: 疾病和病症知识图谱
        '''
        self.device = device
        self.dis_shape = dis_shape
        self.sym_shape = sym_shape
        self.entity_num = dis_shape + sym_shape
        self.slot_set = slot_set
        self.dis = disease
        self.sym = symptoms
        self.nodes = self.dis + self.sym
        self.alpha = alpha
        self.dis_sym_num = dis_sym_num_to_graph
        self.temperature_softmax = TemperatureSoftmax(temperature)
        self.kgraph = kggraph
        self.kgraph.initialize_adj()  # 初始化邻接矩阵, 自动初始化
        self.initial_kg_matrix = copy.deepcopy(self.kgraph.kg_matrix).view(-1, self.entity_num, self.entity_num)
        self.kg_matrix = copy.deepcopy(self.initial_kg_matrix)


    def GetConfirm(self, goal):
        current_confirm_sym = []
        for s in goal['current_slots']['inform_slots'].keys():
            if goal['current_slots']['inform_slots'][s] == dialog_config.TRUE:
                current_confirm_sym.append(s)
        return current_confirm_sym

    def state_representation(self, g):
        state_slot = torch.full((1, self.entity_num), 0.3).to(self.device)  # 实体大小由疾病和症状组成
        current_slots = g['current_slots']
        current_slots_ = []
        for slot in current_slots['inform_slots']:
            if slot != 'disease' and slot not in self.slot_set:
                continue

            if slot == 'disease':
                state_slot[0, self.slot_set[current_slots['inform_slots']['disease']]] = 1
            else:
                state_slot[0, self.slot_set[slot]] = current_slots['inform_slots'][slot]
                current_slots_.append(slot)

        return state_slot

    def get_sym_flag(self, batch_state, total_state):
        ones = torch.ones(batch_state.size()).to(self.device)
        zeros = torch.zeros(batch_state.size()).to(self.device)
        ones_ = torch.ones(total_state.size()).to(self.device)
        zeros_ = torch.zeros(total_state.size()).to(self.device)
        return torch.where(batch_state == 0.3, ones, zeros), torch.where(total_state == 1, ones_, zeros_).unsqueeze(1)

    def score(self, goal):
        '''
        作用:获取当前goal的分数
        输入:goal
        输出:dis_,sym_,表示疾病和病症的概率分布,形状为tensor([[0.8000, 0.1000, 0.1000,...]])
        '''
        self.kgraph.initialize_adj()  # 初始化邻接矩阵, 自动初始化
        current_confirm = self.GetConfirm(goal)
        disease_mask = torch.ones(self.dis_shape).to(self.device).reshape(-1, self.dis_shape)
        symptoms_mask = torch.ones(self.sym_shape).to(self.device).reshape(-1, self.sym_shape)
        self.kgraph.update_adj(current_confirm)  # 修改图谱的邻接矩阵，根据患者自述剪枝

        kg_matrix = copy.deepcopy(self.kgraph.kg_matrix).view(-1, self.entity_num, self.entity_num)
        for i in range(kg_matrix.size(0)):
            for j in range(self.dis_shape):
                if torch.equal(kg_matrix[i, j, :],
                               torch.zeros(kg_matrix[i, j, :].size()).to(self.device)):
                    disease_mask[0][j] = 0.
            for k in range(self.sym_shape):
                if torch.equal(kg_matrix[i, k + self.dis_shape, :],
                               torch.zeros(kg_matrix[i, k + self.dis_shape, :].size()).to(
                                   self.device)):
                    symptoms_mask[0][k] = 0.
        disease_mask_ = copy.deepcopy(disease_mask)
        symptoms_mask_ = copy.deepcopy(symptoms_mask)

        state_slot = self.state_representation(goal)
        sym_flag, _ = self.get_sym_flag(state_slot[:, self.dis_shape:self.dis_shape + self.sym_shape], state_slot)

        dis_, sym_ = self.decision_predict(sym_state=state_slot, sym_flag=sym_flag, disease_mask=disease_mask_,
                                               symptoms_mask=symptoms_mask_)

        return dis_, sym_

    def dis_classify(self, dis_matrix, disease_mask):
        dis_matrix_ = torch.sum(dis_matrix, dim=1)
        # dis_masked = dis_matrix_.eq(0).float()
        dis_masked = disease_mask.eq(0).float()
        x = dis_matrix_ - dis_masked * 1e9
        x = self.temperature_softmax(x)
        return x

    def sym_classify(self, sym_matrix, sym_mask, sym_flag):
        sym_matrix_ = torch.sum(sym_matrix, dim=1)
        sym_masked = sym_mask.eq(0).float()
        x = sym_matrix_ - sym_masked * 1e9
        sym_masked2 = sym_flag.eq(0).float()
        x = x - sym_masked2 * 1e9
        x = self.temperature_softmax(x)
        return x

    def decision_predict(self, sym_state, sym_flag, disease_mask, symptoms_mask):
        rwr_dis = torch.zeros(disease_mask.size(0), self.sym_shape, self.dis_shape).to(self.device)
        rwr_sym = torch.zeros(disease_mask.size(0), self.sym_shape, self.sym_shape).to(self.device)
        for b in range(disease_mask.size(0)):
            ####  当前KG中剩余的疾病和症状  ######
            dis_exit = []
            sym_exit = []
            sym_confirm = []
            dis_mask = disease_mask[b]
            for i in range(len(dis_mask)):
                if dis_mask[i] > 0.:
                    dis_exit.append(self.dis[i])
            sym_mask = symptoms_mask[b]
            for i in range(len(sym_mask)):
                if sym_mask[i] > 0.:
                    sym_exit.append(self.sym[i])
            sym_state_ = sym_state[b, self.dis_shape:self.dis_shape + self.sym_shape]
            for i in range(len(sym_state_)):
                if sym_state_[i] == dialog_config.TRUE:
                    sym_confirm.append(self.sym[i])

            ###### 构建图  剪枝后的 ########
            weight_edges = []
            nodes = []
            for d in self.dis_sym_num.keys():
                for s in self.dis_sym_num[d].keys():
                    if d in dis_exit and s in sym_exit:
                        weight_edges.append((s, d, self.dis_sym_num[d][s]))
                        # weight_edges.append((s, d))
            nodes = self.dis + self.sym
            rwr_ = {}

            #### 以确认症状为中心随机游走  ######
            for s in sym_confirm:
                rwr_[s] = {}
                G = nx.Graph()
                G.add_nodes_from(nodes)
                G.add_weighted_edges_from(weight_edges)
                # G.add_edges_from(weight_edges)
                # print(G.edges)
                a = nx.pagerank(G, alpha=self.alpha, personalization={s: 1})
                rwr_[s] = a
            rwr_dis_matrix = torch.zeros((len(self.sym), len(self.dis)))
            rwr_dis_matrix_ = torch.zeros((len(self.sym), len(self.dis)))
            rwr_sym_matrix = torch.zeros((len(self.sym), len(self.sym)))
            rwr_sym_matrix_ = torch.zeros((len(self.sym), len(self.sym)))

            ##### 随机游走分数矩阵 #####
            for d in self.dis:
                for s in self.sym:
                    if s in rwr_.keys() and d in rwr_[s].keys():
                        rwr_dis_matrix[self.sym.index(s)][self.dis.index(d)] = rwr_[s][d]

            for s1 in self.sym:
                for s in self.sym:
                    if s in rwr_.keys() and s1 in rwr_[s].keys():
                        rwr_sym_matrix[self.sym.index(s)][self.sym.index(s1)] = rwr_[s][s1]

            rwr_dis[b] = copy.deepcopy(torch.tensor(rwr_dis_matrix))
            rwr_sym[b] = copy.deepcopy(torch.tensor(rwr_sym_matrix))

        dis_pro = self.dis_classify(rwr_dis, disease_mask)
        sym_pro = self.sym_classify(rwr_sym, symptoms_mask, sym_flag)

        return dis_pro, sym_pro


