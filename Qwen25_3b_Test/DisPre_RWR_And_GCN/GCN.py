import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from utils import dialog_config


class TemperatureSoftmax(nn.Module):
    def __init__(self, temperature=1):
        super(TemperatureSoftmax, self).__init__()
        self.temperature = temperature

    def forward(self, x):
        x = x / self.temperature
        x = torch.softmax(x, dim=-1)
        return x


class GCNReward(nn.Module):
    def __init__(self, device, kg_node, dis_num, kggraph, slot_set, embed_size=100, temperature=0.5):#初始化
        super(GCNReward, self).__init__()

        self.embed_size = embed_size
        self.kg_node = kg_node
        self.dis_num = dis_num
        self.sym_num = kg_node - dis_num
        self.device = device
        self.slot_set = slot_set

        self.sym_representation = nn.Embedding(self.kg_node, self.embed_size).to(self.device)

        self.gc1 = GCNConv(embed_size, embed_size).to(self.device) # 构建第一层 GCN
        self.gc2 = GCNConv(embed_size, embed_size).to(self.device) # 构建第二层 GCN

        self.classifier = nn.Sequential(  #
            nn.Linear(self.embed_size, self.dis_num),
        ).to(self.device)
        self.loss_func = nn.CrossEntropyLoss()

        self.temperature_softmax = TemperatureSoftmax(temperature)

        self.kgraph = kggraph
        self.kgraph.initialize_adj()  # 初始化邻接矩阵, 自动初始化
        self.initial_kg_matrix = copy.deepcopy(self.kgraph.kg_matrix).view(-1, self.kg_node, self.kg_node)
        self.kg_matrix = copy.deepcopy(self.initial_kg_matrix)


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

    def forward_before(self, label, adj, mask):
        x = torch.tensor([[i for i in range(self.kg_node)] for l in range(mask.size()[0])]).to(self.device)
        x_embed = self.sym_representation(x)
        x = F.relu(self.gc1(x_embed, adj))  # 第一层，并用relu激活
        # x = F.dropout(x, self.dropout, training=self.training)#丢弃一部分特征
        x = self.gc2(x, adj)  # 第二层
        sym_rep = torch.matmul(mask, x).squeeze(1)
        disease_ = self.classifier(sym_rep)
        loss = self.loss_func(disease_, label)
        output_ = disease_.max(1)[1]
        output = 0
        for i in range(len(output_)):
            if output_[i] == label[i]:
                output += 1
        return output, loss

    def forward(self, label, adj, mask):
        x = torch.tensor([[i for i in range(self.kg_node)] for l in range(mask.size()[0])]).to(self.device)
        x_embed = self.sym_representation(x)
        x = F.relu(self.gc1(x_embed, adj))  # 第一层，并用relu激活
        # x = F.dropout(x, self.dropout, training=self.training)#丢弃一部分特征
        x = self.gc2(x, adj)  # 第二层
        sym_rep = torch.matmul(mask, x).squeeze(1)
        # 计算相似度
        disease_mask = torch.ones(self.dis_num).to(self.device).reshape(-1, self.dis_num)
        symptoms_mask = torch.ones(self.sym_num).to(self.device).reshape(-1, self.sym_num)
        kg_matrix = torch.zeros(1, self.kg_node, self.kg_node).to(self.device)
        for s in range(adj.size()[1]):
            kg_matrix[0][adj[0][s]][adj[1][s]] = 1
        for i in range(kg_matrix.size(0)):
            for j in range(self.dis_num):
                if torch.equal(kg_matrix[i, j, :],
                               torch.zeros(kg_matrix[i, j, :].size()).to(self.device)):
                    disease_mask[0][j] = 0.
            for k in range(self.sym_num):
                if torch.equal(kg_matrix[i, k + self.dis_num, :],
                               torch.zeros(kg_matrix[i, k + self.dis_num, :].size()).to(
                                   self.device)):
                    symptoms_mask[0][k] = 0.
        disease_mask_ = copy.deepcopy(disease_mask)
        symptoms_mask_ = copy.deepcopy(symptoms_mask)
        gcn_matrix = torch.zeros(kg_matrix.size()[0], self.kg_node - self.dis_num, self.dis_num).to(self.device)
        # print(mask)
        for s in range(self.kg_node - self.dis_num):
            if mask[0][self.dis_num + s] == 1.:
                for d in range(self.dis_num):
                    # print('here')
                    gcn_matrix[0][s][d] = self.cosine_similarity(x[0][self.dis_num + s], x[0][d])
        # print(gcn_matrix.requires_grad)
        gcn_matrix = gcn_matrix.requires_grad_()
        # print(disease_mask_.requires_grad)
        dis_pro = self.dis_classify(gcn_matrix, disease_mask_)
        # dis_pro = dis_pro.requires_grad_()
        # sym_pro = self.sym_classify(gat_sym_sym, symptoms_mask, sym_flag)
        loss = self.loss_func(dis_pro, label)
        # print(dis_pro.requires_grad)
        # print(label.requires_grad)
        # print(loss)
        output_ = dis_pro.max(1)[1]
        output = 0
        for i in range(len(output_)):
            if output_[i] == label[i]:
                output += 1
        return output, loss

    def construct_kg_index(self, kg_adj):
        # sym_gcn = torch.zeros(kg_adj.size()[0], self.dis_num, self.sym_num).to(self.device)
        edge_index = [[], []]
        for b in range(kg_adj.size()[0]): # 1
            # edge_index = [[], []]
            for source_node in range(self.kg_node):
                for target_node in range(self.kg_node):
                    if kg_adj[b][source_node][target_node] != 0:
                        edge_index[0].append(source_node)
                        edge_index[1].append(target_node)
            edge_index = torch.tensor(edge_index).to(self.device)
        #     with torch.no_grad():
        #         sym_gcn[b] = self.medical_embed.predict(edge_index)
        # return sym_gcn
        return edge_index

    def predict(self, adj, disease_mask, symptoms_mask, sym_flag, sym_confirm):
        with torch.no_grad():
            edge_index = self.construct_kg_index(adj)
            x = torch.tensor([[i for i in range(self.kg_node)] for l in range(adj.size()[0])]).to(self.device)
            x_embed = self.sym_representation(x)
            x = F.relu(self.gc1(x_embed, edge_index))  # 第一层，并用relu激活
            # x = F.dropout(x, self.dropout, training=self.training)#丢弃一部分特征
            x = self.gc2(x, edge_index)  # 第二层 (1, 130, 100)
            # 计算相似度
            gcn_matrix = torch.zeros(adj.size()[0], self.kg_node-self.dis_num, self.dis_num).to(self.device)
            for s in range(self.kg_node-self.dis_num):
                if sym_confirm[0][s] == 1.:
                    for d in range(self.dis_num):
                        gcn_matrix[0][s][d] = self.cosine_similarity(x[0][self.dis_num+s], x[0][d])

            dis_pro = self.dis_classify(gcn_matrix, disease_mask)
            # sym_pro = self.sym_classify(gat_sym_sym, symptoms_mask, sym_flag)

        return dis_pro, None

    def cosine_similarity(self, A, B):
        dot_product = torch.dot(A, B)
        norm_A = torch.norm(A)
        norm_B = torch.norm(B)

        if norm_A.item() == 0 or norm_B.item() == 0:
            # 处理零向量的情况，避免除以零
            # print('zero!!!!!!')
            return torch.tensor(0.0)
        else:
            # 计算余弦相似度
            similarity = dot_product / (norm_A * norm_B)
            # print(similarity.requires_grad)
            return similarity

    def GetConfirm(self, goal):
        current_confirm_sym = []
        for s in goal['current_slots']['inform_slots'].keys():
            if goal['current_slots']['inform_slots'][s] == dialog_config.TRUE:
                current_confirm_sym.append(s)
        return current_confirm_sym

    def state_representation(self, g):
        state_slot = torch.full((1, self.kg_node), 0.3).to(self.device)  # 实体大小由疾病和症状组成
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
        return torch.where(batch_state == 0.3, ones, zeros), torch.where(batch_state == 1, ones, zeros), torch.where(total_state == 1, ones_, zeros_).unsqueeze(1)

    def score(self, goal):
        '''
        作用:获取当前goal的分数
        输入:goal
        输出:dis_,sym_,表示疾病和病症的概率分布,形状为tensor([[0.8000, 0.1000, 0.1000,...]])
        '''
        self.kgraph.initialize_adj()  # 初始化邻接矩阵, 自动初始化
        current_confirm = self.GetConfirm(goal)
        disease_mask = torch.ones(self.dis_num).to(self.device).reshape(-1, self.dis_num)
        symptoms_mask = torch.ones(self.sym_num).to(self.device).reshape(-1, self.sym_num)
        self.kgraph.update_adj(current_confirm)  # 修改图谱的邻接矩阵，根据患者自述剪枝

        kg_matrix = copy.deepcopy(self.kgraph.kg_matrix).view(-1, self.kg_node, self.kg_node)
        for i in range(kg_matrix.size(0)):
            for j in range(self.dis_num):
                if torch.equal(kg_matrix[i, j, :],
                               torch.zeros(kg_matrix[i, j, :].size()).to(self.device)):
                    disease_mask[0][j] = 0.
            for k in range(self.sym_num):
                if torch.equal(kg_matrix[i, k + self.dis_num, :],
                               torch.zeros(kg_matrix[i, k + self.dis_num, :].size()).to(
                                   self.device)):
                    symptoms_mask[0][k] = 0.
        disease_mask_ = copy.deepcopy(disease_mask)
        symptoms_mask_ = copy.deepcopy(symptoms_mask)

        state_slot = self.state_representation(goal)
        sym_flag, sym_confirm, _ = self.get_sym_flag(state_slot[:, self.dis_num:self.dis_num + self.sym_num], state_slot)

        dis_, sym_ = self.predict(kg_matrix, disease_mask_, symptoms_mask_, sym_flag, sym_confirm)

        return dis_, sym_

