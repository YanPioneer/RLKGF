import sys

import argparse
import ast
import os

import torch
import random
import json
import time

import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.utils import *
from Data.GMD.KG import KGADJ
from utils import dialog_config

from DisPre_RWR_And_GCN.GCN import GCNReward
from DisPre_RWR_And_GCN.RWRModel import RWR
from DisPre_RWR_And_GCN.KGReward import KGReward
from DisPre_RWR_And_GCN.dataset import DialogueDataset, custom_collate_fn
from DisPre_RWR_And_GCN.qwen_ppo import PPO_KG



def set_seed(seed):
    """
    设置随机种子以保证复现性
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True  # For deterministic behavior in CUDA
    torch.backends.cudnn.benchmark = False  # Disable the auto-tuner that picks the best algorithm


def GetAllSym(goal_set):
    '''
    输出形式：
    new_goal_test = {
    'train': [
        {
            'current_slots': {
                'inform_slots': {
                    '咳嗽': dialog_config.TRUE,
                    '胸闷气促': dialog_config.FALSE,
                    '胸骨后疼痛': dialog_config.FALSE,
                    '胸闷': dialog_config.FALSE,
                    '咯血': dialog_config.TRUE,
                    '咳痰': dialog_config.TRUE
                }
            },
            'disease_tag': '食管炎'
        }
        ...
    ],
    'test': [...]
    }
    '''
    new_goal_test = {'train': [], 'test': []}
    for g in goal_set['train']:
        g_ = {'current_slots': {'inform_slots': {}}, 'disease_tag': g['disease_tag']}
        for ex in g['explicit_inform_slots'].keys():
            if g['explicit_inform_slots'][ex]:
                g_['current_slots']['inform_slots'][ex] = dialog_config.TRUE
            elif not g['explicit_inform_slots'][ex]:
                g_['current_slots']['inform_slots'][ex] = dialog_config.FALSE

        for im in g['implicit_inform_slots'].keys():
            if g['implicit_inform_slots'][im] == True:
                 g_['current_slots']['inform_slots'][im] = dialog_config.TRUE
            elif g['implicit_inform_slots'][im] == False:
                 g_['current_slots']['inform_slots'][im] = dialog_config.FALSE

        new_goal_test['train'].append(g_)

    for g in goal_set['test']:
        g_ = {'current_slots': {'inform_slots': {}}, 'disease_tag': g['disease_tag']}
        for ex in g['explicit_inform_slots'].keys():
            if g['explicit_inform_slots'][ex]:
                g_['current_slots']['inform_slots'][ex] = dialog_config.TRUE
            elif not g['explicit_inform_slots'][ex]:
                g_['current_slots']['inform_slots'][ex] = dialog_config.FALSE

        for im in g['implicit_inform_slots'].keys():
            if g['implicit_inform_slots'][im] == True:
                 g_['current_slots']['inform_slots'][im] = dialog_config.TRUE
            elif g['implicit_inform_slots'][im] == False:
                 g_['current_slots']['inform_slots'][im] = dialog_config.FALSE

        new_goal_test['test'].append(g_)

    return new_goal_test

# 数据
embed_size = 100
current_path = os.path.abspath(__file__)
father_path = os.path.abspath(os.path.dirname(current_path))
grand_path = os.path.abspath(os.path.dirname(father_path))
GMD_path = os.path.join(grand_path, 'Data', 'GMD', 'dataset_gmd')
GMD_Disease = text_to_list(os.path.join(GMD_path, 'gmd_disease.txt'))
# print(GMD_Disease)
GMD_Symptom = text_to_list(os.path.join(GMD_path, 'gmd_symptom.txt'))

GMD_goal = load_pickle(os.path.join(GMD_path, 'gmd_new.pk'))

dis_sym_num_to_graph_path = os.path.join(GMD_path, 'dise_sym_num_high_dict(≥3).txt')
# dis_sym_num_to_graph
with open(dis_sym_num_to_graph_path, 'r', encoding='utf-8') as f:
    content = f.readlines()
    dis_sym_num_to_graph = ast.literal_eval(content[0])

gcn_path = os.path.join(grand_path, 'useful_models', 'GMD_GCN', 'gcn_0.7955.pth.tar')
slot_set = text_to_dict('{}/gmd_slot_set.txt'.format(GMD_path))  # all slots with symptoms + all disease

# 设置种子
set_seed(1616)
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
# KG
kg = KGADJ(device, len(GMD_Disease), len(GMD_Symptom), GMD_path)
gcn_tem = 1
rwr_tem = 1
mu = 0.01
# 奖励模型
gcn_reward = GCNReward(device, len(GMD_Disease) + len(GMD_Symptom), len(GMD_Disease), kg, slot_set, embed_size, gcn_tem)  # device, kg_node, dis_num, kggraph, slot_set, embed_size=100, temperature=1
gcn_ = torch.load(gcn_path, map_location=device)
gcn_reward.load_state_dict(gcn_['state_dict'])  # 加载离线训练模型

# alpha
alpha = 0.3
# RWR奖励模型
rwr_reward = RWR(len(GMD_Disease), len(GMD_Symptom), GMD_Disease, GMD_Symptom, slot_set, dis_sym_num_to_graph, kg, device, alpha=alpha, temperature=rwr_tem)
kg_reward = KGReward(gcn_reward, rwr_reward, mu=mu)


# 数据
goal_ = GetAllSym(GMD_goal)
data_type = 'train'  # 训练数据
random.shuffle(goal_['test'])
dataset = DialogueDataset(goal_, data_type)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
batch_size_ = 4

# 定义PPO
for i in [240 - 20 * j for j in range(12)]:
# for i in [220, 260, 300, 320]:
    # if i == 200:
    #     l_ = [6e-6, 7e-7]
    # else:
    #     l_ = [2e-6, 6e-6, 7e-7]
    for l in [2e-6, 1e-6, 7e-6]:
        # LLM
        # 加载模型
        # 加载模型和分词器
        llm_path = "/Qwen1.5-4B-Chat/"  # LLM 互信息及最后生成

        model = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype="auto",
            device_map=device
        )
        tokenizer = AutoTokenizer.from_pretrained(llm_path, padding_side='left')
        # 定义PPO
        update_freq = i
        lr = l
        dispre_ppo_kg = PPO_KG(model, kg_reward, tokenizer, GMD_Symptom, GMD_Disease, device, lr=l, update_freq=update_freq)
        print(dispre_ppo_kg.update_freq)
        print(dispre_ppo_kg.lr)
        # 时间戳
        timeStr = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

        success_rate = 0.
        train_losses = []
        average_loss = [10000.]
        val_accuracies = []
        print('==========初始测试==============')
        initial_test_rate = dispre_ppo_kg.eval_step(goal_['test'])
        for epoch in range(5):  # 简化的训练循环
            print('==========epoch:' + str(epoch) + '训练==============')
            for batch in dataloader:
                goals = batch
                # 进行一步训练
                dispre_ppo_kg.train_step(epoch, timeStr, goals)

            # 每个epoch打印平均损失
            avg_loss = np.mean(dispre_ppo_kg.losses[-len(dataloader):])  # 计算当前epoch的平均损失
            print(f"Epoch {epoch + 1}: Average PPO Loss = {avg_loss:.4f}")
            average_loss.append(avg_loss)

            # # 验证
            print('==========epoch:' + str(epoch) + '测试==============')
            test_rate = dispre_ppo_kg.eval_step(goal_['test'])
            val_accuracies.append(test_rate)
            if test_rate > success_rate or avg_loss < average_loss[-2] or test_rate > initial_test_rate:
                success_rate = test_rate
                # 根据准确率创建保存目录，使用字符串格式化将准确率添加到文件夹名称中
                model_directory = os.path.join(father_path, "gmd_dispre_model_save", str(timeStr), f"model_directory_epoch{epoch}_accuracy_{test_rate:.4f}_loss_{avg_loss:.4f}")

                # 保存模型和 tokenizer
                model.save_pretrained(model_directory)
                tokenizer.save_pretrained(model_directory)

            metrics = {
                "initial_test": initial_test_rate,
                "update_freq": update_freq,
                "batch": batch_size_,
                "lr":lr,
                "alpha": alpha,
                "seed": 1616,
                "gcn_tem": gcn_tem,
                "rwr_tem": rwr_tem,
                "mu": mu,
                "gcn": grand_path,
                "train_losses": average_loss,
                "val_accuracies": val_accuracies
            }
            os.makedirs(os.path.dirname(os.path.join(father_path, "gmd_dispre_model_save", str(timeStr), 'training_metrics.json')), exist_ok=True)
            with open(os.path.join(father_path, "gmd_dispre_model_save", str(timeStr), 'training_metrics.json'), 'w') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=4)
            time.sleep(20)





