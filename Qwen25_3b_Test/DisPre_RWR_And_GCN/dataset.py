import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from utils import dialog_config


class DialogueDataset(Dataset):
    def __init__(self, goals, data_type):  # {'current_slots': {'inform_slots': {}}, 'disease_tag': g['disease_tag']}
        self.goals = goals[data_type]
        # self.symptom = symptom

    def __len__(self):
        return len(self.goals)

    # def get_sample(self, idx):
    #     return {
    #         'goal': self.goals[idx]
    #     }

    def __getitem__(self, idx):
        return {
            'goal': self.goals[idx]
        }
        # """
        # 返回单个样本，拆分出 'current_slots' 和 'disease_tag'。
        # """
        # goal = self.goals[idx]
        # current_slots = goal.get("current_slots", {}).get("inform_slots", {})
        # # 填充缺失的键为默认值 0
        # filled_slots = {key: current_slots.get(key, dialog_config.PADDING) for key in self.symptom}
        #
        # disease_tag = goal.get("disease_tag", None)
        # return {"current_slots": filled_slots, "disease_tag": disease_tag}  # {'current_slots': [{'咳嗽': 1, '胸闷': -1}, {'胸骨后疼痛': 1}],'disease_tag': ['食管炎', '食管炎']}


def custom_collate_fn(batch):
    # 按原始数据格式返回批次
    return batch