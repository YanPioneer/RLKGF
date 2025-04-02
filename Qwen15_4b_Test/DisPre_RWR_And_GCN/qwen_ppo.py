import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import Adam
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, Dataset
import numpy as np
from copy import deepcopy
from utils import dialog_config
import os

import copy


# PPO的优化类
class PPO_KG:
    def __init__(self, model, reward_model, tokenizer, sympton, disease, device, lr=1e-5, gamma=0.99, epsilon=0.2,
                 batch_size=16, update_freq=50):
        self.model = model.to(device)
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.optimizer = Adam(model.parameters(), lr=lr)
        self.lr =lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.symptom = sympton
        self.disease = disease
        self.dis_shape = len(disease)
        self.sym_shape = len(sympton)
        self.old_model = deepcopy(model).to(device)  # 创建一个旧策略模型
        self.old_model.eval()  # 固定旧策略模型权重
        self.device = device
        self.losses = []  # 用于存储每步的loss
        self.update_freq = update_freq  # 更新频率，每N步更新一次策略
        self.step = 0

        self.epoch = 0

        self.prompt1 = '''
        #01 你是一个专科医生。

        #02 你的任务是模拟现实的专科医生进行疾病诊断。任务是根据患者症状信息进行诊断，诊断结果在给定的疾病列表中选择一个进行输出。

        #03 注意，只返回一个疾病作为预测结果，如果无法给出，输出UNKNOW。

        #04 以下是你所在门诊涉及的疾病：

        """

        {{此处替换成疾病}}

        """

        #05 对话示例如下，请严格按照示例给出的输出格式进行输出，无需给出任何解释，如果列表中的疾病都不满足，直接输出UNKNOW：

        """
        示例1：输入:患者恶心呕吐、解稀便、发热、腹泻，是怎么了？, 输出:应该是得了肠炎。

        示例2：输入:患者老是心悸、头昏、胸闷、胸骨后疼痛，无背痛，怎么回事？, 输出:可能是冠心病。

        """"

        '''

        self.kg_prompt1 = '''
        #01 你是一个基于知识图谱进行诊断的专科医生。

        #02 你的任务是模拟现实的专科医生进行疾病诊断。任务是根据患者症状信息、结合给出的知识图谱中包含的疾病和症状的关系进行诊断，诊断结果在给定的疾病列表中选择一个进行输出。

        #03 注意，只返回一个疾病作为预测结果，如果无法给出，输出UNKNOW。

        #04 以下是背景知识图谱信息：

        """

        {{此处替换成KG三元组}}

        """

        #05 以下是你所在门诊涉及的疾病：

        """

        {{此处替换成疾病}}

        """

        #06 对话示例如下，请严格按照示例给出的输出格式进行输出，无需给出任何解释，如果列表中的疾病都不满足，直接输出UNKNOW：

        """
        示例1：输入:患者恶心呕吐、解稀便、发热、腹泻，是怎么了？, 输出:应该是得了肠炎。

        示例2：输入:患者老是心悸、头昏、胸闷、胸骨后疼痛，无背痛，怎么回事？, 输出:可能是冠心病。

        """"

        '''

        self.kg_context_prompt1 = '''
        #01 你是一个基于知识进行诊断的专科医生。

        #02 你的任务根据患者症状信息、结合给出的知识进行诊断，诊断结果在给定的疾病列表中选择一个进行输出。

        #03 注意，只返回一个疾病作为预测结果，如果无法给出，输出UNKNOW。

        #04 以下是背景知识：

        """

        {{此处替换成KG文本}}

        """

        #05 以下是你所在门诊涉及的疾病：

        """

        {{此处替换成疾病}}

        """

        #06 对话示例如下，请严格按照示例给出的输出格式进行输出，如果列表中的疾病都不满足，直接输出UNKNOW：

        """
        示例1：输入:患者恶心呕吐、解稀便、发热、腹泻，是怎么了？, 输出:应该是得了肠炎。

        示例2：输入:患者老是心悸、头昏、胸闷、胸骨后疼痛，无背痛，怎么回事？, 输出:可能是冠心病。

        """"

        '''

        self.prompt = self.prompt1.replace("{{此处替换成疾病}}", str(disease))  # 后续进行修剪
        self.kg_prompt = self.kg_prompt1.replace("{{此处替换成疾病}}", str(disease))  # 后续替换KG
        self.kg_context_prompt = self.kg_context_prompt1.replace("{{此处替换成疾病}}", str(disease))  # 后续替换KG上下文


    def user_generate(self, inform_slots):
        context = '患者'
        confirm = []
        deny = []
        not_sure = []
        for sym in inform_slots.keys():
            if inform_slots[sym] == dialog_config.TRUE:
                confirm.append(sym)
            elif inform_slots[sym] == dialog_config.FALSE:
                deny.append(sym)
            else:
                not_sure.append(sym)

        if len(confirm) > 0:
            context += '确认存在症状:'
            for i in range(len(confirm)):
                context += str(confirm[i])
                if i < len(confirm) - 1:
                    context += '、'
                # else:
                #     context += '；'

        if len(deny) > 0:
            if len(confirm) > 0:
                context += '；'
            context += '不存在症状:'
            for i in range(len(deny)):
                context += str(deny[i])
                if i < len(deny) - 1:
                    context += '、'
        elif len(deny) == 0 and len(not_sure) == 0:
            context += '。'

        if len(not_sure) > 0:
            if (len(confirm) > 0 and len(deny) == 0) or len(deny) > 0:
                context += '；'
            context += '不清楚是否存在症状:'
            for i in range(len(not_sure)):
                context += str(not_sure[i])
                if i < len(not_sure) - 1:
                    context += '、'
            context += '。'
        elif len(not_sure) == 0 and len(deny) != 0:
            context += '。'

        return context

    def get_message(self, user_information):
        content = self.user_generate(user_information)
        message = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": '输入:' + content + '输出:'}
        ]
        # 将多轮对话拼接为输入字符串
        # dialog = ""
        # for turn in message:
        #     role = turn["role"]
        #     content = turn["content"]
        #     dialog += f"{role}: {content}\n"
        # return dialog.strip()
        return message
    
    def get_message_with_kg(self, user_information, prompt):
        content = self.user_generate(user_information)
        message = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": '输入:' + content + '输出:'}
        ]
        return message

    def compute_loss(self, old_log_probs, new_log_probs, advantages):
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)

        # 通过 unsqueeze 扩展 advantages 为 [batch_size, seq_len]
        advantages_expanded = advantages.unsqueeze(1).expand_as(ratio)  # 扩展 advantages 的形状为 [16, 153]

        loss = -torch.min(ratio * advantages_expanded, clipped_ratio * advantages_expanded)
        return loss.mean()

    def update(self, input_ids, generated_ids, old_log_probs, advantages):
        outputs = self.model(**input_ids)
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)

        # 计算新策略的 log_probs
        # new_log_probs = log_probs.gather(2, generated_ids.unsqueeze(-1)).squeeze(-1)
        new_log_probs = log_probs.gather(2, input_ids.input_ids.unsqueeze(-1)).squeeze(-1)

        # 计算损失并更新模型
        loss = self.compute_loss(old_log_probs, new_log_probs, advantages)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.losses.append(loss.item())
        # print('loss:', self.losses)

    def train_step(self, epoch, timeStr, goals_batch):
        self.epoch = epoch
        texts = [
            self.tokenizer.apply_chat_template(self.get_message(goal['goal']['current_slots']['inform_slots']),
                                               tokenize=False, add_generation_prompt=True)
            for goal in goals_batch
        ]
        model_inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)

        # 生成序列
        generated_outputs = self.model.generate(
            **model_inputs,
            max_new_tokens=512,
            return_dict_in_generate=True,
            output_scores=True
        )
        generated_ids = generated_outputs.sequences[:, model_inputs.input_ids.shape[1]:]
        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # os.makedirs(os.path.dirname(os.path.join('./gmd_dispre_model_save', str(timeStr), 'response_record.txt')), exist_ok=True)
        # with open(os.path.join('./gmd_dispre_model_save', str(timeStr), 'response_record.txt'), "a+") as f:
        #     f.write("="*80 + "\n")
        #     f.write(f"epoch: {self.epoch}")
        #     f.write("="*80 + "\n")
        #     for i, text in enumerate(responses):
        #         # 写入生成的文本和对应的输入数据
        #         input_text = goals_batch[i]  # 这里的 batch_data[i] 可以是训练中使用的输入文本
        #         f.write(f"Input Text: {input_text}\n")
        #         f.write(f"Generated Text: {text}\n")
        #         f.write("="*80 + "\n")

        preferred_rewards, baseline_rewards = [], []
        for i, generated_text in enumerate(responses):
            # print(generated_text)
            model_reward, base_reward = self.compute_kg_score(generated_text, goals_batch[i]['goal'])
            preferred_rewards.append(model_reward)
            baseline_rewards.append(base_reward)  # 最大的理论上是最优的
        # 计算优势
        advantages = torch.tensor(preferred_rewards).to(self.device) - torch.tensor(baseline_rewards).to(self.device)

        ### 需要归一化吗？？？
        # 奖励归一化
        # preferred_rewards = torch.tensor(preferred_rewards).to(self.device)
        # baseline_rewards = torch.tensor(baseline_rewards).to(self.device)
        # advantages = (preferred_rewards - baseline_rewards)
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        # 旧策略3
        # old_log_probs = self.get_log_probs(self.old_model, model_inputs)

        # 获取旧策略的 log_probs4
        old_log_probs = self.get_log_probs(epoch, self.old_model, model_inputs, generated_ids)

        # 更新模型3
        # self.update(model_inputs, old_log_probs, advantages)

        # 更新模型
        self.update(model_inputs, generated_ids, old_log_probs, advantages)
        self.step += 1

        # 每`update_freq`步后更新旧策略
        if self.step % self.update_freq == 0:
            # self.old_model = deepcopy(self.model)  # 更新旧策略模型为当前模型
            self.old_model.load_state_dict(self.model.state_dict())  # 更新旧策略模型为当前模型

        return old_log_probs

    def get_log_probs(self, epoch, model, input_ids, generated_ids):  ### 4
        with torch.no_grad():
            outputs = model(**input_ids)
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)

        # dialog_config.log_tensor_shapes(epoch=epoch, generated_ids=generated_ids, logits=logits, model_inputs=input_ids)

        # return log_probs.gather(2, generated_ids.unsqueeze(-1)).squeeze(-1)
        return log_probs.gather(2, input_ids.input_ids.unsqueeze(-1)).squeeze(-1)

    def compute_kg_score(self, generated_texts, goal):
        # 抽取出回复中的症状/疾病
        generated_entity, type = self.extract_symptom(generated_texts)
        dis_reward, _ = self.reward_model.score(goal)  # 输入为batch形式

        kg_score = dis_reward.max(1)[0].view(1, 1).item()
        # generate_score = 0.
        if generated_entity == None:
            generate_score = -1.
        else:
            generate_score = dis_reward[0][self.disease.index(generated_entity)].view(1, 1).item()

        return generate_score, kg_score

    def extract_symptom(self, generated_text):  # 从回复中抽取可能的症状、疾病
        # 提取生成的症状  后续抽取方式需要改变

        for disease in self.disease:
            if disease in generated_text:
                return disease, 'dis'
        # for symptom in self.symptom:
        #     if symptom in generated_text:
        #         return symptom, 'sym'
        return None, '_'
    
    def system_generate(self, disease, symptom, generation):
        for d in disease:
            if d in generation:
                return 1, d, '初步判断得了' + str(d) + '。'
        # for s in symptom:
        #     if s in generation:
        #         return 0, s, '有没有' + str(s) +'。'
        # # 如果没有匹配的，先对齐症状
        # if 'UNKNOW' not in generation:
        #     map_sym = sym_map(generation, GMD_Symptom)
        #     if map_sym is None:
        #         return 2, None, generation
        #     else:
        #         return 0, map_sym, '有没有' + str(map_sym) + '。'

        return 3, None, generation  # 返回UNKNOW

    
    def eval_step(self, goal_test, e=3):
        # 测试三次
        pre_success = []
        for i in range(e):
            dis_predict = 0.
            episode = 0.
            
            for g in goal_test:
                # print("=======================")
                # print(g)
                episode += 1
                test_re = 0
                message_  = self.get_message(g['current_slots']['inform_slots'])
                while True:
                    test_re += 1
                    # print('episode:' + str(episode))
                    text = self.tokenizer.apply_chat_template(
                        message_,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    model_inputs = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(self.device)

                    generated_ids = self.model.generate(
                        **model_inputs,
                        max_new_tokens=512
                    )
                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                    ]

                    answers_content = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    # print('gen_system:', answers_content)
                    flag, action, system_ask = self.system_generate(disease=self.disease, symptom=self.symptom, generation=answers_content)
                    # print('system:', system_ask)

                    if flag == 1:
                        if action == g['disease_tag']:
                            dis_predict += 1
                        break
                    elif flag == 3:
                        break
                    if test_re > 6:
                        break

            print("success rate %.4f" % (dis_predict / episode))
            pre_success.append(dis_predict / episode)
        return np.mean(pre_success)
    
    def test_with_kg_step(self, goal_test, dise_sym_dict, e=3):
        # 基于KGprompt测试
        kg = []
        for d in dise_sym_dict.keys():
            for s in dise_sym_dict[d].keys():
                kg.append((d, '导致', s))
        self.kg_prompt_ = self.kg_prompt.replace("{{此处替换成KG三元组}}", str(kg))
        pre_success = []
        for i in range(e):
            dis_predict = 0.
            episode = 0.
            for g in goal_test:
                episode += 1
                test_re = 0
                message_  = self.get_message_with_kg(g['current_slots']['inform_slots'], self.kg_prompt_)
                while True:
                    test_re += 1
                    # print('episode:' + str(episode))
                    text = self.tokenizer.apply_chat_template(
                        message_,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

                    generated_ids = self.model.generate(
                        **model_inputs,
                        max_new_tokens=512
                    )
                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                    ]

                    answers_content = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    # print('gen_system:', answers_content)
                    flag, action, system_ask = self.system_generate(disease=self.disease, symptom=self.symptom, generation=answers_content)
                    # print('system:', system_ask)

                    if flag == 1:
                        if action == g['disease_tag']:
                            dis_predict += 1
                        break
                    elif flag == 3:
                        break
                    if test_re > 6:
                        break

            print("success rate %.4f" % (dis_predict / episode))
            pre_success.append(dis_predict / episode)
        return np.mean(pre_success)
    
    def test_with_kg_context_step(self, goal_test, dise_sym_dict, e=3):
        # 基于KGprompt测试
        kg = ''
        for d in dise_sym_dict.keys():
            kg += '{}能导致的症状有{}。'.format(d, ','.join(dise_sym_dict[d].keys()))
        self.kg_context_prompt_ = self.kg_context_prompt.replace("{{此处替换成KG文本}}", str(kg))
        pre_success = []
        for i in range(e):
            dis_predict = 0.
            episode = 0.
            for g in goal_test:
                episode += 1
                test_re = 0
                message_  = self.get_message_with_kg(g['current_slots']['inform_slots'], self.kg_context_prompt_)
                while True:
                    test_re += 1
                    # print('episode:' + str(episode))
                    text = self.tokenizer.apply_chat_template(
                        message_,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

                    generated_ids = self.model.generate(
                        **model_inputs,
                        max_new_tokens=512
                    )
                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                    ]

                    answers_content = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    # print('gen_system:', answers_content)
                    flag, action, system_ask = self.system_generate(disease=self.disease, symptom=self.symptom, generation=answers_content)
                    # print('system:', system_ask)

                    if flag == 1:
                        if action == g['disease_tag']:
                            dis_predict += 1
                        break
                    elif flag == 3:
                        break
                    if test_re > 6:
                        break

            print("success rate %.4f" % (dis_predict / episode))
            pre_success.append(dis_predict / episode)
        return np.mean(pre_success)
    
    def test_with_kg_context2_step(self, goal_test, dise_sym_dict, e=3):
        # 基于KGprompt测试
        kg = ''
        for d in dise_sym_dict.keys():
            kg += '{}关联的症状有{}。'.format(d, ','.join(dise_sym_dict[d].keys()))
        self.kg_context_prompt_ = self.kg_context_prompt.replace("{{此处替换成KG文本}}", str(kg))
        pre_success = []
        for i in range(e):
            dis_predict = 0.
            episode = 0.
            for g in goal_test:
                episode += 1
                test_re = 0
                message_  = self.get_message_with_kg(g['current_slots']['inform_slots'], self.kg_context_prompt_)
                while True:
                    test_re += 1
                    # print('episode:' + str(episode))
                    text = self.tokenizer.apply_chat_template(
                        message_,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

                    generated_ids = self.model.generate(
                        **model_inputs,
                        max_new_tokens=512
                    )
                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                    ]

                    answers_content = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    # print('gen_system:', answers_content)
                    flag, action, system_ask = self.system_generate(disease=self.disease, symptom=self.symptom, generation=answers_content)
                    # print('system:', system_ask)

                    if flag == 1:
                        if action == g['disease_tag']:
                            dis_predict += 1
                        break
                    elif flag == 3:
                        break
                    if test_re > 6:
                        break

            print("success rate %.4f" % (dis_predict / episode))
            pre_success.append(dis_predict / episode)
        return np.mean(pre_success)
    
    def test_with_kg_context3_step(self, goal_test, dise_sym_dict, e=3):
        # 基于KGprompt测试
        kg = ''
        for d in dise_sym_dict.keys():
            kg += '{}：症状包括{}。'.format(d, ','.join(dise_sym_dict[d].keys()))
            # kg += '{}通常伴随{}。'.format(d, ','.join(dise_sym_dict[d].keys()))
            # kg += '{}可能伴随{}。'.format(d, ','.join(dise_sym_dict[d].keys()))
        self.kg_context_prompt_ = self.kg_context_prompt.replace("{{此处替换成KG文本}}", str(kg))
        pre_success = []
        for i in range(e):
            dis_predict = 0.
            episode = 0.
            for g in goal_test:
                episode += 1
                test_re = 0
                message_  = self.get_message_with_kg(g['current_slots']['inform_slots'], self.kg_context_prompt_)
                while True:
                    test_re += 1
                    # print('episode:' + str(episode))
                    text = self.tokenizer.apply_chat_template(
                        message_,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

                    generated_ids = self.model.generate(
                        **model_inputs,
                        max_new_tokens=512
                    )
                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                    ]

                    answers_content = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    # print('gen_system:', answers_content)
                    flag, action, system_ask = self.system_generate(disease=self.disease, symptom=self.symptom, generation=answers_content)
                    # print('system:', system_ask)

                    if flag == 1:
                        if action == g['disease_tag']:
                            dis_predict += 1
                        break
                    elif flag == 3:
                        break
                    if test_re > 6:
                        break

            print("success rate %.4f" % (dis_predict / episode))
            pre_success.append(dis_predict / episode)
        return np.mean(pre_success)
    
    def test_with_pruned_kg(self, goal_test, dise_sym_dict, e=3):
        pre_success = []
        for i in range(e):
            dis_predict = 0.
            episode = 0.
            for g in goal_test:
                dis_reward, _ = self.reward_model.score(g)
                disease_mask = torch.ones(self.dis_shape).to(self.device).reshape(-1, self.dis_shape)
                symptoms_mask = torch.ones(self.sym_shape).to(self.device).reshape(-1, self.sym_shape)
                exit_dis = []
                exit_sym = []

                kg_matrix = copy.deepcopy(self.reward_model.kgraph.kg_matrix).view(-1, self.dis_shape + self.sym_shape, self.dis_shape + self.sym_shape)
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
                
                # 记录存在的疾病和症状
                for i in range(self.dis_shape):
                    if disease_mask[0][i] == 1.:
                        exit_dis.append(self.disease[i])
                for j in range(self.sym_shape):
                    if symptoms_mask[0][j] == 1.:
                        exit_sym.append(self.symptom[j])
                
                # 剪枝后的图谱
                kg = []
                for d in dise_sym_dict.keys():
                    if d in exit_dis:
                        for s in dise_sym_dict[d].keys():
                            if s in exit_sym:
                                kg.append((d, '导致', s))
                self.kg_prompt_ = self.kg_prompt1.replace("{{此处替换成KG三元组}}", str(kg)).replace("{{此处替换成疾病}}", str(exit_dis))
                
        
                episode += 1
                test_re = 0
                message_  = self.get_message_with_kg(g['current_slots']['inform_slots'], self.kg_prompt_)
                while True:
                    test_re += 1
                    # print('episode:' + str(episode))
                    text = self.tokenizer.apply_chat_template(
                        message_,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

                    generated_ids = self.model.generate(
                        **model_inputs,
                        max_new_tokens=512
                    )
                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                    ]

                    answers_content = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    # print('gen_system:', answers_content)
                    flag, action, system_ask = self.system_generate(disease=self.disease, symptom=self.symptom, generation=answers_content)
                    # print('system:', system_ask)

                    if flag == 1:
                        if action == g['disease_tag']:
                            dis_predict += 1
                        break
                    elif flag == 3:
                        break
                    if test_re > 6:
                        break

            print("success rate %.4f" % (dis_predict / episode))
            pre_success.append(dis_predict / episode)
        return np.mean(pre_success)

#