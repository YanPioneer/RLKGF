'''
进行症状映射 将模型输出与症状列表进行映射
'''
import os
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util

# from utils import text_to_list

mapping = {
    "有没有吐过": "呕吐",
    "精神不好": "精神萎靡",
    "肚子疼痛吗": "腹痛",
    "肚子痛吗": "腹痛",
    "肚子疼吗": "腹痛"

}


def find_most_similar(output, symptom_list, threshold=0.5):
    # print(output)
    max_similarity = 0
    best_match = None
    for symptom in symptom_list:
        similarity = SequenceMatcher(None, output, symptom).ratio()
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = symptom
    # 如果相似度低于阈值，返回 None
    # print(max_similarity)
    if max_similarity < threshold:
        return None

    return best_match


def find_most_similar_with_bert(output, symptom_list, threshold=0.5):
    # 加载模型
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(symptom_list, convert_to_tensor=True)
    output_embedding = model.encode(output, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(output_embedding, embeddings)
    best_match_idx = similarities.argmax()
    best_similarity = similarities[0, best_match_idx].item()
    # 检查是否超过阈值
    if best_similarity < threshold:
        return None
    return symptom_list[best_match_idx]


def sym_map(model_output, symptom_list):
    if model_output in mapping.keys():
        aligned_output = mapping[model_output]
    else:
        # 方法 2: 相似度匹配
        aligned_output = find_most_similar(model_output.strip().split('有没有')[-1], symptom_list)
    print('第一步', aligned_output)

    # if aligned_output is None:
    #     aligned_output = find_most_similar_with_bert(model_output.strip().split('有没有')[-1], symptom_list)

    print("模型输出:", model_output)
    print("对齐结果:", aligned_output)
    return aligned_output



# current_path = os.path.abspath(__file__)
# father_path = os.path.abspath(os.path.dirname(current_path))
# grand_path = os.path.abspath(os.path.dirname(father_path))
# MZ_path = os.path.join(grand_path, 'Data', 'MZ', 'dataset_mz')
# MZ_Disease = text_to_list(os.path.join(MZ_path, 'diseases.txt'))
# # print(MZ_Disease)
# MZ_Symptom = text_to_list(os.path.join(MZ_path, 'symptoms.txt'))
# s = sym_map('有没有出现呼吸困难的情况', MZ_Symptom)
# map_sym = sym_map('精神萎靡', MZ_Symptom)