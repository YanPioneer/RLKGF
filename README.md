## RLKGF: Reinforcement Learning from Knowledge Graph Feedback Without Human Annotations

Reinforcement Learning from Human Feedback (RLHF) has been shown to effectively align large language models (LLMs) with human knowledge. However, the lack of human preference labels remains a significant bottleneck when applying RLHF to a downstream domain. Humans in RLHF play a critical role in injecting reasoning preferences into LLM, and we assume the reasoning process underlying human assessments may potentially be replaced by reasoning pathways derived from Knowledge Graphs (KGs). Inspired by this assumption, we propose Reinforcement Learning from Knowledge Graph Feedback (RLKGF), a novel method that leverages KG semantics and structure to derive RL rewards in the absence of manual annotations. Unlike Reinforcement Learning from AI Feedback (RLAIF), RLKGF directly integrates human priors encoded in KGs as the reward model, aligning LLM responses with expert knowledge without additional preference labeling or reward model training. RLKGF structures context-relevant facts into knowledge subgraphs and defines rewards by simulating information flow across semantic and logical connections between question and candidate response entities. Experiments on three public and one private medical dialogue dataset demonstrate that RLKGF significantly outperforms the competitive RLAIF in improving LLM diagnostic accuracy.

![img](https://github.com/YanPioneer/RLKGF/blob/main/image/Introduction_yan.png)

### Task Definition and Method
The disease diagnosis via Q\&A task requires the model to predict a disease $d$ in the answer $A$ based on a patient's symptom description $[s_1, s_2, ..., s_n]$ in the question $Q$.
Our focus is on using a medical knowledge graph (MKG) containing factual entities as a reward model to automatically assign feedback $R$ to model responses, i.e., RLKGF. 

The MKG $G = (V, E)$ is constructed based on the standard \cite{li2025quality}, where $V$ includes all symptom and disease entities in the dataset, and $E$ represents the relationships between these entities. In this task, we only consider the relationship < disease, causes, symptom >. After extracting the patient's information from the question, RLKGF first locates the relevant symptom entities in the MKG. It then identifies the disease entities connected to those symptoms and any other symptoms that these diseases may cause. Using the entities and their relationships, we construct the personalized diagnostic subgraph $g = (v, e)$, where $v$ represents the disease entities and related symptoms that may explain the patient's condition, and $e$ represents the corresponding triple relationships.

RLKGF evaluates the correctness of the model's response through path reasoning and semantic aggregation using graph-based random walk with restart (RWR) and GNNs. After acquiring feedback, RLKGF optimizes the model's policy using the proximal policy optimization (PPO) to align LLM responses with domain knowledge.

![img](https://github.com/YanPioneer/RLKGF/blob/main/image/main_figure_yan.png)


### Train

run xxxx/xxx_train.py

### Requirements

See requirement.txt

### Datasets

We utilize three public medical dialogue diagnosis datasets: MZ, DXY, and GMD. These datasets are derived from real-world medical dialogue diagnosis records.

We construct a new dataset, MED-D. MED-D is collected from offline electronic medical records (EMRs). These EMRs are sourced from cooperating hospitals and have been anonymized. We filter 14,277 EMRs and choose 20 diseases that could be diagnosed through Q\&A without additional tests. With the assistance of medical experts, we identify 351 associated symptoms. Subsequently, we extract symptom and disease entities from the selected EMRs using named entity recognition to construct Q\&A pairs. All extracted diseases and symptoms are manually aligned with the corresponding ICD-9 terms and reviewed by domain experts. We use the accuracy of disease prediction as the evaluation metric.

### Main Result
Table2 shows the performance of different LLMs trained using only RLAIF and RLKGF. \textbf{RLKGF Base} represents the accuracy achieved by directly selecting the response entity with the highest feedback score from the knowledge graph. From the experimental results, we observe the following.

\textbf{i. Advantages of RLKGF.} Our results demonstrate that RLKGF outperforms RLAIF by 5.67\%, 10.73\%, 8.38\%, and 1.21\% across four datasets, respectively. This indicates the feasibility and effectiveness of using KGs for feedback on model responses. It validates that leveraging KGs as reward models in the medical domain may be a more reliable approach than LLM-based preference labeling.

\textbf{ii. Small models are limited by instruction adherence.} Among different models, Qwen2.5-0.5b-instruct performs poorly, with only 32.39\% on the MZ dataset. We analyze its outputs before and after training and find that it has poor instruction adherence and fails to make correct predictions from the given diseases. Although training improves its instruction-following ability, knowledge injection remains suboptimal. We present the performance of models trained with supervised fine-tuning, where full-parameter SFT on Qwen2.5-0.5b-instruct achieves only 7.60\% accuracy. Preliminary analysis suggests that the MZ dataset's sparsity is insufficient to correct the initially learned model parameters. Additionally, smaller models may be more sensitive to loss design, and how to better inject knowledge into them requires further investigation.

\textbf{iii. Explore more effective KG feedback methods.} Furthermore, by comparing the prediction accuracy of models using only the KG, KG feedback-trained models, GPT-4o-mini predictions, RLAIF, and SFT, we find that although trained LLMs show some performance improvement, they still fall far short of the optimal target. Therefore, further exploration is needed on how to fully utilize factual knowledge and construct reasonable feedback to guide model training.

### RLKGF demonstrates better generalization capability compared to SFT.
We conduct an experiment to compare the generalization ability of RLKGF and SFT. Specifically, we use Qwen2.5-3B-Instruct as the base model, train it on the GMD dataset using both RLKGF and SFT, and evaluate the models on the DXY dataset. As shown in Table 4, the model trained with RLKGF demonstrates clear generalization to a different dataset, while the SFT-trained model even underperforms the untrained baseline, which highlights the superior generalization capability of RLKGF.
| Backbone            | Method       | DXY    |
| ------------------- | ------------ | ------ |
| Qwen2.5-3b-Instruct | Base         | 0.4531 |
|                     | RLKGF on GMD | 0.6117 |
|                     | SFT on GMD   | 0.3680 |

Table 4. Generalization comparison between RLKGF and SFT.Disease Probability (RWR)


