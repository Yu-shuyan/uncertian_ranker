# Dynamic Uncertainty Ranking
Code for our NAACL 2025 paper [***Dynamic Uncertainty Ranking: Enhancing In-Context Learning for Long-Tail Knowledge in LLMs***](https://arxiv.org/abs/2410.23605) by Shuyang Yu, Runxue Bao, Parminder Bhatia, Taha Kass-Hout, Jiayu Zhou, Cao Xiao.

## Overview
![Dynamic Uncertainty Ranking illustration](framework.pdf)

Large language models (LLMs) can learn vast amounts of knowledge from diverse domains during pre-training. However, long-tail knowledge from specialized domains is often scarce
and underrepresented, rarely appearing in the models’ memorization. Prior work has shown that in-context learning (ICL) with retriever
augmentation can help LLMs better capture long-tail knowledge, reducing their reliance on pre-trained data. Despite these advances, we observe that LLM predictions for long-tail questions remain uncertain to variations in retrieved samples. To take advantage of the uncertainty in ICL for guiding LLM predictions toward correct answers on long-tail samples, we propose a reinforcement learning-based dynamic uncertainty ranking method for retrieval-augmented ICL that accounts for the varying impact of each retrieved sample on LLM predictions. Our approach prioritizes more informative and stable samples while demoting misleading ones, updating rankings based on the feedback from the LLM w.r.t. each retrieved sample. To enhance training efficiency and reduce query costs, we introduce a learnable dynamic ranking threshold, adjusted when the model encounters negative prediction shifts. Experimental results on various question-answering datasets from different domains show that our method outperforms the best baseline by 2.76%, with a notable 5.96% boost in accuracy on long-tail questions that elude zero-shot inference.

## Dataset
|Dataset|source|
--|------------
|Pubmedqa| Published by [Pubmedqa: A dataset for biomedical research question answering](https://arxiv.org/abs/1909.06146).|
|ethos-national|Preprocess following [MetaICL](https://arxiv.org/pdf/2110.15943)|
|eval-climate|Preprocess following [MetaICL](https://arxiv.org/pdf/2110.15943)|
|T-REx|Preprocess following [MetaICL](https://arxiv.org/pdf/2110.15943)|
|NatQA|Preprocess following [MetaICL](https://arxiv.org/pdf/2110.15943)|

## Training the ranker
### Example
```
python learn_rank.py --data_root ethos-national_origin --data_root_test ethos-national_origin --shot_number 5 --label ethos-national_origin_train --cand_number 1000 --train_num 200 --preselection
```

## Testing the ranker

### Example
```
python learn_rank.py --data_root ethos-national_origin --data_root_test ethos-national_origin --shot_number 5 --label ethos-national_origin_train --cand_number 1000 --train_num 200 --preselection --ckpt [The trained ranker path saved in checkpoints] --cand_ckpt [The saved cand_pids path saved in cand_results]
```



## Definition for some important parameters:
|Parameter name | Deifinition|
| ------------- |------------|
|prompt_format|The prompt format template could be referred to Table 4 in the appendix of our paper. E.g., 'SQ-A' for Pubmedqa, and 'Q-A' for other datasets.|
|label|The name of this run. Please identify different names for different runs to avoid conflict, as the name of the results file will be based on this parameter.|
|cand_ckpt|cand_pids (saved candidate sample pids) root path. You can use the candidate sample pids saved during training for testing.|
|score_th|Initialize threshold for retriever score. We usually initialize the threshold as 0, the threshold will slowly increase dynamically during training.|
|case_study|Whether to save the case study results.|



