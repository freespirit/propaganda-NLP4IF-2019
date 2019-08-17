A list of actions and a plan how to solve the challenge.

# Plan:
 - explore the data; just look at it
 - make a pipeline
    - use a dumb baseline (e.g. everything is (not)propaganda)
    - evaluate
 - replace the dumb model with a finer one
 - tune

Source of inspiration: http://karpathy.github.io/2019/04/25/recipe/
 
# Models 
Ideas for a better model:
 - [ ] PyTorch Hub's BERT - https://pytorch.org/hub/huggingface_pytorch-pretrained-bert_bert/
 - [ ] huggingface's pytorch-transformers with [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://github.com/huggingface/pytorch-transformers/releases/tag/1.1.0). Paper is here: https://arxiv.org/abs/1907.11692    
 