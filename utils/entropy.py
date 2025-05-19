import math
from collections import Counter
import torch

def ngram_entropy(vector: torch.Tensor, n: int) -> float:
    
    # tensor를 리스트로 변환
    seq = vector.tolist()
    
    # n-gram 추출: 연속된 n개의 element를 튜플로 생성
    ngrams = [tuple(seq[i:i+n]) for i in range(len(seq) - n + 1)]
    
    # 각 n-gram의 빈도수 계산
    counter = Counter(ngrams)
    total = sum(counter.values())
    
    # 엔트로피 계산: -sum(p * log2(p))
    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log2(p)
        
    return entropy