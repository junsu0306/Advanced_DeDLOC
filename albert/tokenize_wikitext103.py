#!/usr/bin/env python
""" 
이 스크립트는 BERT-tiny 모델을 위한 WikiText-103 데이터셋을 토큰화하고 저장하는 코드입니다.
토큰화(Tokenization): 원본 텍스트를 숫자로 변환하여 모델이 학습할 수 있도록 준비하는 과정
"""

import random
from collections import defaultdict
from functools import partial
from multiprocessing import cpu_count

import nltk
from datasets import load_dataset
from transformers import BertTokenizerFast  # BERT-tiny 토크나이저 사용


def create_instances_from_document(tokenizer, document, max_seq_length):
    """
    하나의 문서를 여러 개의 훈련 인스턴스로 변환합니다.
    - 문장을 여러 개의 덩어리로 나누고, 문장 순서 예측(Sentence Order Prediction) 작업을 위한 학습 데이터를 만듭니다.

    Args:
        tokenizer: BERT-tiny 토크나이저
        document: 하나의 문서 (긴 텍스트)
        max_seq_length: 최대 시퀀스 길이 (예: 128)

    Returns:
        instances: 토큰화된 문장 쌍 목록
    """
    instances = []
    current_chunk = []
    current_length = 0

    # 문서를 문장 단위로 나누기 (NLTK 사용)
    segmented_sents = list(nltk.sent_tokenize(document))

    for i, sent in enumerate(segmented_sents):
        current_chunk.append(sent)
        current_length += len(tokenizer.tokenize(sent))  # 토큰 개수 추가

        # 최대 길이를 초과하거나 문서의 끝에 도달하면 새로운 인스턴스를 생성
        if i == len(segmented_sents) - 1 or current_length >= max_seq_length:
            if len(current_chunk) > 1:
                # 문장들을 앞부분(A)과 뒷부분(B)로 나누기
                a_end = random.randint(1, len(current_chunk) - 1)

                tokens_a = current_chunk[:a_end]  # 첫 번째 문장 그룹
                tokens_b = current_chunk[a_end:]  # 두 번째 문장 그룹

                # 50% 확률로 순서를 바꿈 (Sentence Order Prediction 학습)
                if random.random() < 0.5:
                    is_random_next = True
                    tokens_a, tokens_b = tokens_b, tokens_a
                else:
                    is_random_next = False

                # 문장을 하나로 합친 후 토큰화
                instance = tokenizer(
                    " ".join(tokens_a),
                    " ".join(tokens_b),
                    truncation="longest_first",  # 너무 길면 자름
                    max_length=max_seq_length,
                    return_special_tokens_mask=True,  # 마스크 토큰 추가
                )

                # 최대 길이를 초과하지 않았는지 확인
                assert len(instance["input_ids"]) <= max_seq_length
                instance["sentence_order_label"] = 1 if is_random_next else 0
                instances.append(instance)

            # 현재 문장 덩어리 초기화
            current_chunk = []
            current_length = 0

    return instances


def tokenize_function(tokenizer, examples):
    """
    데이터셋을 토큰화하는 함수

    Args:
        tokenizer: BERT-tiny 토크나이저
        examples: 원본 텍스트 데이터셋

    Returns:
        새로운 토큰화된 데이터셋
    """
    # 빈 문장 제거
    texts = (text for text in examples["text"] if len(text) > 0 and not text.isspace())

    new_examples = defaultdict(list)

    for text in texts:
        instances = create_instances_from_document(tokenizer, text, max_seq_length=128)  # BERT-tiny는 짧은 시퀀스 사용
        for instance in instances:
            for key, value in instance.items():
                new_examples[key].append(value)

    return new_examples


if __name__ == "__main__":
    random.seed(0)
    nltk.download("punkt")
    
    #  BERT-tiny 토크나이저 로드
    tokenizer = BertTokenizerFast.from_pretrained("google/bert_uncased_L-2_H-128_A-2")

    #  WikiText 데이터셋 로드 
    wikitext = load_dataset("wikitext", "wikitext-103-v1", cache_dir="./data/cache")

    #  BERT-tiny에 맞게 데이터셋을 토큰화
    tokenized_datasets = wikitext.map(
        partial(tokenize_function, tokenizer),
        batched=True,
        num_proc=cpu_count(),  # CPU 병렬 처리 사용
        remove_columns=["text"],  # 원본 텍스트 삭제
    )

    #  BERT-tiny용 토큰화된 데이터 저장
    tokenized_datasets.save_to_disk("./data/bert_tiny_tokenized_wikitext103")
    tokenizer.save_pretrained("./data/tokenizer_bert_tiny")
