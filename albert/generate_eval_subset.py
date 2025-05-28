from datasets import load_from_disk, Dataset
import random
import os

# 경로 설정
DATASET_PATH = "./data/bert_tiny_tokenized_wikitext103" 
SPLIT_NAME = "validation"
OUTPUT_DIR = "./eval_subsets"
NUM_SUBSETS = 20
SAMPLES_PER_SUBSET = 200
SEED = 42



# 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 데이터셋 불러오기
dataset = load_from_disk(DATASET_PATH)[SPLIT_NAME]
#기존 validation 크기

# validation = dataset["validation"]
# print("Number of validation samples:", len(validation))


# 랜덤 고정
random.seed(SEED)

# N개의 subset 생성
for i in range(NUM_SUBSETS):
    sampled = random.sample(list(dataset), SAMPLES_PER_SUBSET)
    subset = Dataset.from_list(sampled)
    subset.save_to_disk(os.path.join(OUTPUT_DIR, f"val_split_{i}"))

print(f"{NUM_SUBSETS} evaluation subsets saved to: {OUTPUT_DIR}")