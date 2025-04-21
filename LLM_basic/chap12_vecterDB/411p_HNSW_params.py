import psutil
from faiss.contrib.datasets import DatasetSIFT1M


def get_memory_usage_mb():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)


def get_dataset():
    # ds = CustomDatasetSIFT1M(query_path, database_path, groundtruth_path)  # SIFT1M 데이터셋
    ds = DatasetSIFT1M()
    _xq = ds.get_queries()  # 검색용 데이터
    _xb = ds.get_database()  # 저장된 벡터 데이터
    _gt = ds.get_groundtruth()  # 실제 정답 데이터

    return _xq, _xb, _gt


# 데이터셋 가져오기
xq, xb, gt = get_dataset()

'''
# 결과 출력 (디버깅 용도)
print(f"Queries shape: {xq.shape}")
print(f"Database shape: {xb.shape}")
print(f"Groundtruth shape: {gt.shape}")
'''

# ----- params effects on performance -----
# - param m -
import time
import faiss
import numpy as np

k = 1
d = xq.shape[1]
nq = 1000
xq = xq[:nq]

for m in [8, 16, 32, 64]:
    index = faiss.IndexHNSWFlat(d, m)
    time.sleep(3)
    start_memory = get_memory_usage_mb()
    start_index = time.time()
    index.add(xb)
    end_memory = get_memory_usage_mb()
    end_index = time.time()
    print(f"M: {m} - 색인 시간: {end_index - start_index} s, 메모리 사용량: {end_memory - start_memory} MB")

    t0 = time.time()
    D, I = index.search(xq, k)
    t1 = time.time()

    recall_at_1 = np.equal(I, gt[:nq, :1]).sum() / float(nq)
    print(f"{(t1 - t0) * 1000.0 / nq:.3f} ms per query, R@1 {recall_at_1:.3f}")

# - param ef_construction -
k = 1
d = xq.shape[1]
nq = 1000
xq = xq[:nq]

for ef_construction in [40, 80, 160, 320]:
    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = ef_construction
    time.sleep(3)
    start_memory = get_memory_usage_mb()
    start_index = time.time()
    index.add(xb)
    end_memory = get_memory_usage_mb()
    end_index = time.time()
    print(
        f"efConstruction: {ef_construction} - 색인 시간: {end_index - start_index} s, 메모리 사용량: {end_memory - start_memory} MB")

    t0 = time.time()
    D, I = index.search(xq, k)
    t1 = time.time()

    recall_at_1 = np.equal(I, gt[:nq, :1]).sum() / float(nq)
    print(f"{(t1 - t0) * 1000.0 / nq:.3f} ms per query, R@1 {recall_at_1:.3f}")

# - param ef_search -
for ef_search in [16, 32, 64, 128]:
    index.hnsw.efSearch = ef_search
    t0 = time.time()
    D, I = index.search(xq, k)
    t1 = time.time()

    recall_at_1 = np.equal(I, gt[:nq, :1]).sum() / float(nq)
    print(f"{(t1 - t0) * 1000.0 / nq:.3f} ms per query, R@1 {recall_at_1:.3f}")
