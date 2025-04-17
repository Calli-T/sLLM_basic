'''
주석친 내용 다 쓸모 없다
DatasetSIFT1M은 같은 경로 아래의 /data/sift1M/파일명...을 인식한다
즉, 경로가 이미 지정된 클래스라 거기에 맞춰줘야함
for_ignore에 넣어서 git에 등록이 안되도록 한다음,
실행할 때는 빼서 위의 실행파일.py과 같은 장소에 두고
코드 실행이 끝나면 다시 지우면된다
'''

# ----- 데이터셋 읽기 -----

import psutil


def get_memory_usage_mb():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)


from faiss.contrib.datasets import DatasetSIFT1M
# import numpy as np


'''# 커스텀 DatasetSIFT1M 클래스 정의
class CustomDatasetSIFT1M(DatasetSIFT1M):
    def __init__(self, query_path, database_path, groundtruth_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 사용자 정의 경로
        self.query_path = query_path
        self.database_path = database_path
        self.groundtruth_path = groundtruth_path

    # get_queries() 오버라이드
    def get_queries(self):
        return self.read_fvecs(self.query_path)

    # get_database() 오버라이드
    def get_database(self):
        return self.read_fvecs(self.database_path)

    # get_groundtruth() 오버라이드
    def get_groundtruth(self):
        return self.read_ivecs(self.groundtruth_path)

    # .fvecs 포맷 읽는 함수
    def read_fvecs(self, path):
        with open(path, 'rb') as f:
            # 첫 번째 4바이트는 벡터 차원(d)
            d = np.fromfile(f, dtype=np.int32, count=1)[0]
            data = np.fromfile(f, dtype=np.float32)

            # 데이터 크기 체크 후 reshape 가능 여부 확인
            expected_size = d * (data.size // d)
            if expected_size != data.size:
                raise ValueError(f"Data size mismatch: {data.size} cannot be reshaped to {d}D vectors")

            return data.reshape(-1, d)

    # .ivecs 포맷 읽는 함수
    def read_ivecs(self, path):
        with open(path, 'rb') as f:
            # 첫 번째 4바이트는 벡터 차원(d)
            d = np.fromfile(f, dtype=np.int32, count=1)[0]
            data = np.fromfile(f, dtype=np.int32)
            return data'''


def get_dataset():
    # ds = CustomDatasetSIFT1M(query_path, database_path, groundtruth_path)  # SIFT1M 데이터셋
    ds = DatasetSIFT1M()
    _xq = ds.get_queries()  # 검색용 데이터
    _xb = ds.get_database()  # 저장된 벡터 데이터
    _gt = ds.get_groundtruth()  # 실제 정답 데이터

    return _xq, _xb, _gt


# 데이터셋 가져오기
xq, xb, gt = get_dataset()

# 결과 출력 (디버깅 용도)
print(f"Queries shape: {xq.shape}")
print(f"Database shape: {xb.shape}")
print(f"Groundtruth shape: {gt.shape}")

# ----- 실행 -----
import time
import faiss

k = 1
d = xq.shape[1]
nq = 1000
xq = xq[:nq]

for i in range(1, 10, 2):
    start_memory = get_memory_usage_mb()
    start_indexing = time.time()
    index = faiss.IndexFlatL2(d)
    index.add(xb[:(i + 1) * 100000])
    end_indexing = time.time()
    end_memory = get_memory_usage_mb()

    t0 = time.time()
    D, I = index.search(xq, k)
    t1 = time.time()
    print(f"데이터 {(i + 1) * 100000}개:")
    print(
        f"색인: {(end_indexing - start_indexing) * 1000 :.3f} ms ({end_memory - start_memory:.3f} MB) 검색: {(t1 - t0) * 1000 / nq :.3f} ms")

'''
실행결과
Queries shape: (10000, 128)
Database shape: (1000000, 128)
Groundtruth shape: (10000, 100)
데이터 200000개:
색인: 62.067 ms (98.000 MB) 검색: 1.675 ms
데이터 400000개:
색인: 131.344 ms (97.590 MB) 검색: 2.964 ms
데이터 600000개:
색인: 166.501 ms (97.566 MB) 검색: 4.743 ms
데이터 800000개:
색인: 227.176 ms (97.625 MB) 검색: 7.780 ms
데이터 1000000개:
색인: 278.951 ms (97.629 MB) 검색: 7.949 ms
'''