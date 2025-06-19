import time
from engine.threadpool import ThreadPool

# 학습 task를 흉내낸 더미 클래스
class FakeTask:
    def __init__(self, name, iter_cnt, layer):
        self.name = name
        self.iter_cnt = iter_cnt
        self.layer = layer
        self.priority_ = iter_cnt * 1000 + layer  # 우선순위 계산 방식

    def priority(self):
        return self.priority_

    def run(self):
        print(f"[START] {self.name} (priority={self.priority()})")
        time.sleep(1)
        print(f"[END] {self.name}")

if __name__ == "__main__":
    pool = ThreadPool(num_threads=2)

    tasks = [
        FakeTask("Task-iter0-layer0", iter_cnt=0, layer=0),
        FakeTask("Task-iter0-layer3", iter_cnt=0, layer=3),
        FakeTask("Task-iter1-layer0", iter_cnt=1, layer=0),
    ]

    # enqueue_priority()를 사용하여 우선순위 명시
    for task in tasks:
        pool.enqueue_priority(task.priority(), task.run)

    # 메인 스레드 대기
    pool.wait_all()
    pool.shutdown()
    print("[TEST] Done.")

