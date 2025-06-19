import threading
from queue import PriorityQueue, Empty
from typing import Callable, Any
from concurrent.futures import Future, ThreadPoolExecutor
import functools


class ThreadPool:
    def __init__(self, num_threads: int = 0):
        if num_threads <= 1:
            import multiprocessing
            num_threads = multiprocessing.cpu_count()
            if num_threads <= 1:
                raise RuntimeError("This hardware does not support concurrency.")

        self.num_threads = num_threads
        self.task_queue = PriorityQueue()
        self.finished = False
        self.threads = []
        self.cv = threading.Condition()

        for idx in range(self.num_threads):
            thread = threading.Thread(target=self.worker_loop, args=(idx,), name=f"Worker-{idx}")
            thread.daemon = True
            thread.start()
            self.threads.append(thread)

    def worker_loop(self, thread_idx: int):
        while True:
            with self.cv:
                while self.task_queue.empty() and not self.finished:
                    self.cv.wait()
                if self.finished:
                    break
                priority, func = self.task_queue.get()

            try:
                func()
            except Exception as e:
                print(f"Worker-{thread_idx} task exception: {e}")
            finally:
                self.task_queue.task_done()

    def enqueue_priority(self, priority: int, func: Callable[..., Any], *args, **kwargs) -> Future:
        if self.finished:
            raise RuntimeError("ThreadPool has been terminated.")

        future = Future()

        def wrapped():
            if not future.set_running_or_notify_cancel():
                return
            try:
                result = func(*args, **kwargs)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

        with self.cv:
            self.task_queue.put((priority, wrapped))
            self.cv.notify()

        return future

    def enqueue(self, func: Callable[..., Any], *args, **kwargs) -> Future:
        print("[Warning] Enqueue without priority is not recommended")
        return self.enqueue_priority(0, func, *args, **kwargs)

    def shutdown(self):
        with self.cv:
            self.finished = True
            self.cv.notify_all()
        for thread in self.threads:
            thread.join()

    def wait_all(self):
        self.task_queue.join()

    def n_threads(self):
        return self.num_threads
