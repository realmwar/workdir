# Databricks notebook source
# MAGIC %md
# MAGIC # 07 — Concurrency & Multitasking Experiments
# MAGIC
# MAGIC **Goals:**
# MAGIC - Demonstrate ThreadPoolExecutor / ProcessPoolExecutor for parallel batch predictions.
# MAGIC - Show asyncio patterns for asynchronous API calls.
# MAGIC - Thread synchronization examples (locks, semaphores, conditions).
# MAGIC - Benchmark sequential vs concurrent execution.
# MAGIC - Document GIL implications and when to use threads vs processes.
# MAGIC
# MAGIC **Competency coverage:**
# MAGIC - Programming → Use the basics of concurrency and multitasking (KEY)
# MAGIC - Programming → Use concurrency and multitasking in AI pipelines (KEY)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0) Install dependencies (Serverless)

# COMMAND ----------

# MAGIC %md
# MAGIC Section 8 uses `aiohttp` for async HTTP calls and `nest_asyncio` so we can run an event loop
# MAGIC inside the Databricks notebook (which already owns one). Neither is preinstalled on Serverless,
# MAGIC so we `pip install` them inline and call `dbutils.library.restartPython()` so the kernel picks
# MAGIC up the freshly installed wheels before any imports below run.

# COMMAND ----------

# MAGIC %md
# MAGIC This install cell adds the two async-specific libraries used later in the notebook. `aiohttp`
# MAGIC gives us non-blocking HTTP calls, and `nest_asyncio` lets a Databricks notebook reuse its already
# MAGIC running event loop when we demonstrate asyncio patterns.

# COMMAND ----------

# MAGIC %pip install aiohttp nest_asyncio

# COMMAND ----------

# MAGIC %md
# MAGIC After `%pip install`, Databricks needs a Python restart before the new packages can be imported
# MAGIC like normal modules in the next cells.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Imports & setup

# COMMAND ----------

# MAGIC %md
# MAGIC This cell gathers the standard-library building blocks for threads, processes, queues, and asyncio,
# MAGIC plus NumPy and pandas for small synthetic workloads and benchmark summaries.

# COMMAND ----------

import time
import threading
import multiprocessing
import asyncio
import queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Understanding the GIL
# MAGIC
# MAGIC The **Global Interpreter Lock (GIL)** in CPython means only one thread can execute
# MAGIC Python bytecode at a time. This has important implications:
# MAGIC
# MAGIC | Scenario | Threads | Processes |
# MAGIC |---|---|---|
# MAGIC | **I/O-bound** (API calls, file reads, DB queries) | Effective — threads release GIL during I/O waits | Overkill — overhead of IPC not justified |
# MAGIC | **CPU-bound** (math, ML inference, data transforms) | Ineffective — GIL serializes execution | Effective — each process has its own GIL |
# MAGIC | **Mixed** (inference + API calls) | Moderate — depends on I/O ratio | Best for heavy CPU work |
# MAGIC
# MAGIC **Key takeaway:** Use `ThreadPoolExecutor` for I/O-bound tasks, `ProcessPoolExecutor` for CPU-bound tasks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Simulating workloads

# COMMAND ----------

# MAGIC %md
# MAGIC These helpers create repeatable fake workloads for the concurrency experiments. Instead of tying
# MAGIC the notebook to a real API or production model, we simulate I/O waits, CPU-heavy loops, and chunked
# MAGIC batch predictions so the behavior of each concurrency primitive is easy to observe.

# COMMAND ----------

def simulate_io_task(task_id, sleep_seconds=0.5):
    """Simulate an I/O-bound task (e.g., API call, DB query).

    In reality this would be a network request or file read; we use sleep
    to represent the I/O wait time where the GIL is released.
    """
    time.sleep(sleep_seconds)
    return {"task_id": task_id, "result": f"io_result_{task_id}", "duration": sleep_seconds}


def simulate_cpu_task(task_id, n=5_000_000):
    """Simulate a CPU-bound task (e.g., feature computation, model inference).

    This is pure Python computation — the GIL will NOT be released,
    so threading won't speed this up.
    """
    total = 0
    for i in range(n):
        total += i * i
    return {"task_id": task_id, "result": total, "n": n}


def batch_predict_chunk(chunk_data):
    """Simulate a batch prediction on a data chunk.

    Represents a real-world scenario where we split a large prediction
    batch into smaller chunks and process them in parallel.
    """
    chunk_id, data = chunk_data
    time.sleep(0.1)  # Simulate model loading/overhead.
    predictions = data * 1.5 + np.random.normal(0, 0.1, len(data))
    return chunk_id, predictions

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) Experiment 1: Sequential vs ThreadPoolExecutor (I/O-bound)

# COMMAND ----------

# MAGIC %md
# MAGIC The first experiment compares ordinary sequential execution with a thread pool on an I/O-bound
# MAGIC workload. Because the work spends most of its time waiting, threads should overlap those waits and
# MAGIC deliver a visible speedup.

# COMMAND ----------

N_TASKS = 20

# --- Sequential ---
start = time.time()
seq_results = [simulate_io_task(i) for i in range(N_TASKS)]
seq_time = time.time() - start
print(f"Sequential I/O ({N_TASKS} tasks): {seq_time:.2f}s")

# --- ThreadPoolExecutor ---
start = time.time()
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(simulate_io_task, i): i for i in range(N_TASKS)}
    thread_results = []
    for future in as_completed(futures):
        thread_results.append(future.result())
thread_time = time.time() - start
print(f"ThreadPool I/O ({N_TASKS} tasks, 8 workers): {thread_time:.2f}s")
print(f"Speedup: {seq_time / thread_time:.1f}x")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5) Experiment 2: Sequential vs ProcessPoolExecutor (CPU-bound)

# COMMAND ----------

# MAGIC %md
# MAGIC This experiment switches to CPU-heavy work to show the opposite pattern: threads do not help much
# MAGIC because of the GIL, while separate processes can run in parallel on multiple cores.

# COMMAND ----------

N_CPU_TASKS = 8

# --- Sequential ---
start = time.time()
cpu_seq = [simulate_cpu_task(i, n=2_000_000) for i in range(N_CPU_TASKS)]
cpu_seq_time = time.time() - start
print(f"Sequential CPU ({N_CPU_TASKS} tasks): {cpu_seq_time:.2f}s")

# --- ThreadPoolExecutor (will NOT help due to GIL) ---
start = time.time()
with ThreadPoolExecutor(max_workers=4) as executor:
    cpu_thread = list(executor.map(lambda i: simulate_cpu_task(i, n=2_000_000), range(N_CPU_TASKS)))
cpu_thread_time = time.time() - start
print(f"ThreadPool CPU ({N_CPU_TASKS} tasks, 4 workers): {cpu_thread_time:.2f}s")
print(f"Thread speedup: {cpu_seq_time / cpu_thread_time:.1f}x (expected ~1x due to GIL)")

# --- ProcessPoolExecutor (bypasses GIL) ---
start = time.time()
with ProcessPoolExecutor(max_workers=4) as executor:
    cpu_proc = list(executor.map(partial(simulate_cpu_task, n=2_000_000), range(N_CPU_TASKS)))
cpu_proc_time = time.time() - start
print(f"ProcessPool CPU ({N_CPU_TASKS} tasks, 4 workers): {cpu_proc_time:.2f}s")
print(f"Process speedup: {cpu_seq_time / cpu_proc_time:.1f}x")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6) Experiment 3: Parallel batch predictions (AI pipeline pattern)

# COMMAND ----------

# MAGIC %md
# MAGIC This cell translates the concurrency discussion into an ML pipeline pattern: split a large batch
# MAGIC of prediction inputs into chunks and compare sequential inference with parallel chunk processing.

# COMMAND ----------

# Simulate a large prediction dataset split into chunks.
total_rows = 100_000
chunk_size = 10_000
data = np.random.uniform(0, 100, total_rows)
chunks = [(i, data[i * chunk_size:(i + 1) * chunk_size]) for i in range(total_rows // chunk_size)]

# --- Sequential ---
start = time.time()
seq_preds = [batch_predict_chunk(c) for c in chunks]
seq_pred_time = time.time() - start
print(f"Sequential batch predict: {seq_pred_time:.2f}s")

# --- Parallel with ThreadPoolExecutor ---
start = time.time()
with ThreadPoolExecutor(max_workers=4) as executor:
    par_preds = list(executor.map(batch_predict_chunk, chunks))
par_pred_time = time.time() - start
print(f"Parallel batch predict (4 workers): {par_pred_time:.2f}s")
print(f"Speedup: {seq_pred_time / par_pred_time:.1f}x")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7) Thread synchronization: Locks, Semaphores, Conditions

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7a) Lock — protecting shared state

# COMMAND ----------

# MAGIC %md
# MAGIC A lock is the simplest synchronization primitive for shared mutable state. This example uses a
# MAGIC shared counter to show how a critical section prevents race conditions during concurrent increments.

# COMMAND ----------

# Without a lock, concurrent counter increments cause race conditions.
class SharedCounter:
    """Thread-safe counter using a Lock to prevent race conditions.

    Without the lock, concurrent increments could read stale values
    and produce incorrect totals.
    """

    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()

    def increment(self, n=1):
        with self._lock:  # Acquire lock before modifying shared state.
            current = self._value
            time.sleep(0.0001)  # Exaggerate the race condition window.
            self._value = current + n

    @property
    def value(self):
        return self._value


# Demonstrate: 100 threads incrementing a counter.
counter = SharedCounter()
threads = [threading.Thread(target=counter.increment) for _ in range(100)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"Thread-safe counter: {counter.value} (expected 100)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7b) Semaphore — limiting concurrent access

# COMMAND ----------

# MAGIC %md
# MAGIC A semaphore limits how many workers can enter a code region at once. That maps naturally to
# MAGIC real-world throttling scenarios such as outbound API calls or scarce shared resources.

# COMMAND ----------

# Semaphore limits how many threads can access a resource simultaneously.
# Use case: limiting concurrent API connections to avoid throttling.
semaphore = threading.Semaphore(3)  # Max 3 concurrent workers.
active_count = []
active_lock = threading.Lock()


def rate_limited_task(task_id):
    with semaphore:
        with active_lock:
            active_count.append(threading.active_count())
        time.sleep(0.2)
        return task_id


with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(rate_limited_task, range(15)))

print(f"Semaphore demo: {len(results)} tasks completed")
print(f"Max concurrent threads observed: {max(active_count)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7c) Condition — producer/consumer pattern

# COMMAND ----------

# MAGIC %md
# MAGIC Conditions coordinate threads when one side needs to wait for a state change. Here the consumer
# MAGIC sleeps until the producer notifies it that new data has been placed into the shared buffer.

# COMMAND ----------

# Producer-consumer with a Condition variable.
# The consumer waits until the producer signals that data is ready.
buffer = queue.Queue(maxsize=5)
condition = threading.Condition()
SENTINEL = object()


def producer(items):
    """Produces items into the shared buffer, notifying consumers."""
    for item in items:
        with condition:
            buffer.put(item)
            condition.notify()
            time.sleep(0.05)
    with condition:
        buffer.put(SENTINEL)
        condition.notify()


def consumer():
    """Consumes items from the buffer, waiting for notifications."""
    consumed = []
    while True:
        with condition:
            condition.wait_for(lambda: not buffer.empty())
            item = buffer.get()
            if item is SENTINEL:
                break
            consumed.append(item)
    return consumed


# Run producer and consumer in separate threads.
items_to_produce = list(range(10))
consumer_results = []

prod_thread = threading.Thread(target=producer, args=(items_to_produce,))
cons_thread = threading.Thread(target=lambda: consumer_results.extend(consumer()))

prod_thread.start()
cons_thread.start()
prod_thread.join()
cons_thread.join()

print(f"Produced: {items_to_produce}")
print(f"Consumed: {consumer_results}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8) Asyncio patterns for asynchronous operations

# COMMAND ----------

# MAGIC %md
# MAGIC This section demonstrates asynchronous I/O with coroutines. The example is intentionally written
# MAGIC like a mini serving client: build a batch of payloads, send them concurrently, and collect all
# MAGIC responses without blocking one request on another.

# COMMAND ----------

import aiohttp
import asyncio

async def async_predict(session, url, payload):
    """Asynchronous API call — non-blocking I/O with coroutines.

    This pattern is ideal for calling multiple prediction endpoints
    concurrently without blocking the event loop.
    """
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            return await resp.json()
    except Exception as e:
        return {"error": str(e)}


async def batch_async_predictions(api_url, payloads):
    """Run multiple prediction requests concurrently using asyncio.

    Uses asyncio.gather to await all coroutines simultaneously,
    maximizing throughput for I/O-bound prediction calls.
    """
    async with aiohttp.ClientSession() as session:
        tasks = [async_predict(session, api_url, p) for p in payloads]
        return await asyncio.gather(*tasks)


# Demonstrate the pattern (will fail gracefully if no API is running).
sample_payloads = [
    {"instances": [{"store_id": i, "day_of_week": 1, "is_weekend": 0,
                     "is_open": 1, "is_promo": 0, "state_holiday_code": 0,
                     "is_school_holiday": 0, "store_type": 0, "assortment_type": 0,
                     "competition_distance_km": 5.0, "has_promo2": 0,
                     "is_promo2_active": 0, "is_promo_interval_month": 0}]}
    for i in range(1, 6)
]

try:
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # In Databricks/Jupyter, the event loop is already running.
        import nest_asyncio
        nest_asyncio.apply()
    results = asyncio.get_event_loop().run_until_complete(
        batch_async_predictions("http://localhost:8000/predict/rossmann", sample_payloads)
    )
    print(f"Async results: {len(results)} responses received")
except Exception as e:
    print(f"Async demo (expected if API not running): {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9) Shared Memory — multiprocessing.Value and Array

# COMMAND ----------

# MAGIC %md
# MAGIC Processes do not share Python objects by default, so shared memory needs explicit primitives.
# MAGIC This example writes into a synchronized multiprocessing array from child processes to show how
# MAGIC cross-process state can still be coordinated safely.

# COMMAND ----------

def worker_shared_mem(shared_array, index, value):
    """Write to a shared memory array from a child process.

    multiprocessing.Array provides a synchronized shared memory block
    that multiple processes can read/write without corruption.
    """
    shared_array[index] = value


shared_arr = multiprocessing.Array('d', 10)  # 'd' = double, 10 elements.
processes = []
for i in range(10):
    p = multiprocessing.Process(target=worker_shared_mem, args=(shared_arr, i, i * 3.14))
    processes.append(p)
    p.start()

for p in processes:
    p.join()

print(f"Shared memory array: {list(shared_arr)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10) Benchmark summary

# COMMAND ----------

# MAGIC %md
# MAGIC The summary table consolidates the timing measurements from the earlier experiments into one place.
# MAGIC Grouping the results by workload category makes the speedup patterns easier to compare side by side.

# COMMAND ----------

benchmark = pd.DataFrame({
    "experiment": [
        "I/O Sequential", "I/O ThreadPool",
        "CPU Sequential", "CPU ThreadPool", "CPU ProcessPool",
        "Batch Predict Sequential", "Batch Predict Parallel",
    ],
    "time_seconds": [
        seq_time, thread_time,
        cpu_seq_time, cpu_thread_time, cpu_proc_time,
        seq_pred_time, par_pred_time,
    ],
    "category": ["io", "io", "cpu", "cpu", "cpu", "ml", "ml"],
})

benchmark["speedup_vs_sequential"] = benchmark.groupby("category")["time_seconds"].transform(
    lambda x: x.iloc[0] / x
)

print(benchmark.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC The final chart is a compact visual summary of the notebook: I/O, CPU, and ML-style workloads each
# MAGIC get their own panel so the trade-offs between sequential, threaded, and process-based execution are
# MAGIC easy to scan.

# COMMAND ----------

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for i, cat in enumerate(["io", "cpu", "ml"]):
    subset = benchmark[benchmark["category"] == cat]
    axes[i].barh(subset["experiment"], subset["time_seconds"], color=["#4C72B0", "#DD8452", "#55A868"][:len(subset)])
    axes[i].set_xlabel("Time (seconds)")
    axes[i].set_title(f"{cat.upper()}-bound benchmarks")
    axes[i].invert_yaxis()

plt.tight_layout()
plt.savefig("/tmp/concurrency_benchmarks.png", dpi=100)
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11) Summary
# MAGIC
# MAGIC | Concept | What was demonstrated |
# MAGIC |---|---|
# MAGIC | GIL | Explained implications for threads vs processes |
# MAGIC | ThreadPoolExecutor | Parallel I/O-bound tasks (API calls, file reads) |
# MAGIC | ProcessPoolExecutor | Parallel CPU-bound tasks (bypasses GIL) |
# MAGIC | Lock | Thread-safe shared state modification |
# MAGIC | Semaphore | Rate-limiting concurrent resource access |
# MAGIC | Condition | Producer-consumer coordination |
# MAGIC | asyncio | Non-blocking I/O with coroutines and gather |
# MAGIC | Shared Memory | Cross-process data sharing via multiprocessing.Array |
# MAGIC | Batch prediction | Chunked parallel inference pattern for ML pipelines |
# MAGIC
# MAGIC **Next step:** `08_code_assessment.py` — profiling, refactoring, DS&A examples.
