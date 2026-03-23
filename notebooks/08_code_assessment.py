# Databricks notebook source
# MAGIC %md
# MAGIC # 08 — Code Quality Assessment, DS&A, and Refactoring
# MAGIC
# MAGIC **Goals:**
# MAGIC - Code profiling of key pipeline stages.
# MAGIC - Refactoring examples with before/after patterns.
# MAGIC - Design pattern demonstrations applicable to our pipeline.
# MAGIC - Data structures and algorithms: sorting, searching, hash tables, trees, queues.
# MAGIC - Big O complexity analysis for each algorithm.
# MAGIC
# MAGIC **Competency coverage:**
# MAGIC - Programming → Use basic algorithms and data structures (KEY)
# MAGIC - Programming → Use advanced data structures and algorithms (KEY)
# MAGIC - Programming → Use advanced software engineering concepts (KEY)
# MAGIC - Programming → Perform code assessments and propose improvements (KEY)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Imports

# COMMAND ----------

import time
import cProfile
import pstats
import io
from collections import deque, defaultdict
from typing import Any, List, Optional, Protocol
import heapq
import numpy as np
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## PART A: Data Structures & Algorithms
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Big O Notation — Complexity Reference
# MAGIC
# MAGIC | Notation | Name | Example |
# MAGIC |---|---|---|
# MAGIC | O(1) | Constant | Hash table lookup, array index access |
# MAGIC | O(log n) | Logarithmic | Binary search, balanced BST lookup |
# MAGIC | O(n) | Linear | Linear search, single loop |
# MAGIC | O(n log n) | Linearithmic | Merge sort, quicksort (avg) |
# MAGIC | O(n²) | Quadratic | Bubble sort, nested loops |
# MAGIC | O(2ⁿ) | Exponential | Recursive Fibonacci (naive) |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Basic data structures: Lists, Dicts, Sets

# COMMAND ----------

# --- Lists (dynamic arrays) ---
# Time complexity: access O(1), append O(1) amortized, insert O(n), search O(n).
sample_list = list(range(10_000))

# Benchmark: list access vs search.
start = time.time()
_ = sample_list[5000]  # O(1) index access.
access_time = time.time() - start

start = time.time()
_ = 5000 in sample_list  # O(n) linear search.
search_time = time.time() - start

print(f"List access (O(1)): {access_time:.8f}s")
print(f"List search (O(n)): {search_time:.8f}s")

# --- Dictionaries (hash tables) ---
# Time complexity: get/set/delete O(1) average, O(n) worst case (hash collisions).
sample_dict = {i: f"value_{i}" for i in range(10_000)}

start = time.time()
_ = sample_dict[5000]  # O(1) hash lookup.
dict_time = time.time() - start
print(f"Dict lookup (O(1)): {dict_time:.8f}s")

# --- Sets (hash sets) ---
# Time complexity: add/remove/lookup O(1) average.
set_a = set(range(0, 10_000, 2))     # Even numbers.
set_b = set(range(0, 10_000, 3))     # Multiples of 3.

start = time.time()
union = set_a | set_b                  # O(len(a) + len(b))
intersect = set_a & set_b             # O(min(len(a), len(b)))
diff = set_a - set_b                   # O(len(a))
set_time = time.time() - start

print(f"Set operations (union/intersect/diff): {set_time:.8f}s")
print(f"  Union: {len(union)}, Intersection: {len(intersect)}, Difference: {len(diff)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) Sorting algorithms

# COMMAND ----------

def bubble_sort(arr):
    """Bubble sort — O(n²) time, O(1) space.

    Repeatedly swaps adjacent elements if they are in the wrong order.
    Simple but inefficient for large datasets.
    """
    a = arr.copy()
    n = len(a)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                swapped = True
        if not swapped:
            break
    return a


def insertion_sort(arr):
    """Insertion sort — O(n²) time, O(1) space.

    Builds the sorted array one element at a time by inserting each
    element into its correct position. Efficient for small/nearly sorted data.
    """
    a = arr.copy()
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
    return a


def quicksort(arr):
    """Quicksort — O(n log n) average, O(n²) worst case, O(log n) space.

    Divide-and-conquer: picks a pivot, partitions the array into elements
    less/greater than the pivot, and recursively sorts each partition.
    """
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)


def merge_sort(arr):
    """Merge sort — O(n log n) time, O(n) space.

    Divide-and-conquer: splits the array in half, recursively sorts each
    half, then merges them back in sorted order. Stable and predictable.
    """
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return _merge(left, right)


def _merge(left, right):
    """Merge two sorted lists into one sorted list."""
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


# Benchmark sorting algorithms.
test_data = list(np.random.randint(0, 100_000, 2000))

benchmarks = {}
for name, func in [("bubble_sort", bubble_sort), ("insertion_sort", insertion_sort),
                     ("quicksort", quicksort), ("merge_sort", merge_sort),
                     ("python_sorted", sorted)]:
    start = time.time()
    result = func(test_data)
    elapsed = time.time() - start
    benchmarks[name] = elapsed
    print(f"  {name:20s}: {elapsed:.4f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5) Searching algorithms

# COMMAND ----------

def linear_search(arr, target):
    """Linear search — O(n) time, O(1) space.

    Scans every element sequentially until the target is found.
    Works on unsorted data.
    """
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1


def binary_search(arr, target):
    """Binary search — O(log n) time, O(1) space.

    Requires a sorted array. Repeatedly halves the search space
    by comparing the target to the middle element.
    """
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1


# Benchmark.
sorted_data = sorted(test_data)
target = sorted_data[1500]

start = time.time()
for _ in range(10_000):
    linear_search(sorted_data, target)
linear_time = time.time() - start

start = time.time()
for _ in range(10_000):
    binary_search(sorted_data, target)
binary_time = time.time() - start

print(f"Linear search (10k runs): {linear_time:.4f}s  — O(n)")
print(f"Binary search (10k runs): {binary_time:.4f}s  — O(log n)")
print(f"Speedup: {linear_time / binary_time:.1f}x")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6) Advanced data structures: Hash tables, Trees, Queues

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6a) Hash Table (custom implementation)

# COMMAND ----------

class HashTable:
    """Simple hash table with separate chaining for collision resolution.

    Time complexity: O(1) average for get/set/delete, O(n) worst case.
    Space complexity: O(n).
    """

    def __init__(self, capacity=64):
        self._capacity = capacity
        self._size = 0
        self._buckets: List[List] = [[] for _ in range(capacity)]

    def _hash(self, key) -> int:
        return hash(key) % self._capacity

    def set(self, key, value):
        idx = self._hash(key)
        for i, (k, v) in enumerate(self._buckets[idx]):
            if k == key:
                self._buckets[idx][i] = (key, value)
                return
        self._buckets[idx].append((key, value))
        self._size += 1

    def get(self, key, default=None):
        idx = self._hash(key)
        for k, v in self._buckets[idx]:
            if k == key:
                return v
        return default

    def delete(self, key) -> bool:
        idx = self._hash(key)
        for i, (k, v) in enumerate(self._buckets[idx]):
            if k == key:
                self._buckets[idx].pop(i)
                self._size -= 1
                return True
        return False

    def __len__(self):
        return self._size


ht = HashTable()
for i in range(100):
    ht.set(f"key_{i}", i * 10)

print(f"HashTable size: {len(ht)}")
print(f"get('key_50'): {ht.get('key_50')}")
ht.delete("key_50")
print(f"After delete: get('key_50'): {ht.get('key_50', 'NOT FOUND')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6b) Binary Search Tree (BST)

# COMMAND ----------

class BSTNode:
    """Node in a Binary Search Tree."""

    def __init__(self, key, value=None):
        self.key = key
        self.value = value
        self.left: Optional["BSTNode"] = None
        self.right: Optional["BSTNode"] = None


class BinarySearchTree:
    """Binary Search Tree — O(log n) avg, O(n) worst for search/insert/delete.

    Maintains the BST property: left.key < node.key < right.key.
    Unbalanced worst case degenerates to a linked list.
    """

    def __init__(self):
        self.root: Optional[BSTNode] = None
        self._size = 0

    def insert(self, key, value=None):
        if self.root is None:
            self.root = BSTNode(key, value)
        else:
            self._insert_recursive(self.root, key, value)
        self._size += 1

    def _insert_recursive(self, node, key, value):
        if key < node.key:
            if node.left is None:
                node.left = BSTNode(key, value)
            else:
                self._insert_recursive(node.left, key, value)
        elif key > node.key:
            if node.right is None:
                node.right = BSTNode(key, value)
            else:
                self._insert_recursive(node.right, key, value)
        else:
            node.value = value

    def search(self, key) -> Optional[Any]:
        return self._search_recursive(self.root, key)

    def _search_recursive(self, node, key):
        if node is None:
            return None
        if key == node.key:
            return node.value
        elif key < node.key:
            return self._search_recursive(node.left, key)
        else:
            return self._search_recursive(node.right, key)

    def inorder(self) -> list:
        """In-order traversal returns keys in sorted order."""
        result = []
        self._inorder_recursive(self.root, result)
        return result

    def _inorder_recursive(self, node, result):
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.key)
            self._inorder_recursive(node.right, result)

    def __len__(self):
        return self._size


bst = BinarySearchTree()
for val in [50, 30, 70, 20, 40, 60, 80]:
    bst.insert(val, f"data_{val}")

print(f"BST size: {len(bst)}")
print(f"search(40): {bst.search(40)}")
print(f"Inorder traversal (sorted): {bst.inorder()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6c) Priority Queue (min-heap)

# COMMAND ----------

class PriorityQueue:
    """Min-heap priority queue — O(log n) push/pop, O(1) peek.

    Use case: scheduling tasks by priority, e.g., processing high-priority
    prediction requests before low-priority ones.
    """

    def __init__(self):
        self._heap: list = []
        self._counter = 0  # Tie-breaker for equal priorities.

    def push(self, priority: float, item: Any):
        heapq.heappush(self._heap, (priority, self._counter, item))
        self._counter += 1

    def pop(self):
        if not self._heap:
            raise IndexError("Priority queue is empty")
        priority, _, item = heapq.heappop(self._heap)
        return priority, item

    def peek(self):
        if not self._heap:
            return None
        return self._heap[0][0], self._heap[0][2]

    def __len__(self):
        return len(self._heap)

    @property
    def is_empty(self):
        return len(self._heap) == 0


pq = PriorityQueue()
pq.push(3.0, "low priority batch")
pq.push(1.0, "urgent real-time request")
pq.push(2.0, "normal API request")

print("Priority Queue dequeue order:")
while not pq.is_empty:
    priority, item = pq.pop()
    print(f"  priority={priority} -> {item}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6d) Deque (double-ended queue)

# COMMAND ----------

# collections.deque: O(1) append/pop from both ends (vs O(n) for list.insert(0)).
dq = deque(maxlen=5)
for i in range(8):
    dq.append(i)

print(f"Deque (maxlen=5): {list(dq)}")
print(f"popleft: {dq.popleft()}, pop: {dq.pop()}")
print(f"After pops: {list(dq)}")

# Use case: sliding window buffer for streaming predictions.
print("\nSliding window demo:")
window = deque(maxlen=3)
for val in [10, 20, 30, 40, 50]:
    window.append(val)
    print(f"  Added {val} -> window: {list(window)}, mean: {sum(window)/len(window):.1f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## PART B: Code Assessment & Refactoring
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7) Code Profiling

# COMMAND ----------

def profile_function(func, *args, **kwargs):
    """Profile a function and return formatted stats."""
    profiler = cProfile.Profile()
    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(15)
    print(stream.getvalue())
    return result


# Profile the sorting algorithms.
large_data = list(np.random.randint(0, 1_000_000, 5000))
print("=== Profiling quicksort ===")
_ = profile_function(quicksort, large_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8) Refactoring examples — Before / After

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8a) Extract Method + Single Responsibility

# COMMAND ----------

# BEFORE: monolithic function doing too many things.
def process_data_bad(raw_data):
    cleaned = []
    for row in raw_data:
        if row.get("sales") is not None and row["sales"] >= 0:
            cleaned.append(row)
    features = []
    for row in cleaned:
        feat = {"store": row["store_id"], "sales_log": np.log1p(row["sales"])}
        features.append(feat)
    predictions = []
    for feat in features:
        predictions.append(feat["sales_log"] * 1.5)
    return predictions


# AFTER: separated into focused functions (Extract Method refactoring).
def clean_rows(raw_data):
    """Remove invalid rows — single responsibility."""
    return [r for r in raw_data if r.get("sales") is not None and r["sales"] >= 0]


def extract_features(cleaned_data):
    """Transform cleaned rows into feature dicts."""
    return [{"store": r["store_id"], "sales_log": np.log1p(r["sales"])} for r in cleaned_data]


def make_predictions(features):
    """Run predictions on feature dicts."""
    return [f["sales_log"] * 1.5 for f in features]


def process_data_good(raw_data):
    """Orchestrator — composes clean→features→predict pipeline."""
    cleaned = clean_rows(raw_data)
    features = extract_features(cleaned)
    return make_predictions(features)


sample = [{"store_id": 1, "sales": 5000}, {"store_id": 2, "sales": -1}, {"store_id": 3, "sales": None}]
print(f"Before: {process_data_bad(sample)}")
print(f"After:  {process_data_good(sample)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8b) Strategy pattern — replacing conditionals with polymorphism

# COMMAND ----------

# BEFORE: branching logic with if/elif.
def normalize_bad(data, method):
    if method == "minmax":
        mn, mx = min(data), max(data)
        return [(x - mn) / (mx - mn + 1e-9) for x in data]
    elif method == "zscore":
        mean = sum(data) / len(data)
        std = (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5
        return [(x - mean) / (std + 1e-9) for x in data]
    elif method == "robust":
        sorted_d = sorted(data)
        q1, q3 = sorted_d[len(data)//4], sorted_d[3*len(data)//4]
        median = sorted_d[len(data)//2]
        return [(x - median) / (q3 - q1 + 1e-9) for x in data]
    else:
        raise ValueError(f"Unknown method: {method}")


# AFTER: Strategy pattern — each normalizer is a class.
class Normalizer(Protocol):
    def normalize(self, data: list) -> list: ...


class MinMaxNormalizer:
    def normalize(self, data):
        mn, mx = min(data), max(data)
        rng = mx - mn + 1e-9
        return [(x - mn) / rng for x in data]


class ZScoreNormalizer:
    def normalize(self, data):
        mean = sum(data) / len(data)
        std = (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5 + 1e-9
        return [(x - mean) / std for x in data]


class RobustNormalizer:
    def normalize(self, data):
        s = sorted(data)
        q1, median, q3 = s[len(data)//4], s[len(data)//2], s[3*len(data)//4]
        iqr = q3 - q1 + 1e-9
        return [(x - median) / iqr for x in data]


# Usage: select strategy at runtime without modifying the normalization logic.
normalizers = {"minmax": MinMaxNormalizer(), "zscore": ZScoreNormalizer(), "robust": RobustNormalizer()}

test_vals = [10, 20, 30, 40, 50, 100]
for name, norm in normalizers.items():
    result = norm.normalize(test_vals)
    print(f"  {name}: {[round(x, 3) for x in result]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8c) Singleton pattern — configuration manager

# COMMAND ----------

class PipelineConfig:
    """Singleton configuration manager.

    Ensures a single shared config instance across the pipeline.
    Uses the classic __new__ override pattern.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._settings = {}
        return cls._instance

    def set(self, key, value):
        self._settings[key] = value

    def get(self, key, default=None):
        return self._settings.get(key, default)


config1 = PipelineConfig()
config1.set("catalog", "demo")
config1.set("rebuild_all", True)

config2 = PipelineConfig()
print(f"Same instance: {config1 is config2}")
print(f"config2.get('catalog'): {config2.get('catalog')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9) Memory allocation & complexity summary

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC | Data Structure | Access | Search | Insert | Delete | Space | Notes |
# MAGIC |---|---|---|---|---|---|---|
# MAGIC | Array/List | O(1) | O(n) | O(n) | O(n) | O(n) | Contiguous memory, cache-friendly |
# MAGIC | Linked List | O(n) | O(n) | O(1) | O(1) | O(n) | Dynamic, pointer overhead |
# MAGIC | Hash Table | — | O(1) avg | O(1) avg | O(1) avg | O(n) | Amortized; worst case O(n) |
# MAGIC | BST | — | O(log n) avg | O(log n) avg | O(log n) avg | O(n) | Degrades to O(n) if unbalanced |
# MAGIC | Heap / PQ | — | O(n) | O(log n) | O(log n) | O(n) | Fast min/max access: O(1) |
# MAGIC | Deque | O(1) ends | O(n) | O(1) ends | O(1) ends | O(n) | Double-ended, sliding windows |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10) Summary
# MAGIC
# MAGIC ### Data Structures & Algorithms covered:
# MAGIC - Lists, Dicts, Sets with complexity analysis
# MAGIC - Sorting: Bubble, Insertion, Quicksort, Merge Sort
# MAGIC - Searching: Linear, Binary
# MAGIC - Hash Table (custom implementation with chaining)
# MAGIC - Binary Search Tree
# MAGIC - Priority Queue (min-heap)
# MAGIC - Deque (double-ended queue)
# MAGIC
# MAGIC ### Code Assessment covered:
# MAGIC - Code profiling with cProfile
# MAGIC - Refactoring: Extract Method, Single Responsibility
# MAGIC - Design Patterns: Strategy, Singleton
# MAGIC - Before/after comparison for code quality improvement
# MAGIC - Memory allocation and operation complexity documentation
# MAGIC
# MAGIC **Next step:** `workdir/jobs/` — Databricks job orchestration YAML.
