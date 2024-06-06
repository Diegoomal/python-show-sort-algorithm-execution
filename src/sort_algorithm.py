import matplotlib.pyplot as plt
import numpy as np
import random
import time
from tqdm import tqdm

# Função para atualizar o gráfico de barras
def update_bars(array, bars, ax, iteration, comparisons, sorted_idx):
    for idx, (bar, val) in enumerate(zip(bars, array)):
        bar.set_height(val)
        if idx in comparisons:
            bar.set_color('red')
        elif idx in sorted_idx:
            bar.set_color('green')
        else:
            bar.set_color('blue')
    ax.set_title(f'Iteration {iteration}')
    plt.pause(0.01)

class SortAlgorithm:

    def __init__(self, arr):
        self.arr = arr

    def sort(self, visualize=False):
        raise NotImplementedError("O método sort deve ser implementado na subclasse.")

class BingoSort(SortAlgorithm):

    def sort(self, visualize=False):
        arr = self.arr
        size = len(arr)
        bingo = min(arr)
        largest = max(arr)
        nextBingo = largest
        nextPos = 0

        sorted_idx = set()
        if visualize:
            fig, ax = plt.subplots()
            bars = ax.bar(range(size), arr)
            plt.ion()

        while bingo < nextBingo:
            startPos = nextPos
            for i in range(startPos, size):
                if arr[i] == bingo:
                    arr[i], arr[nextPos] = arr[nextPos], arr[i]
                    nextPos += 1
                elif arr[i] < nextBingo:
                    nextBingo = arr[i]
                if visualize:
                    update_bars(arr, bars, ax, f'{startPos}-{i}', {i}, sorted_idx)
            bingo = nextBingo
            nextBingo = largest
            sorted_idx.update(range(nextPos))

        if visualize:
            update_bars(arr, bars, ax, 'Completed', set(), set(range(size)))
            plt.ioff()
            plt.show()

        return arr

class BitonicSort(SortAlgorithm):

    def sort(self, visualize=False):
        self.bitonic_sort(0, len(self.arr), 1, visualize)
        return self.arr

    def comp_and_swap(self, i, j, dire, visualize, bars, ax, sorted_idx):
        if (dire == 1 and self.arr[i] > self.arr[j]) or (dire == 0 and self.arr[i] < self.arr[j]):
            self.arr[i], self.arr[j] = self.arr[j], self.arr[i]
        if visualize:
            update_bars(self.arr, bars, ax, f'{i}-{j}', {i, j}, sorted_idx)

    def bitonic_merge(self, low, cnt, dire, visualize, bars, ax, sorted_idx):
        if cnt > 1:
            k = cnt // 2
            for i in range(low, low + k):
                self.comp_and_swap(i, i + k, dire, visualize, bars, ax, sorted_idx)
            self.bitonic_merge(low, k, dire, visualize, bars, ax, sorted_idx)
            self.bitonic_merge(low + k, k, dire, visualize, bars, ax, sorted_idx)

    def bitonic_sort(self, low, cnt, dire, visualize):
        if visualize:
            fig, ax = plt.subplots()
            bars = ax.bar(range(len(self.arr)), self.arr)
            plt.ion()
        
        if cnt > 1:
            k = cnt // 2
            self.bitonic_sort(low, k, 1, visualize)
            self.bitonic_sort(low + k, k, 0, visualize)
            self.bitonic_merge(low, cnt, dire, visualize, bars, ax, set())

        if visualize:
            update_bars(self.arr, bars, ax, 'Completed', set(), set(range(len(self.arr))))
            plt.ioff()
            plt.show()

class BogoSort(SortAlgorithm):

    def sort(self, visualize=False):
        n = len(self.arr)
        sorted_idx = set()
        if visualize:
            fig, ax = plt.subplots()
            bars = ax.bar(range(n), self.arr)
            plt.ion()
        
        while not self.is_sorted():
            self.shuffle()
            if visualize:
                update_bars(self.arr, bars, ax, 'Shuffling', set(), sorted_idx)
        
        if visualize:
            update_bars(self.arr, bars, ax, 'Completed', set(), set(range(n)))
            plt.ioff()
            plt.show()

        return self.arr

    def is_sorted(self):
        n = len(self.arr)
        for i in range(n - 1):
            if self.arr[i] > self.arr[i + 1]:
                return False
        return True

    def shuffle(self):
        n = len(self.arr)
        for i in range(n):
            r = random.randint(0, n - 1)
            self.arr[i], self.arr[r] = self.arr[r], self.arr[i]

class BrickSort(SortAlgorithm):

    def sort(self, visualize=False):
        isSorted = False
        n = len(self.arr)
        sorted_idx = set()
        if visualize:
            fig, ax = plt.subplots()
            bars = ax.bar(range(n), self.arr)
            plt.ion()
        
        while not isSorted:
            isSorted = True
            for i in range(1, n - 1, 2):
                if self.arr[i] > self.arr[i + 1]:
                    self.arr[i], self.arr[i + 1] = self.arr[i + 1], self.arr[i]
                    isSorted = False
                if visualize:
                    update_bars(self.arr, bars, ax, 'Odd Phase', {i, i + 1}, sorted_idx)
            for i in range(0, n - 1, 2):
                if self.arr[i] > self.arr[i + 1]:
                    self.arr[i], self.arr[i + 1] = self.arr[i + 1], self.arr[i]
                    isSorted = False
                if visualize:
                    update_bars(self.arr, bars, ax, 'Even Phase', {i, i + 1}, sorted_idx)

        if visualize:
            update_bars(self.arr, bars, ax, 'Completed', set(), set(range(n)))
            plt.ioff()
            plt.show()
        
        return self.arr

class BubbleSort(SortAlgorithm):

    def sort(self, visualize=False):
        n = len(self.arr)
        sorted_idx = set()
        if visualize:
            fig, ax = plt.subplots()
            bars = ax.bar(range(n), self.arr)
            plt.ion()
        
        for i in range(n):
            for j in range(0, n-i-1):
                if self.arr[j] > self.arr[j+1]:
                    self.arr[j], self.arr[j+1] = self.arr[j+1], self.arr[j]
                if visualize:
                    update_bars(self.arr, bars, ax, f'{i}-{j}', {j, j+1}, sorted_idx)
            sorted_idx.add(n-i-1)

        if visualize:
            update_bars(self.arr, bars, ax, 'Completed', set(), set(range(n)))
            plt.ioff()
            plt.show()
        
        return self.arr

class CocktailSort(SortAlgorithm):
    def sort(self, visualize=False):
        n = len(self.arr)
        swapped = True
        start = 0
        end = n - 1
        sorted_idx = set()

        if visualize:
            fig, ax = plt.subplots()
            bars = ax.bar(range(n), self.arr)
            plt.ion()

        while swapped:
            swapped = False
            for i in range(start, end):
                if self.arr[i] > self.arr[i + 1]:
                    self.arr[i], self.arr[i + 1] = self.arr[i + 1], self.arr[i]
                    swapped = True
                if visualize:
                    update_bars(self.arr, bars, ax, 'Forward', {i, i+1}, sorted_idx)
            if not swapped:
                break
            swapped = False
            end -= 1
            for i in range(end - 1, start - 1, -1):
                if self.arr[i] > self.arr[i + 1]:
                    self.arr[i], self.arr[i + 1] = self.arr[i + 1], self.arr[i]
                    swapped = True
                if visualize:
                    update_bars(self.arr, bars, ax, 'Backward', {i, i+1}, sorted_idx)
            start += 1

        if visualize:
            update_bars(self.arr, bars, ax, 'Completed', set(), set(range(n)))
            plt.ioff()
            plt.show()
        return self.arr

class CombSort(SortAlgorithm):
    def sort(self, visualize=False):
        n = len(self.arr)
        gap = n
        swapped = True
        sorted_idx = set()

        if visualize:
            fig, ax = plt.subplots()
            bars = ax.bar(range(n), self.arr)
            plt.ion()

        while gap != 1 or swapped:
            gap = self.get_next_gap(gap)
            swapped = False
            for i in range(0, n - gap):
                if self.arr[i] > self.arr[i + gap]:
                    self.arr[i], self.arr[i + gap] = self.arr[i + gap], self.arr[i]
                    swapped = True
                if visualize:
                    update_bars(self.arr, bars, ax, f'Gap {gap}', {i, i+gap}, sorted_idx)

        if visualize:
            update_bars(self.arr, bars, ax, 'Completed', set(), set(range(n)))
            plt.ioff()
            plt.show()
        return self.arr

    def get_next_gap(self, gap):
        gap = (gap * 10) // 13
        if gap < 1:
            return 1
        return gap

class CountSort(SortAlgorithm):

    def sort(self, visualize=False):
        arr = self.arr
        M = max(arr)
        count_array = [0] * (M + 1)
        sorted_idx = set()
        
        if visualize:
            fig, ax = plt.subplots()
            bars = ax.bar(range(len(arr)), arr)
            plt.ion()

        for num in arr:
            count_array[num] += 1
            if visualize:
                update_bars(arr, bars, ax, 'Counting', {arr.index(num)}, sorted_idx)

        for i in range(1, M + 1):
            count_array[i] += count_array[i - 1]

        output_array = [0] * len(arr)
        for i in range(len(arr) - 1, -1, -1):
            output_array[count_array[arr[i]] - 1] = arr[i]
            count_array[arr[i]] -= 1
            sorted_idx.add(i)
            if visualize:
                update_bars(output_array, bars, ax, 'Output', {i}, sorted_idx)
        
        arr[:] = output_array[:]

        if visualize:
            update_bars(arr, bars, ax, 'Completed', set(), set(range(len(arr))))
            plt.ioff()
            plt.show()

        return arr

class CycleSort(SortAlgorithm):
    def sort(self, visualize=False):
        writes = 0
        sorted_idx = set()

        if visualize:
            fig, ax = plt.subplots()
            bars = ax.bar(range(len(self.arr)), self.arr)
            plt.ion()

        for cycle_start in range(0, len(self.arr) - 1):
            item = self.arr[cycle_start]
            pos = cycle_start
            for i in range(cycle_start + 1, len(self.arr)):
                if self.arr[i] < item:
                    pos += 1
            if pos == cycle_start:
                continue
            while item == self.arr[pos]:
                pos += 1
            self.arr[pos], item = item, self.arr[pos]
            writes += 1
            sorted_idx.add(pos)
            if visualize:
                update_bars(self.arr, bars, ax, f'Cycle {cycle_start}', {pos}, sorted_idx)
            while pos != cycle_start:
                pos = cycle_start
                for i in range(cycle_start + 1, len(self.arr)):
                    if self.arr[i] < item:
                        pos += 1
                while item == self.arr[pos]:
                    pos += 1
                self.arr[pos], item = item, self.arr[pos]
                writes += 1
                sorted_idx.add(pos)
                if visualize:
                    update_bars(self.arr, bars, ax, f'Cycle {cycle_start}', {pos}, sorted_idx)

        if visualize:
            update_bars(self.arr, bars, ax, 'Completed', set(), set(range(len(self.arr))))
            plt.ioff()
            plt.show()
        return self.arr

class GnomeSort(SortAlgorithm):
    def sort(self, visualize=False):
        index = 0
        sorted_idx = set()

        if visualize:
            fig, ax = plt.subplots()
            bars = ax.bar(range(len(self.arr)), self.arr)
            plt.ion()

        while index < len(self.arr):
            if index == 0 or self.arr[index] >= self.arr[index - 1]:
                sorted_idx.add(index)
                index += 1
            else:
                self.arr[index], self.arr[index - 1] = self.arr[index - 1], self.arr[index]
                sorted_idx.add(index - 1)
                index -= 1
            if visualize:
                update_bars(self.arr, bars, ax, f'Index {index}', {index, index-1}, sorted_idx)

        if visualize:
            update_bars(self.arr, bars, ax, 'Completed', set(), set(range(len(self.arr))))
            plt.ioff()
            plt.show()
        return self.arr

class HeapSort(SortAlgorithm):
    def sort(self, visualize=False):
        n = len(self.arr)
        sorted_idx = set()

        if visualize:
            fig, ax = plt.subplots()
            bars = ax.bar(range(n), self.arr)
            plt.ion()

        for i in range(n // 2 - 1, -1, -1):
            self.heapify(n, i, visualize, bars, ax, sorted_idx)
        for i in range(n - 1, 0, -1):
            self.arr[i], self.arr[0] = self.arr[0], self.arr[i]
            sorted_idx.add(i)
            self.heapify(i, 0, visualize, bars, ax, sorted_idx)

        if visualize:
            update_bars(self.arr, bars, ax, 'Completed', set(), set(range(n)))
            plt.ioff()
            plt.show()
        return self.arr

    def heapify(self, n, i, visualize, bars, ax, sorted_idx):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2

        if l < n and self.arr[i] < self.arr[l]:
            largest = l
        if r < n and self.arr[largest] < self.arr[r]:
            largest = r
        if largest != i:
            self.arr[i], self.arr[largest] = self.arr[largest], self.arr[i]
            if visualize:
                update_bars(self.arr, bars, ax, f'Heapify {i}', {i, largest}, sorted_idx)
            self.heapify(n, largest, visualize, bars, ax, sorted_idx)

class InsertionSort(SortAlgorithm):
    def sort(self, visualize=False):
        sorted_idx = set()

        if visualize:
            fig, ax = plt.subplots()
            bars = ax.bar(range(len(self.arr)), self.arr)
            plt.ion()

        for i in range(1, len(self.arr)):
            key = self.arr[i]
            j = i - 1
            while j >= 0 and key < self.arr[j]:
                self.arr[j + 1] = self.arr[j]
                j -= 1
            self.arr[j + 1] = key
            sorted_idx.add(j + 1)
            if visualize:
                update_bars(self.arr, bars, ax, f'Insert {i}', {j+1, j+2}, sorted_idx)

        if visualize:
            update_bars(self.arr, bars, ax, 'Completed', set(), set(range(len(self.arr))))
            plt.ioff()
            plt.show()
        return self.arr

class MergeSort(SortAlgorithm):
    def sort(self, visualize=False):
        self.merge_sort(self.arr, 0, len(self.arr) - 1, visualize)
        return self.arr

    def merge(self, array, left, mid, right, visualize, bars, ax, sorted_idx):
        subArrayOne = mid - left + 1
        subArrayTwo = right - mid
        leftArray = [0] * subArrayOne
        rightArray = [0] * subArrayTwo

        for i in range(subArrayOne):
            leftArray[i] = array[left + i]
        for j in range(subArrayTwo):
            rightArray[j] = array[mid + 1 + j]

        indexOfSubArrayOne = 0
        indexOfSubArrayTwo = 0
        indexOfMergedArray = left

        while indexOfSubArrayOne < subArrayOne and indexOfSubArrayTwo < subArrayTwo:
            if leftArray[indexOfSubArrayOne] <= rightArray[indexOfSubArrayTwo]:
                array[indexOfMergedArray] = leftArray[indexOfSubArrayOne]
                indexOfSubArrayOne += 1
            else:
                array[indexOfMergedArray] = rightArray[indexOfSubArrayTwo]
                indexOfSubArrayTwo += 1
            indexOfMergedArray += 1
            if visualize:
                update_bars(array, bars, ax, f'Merge {left}-{right}', {indexOfMergedArray-1}, sorted_idx)

        while indexOfSubArrayOne < subArrayOne:
            array[indexOfMergedArray] = leftArray[indexOfSubArrayOne]
            indexOfSubArrayOne += 1
            indexOfMergedArray += 1
            if visualize:
                update_bars(array, bars, ax, f'Merge {left}-{right}', {indexOfMergedArray-1}, sorted_idx)

        while indexOfSubArrayTwo < subArrayTwo:
            array[indexOfMergedArray] = rightArray[indexOfSubArrayTwo]
            indexOfSubArrayTwo += 1
            indexOfMergedArray += 1
            if visualize:
                update_bars(array, bars, ax, f'Merge {left}-{right}', {indexOfMergedArray-1}, sorted_idx)

    def merge_sort(self, array, begin, end, visualize):
        if begin >= end:
            return

        mid = begin + (end - begin) // 2
        self.merge_sort(array, begin, mid, visualize)
        self.merge_sort(array, mid + 1, end, visualize)
        self.merge(array, begin, mid, end, visualize, bars=None, ax=None, sorted_idx=set())

class PancakeSort(SortAlgorithm):
    def sort(self, visualize=False):
        self.pancake_sort(self.arr, len(self.arr), visualize)
        return self.arr

    def flip(self, arr, i):
        start = 0
        while start < i:
            arr[start], arr[i] = arr[i], arr[start]
            start += 1
            i -= 1

    def find_max(self, arr, n):
        mi = 0
        for i in range(0, n):
            if arr[i] > arr[mi]:
                mi = i
        return mi

    def pancake_sort(self, arr, n, visualize):
        sorted_idx = set()

        if visualize:
            fig, ax = plt.subplots()
            bars = ax.bar(range(n), arr)
            plt.ion()

        curr_size = n
        while curr_size > 1:
            mi = self.find_max(arr, curr_size)
            if mi != curr_size - 1:
                self.flip(arr, mi)
                if visualize:
                    update_bars(arr, bars, ax, f'Flip {mi}', set(range(mi+1)), sorted_idx)
                self.flip(arr, curr_size - 1)
                if visualize:
                    update_bars(arr, bars, ax, f'Flip {curr_size-1}', set(range(curr_size)), sorted_idx)
            sorted_idx.add(curr_size - 1)
            curr_size -= 1

        if visualize:
            update_bars(arr, bars, ax, 'Completed', set(), set(range(n)))
            plt.ioff()
            plt.show()

class PigeonholeSort(SortAlgorithm):
    def sort(self, visualize=False):
        if visualize:
            fig, ax = plt.subplots()
            bars = ax.bar(range(len(self.arr)), self.arr)
            plt.ion()
        self.pigeonhole_sort(self.arr, visualize, bars, ax, set())
        if visualize:
            update_bars(self.arr, bars, ax, 'Completed', set(), set(range(len(self.arr))))
            plt.ioff()
            plt.show()
        return self.arr

    def pigeonhole_sort(self, a, visualize, bars, ax, sorted_idx):
        my_min = np.min(a)
        my_max = np.max(a)
        size = my_max - my_min + 1
        holes = np.zeros(size, dtype=np.int32)

        for x in a:
            holes[x - my_min] += 1
            if visualize:
                update_bars(a, bars, ax, 'Filling holes', set(), sorted_idx)

        i = 0
        for count in range(size):
            while holes[count] > 0:
                holes[count] -= 1
                a[i] = count + my_min
                i += 1
                if visualize:
                    update_bars(a, bars, ax, 'Placing elements back', {i}, sorted_idx)

class QuickSort(SortAlgorithm):
    def partition(self, low, high, visualize, bars, ax, sorted_idx):
        pivot = self.arr[high]
        i = low - 1
        for j in range(low, high):
            if self.arr[j] <= pivot:
                i += 1
                self.arr[i], self.arr[j] = self.arr[j], self.arr[i]
            if visualize:
                update_bars(self.arr, bars, ax, f'Partition {low}-{high}', {i, j}, sorted_idx)
        self.arr[i + 1], self.arr[high] = self.arr[high], self.arr[i + 1]
        return i + 1

    def quick_sort(self, low, high, visualize, bars, ax, sorted_idx):
        if low < high:
            pi = self.partition(low, high, visualize, bars, ax, sorted_idx)
            self.quick_sort(low, pi - 1, visualize, bars, ax, sorted_idx)
            self.quick_sort(pi + 1, high, visualize, bars, ax, sorted_idx)

    def sort(self, visualize=False):
        if visualize:
            fig, ax = plt.subplots()
            bars = ax.bar(range(len(self.arr)), self.arr)
            plt.ion()
        self.quick_sort(0, len(self.arr) - 1, visualize, bars, ax, set())
        if visualize:
            update_bars(self.arr, bars, ax, 'Completed', set(), set(range(len(self.arr))))
            plt.ioff()
            plt.show()
        return self.arr

class RadixSort(SortAlgorithm):
    def counting_sort(self, exp1, visualize, bars, ax, sorted_idx):
        n = len(self.arr)
        output = [0] * n
        count = [0] * 10

        for i in range(n):
            index = self.arr[i] // exp1
            count[index % 10] += 1

        for i in range(1, 10):
            count[i] += count[i - 1]

        i = n - 1
        while i >= 0:
            index = self.arr[i] // exp1
            output[count[index % 10] - 1] = self.arr[i]
            count[index % 10] -= 1
            i -= 1

        for i in range(n):
            self.arr[i] = output[i]
            if visualize:
                update_bars(self.arr, bars, ax, f'Exp {exp1}', {i}, sorted_idx)

    def radix_sort(self, visualize, bars, ax, sorted_idx):
        max1 = max(self.arr)
        exp = 1
        while max1 / exp >= 1:
            self.counting_sort(exp, visualize, bars, ax, sorted_idx)
            exp *= 10

    def sort(self, visualize=False):
        if visualize:
            fig, ax = plt.subplots()
            bars = ax.bar(range(len(self.arr)), self.arr)
            plt.ion()
        self.radix_sort(visualize, bars, ax, set())
        if visualize:
            update_bars(self.arr, bars, ax, 'Completed', set(), set(range(len(self.arr))))
            plt.ioff()
            plt.show()
        return self.arr

class SelectionSort(SortAlgorithm):
    def sort(self, visualize=False):
        n = len(self.arr)
        sorted_idx = set()

        if visualize:
            fig, ax = plt.subplots()
            bars = ax.bar(range(n), self.arr)
            plt.ion()

        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if self.arr[min_idx] > self.arr[j]:
                    min_idx = j
                if visualize:
                    update_bars(self.arr, bars, ax, f'Iter {i}', {min_idx, j}, sorted_idx)
            self.arr[i], self.arr[min_idx] = self.arr[min_idx], self.arr[i]
            sorted_idx.add(i)

        if visualize:
            update_bars(self.arr, bars, ax, 'Completed', set(), set(range(n)))
            plt.ioff()
            plt.show()
        return self.arr

class ShellSort(SortAlgorithm):
    def shell_sort(self, arr, n, visualize, bars, ax, sorted_idx):
        gap = n // 2
        while gap > 0:
            j = gap
            while j < n:
                i = j - gap
                while i >= 0:
                    if arr[i + gap] > arr[i]:
                        break
                    else:
                        arr[i + gap], arr[i] = arr[i], arr[i + gap]
                    i -= gap
                    if visualize:
                        update_bars(arr, bars, ax, f'Gap {gap}', {i, i+gap}, sorted_idx)
                j += 1
            gap //= 2

    def sort(self, visualize=False):
        n = len(self.arr)
        if visualize:
            fig, ax = plt.subplots()
            bars = ax.bar(range(n), self.arr)
            plt.ion()
        self.shell_sort(self.arr, n, visualize, bars, ax, set())
        if visualize:
            update_bars(self.arr, bars, ax, 'Completed', set(), set(range(n)))
            plt.ioff()
            plt.show()
        return self.arr

class StrandSort(SortAlgorithm):
    def merge_lists(self, list1, list2):
        result = []
        while list1 and list2:
            if list1[0] < list2[0]:
                result.append(list1.pop(0))
            else:
                result.append(list2.pop(0))
        result += list1
        result += list2
        return result

    def strand_sort(self, ip, visualize, bars, ax, sorted_idx):
        if len(ip) <= 1:
            return ip

        sublist = [ip.pop(0)]
        i = 0
        while i < len(ip):
            if ip[i] > sublist[-1]:
                sublist.append(ip.pop(i))
            else:
                i += 1

        if visualize:
            update_bars(self.arr, bars, ax, 'Strand Sorting', set(range(len(self.arr))), sorted_idx)

        sorted_sublist = sublist
        remaining_list = self.strand_sort(ip, visualize, bars, ax, sorted_idx)
        return self.merge_lists(sorted_sublist, remaining_list)

    def sort(self, visualize=False):
        if visualize:
            fig, ax = plt.subplots()
            bars = ax.bar(range(len(self.arr)), self.arr)
            plt.ion()
        self.arr = self.strand_sort(self.arr, visualize, bars, ax, set())
        if visualize:
            update_bars(self.arr, bars, ax, 'Completed', set(), set(range(len(self.arr))))
            plt.ioff()
            plt.show()
        return self.arr

class TimSort(SortAlgorithm):

    def sort(self, visualize=False):
        return self.tim_sort(self.arr, visualize)

    def calc_min_run(self, n):
        r = 0
        MIN_MERGE = 32
        while n >= MIN_MERGE:
            r |= n & 1
            n >>= 1
        return n + r

    def insertion_sort(self, arr, left, right, visualize, bars, ax, sorted_idx):
        for i in range(left + 1, right + 1):
            j = i
            while j > left and arr[j] < arr[j - 1]:
                arr[j], arr[j - 1] = arr[j - 1], arr[j]
                j -= 1
            if visualize:
                update_bars(arr, bars, ax, f'Insertion {left}-{right}', {j, j+1}, sorted_idx)

    def merge(self, arr, l, m, r, visualize, bars, ax, sorted_idx):
        len1, len2 = m - l + 1, r - m
        left, right = [], []
        for i in range(0, len1):
            left.append(arr[l + i])
        for i in range(0, len2):
            right.append(arr[m + 1 + i])
        i, j, k = 0, 0, l
        while i < len1 and j < len2:
            if left[i] <= right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            if visualize:
                update_bars(arr, bars, ax, f'Merging {l}-{r}', {k}, sorted_idx)
            k += 1
        while i < len1:
            arr[k] = left[i]
            i += 1
            k += 1
        while j < len2:
            arr[k] = right[j]
            j += 1
            k += 1

    def tim_sort(self, arr, visualize):
        n = len(arr)
        minRun = self.calc_min_run(n)

        sorted_idx = set()
        if visualize:
            fig, ax = plt.subplots()
            bars = ax.bar(range(n), arr)
            plt.ion()

        for start in range(0, n, minRun):
            end = min(start + minRun - 1, n - 1)
            self.insertion_sort(arr, start, end, visualize, bars, ax, sorted_idx)
            if visualize:
                sorted_idx.update(range(start, end + 1))

        size = minRun
        while size < n:
            for left in range(0, n, 2 * size):
                mid = min(n - 1, left + size - 1)
                right = min((left + 2 * size - 1), (n - 1))
                if mid < right:
                    self.merge(arr, left, mid, right, visualize, bars, ax, sorted_idx)
            size = 2 * size

        if visualize:
            update_bars(arr, bars, ax, 'Completed', set(), set(range(n)))
            plt.ioff()
            plt.show()

        return arr
