# Sorting Algorithm Visualizer

This project provides a collection of sorting algorithms implemented in Python, each capable of visualizing the sorting process step by step using matplotlib. The available sorting algorithms include well-known ones like QuickSort, MergeSort, and BubbleSort, as well as lesser-known algorithms such as PigeonholeSort and StrandSort.

## Features

- Multiple Sorting Algorithms: Includes a variety of sorting algorithms such as BingoSort, BitonicSort, BrickSort, BubbleSort, CocktailSort, CombSort, CountSort, CycleSort, GnomeSort, HeapSort, InsertionSort, MergeSort, PancakeSort, PigeonholeSort, QuickSort, RadixSort, SelectionSort, ShellSort, StrandSort, and TimSort.

- Visualization: Visualize the sorting process in real-time using matplotlib.

- Random Array Generation: Generate random arrays of integers for sorting.

## Usage

1. Clone the repository.
2. Ensure you have the required dependencies installed (e.g., matplotlib, numpy).
3. Run the main.py script to visualize a sorting algorithm of your choice.

```

conda env create -n show-sort-alg-exec-env -f ./env.yml

conda activate show-sort-alg-exec-env

python src/main.py

```

By default, the script uses BubbleSort for visualization. To use a different sorting algorithm, modify the sort_algorithm variable in main.py to one of the available algorithms listed in sort_algorithms.

## Example

To visualize the sorting process using QuickSort:

```
sort_algorithm = sort_algorithms['quick']
```

Run the script, and a real-time visualization of the QuickSort algorithm will be displayed.

## Links

[github](https://github.com/Diegoomal)