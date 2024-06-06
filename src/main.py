import random
from sort_algorithm import (
    BingoSort, BitonicSort, BrickSort, BubbleSort,
    CocktailSort, CombSort, CountSort, CycleSort, 
    GnomeSort, HeapSort, InsertionSort, MergeSort,
    PancakeSort, PigeonholeSort, QuickSort, RadixSort,
    SelectionSort, ShellSort, StrandSort, TimSort
)


# Função para gerar um array de inteiros aleatórios
def generate_random_array(size):
    return [random.randint(1, size) for _ in range(size)]


if __name__ == "__main__":
    
    # Gera um array de inteiros aleatórios
    unsorted_array = generate_random_array(100)

    sort_algorithms = {
        'bingo': BingoSort(unsorted_array.copy()),
        'bitonic': BitonicSort(unsorted_array.copy()),
        'brick': BrickSort(unsorted_array.copy()),
        'bubble': BubbleSort(unsorted_array.copy()),
        'cocktailSort': CocktailSort(unsorted_array.copy()),
        'comb': CombSort(unsorted_array.copy()),
        'count': CountSort(unsorted_array.copy()),
        'cycle': CycleSort(unsorted_array.copy()),    
        'gnome': GnomeSort(unsorted_array.copy()),
        'heap': HeapSort(unsorted_array.copy()),
        'insertion': InsertionSort(unsorted_array.copy()),
        'merge': MergeSort(unsorted_array.copy()),
        'pancake': PancakeSort(unsorted_array.copy()),
        'pingeonhole': PigeonholeSort(unsorted_array.copy()),    
        'quick': QuickSort(unsorted_array.copy()),
        'radix': RadixSort(unsorted_array.copy()),
        'selection': SelectionSort(unsorted_array.copy()),
        'shell': ShellSort(unsorted_array.copy()),
        'strand': StrandSort(unsorted_array.copy()),
        'tim': TimSort(unsorted_array.copy()),
    }

    # Inicializa o algoritmo de ordenação com visualização
    sort_algorithm = sort_algorithms['bubble']

    # Executa a ordenação com visualização
    sorted_array = sort_algorithm.sort(visualize=True)
