import numpy as np

def most_frequent_indices(list):
    frequency_index = {}

    most_frequent_filled = []

    for y in list:
        for i, v in enumerate(y):
            if v == 1:
                if i in frequency_index:
                    frequency_index[i] = frequency_index[i] + 1
                else:
                    frequency_index[i] = 1
    print(type(frequency_index.items()))
    most_frequent = list(reversed(sorted(frequency_index.items(), key=lambda item: item[1])))

    for index in most_frequent:
        most_frequent_filled.append(index[0])

    return most_frequent_filled
