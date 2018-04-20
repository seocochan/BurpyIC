import numpy as np
from PIL import Image
import io

def process_result_dict(result, size):
    temp_result = result
    sorted_dict = dict()

    # if you want to resize 'size', consider Unity Burpy Project!
    for i in range(size):
        highest_predict_key = max(temp_result.keys(), key=(lambda k: temp_result[k]))
        highest_predict_value = temp_result[highest_predict_key]
        del temp_result[highest_predict_key]

        order = "r" + str(i+1) + "th"
        sorted_dict[order] = {highest_predict_key:highest_predict_value}

    return sorted_dict
