import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def final_score(pieapp_score):
    # apply sigmoid function to normalize the score
    sigmoid_normalized_score = sigmoid(pieapp_score)
    
    # get the original values at the boundary points
    original_at_zero = (1 - (np.log10(sigmoid(0) + 1) / np.log10(3.5))) ** 2.5
    original_at_two = (1 - (np.log10(sigmoid(2.0) + 1) / np.log10(3.5))) ** 2.5
    
    # calculate the original value at the current point
    original_value = (1 - (np.log10(sigmoid_normalized_score + 1) / np.log10(3.5))) ** 2.5
    
    # scale to the new range [1, 0]
    scaled_value = 1 - ((original_value - original_at_zero) / (original_at_two - original_at_zero))
    
    return scaled_value

for x in range(0, 20, 1):
    x = x / 10.0
    print(final_score(x))