# RNN from scratch, only using numpy
# By: Teddy Ordoñez
# Source: https://www.youtube.com/watch?v=Wo5dMEP_BbI&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3

import numpy as np

inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

output = np.dot(weights, inputs) + biases   # The order matter due to shapes of weights and inputs
print(output)
'''
layer_outputs = []  # Output of current layer (size will change)
for neuron_weigths, neuron_bias in zip(weights, biases):
    neuron_output = 0   # Output for a specific neuron   
    
    for n_input, weight in zip(inputs, neuron_weigths):
        neuron_output = n_input * weight    # Multiplying the input times the weight
    
    neuron_output += neuron_bias    # Adding the bias of the specific neuron to the output
    layer_outputs.append(neuron_output) # Appending the result of the calculation to the list of results

print(layer_outputs)
'''