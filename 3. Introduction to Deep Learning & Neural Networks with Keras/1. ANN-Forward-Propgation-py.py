# Let's start by randomly initializing the weights and the biases in the network.
#  We have 6 weights and 3 biases, one for each node in the hidden layer as well as for each node in the output layer.

import numpy as np

# initialize the weights
from typing import List

weights = np.around(np.random.uniform(size=6), decimals=2)
# initialize the biases
biases = np.round(np.random.uniform(size=3), decimals=2)
print(weights)
print(biases)

# create the input
x_1 = 0.5
x_2 = 0.85
print("Input x_1 is {} and x_2 is {}".format(x_1, x_2))

# compute the wighted sum of the inputs,  𝑧_1,1 , at the first node of the hidden layer.
z_11 = x_1 * weights[0] + x_2 * weights[1] + biases[0]
print("The weighted sum of the inputs at the first node in the hidden layer is {}".format(z_11))

# compute the wighted sum of the inputs,  𝑧_1,2 , at the first node of the hidden layer.
z_12 = x_1 * weights[2] + x_2 * weights[3] + biases[1]
print("The weighted sum of the inputs at the second node in the hidden layer is {}".format(z_12))

# go through sigmoid activation function
a_11 = np.around(1.0 / (1.0 + np.exp(-z_11)), decimals=4)
a_12 = np.around(1.0 / (1.0 + np.exp(-z_12)), decimals=4)
print("The activations of the hidden layer are {} and {}".format(a_11, a_12))

# compute the weighted sum of these inputs to the node in the output layer. Assign the value to z_2.
z_2 = a_11 * weights[4] + a_12 * weights[5] + biases[2]
print('The weighted sum of the inputs at the node in the output layer is {}'.format(np.around(z_2, decimals=4)))
# through activation function
a_2 = np.around(1.0 / (1.0 + np.exp(-z_2)), decimals=4)
print('The output of the network for x1 = 0.5 and x2 = 0.85 is {}'.format(a_2))


# =========================== generalize our network ===============================

# # define the structure of the network
# n = 2  # number of inputs
# num_hidden_layers = 2  # number of hidden layers
# m = [2, 2]  # number of nodes in each hidden layer, which means len(m) = num_hidden_layers
# num_nodes_output = 1  # number of nodes in the output layer
#
# # let's initialize the weights and the biases in the networks
# import numpy as np
# num_nodes_previous = n
# network = {}
#
# # loop through each layer and randomly initialize the weights and biases associated with each node
# # notice how we are adding 1 to the number of hidden layers in order to include the output layer
# for layer in range(num_hidden_layers + 1):
#
#     # determine name of layer
#     if layer == num_hidden_layers:
#         layer_name = "output"
#         num_nodes = num_nodes_output
#     else:
#         layer_name = "layer_{}".format(layer+1)
#         num_nodes = m[layer]
#
#     # initialize weights and biases associated with each node in the current layer
#     network[layer_name] = {}
#     for node in range(num_nodes):
#         node_name = 'node_{}'.format(node+1)
#         network[layer_name][node_name] = {
#             "weights": np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
#             "bias": np.around(np.random.uniform(size=1), decimals=2)
#         }
#
#     num_nodes_previous = num_nodes
#
# print(network)

########################################################################################
# let's put this code in a function so that we are able to repetitively
# execute all this code whenever we want to construct a neural network.

def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
    num_nodes_previous = num_inputs  # number of nodes in the previous layer
    network = {}

    # loop through each layer and randomly initialize the weights and biases associated with each layer
    for layer in range(num_hidden_layers + 1):

        if layer == num_hidden_layers:
            layer_name = "output"
            num_nodes = num_nodes_output
        else:
            layer_name = "layer_{}".format(layer + 1)
            num_nodes = num_nodes_hidden[layer]

        network[layer_name] = {}

        # loop through each node in this hidden layer to get the weights and biases
        for node in range(num_nodes):
            node_name = "node_{}".format(node + 1)
            network[layer_name][node_name] = {
                "weights": np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
                "bias": np.around(np.random.uniform(size=1), decimals=2)
            }

        # because we are going to iterate through the next layer so the num_nodes_previous should also shift
        num_nodes_previous = num_nodes

    return network


# ============== Test the initialize_network function ====================
# takes 5 inputs
# has three hidden layers
# has 3 nodes in the first layer, 2 nodes in the second layer, and 3 nodes in the third layer
# has 1 node in the output layer
small_network = initialize_network(5, 3, [3, 2, 3], 1)
print(small_network)


# =============== Compute Weighted Sum at Each Node =========================

def compute_weighted_sum(inputs, weights, bias):
    return np.sum(inputs * weights) + bias


from random import seed
import numpy as np

np.random.seed(12)
inputs = np.around(np.random.uniform(size=5), decimals=2)
print('The inputs to the network are {}'.format(inputs))

# compute the weighted sum at the first node in the first hidden layer.
sum_first_node = compute_weighted_sum(inputs, small_network["layer_1"]["node_1"]["weights"],
                                      small_network["layer_1"]["node_1"]["bias"])
# sum_first_node is a list structure [...]
print('The weighted sum at the first node in the hidden layer is {}'.format(np.around(sum_first_node[0], decimals=4)))


# compute Node Activation
def node_activation(weighted_sum):
    return 1.0 / (1.0 + np.exp(-1 * weighted_sum))


# compute the output of the first node in the first hidden layer
output_first_node = np.around(node_activation(sum_first_node[0]), decimals=4)
print("The first output of the first node in the first hidden layer is {}".format(output_first_node))


# ========================= Forward Propagation =================================
# Start with the input layer as the input to the first hidden layer.
# Compute the weighted sum at the nodes of the current layer.
# Compute the output of the nodes of the current layer.
# Set the output of the current layer to be the input to the next layer.
# Move to the next layer in the network.
# Repeat steps 2 - 4 until we compute the output of the output layer.


# inside of the network you could first use the "initialize_network function" to initialize the layers, nodes, output
def forward_propagate(network, inputs):

    # make sure we go the list type of input
    if type(inputs) != list:
        layer_inputs = [float(i) for i in inputs]
    else:
        layer_inputs = inputs

    layer_final_outputs = 0

    # iterate through all layers
    for layer in network:
        # get the node info out of this layer
        layer_data = network[layer]

        # initialize the layer outputs, also for the use of the next iteration as layer input
        layer_current_outputs = []

        # iterate through all the node
        for node in layer_data:
            # get the weight and bias info out of this node
            node_data = layer_data[node]
            # compute the weight sum of this node, be aware that weighted_sum is a list type [...]
            weighted_sum = compute_weighted_sum(layer_inputs, node_data["weights"], node_data["bias"])
            # compute the output through the activation function
            current_output = np.around(node_activation(weighted_sum[0]), decimals=4)
            # store the output in to the output list
            layer_current_outputs.append(current_output)

        if layer != 'output':
            print('The outputs of the nodes in hidden layer number {} is {}'.format(layer.split('_')[1], layer_current_outputs))

        layer_inputs = layer_current_outputs
        layer_final_outputs = layer_current_outputs

    network_predictions = layer_final_outputs
    return network_predictions

print(" ")
# ================ Use the forward_propagate function to compute the prediction of our small network
small_network_prediction = forward_propagate(small_network, inputs)
print("The final prediction is {}".format(small_network_prediction))


print(" ")
# ================= let's try another example and network structure ==================
np.random.seed(10)
my_network = initialize_network(10, 5, [8, 7, 8, 7, 9], 3)
inputs = np.around(np.random.uniform(size=10), decimals=2)
predictions = forward_propagate(my_network, inputs)
print('The predicted values by the network for the given input are {}'.format(predictions))