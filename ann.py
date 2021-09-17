import numpy
import scipy.special

input_node = 3
hidden_node = 3
output_node = 3

wih = numpy.random.normal(0.0,pow(input_node,-0.5),(hidden_node,input_node))
who = numpy.random.normal(0.0,pow(hidden_node,-0.5),(output_node,hidden_node))

lr = 0.3
activation_function = lambda x : scipy.special.expit(x)

input_list = [0.2,0.3,0.4]
target_list = [0.3,0.2,0.1]

inputs = numpy.array(input_list,ndmin = 2).T
targets = numpy.array(target_list,ndmin = 2).T

hidden_input = numpy.dot(wih,inputs)
hidden_output = activation_function(hidden_input)

final_input = numpy.dot(who,hidden_output)
final_output = activation_function(final_input)

output_error = targets - final_output
hidden_error = numpy.dot(who.T,output_error)

who += lr * numpy.dot((output_error*final_output)*(1.0-final_output),numpy.transpose(hidden_output))
wih += lr * numpy.dot((hidden_error*hidden_output)*(1.0-hidden_output),numpy.transpose(inputs))

input = numpy.array(input_list,ndmin=2).T
hidden_input = numpy.dot(wih,inputs)
hidden_output = activation_function(hidden_input)
final_input = numpy.dot(who,hidden_output)
final_output = activation_function(final_input)
k = "Kushal Pokhrel"
print("Hey!,", k, "This is the Final Output for your Artificial Neural Network")
print(final_output)





