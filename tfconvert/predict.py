import load_graph
import tensorflow as tf
import numpy as np

graph = load_graph.load_graph('ck_model/frozen_model.pb')

for op in graph.get_operations():
    print(op.name)

y = graph.get_tensor_by_name('prefix/actor/actor_out/Softmax:0')
x = graph.get_tensor_by_name('prefix/actor/InputData/X:0')

print(x)
print(y)

a = np.arange(0,48).reshape(1, 6, 8)
with tf.Session(graph=graph) as sess:
    y_out = sess.run(y, feed_dict={x:a})

print(y_out)