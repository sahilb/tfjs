1. rename to actor_out : out = tflearn.fully_connected(dense_net_0, self.a_dim, activation='softmax', name='actor_out')
2. changes for multi


#if epoch % MODEL_SAVE_INTERVAL == 0:
# g = sess.graph
# for op in g.get_operations():
#     if('actor_out' in op.name):
#         print(op.name)

# print('actor inputs')
# print(actor.inputs)
#   
# actor_in = g.get_tensor_by_name('actor/InputData/X:0')
# print(actor_in)
# actor_out = g.get_tensor_by_name('actor/actor_out/Softmax:0')
# print(actor_out)
          

3. virtual env: p2.7 train model

4. python freeze model using:   python freeze.py --model_dir ck_model/ --output_node_names="actor/actor_out/Softmax"
ensure frozen.pb in ck_model

5. separate virtual env (p2.7) for conversion to tfjs.  install only:  pip install tensorflowjs==0.8.5

6. tensorflowjs_converter --input_format=tf_frozen_model --output_node_names=actor/actor_out/Softmax --saved_model_tags=serve  --output_json=true  ./frozen_model/frozen_model.pb  ./exported_frozen



