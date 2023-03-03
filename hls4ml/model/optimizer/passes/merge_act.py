from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.layers import Activation, Dense, Conv2D, Conv2DBatchnorm

class MergeAct(OptimizerPass):
    def match(self, node):
        backends = ['VivadoAccelerator', 'Vivado']
        supported_layers = ['Dense'] 	#, 'Conv2D', 'Conv2DBatchNorm'
        supported_activations = ['relu', 'sigmoid', 'softmax', 'tanh']
        # By the time this optimization pass runs, the Layer nodes' class names
        # have been prepended with the name of the backend, e.g., a Conv2D
        # layer is renamed VivadoAcceleratorConv2D. Thus, we strip the backend
        # name for more streamlined matching. 
        input_node_class = node.get_input_node().__class__.__name__
        curr_node_class = node.__class__.__name__
        for b in backends:
            input_node_class = input_node_class.replace(b, '')
            curr_node_class = curr_node_class.replace(b, '')

        is_match = input_node_class in supported_layers
        # activation layers are of class Activation
        # is_match = is_match and (curr_node_class == 'Activation') 
        is_match = is_match and (node.get_attr('activation') in supported_activations)
    
        return is_match
    

    def transform(self, model, node):
        # Merge ReLU and Convolution/Dense layer
        previous_node = node.get_input_node()
        previous_node.set_merged_act(True) # Turn on merged_act flag for this Conv/Dense layer
        print("******** Merged activation ********")
        if 'Conv2D' in previous_node.__class__.__name__:
            if previous_node.get_attr('data_format') == 'channels_last':
                shape = [previous_node.attributes['out_height'], previous_node.attributes['out_width'], previous_node.attributes['n_filt']]
                dims = ['OUT_HEIGHT_{}'.format(previous_node.index), 'OUT_WIDTH_{}'.format(previous_node.index), 'N_FILT_{}'.format(previous_node.index)]
            else:
                shape = [previous_node.attributes['n_filt'], previous_node.attributes['out_height'], previous_node.attributes['out_width']]
                dims = ['N_FILT_{}'.format(previous_node.index), 'OUT_HEIGHT_{}'.format(previous_node.index), 'OUT_WIDTH_{}'.format(previous_node.index)]
            activation_precision, _ = model.config.get_precision(node, var='result')
            previous_node.add_output_variable(shape, dims, precision=activation_precision)
            if not node.get_output_nodes():
                print("WARNING: {} is the output layer! No rewiring performed.".format(node.name))
                model.remove_node(node, rewire=False)
            else:
                model.remove_node(node, rewire=True)
            return True 
        elif 'Dense' in previous_node.__class__.__name__:
            shape = previous_node.get_input_variable().shape[:]
            shape[-1] = previous_node.attributes['n_out']
            if len(shape) > 1:
                dims = ['N_LAYER_{}_{}'.format(i, previous_node.index) for i in range(1, len(shape) + 1)]
            else:
                dims = ['N_LAYER_{}'.format(previous_node.index)]
            print('shape: {}'.format(shape))
            print('dims: {}'.format(dims))
            activation_precision, _ = model.config.get_precision(node, var='result')
            previous_node.add_output_variable(shape, dims, precision=activation_precision)
            if not node.get_output_nodes():
                print("WARNING: {} is the output layer! No rewiring performed.".format(node.name))
                model.remove_node(node, rewire=False)
            else:
                model.remove_node(node, rewire=True)
            return True