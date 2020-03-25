import tensorflow as tf
class Gate(tf.keras.layers.Layer):
    def __init__(self, layers, context, use_bias=True, layer_activation=lambda x:x, output_size=None):
        super(Gate, self).__init__()
        self.layers = layers
        self.use_bias = use_bias
        self.activation = layer_activation
        self.context_shape = context.shape
        self.layer_output_size = output_size
        self.layer_num = len(self.layers)

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
                                        shape=[self.context_shape[-1],
                                               self.layer_num])
        if self.use_bias:
            self.bias = self.add_weight("bias",
                                        shape=[self.layer_num])

    def call(self, inputs, context):
        #Situation analysis
        if self.use_bias: 
            pred = tf.nn.softmax(tf.matmul(context, self.kernel)+self.bias)
        else:
            pred = tf.nn.softmax(tf.matmul(context, self.kernel))
        #Propositions by the different layers
        all_layers = tf.stack([layer(inputs) for layer in self.layers], axis=1)
        #pred_per_layer = tf.reshape(pred, all_layers.shape)
        if not self.layer_output_size:
            pred_per_layer = tf.reshape(pred, (-1, self.layer_num, 1))
        else:
            pred_per_layer = tf.reshape(pred, (-1, self.layer_num, *self.layer_output_size))
        #mult_layers = tf.multiply(pred_per_layer, all_layers)
        mult_layers = pred_per_layer * all_layers
        one_layer = tf.math.reduce_sum(mult_layers, axis=1)
        #Only uses the Chosen One
        return self.activation(one_layer)