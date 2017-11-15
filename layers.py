import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import _linear
import numpy as np

def layer_norm(inp, eps=1e-5, scope=None):
    """
        args:
            inp: [tensor] a 2D tensor with shape: [batch_size, num_hidden]
            eps: [float] for math stability
            scope: [str] variable scope
        return:
            ln_inp: [tensor] layer normed input, the same shape with 'inp'
    """
    assert (len(inp.get_shape()) == 2)
    mean, var = tf.nn.moments(inp, [1], keep_dims=False)
    scope = '' if scope == None else scope
    with tf.variable_scope(scope + 'layer_norm'):
        gain = tf.get_variable('gain', shape=[inp.get_shape()[1]], initializer=tf.constant_initializer(1))
        bias = tf.get_variable('bias', shape=[inp.get_shape()[1]], initializer=tf.constant_initializer(0))

    ln_inp = (inp - mean) / tf.sqrt(var + eps)
    ln_inp = gain * ln_inp + bias   
    return ln_inp

def dynamic_layer_norm(inp, eps=1e-5, scope=None):
    pass


def hyper_norm(layer, hyper_output, embedding_size, num_units,
               scope="hyper", use_bias=True):
    """
        args:
            layer: [tensor] input tensor need to be normed, with shape: [batch_size, num_units]
            hyper_output: [tensor] output tensor with shape: [batch_size, embedding_size] 
            num_units: [int] num of main network's hidden units

            init_gamma= 0.10
            
            
    """
    init_gamma = 0.10

    with tf.variable_scope(scope):
        with tf.variable_scope('zw'):
            zw = _linear(hyper_output, embedding_size, False)
        with tf.variable_scope('alpha'):    
            alpha = _linear(zw, num_units, False)

        result = tf.multiply(alpha, layer)

    return result


def hyper_bias(layer, hyper_output, embedding_size, num_units, 
               scope="hyper"):
    with tf.variable_scope(scope):
        with tf.variable_scope('zb'):
            zb = _linear(hyper_output, embedding_size, False)
        with tf.variable_scope('beta'):
            beta = _linear(zb, num_units, False)

    return layer + beta

def get_mixture_coef(output, params, num_units):
    """
        args: 
            output: [tensor] an 2D tensor got from network
            params: [int] how many params need to be trained
            KMIX: [int] num of mixture networks
            num_units: [int] num of hidden units in main network
        return: 
            normed params of 'output', each param should be a 2D tensor, used for describing distrubtion
    """
    # out_pi = tf.placeholder(dtype=tf.float32, shape=[
    #                         None, KMIX], name="mixparam")
    # out_sigma = tf.placeholder(dtype=tf.float32, shape=[
    #                            None, KMIX], name="mixparam")
    # out_mu = tf.placeholder(dtype=tf.float32, shape=[
    #                         None, KMIX], name="mixparam")

    assert params == 2 * num_units + 1, "please confirm params = 2*num_units+1"

    # a list with length [params],
    # each element is tensor with shape: [batch_size, KMIX]
    split_params = tf.split(output, params, 1)

    out_mu_list = split_params[:num_units]
    out_sigma_list = split_params[num_units:-1]
    out_pi = split_params[-1]

    out_sigma_list = [tf.exp(out_sigma) for out_sigma in out_sigma_list]

    max_pi = tf.reduce_max(out_pi, 1, keep_dims=True)
    out_pi = tf.subtract(out_pi, max_pi)
    out_pi = tf.exp(out_pi)
    normalize_pi = tf.reciprocal(tf.reduce_sum(out_pi, 1, keep_dims=True))
    out_pi = tf.multiply(normalize_pi, out_pi)
    return out_mu_list, out_sigma_list, out_pi


def get_pi_idx(out_pi):
        """ transforms result into random ensembles
            args: 
                out_pi: [tensor] a tensor with shape [batch_size, KMIX]
            return: 
                idx_pi:  [int] index of mixture network
        """
        
        stop = tf.random_normal([1])

        num_pi = out_pi.get_shape().as_list()[1]

        for i in range(num_pi):
            cum += out_pi[:, i]
            idx_result = tf.cond(tf.less(stop, cum[0])[0], lambda: i, lambda: -1)

            if idx_result>0:
                return idx_result
        print("No Pi is drawn, ERROR!")
        return idx_result


def generate_ensemble(out_mu_list, out_sigma_list, out_pi):
    """sample based on normal distribution
        args:
            out_mu_list: [list] a list with length 'num_units', 
                                each element is a tensor with lenth 'KMIX'
            out_sigma_list: [list] a list with length 'num_units', 
                                each element is a tensor with lenth 'KMIX'
            out_pi: [list] a tensor with length 'KMIX', sum(out_pi) = 1
        return:
            result: [tensor] a 2D tensor with shape [batch_size, num_units]
    """

    assert len(out_mu_list) == len(out_sigma_list), "please confirm both lists have same length"
    num_units = len(out_mu_list)

    # initially random [0, 1]
    
    
    gen_weight_list = []
    # # normal random matrix (0.0, 1.0)
    # rn = tf.random_normal([num_units])

    mu = 0
    std = 0
    idx = 0
 
    
    for i in range(num_units):
        # idx = get_pi_idx(out_pi)
        idx=0
        mu = out_mu_list[i][:, idx]
        std = out_sigma_list[i][:, idx]
        gen_weight_list.append(mu + tf.multiply(tf.random_normal([1]), std)) 
    gen_weight = tf.stack(gen_weight_list,1)

    return gen_weight




def hyper_mix_norm(layer, hyper_output, embedding_size, num_units,
                   scope="hyper", use_bias=True):
    """
        args:
            layer: [tensor] input tensor need to be normed, with shape: [batch_size, num_units]
            hyper_output: [tensor] output tensor with shape: [batch_size, embedding_size] 
            num_units: [int] num of main network's hidden units

            init_gamma= 0.10
        return:
            result: [tensor] normed output with the same shape of 'layer': [batch_size, num_units]
    """

    KMIX = 2  # mix 5 networks
    #  there are KMIX networks in sum, and each network has 2 params: mu, stdeve
    # Assume each unit is irrelevant, and include a weight 'theta'
    params = 2 * num_units + 1 #257
    NOUT = KMIX * params 
    
    with tf.variable_scope(scope):
        with tf.variable_scope('zw'):
            # zw is a tensor with shape: [batch_size, embedding_size]
            zw = _linear(hyper_output, embedding_size, False)
        with tf.variable_scope('alpha'):
            # alpha is a tensor with shpae: [batch_size, NOUT]
            alpha = _linear(zw, NOUT, False)

        out_mu_list, out_sigma_list, out_pi = get_mixture_coef(alpha, params, num_units)


        gen_weight = generate_ensemble(out_mu_list, out_sigma_list, out_pi)
        result = tf.multiply(gen_weight, layer)
    print("7777777777")
    print(result)
    print("7777777777")
    return result

class HyperLSTMCell(tf.contrib.rnn.RNNCell):
    def __init__(self, num_units, forget_bias=1.0, use_layer_norm=True, 
                 use_recurrent_dropout=False, dropout_keep_prob=0.90,
                 hyper_num_units=128, hyper_embedding_size=16):
        """
            arg: 
                num_units: [int] hidden units num for main network
                forget_bias: [float] forget_bias
                use_layer_norm: [bool] whether use layer norm for main netwrok
                user_drop_out: [bool] whether use drop out for main network
                drop_out_pro: [float] probability of drop out
                hyper_num_units: [int] hidden units num for hyper network 
                hyper_embedding_size: [int] output units num for hyper network,
                                            always smaller than hyper_num_units
        """
        self.num_units = num_units
        self.forget_bias = forget_bias
        self.use_layer_norm = use_layer_norm
        self.use_recurrent_dropout = use_recurrent_dropout
        self.dropout_keep_prob = dropout_keep_prob
        self.hyper_num_units = hyper_num_units
        self.hyper_embedding_size = hyper_embedding_size

        #define training units num, 
        #both hidden units in main netowrk and hyper network
        self.total_num_units = self.num_units + self.hyper_num_units

        #define hyper cell 
        self.hyper_cell = tf.contrib.rnn.BasicLSTMCell(hyper_num_units)


    @property
    def output_size(self):
        return self.num_units
    @property
    def state_size(self):
        return tf.contrib.rnn.LSTMStateTuple(self.num_units + self.hyper_num_units, 
                                             self.num_units + self.hyper_num_units)

    def __call__(self, x, state, scope=None):
        """
            arg: 
                x: [tensor] input tensor at each time step with shape: [batch_szie, num_input]
                state: [tuple] state at last time step 
        """
        with tf.variable_scope(scope or type(self).__name__):
            total_c, total_h = state
            c = total_c[:, 0:self.num_units]
            h = total_h[:, 0:self.num_units]
            hyper_state = tf.contrib.rnn.LSTMStateTuple(total_c[:, self.num_units:],
                                                        total_h[:, self.num_units:])
            x_size = x.get_shape().as_list()[1]
            batch_size = x.get_shape().as_list()[0]
            embedding_size = self.hyper_embedding_size
            num_units = self.num_units

            #define weights and bias for main network
            W_xh = tf.get_variable('W_xh', initializer=tf.random_normal([x_size, 4*num_units]), dtype=tf.float32)
            W_hh = tf.get_variable('W_hh', initializer=tf.random_normal([num_units, 4*num_units]), dtype=tf.float32)
            bias = tf.get_variable('bias', initializer=tf.random_normal([4*num_units]), dtype=tf.float32)

            #define hyper network input, shape : [batch_size, x_size+num_units]
            hyper_input = tf.concat([x,h], 1)
            hyper_output, hyper_new_state = self.hyper_cell(hyper_input, hyper_state)

            xh = tf.matmul(x, W_xh)
            hh = tf.matmul(h, W_hh)

            #split Wxh contributions
            ix, jx, fx, ox = tf.split(xh, 4, 1)

            ix = hyper_norm(ix, hyper_output, embedding_size, num_units, 'hyper_ix')
            jx = hyper_norm(jx, hyper_output, embedding_size, num_units, 'hyper_jx')
            fx = hyper_norm(fx, hyper_output, embedding_size, num_units, 'hyper_fx')
            ox = hyper_norm(ox, hyper_output, embedding_size, num_units, 'hyper_ox')

            #split Whh contributions
            ih, jh, fh, oh = tf.split(hh, 4, 1)
            ih = hyper_norm(ih, hyper_output, embedding_size, num_units, 'hyper_ih')
            jh = hyper_norm(jh, hyper_output, embedding_size, num_units, 'hyper_jh')
            fh = hyper_norm(fh, hyper_output, embedding_size, num_units, 'hyper_fh')
            oh = hyper_norm(oh, hyper_output, embedding_size, num_units, 'hyper_oh')

            #split bias      
            ib, jb, fb, ob = tf.split(bias, 4, 0)
            ib = hyper_bias(ib, hyper_output, embedding_size, num_units, 'hyper_ib')
            jb = hyper_bias(jb, hyper_output, embedding_size, num_units, 'hyper_jb')
            fb = hyper_bias(fb, hyper_output, embedding_size, num_units, 'hyper_fb')
            ob = hyper_bias(ob, hyper_output, embedding_size, num_units, 'hyper_ob')

            #i = input_gate, j = new_input, f= forget_gate, o = output_gate
            i = ix + ih + ib
            j = jx + jh + jb
            f = fx + fh + fb
            o = ox + oh + ob

            if self.use_layer_norm:
                i = layer_norm(i, scope='ln_i/')
                j = layer_norm(j, scope='ln_j/')
                f = layer_norm(f, scope='ln_f/')
                o = layer_norm(o, scope='ln_o/')

            if self.use_recurrent_dropout:
                g = tf.nn.dropout(tf.tanh(j), self.dropout_keep_prob)
            else:
                g = tf.tanh(j)

            new_c = c*tf.sigmoid(f+self.forget_bias) + tf.sigmoid(i)*g
            if self.use_layer_norm:
                new_h = tf.tanh(layer_norm(new_c, scope='ln_c/')) * tf.sigmoid(o)
            else:
                new_h = tf.tanh(new_c) * tf.sigmoid(o)

            hyper_c, hyper_h = hyper_new_state
            new_total_c = tf.concat([new_c, hyper_c], 1)
            new_total_h = tf.concat([new_h, hyper_h], 1)

        return new_h, tf.contrib.rnn.LSTMStateTuple(new_total_c, new_total_h)         


class MLPLSTMCell(tf.contrib.rnn.RNNCell):

    def __init__(self, num_units, forget_bias=1.0, use_layer_norm=True, 
                 use_recurrent_dropout=False, dropout_keep_prob=0.90,
                 hyper_num_units=128, hyper_embedding_size=16):
        """ Multi-Layer Percepertion in LSTM network
            arg: 
                num_units: [int] hidden units num for main network
                forget_bias: [float] forget_bias
                use_layer_norm: [bool] whether use layer norm for main netwrok
                user_drop_out: [bool] whether use drop out for main network
                drop_out_pro: [float] probability of drop out
                hyper_num_units: [int] hidden units num for hyper network 
                hyper_embedding_size: [int] output units num for hyper network,
                                            always smaller than hyper_num_units
        """
        self.num_units = num_units
        self.forget_bias = forget_bias
        self.use_layer_norm = use_layer_norm
        self.use_recurrent_dropout = use_recurrent_dropout
        self.dropout_keep_prob = dropout_keep_prob
        self.hyper_num_units = hyper_num_units
        self.hyper_embedding_size = hyper_embedding_size

        #define training units num, 
        #both hidden units in main netowrk and hyper network
        # self.total_num_units = self.num_units + self.hyper_num_units


    @property
    def output_size(self):
        return self.num_units
    @property
    def state_size(self):
        return tf.contrib.rnn.LSTMStateTuple(self.num_units, 
                                             self.num_units)

    def __call__(self, x, state, scope=None):
        """
            arg: 
                x: [tensor] input tensor at each time step with shape: [batch_szie, num_input]
                state: [tuple] state at last time step 
        """
        with tf.variable_scope(scope or type(self).__name__):
            # total_c, total_h = state
            # c = total_c[:, 0:self.num_units]
            # h = total_h[:, 0:self.num_units]
            c, h = state
            # hyper_state = tf.contrib.rnn.LSTMStateTuple(total_c[:, self.num_units:],
            #                                              total_h[:, self.num_units:])
            x_size = x.get_shape().as_list()[1]
            batch_size = x.get_shape().as_list()[0]
            embedding_size = self.hyper_embedding_size
            num_units = self.num_units

            #define weights and bias for main network
            W_xh = tf.get_variable('W_xh', initializer=tf.random_normal([x_size, 4*num_units]), dtype=tf.float32)
            W_hh = tf.get_variable('W_hh', initializer=tf.random_normal([num_units, 4*num_units]), dtype=tf.float32)
            bias = tf.get_variable('bias', initializer=tf.random_normal([4*num_units]), dtype=tf.float32)

            #define hyper network input, shape : [batch_size, x_size+num_units]
            hyper_input = tf.concat([x,h], 1)
            hyper_output = hyper_input
            # hyper_output, hyper_new_state = self.hyper_cell(hyper_input, hyper_state)

            xh = tf.matmul(x, W_xh)
            hh = tf.matmul(h, W_hh)

            #split Wxh contributions
            ix, jx, fx, ox = tf.split(xh, 4, 1)

            ix = hyper_norm(ix, hyper_output, embedding_size, num_units, 'hyper_ix')
            jx = hyper_norm(jx, hyper_output, embedding_size, num_units, 'hyper_jx')
            fx = hyper_norm(fx, hyper_output, embedding_size, num_units, 'hyper_fx')
            ox = hyper_norm(ox, hyper_output, embedding_size, num_units, 'hyper_ox')

            #split Whh contributions
            ih, jh, fh, oh = tf.split(hh, 4, 1)
            ih = hyper_norm(ih, hyper_output, embedding_size, num_units, 'hyper_ih')
            jh = hyper_norm(jh, hyper_output, embedding_size, num_units, 'hyper_jh')
            fh = hyper_norm(fh, hyper_output, embedding_size, num_units, 'hyper_fh')
            oh = hyper_norm(oh, hyper_output, embedding_size, num_units, 'hyper_oh')

            #split bias      
            ib, jb, fb, ob = tf.split(bias, 4, 0)
            ib = hyper_bias(ib, hyper_output, embedding_size, num_units, 'hyper_ib')
            jb = hyper_bias(jb, hyper_output, embedding_size, num_units, 'hyper_jb')
            fb = hyper_bias(fb, hyper_output, embedding_size, num_units, 'hyper_fb')
            ob = hyper_bias(ob, hyper_output, embedding_size, num_units, 'hyper_ob')

            #i = input_gate, j = new_input, f= forget_gate, o = output_gate
            i = ix + ih + ib
            j = jx + jh + jb
            f = fx + fh + fb
            o = ox + oh + ob

            if self.use_layer_norm:
                i = layer_norm(i, scope='ln_i/')
                j = layer_norm(j, scope='ln_j/')
                f = layer_norm(f, scope='ln_f/')
                o = layer_norm(o, scope='ln_o/')

            if self.use_recurrent_dropout:
                g = tf.nn.dropout(tf.tanh(j), self.dropout_keep_prob)
            else:
                g = tf.tanh(j)

            new_c = c*tf.sigmoid(f+self.forget_bias) + tf.sigmoid(i)*g
            if self.use_layer_norm:
                new_h = tf.tanh(layer_norm(new_c, scope='ln_c/')) * tf.sigmoid(o)
            else:
                new_h = tf.tanh(new_c) * tf.sigmoid(o)

            # hyper_c, hyper_h = hyper_new_state
            # new_total_c = tf.concat([new_c, hyper_c], 1)
            # new_total_h = tf.concat([new_h, hyper_h], 1)
            new_total_c = new_c
            new_total_h = new_h

        return new_h, tf.contrib.rnn.LSTMStateTuple(new_total_c, new_total_h)         



class LayerNormLSTMCell(tf.contrib.rnn.RNNCell):

    def __init__(self, num_units, forget_bias=1.0, activation=tf.nn.tanh):
        # forget bias is pretty important for training
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation

    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(self._num_units, self._num_units)    

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            c, h = state
            concat = _linear([inputs, h], 4*self._num_units, False)
            i,j,f,o = tf.split(concat, 4, 1)

            #add layer normalization for each gate before activation
            i = layer_norm(i, scope='i/')
            j = layer_norm(j, scope='j/')
            f = layer_norm(f, scope='f/')
            o = layer_norm(o, scope='o/')

            new_c = c * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) * self._activation(j)

            #add layer normalization in calculation of new hidden state
            new_h = self._activation(layer_norm(new_c, scope='h/')) * tf.nn.sigmoid(o)
            
            new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)

            return new_h, new_state

class DynamicLayerNormLSTMCell(tf.contrib.rnn.RNNCell):
    def __init__(self, num_units, forget_bias=1.0, dynamic_num_units=128, dynamic_embedding_size=16):
        """ code for 'Dynamic Layer Normalization for Adaptive Neural Acoustic Modeling in Speech Recognition'
            the main idea is to creative two network: main netowrk and dynamic network. The dynamic one could 
            generate scaling factor 'alpha' and shift factor 'beta' for layer normalization. Then, the main 
            netowrk uses layer normalization for each gate (i, j, f, o) in LSTM.

            args:
                num_units: [int] hidden units num
                forget_bias: [float] foget gate bias
                dynamic_num_units: [int] hidden units num for dynamic network
                dynamic_embedding_size: [int] output units num for dynamic network,
                                              always smaller than 'dynamic_num_units'                                              
        """
        self.num_units = num_units
        self.forget_bias = forget_bias
        self.dynamic_num_units = dynamic_num_units
        self.dynamic_embedding_size = dynamic_embedding_size

        #define training units num
        self.total_num_units = self.num_units + self.dynamic_num_units
        #define dynamic cell
        self.dynamic_cell = tf.contrib.rnn.BasicLSTMCell(self.dynamic_num_units)


    @property
    def output_size(self):
        return self.num_units
    @property
    def state_size(self):
        #In fact, both main network and dynamic netowrk need to be trained.
        #Therefore, the state size should inlcude hidden units num of both network
        return tf.contrib.rnn.LSTMStateTuple(self.num_units+self.dynamic_num_units,
                                             self.num_units+self.dynamic_num_units)

    def __call__(self, x, state, scope=None):
        """
            arg: 
                x: [tensor] input tensor at a single time step with shape: [batch_szie, num_input]
                state: [tuple] state at last time step 
        """
        with tf.variable_scope(scope or type(self).__name__):
            total_c, total_h = state
            c = total_c[:, 0:self.num_units]
            h = total_h[:, 0:self.num_units]
            dynamic_state = tf.contrib.rnn.LSTMStateTuple(total_c[:, self.num_units:],
                                                          tocal_h[:, self.num_units:])
            x_size = x.get_shape().as_list()[1]
            batch_size = x.get_shape().as_list()[0]
            embedding_size = self.dynamic_embedding_size
            num_units = self.num_units

            concat = _linear([x, h], 4*self.num_units, False)
            #each gate has shape: [batch_size, num_units]
            i, j, f, o = tf.split(concat, 4, 1)

            #add dynamic layer normalization for each gate before activation
            #how to get 'alpha' at each layer. 











# test if the layer normarlization works well
if __name__ == '__main__':
    #shape is [batch_size, time_steps, num_inputs]
    num_input = 28
    batch_size = 128
    time_steps = 28
    num_hidden = 128
    
    inputs = tf.placeholder(tf.float32, [batch_size, time_steps,num_input]) 
    inp = tf.placeholder(tf.float32, [batch_size, num_hidden])

    cell = HyperLSTMCell(num_hidden)
    print cell
    # cell = LayerNormLSTMCell(num_hidden)
    # outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=inputs.dtype)
    
    # X = np.random.randn(batch_size, time_steps,num_input)
    # with tf.Session() as sess:
    #     init = tf.global_variables_initializer()
    #     sess.run(init)
    #     for _ in range(3):
    #         result = sess.run(inputs, feed_dict={inputs: X})
    #         print result

    # result = generate_sum_feature(inputs)
    # inputs = tf.unstack(inputs, time_steps, 1)
    # danamic_layer_norm(inp, inputs)