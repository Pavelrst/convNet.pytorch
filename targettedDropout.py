from torch.nn.modules import Module
from torch.nn import functional as F

# def targeted_weight_dropout(w, params, is_training):
#   drop_rate = params.drop_rate
#   targ_perc = params.targ_rate
#
#   w_shape = w.shape
#   w = tf.reshape(w, [-1, w_shape[-1]])
#   norm = tf.abs(w)
#   idx = tf.to_int32(targ_perc * tf.to_float(tf.shape(w)[0]))
#   threshold = tf.contrib.framework.sort(norm, axis=0)[idx]
#   mask = norm < threshold[None, :]
#
#   if not is_training:
#     w = (1. - tf.to_float(mask)) * w
#     w = tf.reshape(w, w_shape)
#     return w
#
#   mask = tf.to_float(
#       tf.logical_and(tf.random_uniform(tf.shape(w)) < drop_rate, mask))
#   w = (1. - mask) * w
#   w = tf.reshape(w, w_shape)
#   return w


class _targetedDropout(Module):
    def __init__(self, drop_rate, targeted_percentage, inplace=False):
        super(_targetedDropout, self).__init__()
        if drop_rate < 0 or drop_rate > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = drop_rate
        self.targeted_percentage = targeted_percentage
        self.inplace = inplace
    def extra_repr(self):
        return 'dropRate={} , targetedPercentage={} , inplace={}'.format(self.drop_rate,self.targeted_percentage,self.inplace)

class targeted_weight_dropout(_targetedDropout):
    def forward(self,input):
        # TODO: need to be implemented
        print("targeted_weight_dropout called")
        '''
        Add your code here.
        '''
        return F.dropout(input, self.p, self.training, self.inplace)

class targeted_unit_dropout(_targetedDropout):
    '''
    Unit - i.e. a column of a matrix.
    The threshold is calculated by... 
    '''
    def forward(self, input):
        # TODO: need to be implemented
        print("targeted_unit_dropout called")
        '''
        Add your code here.
        '''
        return F.dropout(input, self.p, self.training, self.inplace)

class ramping_targeted_weight_dropout(_targetedDropout):
    '''
    Ramping targeted dropout simply anneals the targeted_percentage from zero,
    to the specified final targeted_percentage throughout the course of training.
    For our ResNet experiments, we anneal from zero to 95% of targeted_percentage
    over the first forty-nine epochs, and then from 95% of targeted_percentage
    to 100% of γ over the subsequent forty-nine.
    '''
    def forward(self,input):
        # TODO: need to be implemented
        print("ramping_targeted_weight_dropout")
        '''
        Add your code here.
        '''
        return F.dropout(input, self.p, self.training, self.inplace)

class ramping_targeted_unit_dropout(_targetedDropout):
    '''
    Ramping targeted dropout simply anneals the targeted_percentage from zero,
    to the specified final targeted_percentage throughout the course of training.
    For our ResNet experiments, we anneal from zero to 95% of targeted_percentage
    over the first forty-nine epochs, and then from 95% of targeted_percentage
    to 100% of γ over the subsequent forty-nine.
    '''
    def forward(self, input):
        # TODO: need to be implemented
        print("ramping_targeted_unit_dropout")
        '''
        Add your code here.
        '''
        return F.dropout(input, self.p, self.training, self.inplace)