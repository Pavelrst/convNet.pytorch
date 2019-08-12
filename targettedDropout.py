

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



class __targetedDropout(Module):
    def __init__(self, drop_rate , targeted_percentage ,inplace=False):
        super(_DropoutNd, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.drop_rate = drop_rate
        self.targeted_percentage = targeted_percentage
        self.inplace = inplace

    def extra_repr(self):
        return 'dropRate={} , targetedPercentage={} , inplace={}'.format(self.drop_rate,self.targeted_percentage,self.inplace)

class targetedDropout(__targetedDropout):
    def forward(self,input):
        # need to be implemented