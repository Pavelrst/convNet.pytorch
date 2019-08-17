import torch
from torch.nn.modules import Module
from torch.nn import functional as F
import  numpy as np

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
#   mask = tf.to_float(tf.logical_and(tf.random_uniform(tf.shape(w)) < drop_rate, mask))
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
        Test = False
        if Test:
            torch.set_printoptions(threshold=5000)
            torch.set_printoptions(precision=2)
            self.p = 1

        # Reshape - remove redundant dimensions.
        # weight: (out_channels , in_channels , kH , kW)
        # New matrix shape will be:
        # ( in_channels * kH * kW , out_channels )
        initial_shape = input.shape

        input = input.view(initial_shape[0], -1)
        input = torch.abs(input)
        input = torch.t(input)

        idx = int(self.targeted_percentage * input.shape[0])
        sorted = torch.sort(input, dim=0)

        # For each column in w_abs calc the threshold.
        threshoulds = sorted[0][int(idx)].repeat(input.shape[0], 1)

        # As a result all elements which are '0' - protected from being dropped out.
        mask = torch.where(input > threshoulds, torch.zeros(input.shape), torch.ones(input.shape))

        # TODO: TBD!!!
        #   if not is_training:
        #     w = (1. - tf.to_float(mask)) * w
        #     w = tf.reshape(w, w_shape)
        #     return w

        # mask_2 = matrix of {1/0} of (Uni < drop_rate)
        mask_2 = torch.where(torch.empty(input.shape).uniform_(0.1) > self.p,
                             torch.zeros(input.shape), torch.ones(input.shape))

        # final_mask = mask_1 LOGIC_AND mask_2.
        final_mask = (1 - (mask.byte() & mask_2.byte())).double().float()

        out_w = input.float() * final_mask

        if Test:
            self.self_test(input, out_w, threshoulds)

        out_w = torch.t(out_w)
        out_w = out_w.view(initial_shape)

        return out_w

    def self_test(self, w_in, w_out, thresholds):
        '''
        :param w_in: tensor of shape ( in_channels * kH * kW , out_channels )
        :param w_out: tensor of shape ( in_channels * kH * kW , out_channels )
        :return: Pass / Fail
        '''
        w_in = w_in.detach().numpy()
        w_out = w_out.detach().numpy()

        thresholds = thresholds[0, :].detach().numpy()
        thresholds_list = []

        idx = int(self.targeted_percentage * len(w_in[:, 0]))
        for col in range(w_in.shape[1]):
            thresh = sorted(w_in[:, col])[idx]
            thresholds_list.append(thresh)
            for row in range(w_in.shape[0]):
                if w_in[row][col] <= thresh:
                    # Drops with no respect to self.p
                    w_in[row][col] = 0

        thresholds_list = np.array(thresholds_list)

        if np.array_equal(thresholds, thresholds_list) and np.array_equal(w_in, w_out):
            print("Test passed")
        elif np.allclose(thresholds, thresholds_list) and np.allclose(w_in, w_out):
            print("Test passed - with respect to tolerance")
        else:
            print("Test FAILED")


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