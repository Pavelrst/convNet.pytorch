import torch
from torch.nn.modules import Module
from torch.nn import functional as F
import  numpy as np


class _targetedDropout(Module):
    def __init__(self, drop_rate, targeted_percentage, device='cpu' , inplace=False):
        super(_targetedDropout, self).__init__()
        if drop_rate < 0 or drop_rate > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = drop_rate
        self.targeted_percentage = targeted_percentage
        self.inplace = inplace
        self.device = device
    def extra_repr(self):
        return 'dropRate={} , targetedPercentage={} , inplace={}'.format(self.p,self.targeted_percentage,self.inplace)

class targeted_weight_dropout(_targetedDropout):
    def forward(self,input, is_training):
        Test = False
        if Test:
            torch.set_printoptions(threshold=5000)
            torch.set_printoptions(precision=2)
            self.p = 1

        if self.targeted_percentage == 0:
            # Equal to not doing dropout.
            # It's true for both train and test phase.
            return input

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
        mask = torch.where(input > threshoulds, torch.zeros(input.shape).to(self.device), torch.ones(input.shape).to(self.device))


        if not is_training:
            # When not training we set to zero all weights
            # which are less than threshold, as it would be
            # if the model was pruned.
            # TODO: This code is not tested.
            out_w = (1 - mask) * input
            out_w = out_w.view(initial_shape)
            return out_w

        # mask_2 = matrix of {1/0} of (Uni < drop_rate)
        mask_2 = torch.where(torch.empty(input.shape).uniform_(0, 1).to(self.device) > self.p, torch.zeros(input.shape).to(self.device), torch.ones(input.shape).to(self.device))

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

    # weight: (out_channels , in_channels , kH , kW)
    # New matrix shape will be:
    # ( out_channels, in_channels * kH * kW )
    '''
    def forward(self, input , is_training):
        Test = False

        if self.targeted_percentage == 0:
            # Equal to not doing dropout.
            # It's true for both train and test phase.
            return input
        
        initial_shape = input.shape
        input = input.view(initial_shape[0], -1)
        norm = input.norm(dim=1)
        idx = int(self.targeted_percentage * (input.shape[0]-1))
        sorted_norms = torch.sort(norm)[0]
        threshold = sorted_norms[idx]
        mask = torch.where(norm.to(self.device) > threshold.to(self.device), torch.zeros(norm.shape).to(self.device), torch.ones(norm.shape).to(self.device))
        mask = torch.t(mask.repeat(input.shape[1] , 1)).to(self.device)

        if not is_training:
            # When not training we set to zero all weights
            # which are less than threshold, as it would be
            # if the model was pruned.
            # TODO: This code is not tested.
            out_w = (1 - mask) * input
            out_w = out_w.view(initial_shape)
            return out_w

        tmp = (1 - self.p < torch.empty(input.shape).uniform_(0, 1)).to(self.device)
        mask_temp = torch.where((tmp.byte() & mask.byte()),
                             torch.zeros(input.shape).to(self.device), torch.ones(input.shape).to(self.device)).to(self.device)

        after_dropout = mask_temp * input.to(self.device)

        if Test:
            self.self_test(input, after_dropout, threshold)

        final_weights = after_dropout.view(initial_shape)
        return final_weights

    def self_test(self, w_in, w_out, thresh):
        '''
        targeted_unit_dropout self test
        :param w_in: tensor of shape ( out_channels, in_channels * kH * kW )
        :param w_out: tensor of shape ( out_channels, in_channels * kH * kW )
        :return: Pass / Fail
        '''
        w_in = w_in.detach().numpy()
        w_out = w_out.detach().numpy()
        thresh = thresh.detach().numpy()

        num_of_rows = w_in.shape[0]

        idx = int(self.targeted_percentage * num_of_rows)

        norm_list = []
        for row in range(num_of_rows):
            norm = np.linalg.norm(w_in[row, :])
            norm_list.append(norm)

        sorted_norm_list = sorted(norm_list)

        threshold = sorted_norm_list[idx]

        for row in range(num_of_rows):
            norm = np.linalg.norm(w_in[row, :])
            if norm <= threshold:
                # drop
                w_in[row, :] = np.zeros(w_in[row, :].shape)

        #for row in range(num_of_rows):
        #    print(np.linalg.norm(w_in[row, :]), '=?=', np.linalg.norm(w_out[row, :]))

        if np.array_equal(thresh, threshold) and np.array_equal(w_in, w_out):
            print("Test passed")
        elif np.allclose(thresh, threshold) and np.allclose(w_in, w_out):
            print("Test passed - with respect to tolerance")
        else:
            print("Test FAILED")

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