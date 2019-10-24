import torch
from torch.nn import Conv2d
from math import sqrt, ceil, floor
import matplotlib.pyplot as plt
import numpy as np

def register_stats_collectors(model):
    '''
    Registering forward hooks for each Conv layer.
    :param model: our model.
    :return: model with hooks.
    '''
    print("Registering forward hooks")
    for name, module in model.named_modules():
        if type(module) == Conv2d:
            module.register_forward_hook(statistic_collector_forward_hook)
    return model

def register_hist_collectors(model):
    '''
    Registering forward hooks for each Conv layer.
    :param model: our model.
    :return: model with hooks.
    '''
    print("Registering forward hooks")
    for name, module in model.named_modules():
        if type(module) == Conv2d:
            module.register_forward_hook(histograms_collector_forward_hook)
    return model


def dump_buffers(model):
    '''
    Iterate over all layers with buffers (and hooks) and dump statistics to file.
    :param model: our model
    '''
    # TODO: dump all collected data to csv file.
    # TODO: save all images with corresponding names.
    for name, module in model.named_modules():
        if type(module) == Conv2d:
            print('=======', name)
            max = module._buffers['max'].cpu().item()
            min = module._buffers['min'].cpu().item()
            for key, value in module._buffers.items():
                if key=='hist':
                    print('histogram')
                    #for val in value.cpu().numpy():
                        #print(val)
                    x_data = np.linspace(min, max, num=201)
                    plt.plot(x_data, module._buffers['hist'].cpu())
                    plt.show()
                else:
                    print(key, value)


def statistic_collector_forward_hook(module, input, output):
    '''
    This hook gathers all required statistics for quantization:
    Minimum value across all data set.
    Maximum value across all data set.
    Mean value across all data set.
    Num of values in total in the data set.
    Estimation of Std value across all data set.
    '''
    min = torch.min(output).detach()
    max = torch.max(output).detach()

    if 'max' not in module._buffers.keys():
        module._buffers['max'] = max
    else:
        if max > module._buffers['max']:
            module._buffers['max'] = max

    if 'min' not in module._buffers.keys():
        module._buffers['min'] = min
    else:
        if min < module._buffers['min']:
            module._buffers['min'] = min

    if 'std' not in module._buffers.keys():
        module._buffers['std'] = torch.std(output).detach()
        module._buffers['sum'] = torch.sum(output).detach()
        module._buffers['num'] = torch.numel(output)
        module._buffers['mean'] = torch.mean(output).detach()
    else:
        old_std = module._buffers['std']
        old_mean = module._buffers['mean']
        old_num = module._buffers['num']

        module._buffers['sum'] += torch.sum(output).detach()
        module._buffers['num'] += torch.numel(output)

        module._buffers['mean'] = module._buffers['sum']/module._buffers['num']
        module._buffers['std'] = update_std(output, old_std, old_mean,
                                            module._buffers['mean'], old_num)


def histograms_collector_forward_hook(module, input, output):
    '''
    This hook gathers histograms of the activations,
    which is not must for quantization. Should be registered after
    first pass of the data.
    '''
    #if 'max' not in module._buffers.keys():
    max = module._buffers['max']

    #if 'min' not in module._buffers.keys():
    min = module._buffers['min']

    if 'hist' not in module._buffers.keys():
        module._buffers['hist'] = torch.histc(output, bins=201, min=min, max=max)
    else:
        module._buffers['hist'] += torch.histc(output, bins=201, min=min, max=max)


def update_std(values, old_std, old_mean, new_mean, total_values_so_far):
    # See here:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_Online_algorithm
    numel = values.numel() if isinstance(values, torch.Tensor) else values.size
    M = (old_std ** 2) * (total_values_so_far - 1)
    mean_diffs = (values - old_mean) * (values - new_mean)
    M += mean_diffs.sum()
    return sqrt((M / (total_values_so_far + numel - 1)).item())
