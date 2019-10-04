import torch
import seaborn as sns
from datetime import datetime
from os import path
import matplotlib.pyplot as plt

def validate_prune_args(args):
    if (args.pruning_policy is not None) and (args.eval_path is not None) and (args.pruning_perc is not None):
        print('The model will be pruned with pruning percentage: ', args.pruning_perc)
        return 'single_run'
    elif (args.pruning_policy is not None) and (args.eval_path is None) and (args.pruning_perc is not None):
        print('If you want to prune and evaluate a model, define a path to the model using --eval-path <some path>')
        return 'wrong_args'
    elif (args.pruning_policy is None) and (args.eval_path is not None) and (args.pruning_perc is not None):
        print('If you want to prune and evaluate a model, set pruning policy using --pruning-policy <unit/weigh>')
        return 'wrong_args'
    elif (args.pruning_policy is not None) and (args.eval_path is not None) and (args.pruning_perc is None):
        print('The model will be pruned for following percentages: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9')
        return 'multiple_run'
    else:
        return 'wrong_args'

def unit_prune(state_dict, key, prune_percentage=0):
    weights = state_dict[key]

    if prune_percentage == 0:
        return weights

    initial_shape = weights.shape
    # if initial_shape[0] == 64:
    #     print('!')

    input = weights.view(initial_shape[0], -1)

    heat_maps_dir = 'C:\\Users\\Pavel\\Desktop\\targeted_dropout_pytorch\\pics\\experiment_2'
    plot = sns.heatmap(input, center=0)
    name = str(datetime.now()).replace(':', '_').replace('-', '_').replace('.', '_').replace(' ', '_') + 'before.png'
    plot.get_figure().savefig(path.join(heat_maps_dir, name))
    plt.clf()

    norm = torch.abs(input).sum(dim=1)
    idx = int(prune_percentage * (input.shape[0] - 1))
    sorted_norms = torch.sort(norm)[0]
    threshold = sorted_norms[idx]
    mask = torch.where(norm > threshold, torch.zeros(norm.shape), torch.ones(norm.shape))
    mask = torch.t(mask.repeat(input.shape[1], 1))

    out_w = (1 - mask) * input

    plot = sns.heatmap(out_w, center=0)
    name = str(datetime.now()).replace(':', '_').replace('-', '_').replace('.', '_').replace(' ', '_') + 'after.png'
    plot.get_figure().savefig(path.join(heat_maps_dir, name))
    plt.clf()

    out_w = out_w.view(initial_shape)

    state_dict[key] = out_w

def main():
    load_path = 'C:\\Users\\Pavel\\Desktop\\2019-08-29_17-49-37\\model_best.pth.tar'
    # TODO:Load a model
    model = torch.load(load_path, map_location='cpu')

    # TODO: Evaluate the model before pruning.

    # TODO: Iterate over all Conv later weights and prune them according to given policy.
    for key in model['state_dict']:
        #print(key)
        if 'conv' in key and key != 'conv1.weight': # Except the first convolution layer with 3 channels.
            print('Pruning ', key, ' of shape ', model['state_dict'][key].shape)

            unit_prune(model['state_dict'], key, prune_percentage=0.5)



    # TODO: Evaluate the pruned model.

if __name__ == "__main__":
    main()