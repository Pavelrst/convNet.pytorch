import torch

def validate_prune_args(args):
    if (args.pruning_policy is not None) and (args.pruning_modelpath is not None):
        print('The model will be pruned with pruning percentage: ', args.pruning_perc)
        return True
    elif (args.pruning_policy is not None) and (args.pruning_modelpath is None):
        print('If you want to prune and evaluate a model, define a path to the model using --pruning-modelpath <some path>')
        return False
    elif (args.pruning_policy is None) and (args.pruning_modelpath is not None):
        print('If you want to prune and evaluate a model, set pruning policy using --pruning-policy <unit/weigh>')
        return False
    else:
        return False

def unit_prune(state_dict, key, prune_percentage=0):
    weights = state_dict[key]

    if prune_percentage == 0:
        return weights

    initial_shape = weights.shape
    input = weights.view(initial_shape[0], -1)
    norm = torch.abs(input).sum(dim=1)
    idx = int(prune_percentage * (input.shape[0] - 1))
    sorted_norms = torch.sort(norm)[0]
    threshold = sorted_norms[idx]
    mask = torch.where(norm > threshold, torch.zeros(norm.shape), torch.ones(norm.shape))
    mask = torch.t(mask.repeat(input.shape[1], 1))

    out_w = (1 - mask) * input
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