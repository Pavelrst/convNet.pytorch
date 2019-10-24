python main.py --dataset cifar10 --model resnet --depth 32 --batch-size 64 --device cuda --model-config "{'dropout' : 0.5 , 'dp_percentage' : 0.5 , 'dp_type' : 'weight' , 'device' : 'cuda'}"

python main.py --dataset cifar10 --model resnet --depth 32 --batch-size 64 --device cuda --model-config "{'dropout' : 0 , 'dp_percentage' : 0 , 'dp_type' : 'weight' , 'device' : 'cuda'}"