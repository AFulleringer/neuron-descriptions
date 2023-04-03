"""Dissect a pretrained vision model."""
import argparse
import pathlib

from src.exemplars import compute, datasets, models
from src.utils import env

from torch import cuda
import torch
from os import path

parser = argparse.ArgumentParser(description='compute unit exemplars')
parser.add_argument('model', help='model architecture')
parser.add_argument('dataset', help='dataset of unseen examples for model')
parser_ex = parser.add_mutually_exclusive_group()
parser_ex.add_argument('--layer-names',
                       nargs='+',
                       help='layer names to compute exemplars for')
parser_ex.add_argument('--layer-indices',
                       type=int,
                       nargs='+',
                       help='layer indices to compute exemplars for; '
                       'cannot be used with --layers')
parser.add_argument(
    '--units',
    type=int,
    help='only compute exemplars for first n units (default: all)')
parser.add_argument('--data-root',
                    type=pathlib.Path,
                    help='link results (in --results-root) to this directory '
                    '(default: <project data dir> / model / dataset)')
parser.add_argument('--results-root',
                    type=pathlib.Path,
                    help='exemplars results root '
                    '(default: <project results dir> / exemplars)')
parser.add_argument('--viz-root',
                    type=pathlib.Path,
                    help='exemplars visualization root '
                    '(default: <project results dir> / exemplars / viz)')
parser.add_argument('--model-file',
                    type=pathlib.Path,
                    help='path to model weights')
parser.add_argument('--dataset-path',
                    type=pathlib.Path,
                    help='path to dataset')
parser.add_argument('--no-viz',
                    action='store_true',
                    help='do not compute visualization')
parser.add_argument('--no-link',
                    action='store_true',
                    help='do not link results to data dir')
parser.add_argument('--num-workers',
                    type=int,
                    default=16,
                    help='number of worker threads (default: 16)')
parser.add_argument('--device', help='manually set device (default: guessed)')

#This is for me!
parser.add_argument("--results-directory", type=str)
parser.add_argument("--use-our-model", action='store_true')
parser.add_argument("--test-symlink", action='store_true') #Used for debugging. If the symlink fails, stop regeneration of exemplars


args = parser.parse_args()

device = args.device or 'cuda' if cuda.is_available() else 'cpu'

model, layers, config = models.load(f'{args.model}/{args.dataset}',
                                    map_location=device,
                                    path=args.model_file)

print(args)
exit(0)
def get_my_model_dict_and_layer(results_directory):
    if not path.exists(results_directory):
        print(f'ERROR, the results directory {results_directory} does not exist!')
        print('Exiting program')
        exit(0)

    def get_tuple_from_config_dict(my_dict, dict_key):
        output = my_dict[dict_key]
        output = output.strip('[')
        output = output.strip(']')
        output = output.split(',')
        return output

    config_dict = {}
    with open(path.join(args.results_directory, "configuration.txt")) as f:
        for line in f:
            (key, val) = line.split(':')
            config_dict[key] = val.strip()
    #print(config_dict)
    steps = [i for i in
             range(0, int(config_dict['nsteps']) + 1, int(config_dict['save_interval']))]

    #If you're not tracking intermediate steps, take only the first and final versions of the model
    #steps = [steps[0], steps[-1]]

    # Feature name is arbitrary but necessary for the hook
    feature_name = 'activations'
    layers = get_tuple_from_config_dict(config_dict, 'layer')
    my_layer = layers[0]
    # used for converting our name of the model version to theirs.
    # ('conv1', 'conv2', 'conv3', 'conv4', 'conv5')
    layer_dict = {
        'features_0': 'conv1',
        'features_3': 'conv2',
        'features_6': 'conv3',
        'features_8': 'conv4',
        'features_10': 'conv5',
    }
    milan_layer = layer_dict[my_layer]
    model_dict = torch.load(path.join(results_directory, f"model_checkpoint_step_{steps[-1]}.pt"))
    return model_dict

if args.use_our_model:
    my_model_dict,  = get_my_model_dict_and_layer(args.results_directory)
    model = model.load_state_dict(my_model_dict)


dataset, generative = args.dataset, False
print(f'dataset: {dataset}')
if isinstance(config.exemplars, models.GenerativeModelExemplarsConfig):
    dataset = config.exemplars.dataset
    generative = True
# TODO(evandez): Yuck, push this into config.
elif dataset == datasets.KEYS.IMAGENET_BLURRED:
    dataset = datasets.KEYS.IMAGENET

dataset = datasets.load(dataset, path=args.dataset_path)

if args.layer_names:
    layers = args.layer_names
elif args.layer_indices:
    layers = [layers[index] for index in args.layer_indices]
assert layers is not None, 'should always be >= 1 layer'

units = None
if args.units:
    units = range(args.units)

data_root = args.data_root
if data_root is None:
    data_root = env.data_dir()
data_dir = data_root / args.model / args.dataset

results_root = args.results_root
if results_root is None:
    results_root = env.results_dir() / 'exemplars'
results_dir = results_root / args.model / args.dataset

viz_root = args.viz_root
viz_dir = None
if viz_root is not None:
    viz_dir = viz_root / args.model / args.dataset
elif not args.no_viz:
    viz_dir = results_root / 'viz' / args.model / args.dataset

if not args.test_symlink:
    for layer in layers:
        if generative:
            compute.generative(model,
                               dataset,
                               layer=layer,
                               units=units,
                               results_dir=results_dir,
                               viz_dir=viz_dir,
                               save_viz=not args.no_viz,
                               device=device,
                               num_workers=args.num_workers,
                               **config.exemplars.kwargs)
        else:
            compute.discriminative(model,
                                   dataset,
                                   layer=layer,
                                   units=units,
                                   results_dir=results_dir,
                                   viz_dir=viz_dir,
                                   save_viz=not args.no_viz,
                                   device=device,
                                   num_workers=args.num_workers,
                                   **config.exemplars.kwargs)

print('data_dir: ', data_dir)
print('results_dir: ', results_dir)
if not args.no_link:
    data_dir.symlink_to(results_dir, target_is_directory=True)
