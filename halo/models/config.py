from models.data.input_helpers import random_rotate
import yaml
from torchvision import transforms
from models import naive
from models import data
# from artihand import nasa

# from im2mesh import data
# from im2mesh import onet, r2n2, psgn, pix2mesh, dmc
# from im2mesh import preprocess

# method_dict = {
#     'onet': onet,
#     'r2n2': r2n2,
#     'psgn': psgn,
#     'pix2mesh': pix2mesh,
#     'dmc': dmc,
# }

method_dict = {
    'naive': naive
}


# General config
def load_config(path, default_path=None):
    ''' Loads config file.

    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f, Loader=yaml.FullLoader )

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


# Models
def get_model(cfg, device=None, dataset=None):
    ''' Returns the model instance.

    Args:
        cfg (dict): config dictionary
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    method = cfg['method']
    model = method_dict[method].config.get_model(
        cfg, device=device, dataset=dataset)
    return model


# Trainer
def get_trainer(model, optimizer, cfg, device):
    ''' Returns a trainer instance.

    Args:
        model (nn.Module): the model which is used
        optimizer (optimizer): pytorch optimizer
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    trainer = method_dict[method].config.get_trainer(
        model, optimizer, cfg, device)
    return trainer


# Generator for final mesh extraction
def get_generator(model, cfg, device):
    ''' Returns a generator instance.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    generator = method_dict[method].config.get_generator(model, cfg, device)
    return generator


# Datasets
def get_dataset(mode, cfg, return_idx=False, return_category=False):
    ''' Returns the dataset.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        return_idx (bool): whether to include an ID field
    '''
    method = cfg['method']
    dataset_type = cfg['data']['dataset']
    dataset_folder = cfg['data']['path']
    use_bps = cfg['model']['use_bps']
    # categories = cfg['data']['classes']
    rand_rotate = cfg['data']['random_rotate']

    # Get split
    splits = {
        'train': cfg['data']['train_split'],
        'val': cfg['data']['val_split'],
        'test': cfg['data']['test_split'],
    }

    split = splits[mode]
    # Create dataset
    if dataset_type == 'obman':
        # print("sample hand dataset")
        # print("folder", dataset_folder)

        # Method specific data
        # data_loader_helpers = method_dict[method].config.get_data_helpers(mode, cfg)
        # transforms_dict = method_dict[method].config.get_data_transforms(mode, cfg)
        # Input data
        # input_helper = get_inputs_helper(mode, cfg)
        # if input_helper is not None:
        #     data_loader_helpers['inputs'] = input_helper

        # input_helper = None
        # dataset = data.SampleHandDataset(
        #     dataset_folder, data_loader_helpers, # input_helper,
        #     transforms=transforms_dict,
        #     split=split,
        #     no_except=False,
        #     return_idx=return_idx
        # )
        dataset = data.ObmanDataset(
            dataset_folder,
            split=split,
            no_except=False,
            use_bps=use_bps,
            return_idx=return_idx
        )
    elif dataset_type == 'inference':
        dataset = data.InferenceDataset(
            dataset_folder,
            split=split,
            no_except=False,
            use_bps=use_bps,
            random_rotate=rand_rotate,
            return_idx=return_idx
        )
    else:
        raise ValueError('Invalid dataset "%s"' % cfg['data']['dataset'])

    return dataset


def get_inputs_helper(mode, cfg):
    ''' Returns the inputs fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): config dictionary
    '''
    input_type = cfg['data']['input_type']
    with_transforms = cfg['data']['with_transforms']

    if input_type is None:
        inputs_field = None
    
    elif input_type == 'trans_matrix':
        inputs_helper = data.TransMatInputHelper(
            cfg['data']['transmat_file'],
            use_bone_length=cfg['model']['use_bone_length'],
            unpackbits=cfg['data']['points_unpackbits']
        )

    # elif input_type == 'img':
    #     if mode == 'train' and cfg['data']['img_augment']:
    #         resize_op = transforms.RandomResizedCrop(
    #             cfg['data']['img_size'], (0.75, 1.), (1., 1.))
    #     else:
    #         resize_op = transforms.Resize((cfg['data']['img_size']))

    #     transform = transforms.Compose([
    #         resize_op, transforms.ToTensor(),
    #     ])

    #     with_camera = cfg['data']['img_with_camera']

    #     if mode == 'train':
    #         random_view = True
    #     else:
    #         random_view = False

    #     inputs_field = data.ImagesField(
    #         cfg['data']['img_folder'], transform,
    #         with_camera=with_camera, random_view=random_view
    #     )
    # elif input_type == 'pointcloud':
    #     transform = transforms.Compose([
    #         data.SubsamplePointcloud(cfg['data']['pointcloud_n']),
    #         data.PointcloudNoise(cfg['data']['pointcloud_noise'])
    #     ])
    #     with_transforms = cfg['data']['with_transforms']
    #     inputs_field = data.PointCloudField(
    #         cfg['data']['pointcloud_file'], transform,
    #         with_transforms=with_transforms
    #     )
    # elif input_type == 'voxels':
    #    inputs_field = data.VoxelsField(
    #        cfg['data']['voxels_file']
    #    )
    # elif input_type == 'idx':
    #     inputs_field = data.IndexField()
    else:
        raise ValueError(
            'Invalid input type (%s)' % input_type)
    return inputs_helper


def get_preprocessor(cfg, dataset=None, device=None):
    ''' Returns preprocessor instance.

    Args:
        cfg (dict): config dictionary
        dataset (dataset): dataset
        device (device): pytorch device
    '''
    p_type = cfg['preprocessor']['type']
    cfg_path = cfg['preprocessor']['config']
    model_file = cfg['preprocessor']['model_file']

    if p_type == 'psgn':
        preprocessor = preprocess.PSGNPreprocessor(
            cfg_path=cfg_path,
            pointcloud_n=cfg['data']['pointcloud_n'],
            dataset=dataset,
            device=device,
            model_file=model_file,
        )
    elif p_type is None:
        preprocessor = None
    else:
        raise ValueError('Invalid Preprocessor %s' % p_type)

    return preprocessor