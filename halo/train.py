
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np
import os
import time

from models import config, data
from models.checkpoints import CheckpointIO


# Arguments
parser = argparse.ArgumentParser(
    description='Train a deep structured implicit function model for hand reconstruction.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of seconds'
                         'with exit code 2.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

# Set t0
t0 = time.time()
ts = t0

# Shorthands
out_dir = cfg['training']['out_dir']
batch_size = cfg['training']['batch_size']
backup_every = cfg['training']['backup_every']
exit_after = args.exit_after

model_selection_metric = cfg['training']['model_selection_metric']
if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1
else:
    raise ValueError('model_selection_mode must be '
                     'either maximize or minimize.')

# Output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Random rotate
cfg['data']['random_rotate'] = False
# Dataset
train_dataset = config.get_dataset('train', cfg)
val_dataset = config.get_dataset('val', cfg)

# for i in train_dataset:
#     import pdb; pdb.set_trace()

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=4, shuffle=True,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=10, num_workers=4, shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

# For visualizations
# vis_loader = torch.utils.data.DataLoader(
#     val_dataset, batch_size=12, shuffle=True,
#     collate_fn=data.collate_remove_none,
#     worker_init_fn=data.worker_init_fn)
# data_vis = next(iter(vis_loader))

# Model
model = config.get_model(cfg, device=device, dataset=train_dataset)

# Intialize training
npoints = 1000
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
trainer = config.get_trainer(model, optimizer, cfg, device=device)

# Load pre-trained model if existing
kwargs = {
    'model': model,
    'optimizer': optimizer,
}
checkpoint_io = CheckpointIO(
    out_dir, initialize_from=cfg['model']['initialize_from'],
    initialization_file_name=cfg['model']['initialization_file_name'],
    **kwargs)
# checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)
try:
    load_dict = checkpoint_io.load('model.pt')
except FileExistsError:
    load_dict = dict()
epoch_it = load_dict.get('epoch_it', -1)
it = load_dict.get('it', -1)
metric_val_best = load_dict.get(
    'loss_val_best', -model_selection_sign * np.inf)

# metric_val_best = np.inf

if metric_val_best == np.inf or metric_val_best == -np.inf:
    metric_val_best = -model_selection_sign * np.inf

print('Current best validation metric (%s): %.8f'
      % (model_selection_metric, metric_val_best))

# Load HALO mesh model if needed
if cfg['model']['use_mesh_model']:
    model.initialize_halo(cfg['model']['halo_config_file'])

logger = SummaryWriter(os.path.join(out_dir, 'logs'))

# Shorthands
print_every = cfg['training']['print_every']
checkpoint_every = cfg['training']['checkpoint_every']
validate_every = cfg['training']['validate_every']
visualize_every = cfg['training']['visualize_every']
encoder_freeze_after = cfg['training']['encoder_freeze_after']

# Print model
nparameters = sum(p.numel() for p in model.parameters())
print(model)
print('Total number of parameters: %d' % nparameters)

# Freeze object encoder
if epoch_it >= encoder_freeze_after:
    for param in model.obj_encoder.parameters():
        param.requires_grad = False

print_every = 5
while True:
    epoch_it += 1
    if epoch_it == encoder_freeze_after:
        for param in model.obj_encoder.parameters():
            param.requires_grad = False
    # scheduler.step()

    for batch in train_loader:
        it += 1
        with torch.autograd.set_detect_anomaly(True):
            loss_dict = trainer.train_step(batch, epoch_it)
        loss = loss_dict['total']
        for k, v in loss_dict.items():
            logger.add_scalar('train/loss/%s' % k, v, it)
        # logger.add_scalar('train/loss', loss, it)

        # Print output
        if print_every > 0 and (it % print_every) == 0:
            print('[Epoch %02d] it=%03d, loss=%.4f'
                  % (epoch_it, it, loss))
            print('time per batch: %.2f, total time: %.2f' 
                  % (time.time() - ts, time.time() - t0))
            ts = time.time()
            out_str = ''
            for k, v in loss_dict.items():
                out_str += '%s: %.5f, ' % (k, v)
            print(out_str)  # loss_dict

        # visualize_every = 2000
        # Visualize output
        if visualize_every > 0 and (it % visualize_every) == 0:
            print('Visualizing')
            trainer.visualize(batch, epoch_it)  # (data_vis)

        # Save checkpoint
        if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
            print('Saving checkpoint')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
            # sys.exit() ##################

        # Backup if necessary
        if (backup_every > 0 and (it % backup_every) == 0):
            print('Backup checkpoint')
            checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)

        # Run validation
        if validate_every > 0 and (it % validate_every) == 0:
            eval_dict = trainer.evaluate(val_loader)
            metric_val = eval_dict[model_selection_metric]
            print('Validation metric (%s): %.4f'
                  % (model_selection_metric, metric_val))

            for k, v in eval_dict.items():
                logger.add_scalar('val/%s' % k, v, it)

            if model_selection_sign * (metric_val - metric_val_best) > 0:
                metric_val_best = metric_val
                print('New best model (loss %.4f)' % metric_val_best)
                checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)

        # Exit if necessary
        if exit_after > 0 and (time.time() - t0) >= exit_after:
            print('Time limit reached. Exiting.')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
            exit(3)
