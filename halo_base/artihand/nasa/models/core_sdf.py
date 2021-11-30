import torch
from torch._C import dtype
import torch.nn as nn
from torch import distributions as dist


class ArticulatedHandNetSdf(nn.Module):
    ''' Occupancy Network class.
    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        encoder_latent (nn.Module): latent encoder network
        p0_z (dist): prior distribution for latent code z
        device (device): torch device
    '''

    def __init__(self, decoder, encoder=None, encoder_latent=None, use_bone_length=False, per_part_output=False,
                 p0_z=None, device=None):
        super().__init__()
        if p0_z is None:
            p0_z = dist.Normal(torch.tensor([]), torch.tensor([]))

        self.decoder = decoder.to(device)
        self.use_bone_length = use_bone_length
        self.use_sdf = True
        
        # If true, return predicted occupancies for each part-model
        self.per_part_output = per_part_output

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        self._device = device
        # self.p0_z = p0_z

    def forward(self, p, inputs, bone_lengths=None, sample=True, **kwargs):
        ''' Performs a forward pass through the network.
        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
        '''

        c = self.encode_inputs(inputs)
        # z = self.get_z_from_prior((batch_size,), sample=sample)
        # p_r = self.decode(p, z, c, **kwargs)
        p_r = self.decode(p, c, bone_lengths=bone_lengths, **kwargs)
        return p_r

    def encode_inputs(self, inputs):
        ''' Encodes the input.
        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            # Return inputs
            # c = torch.empty(inputs.size(0), 0)
            c = inputs

        return c
    
    def decode(self, p, c, bone_lengths=None, reduce_part=True, return_model_indices=False, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            z (tensor): latent code z
            c (tensor): latent conditioned code c
            # joints (tensor): joint locations
            reduce_part (bool): whether to reduce the last (sub-model) dimention for 
                part-base model (with max() or logSumExp()). Only considered if part-base model is used.
                True when training normal occupancy, and False when training skinning weight.
            return_model_indices (bool): only for geration
        '''
        ############### Expand latent code to match the number of points here
        # reshape to [batch x points, latent size]
        # sdf_data = (samples.cuda()).reshape(
                        #     num_samp_per_scene * scene_per_subbatch, 5 # 4
                        # )
        ##### repeat interleave
        batch_size, points_size, p_dim = p.size()
        p = p.reshape(batch_size * points_size, p_dim)
        # print("c shape", c.size())
        c = c.repeat_interleave(points_size, dim=0)
        if bone_lengths is not None:
            bone_lengths = bone_lengths.repeat_interleave(points_size, dim=0)
        # print("c shape", c.size())
        # import pdb; pdb.set_trace()
        # True during testing
        if return_model_indices:
            # If part-labels are needed, get [batch x bones] probabilities from the model and find argmax externally
            p_r = self.decoder(p, c, bone_lengths, reduce_part=False)
            # No sigmoid at the end
            # p_r = self.decoder.sigmoid(p_r)
            # No support for smooth_max for now
            # if self.decoder.smooth_max:
            #     # _, sub_model_indices = p_r.max(1, keepdim=True)
            #     # # p_r = p_r.logsumexp(1, keepdim=True)
            #     # weights = nn.functional.softmax(5.0 * p_r, dim=1)
            #     # p_r = torch.sum(weights * p_r, dim=1)
            #     p_r, sub_model_indices = p_r.min(1, keepdim=True)
            # if not self.decoder.combine_final:
            #     p_r = p_r[:, 0, None]
            # sub_model_indices = torch.zeros_like(p_r).int()

            # Find min among all sub-models
            # p_r, sub_model_indices = p_r.min(1, keepdim=True)
            # import pdb; pdb.set_trace()
            sub_model_indices = torch.zeros_like(p_r, dtype=torch.int)

            # p_r = self.decoder.sigmoid(p_r)
            sub_model_indices = sub_model_indices.reshape(batch_size, points_size)
            return p_r, sub_model_indices

        else:
            p_r = self.decoder(p, c, bone_lengths, reduce_part=reduce_part)
            # No sigmoid at the end
            # Use min instead of max (occupancy and sdf)
            # p_r = self.decoder.sigmoid(p_r)
            # if reduce_part:
                # if self.decoder.smooth_max:
                #     # p_r = p_r.logsumexp(1, keepdim=True)
                #     # weights = nn.functional.softmax(5.0 * p_r, dim=1)
                #     # p_r = torch.sum(weights * p_r, dim=1)
                #     p_r, _ = p_r.min(1, keepdim=True)
                # if not self.decoder.combine_final:
                #     # import pdb; pdb.set_trace()
                #     # p_r, _ = p_r.max(1, keepdim=True)
                #     p_r = p_r[:, 0, None]
                # p_r, _ = p_r.min(1, keepdim=True)
            # import pdb; pdb.set_trace()
            p_r = p_r.reshape(batch_size, points_size)
            # p_r = self.decoder.sigmoid(p_r) # 
        
        return p_r

    def to(self, device):
        ''' Puts the model to the device.
        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model