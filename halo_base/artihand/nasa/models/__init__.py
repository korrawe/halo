from artihand.nasa.models.core import ArticulatedHandNet
from artihand.nasa.models.core_sdf import ArticulatedHandNetSdf

from artihand.nasa.models import decoder

# Encoder latent dictionary
# encoder_latent_dict = {
#     'simple': encoder_latent.Encoder,
# }

# Decoder dictionary
# decoder_dict = {
#     'simple': decoder.Decoder,
#     'cbatchnorm': decoder.DecoderCBatchNorm,
#     'cbatchnorm2': decoder.DecoderCBatchNorm2,
#     'batchnorm': decoder.DecoderBatchNorm,
#     'cbatchnorm_noresnet': decoder.DecoderCBatchNormNoResnet,
# }

decoder_dict = {
    'simple': decoder.SimpleDecoder,
    'piece_rigid': decoder.PiecewiseRigidDecoder,
    'piece_deform': decoder.PiecewiseDeformableDecoder,
    'sdf_simple': decoder.SdfDecoder
}
