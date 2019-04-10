import torch
from torch import nn
import torchvision

from models.captioning_model import CaptioningModelDecoder, update_params

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    DEFAULT_MODEL_PARAMS = {"fine_tune_encoder": False, "joint_embeddings_size": 1024}
    DEFAULT_OPTIMIZER_PARAMS = {"encoder_learning_rate": 1e-4}

    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = update_params(self.DEFAULT_MODEL_PARAMS, params)

        # TODO use resnet 152?
        resnet = torchvision.models.resnet101(pretrained=True)

        # TODO check that this does not disable gradients for new layers
        # Disable calculation of all gradients
        for p in resnet.parameters():
            p.requires_grad = False

        # Enable calculation of some gradients for fine tuning
        self.set_fine_tuning_enabled(self.params["fine_tune_encoder"])

        # Replace average pool with adaptive average pool
        resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Replace the last fully connected layer of the CNN with a new one that outputs into the correct embedding
        # space size
        resnet.fc = nn.Linear(
            resnet.fc.in_features, self.params["joint_embeddings_size"]
        )

        self.model = resnet

    def forward(self, images):
        """
        Forward propagation.

        :param images: input images, shape: (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.model(images)  # output shape: (batch_size, joint_embeddings_size)
        return out

    def set_fine_tuning_enabled(self, enable_fine_tuning):
        """
        Enable or disable the computation of gradients for the convolutional blocks 2-4 of the encoder.

        :param enable_fine_tuning: Set to True to enable fine tuning
        """
        # The convolutional blocks 2-4 are found at position 5-7 in the model
        for c in list(self.model.children())[5:]:
            for p in c.parameters():
                p.requires_grad = enable_fine_tuning
