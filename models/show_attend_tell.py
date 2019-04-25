import torch
from torch import nn
import torchvision

from models.captioning_model import CaptioningModelDecoder, update_params

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    DEFAULT_MODEL_PARAMS = {"encoded_image_size": 14, "fine_tune_encoder": False}
    DEFAULT_OPTIMIZER_PARAMS = {"encoder_learning_rate": 1e-4}

    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = update_params(self.DEFAULT_MODEL_PARAMS, params)

        resnet = torchvision.models.resnet101(pretrained=True)

        # Remove linear and pool layers, these are only used for classification
        modules = list(resnet.children())[:-2]
        self.model = nn.Sequential(*modules)

        # Resize input image to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (self.params["encoded_image_size"], self.params["encoded_image_size"])
        )

        # Disable calculation of all gradients
        for p in self.model.parameters():
            p.requires_grad = False

        # Enable calculation of some gradients for fine tuning
        self.set_fine_tuning_enabled(self.params["fine_tune_encoder"])

    def forward(self, images):
        """
        Forward propagation.

        :param images: input images, shape: (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.model(
            images
        )  # output shape: (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(
            out
        )  # output shape: (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(
            0, 2, 3, 1
        )  # output shape: (batch_size, encoded_image_size, encoded_image_size, 2048)
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


class SATDecoder(CaptioningModelDecoder):
    DEFAULT_MODEL_PARAMS = {
        "word_embeddings_size": 512,
        "attention_dim": 512,
        "encoder_dim": 2048,
        "decoder_dim": 512,
        "teacher_forcing_ratio": 1,
        "dropout_ratio": 0.5,
        "alpha_c": 1.0,
        "max_caption_len": 100,
        "fine_tune_decoder_word_embeddings": True,
    }
    DEFAULT_OPTIMIZER_PARAMS = {"decoder_learning_rate": 4e-4}

    def __init__(self, word_map, params, pretrained_embeddings=None):
        super(SATDecoder, self).__init__(word_map, params, pretrained_embeddings)

        self.attention = AttentionModule(
            self.params["encoder_dim"],
            self.params["decoder_dim"],
            self.params["attention_dim"],
        )

        # Linear layers to find initial states of LSTMs
        self.init_h = nn.Linear(self.params["encoder_dim"], self.params["decoder_dim"])
        self.init_c = nn.Linear(self.params["encoder_dim"], self.params["decoder_dim"])

        # Gating scalars and sigmoid layer (cf. section 4.2.1 of the paper)
        self.f_beta = nn.Linear(self.params["decoder_dim"], self.params["encoder_dim"])
        self.sigmoid = nn.Sigmoid()

        # LSTM
        self.decode_step = nn.LSTMCell(
            self.params["word_embeddings_size"] + self.params["encoder_dim"],
            self.params["decoder_dim"],
            bias=True,
        )

        # Dropout layer
        self.dropout = nn.Dropout(p=self.params["dropout_ratio"])

        # Linear layers for output generation
        self.linear_o = nn.Linear(self.params["word_embeddings_size"], self.vocab_size)
        self.linear_h = nn.Linear(
            self.params["decoder_dim"], self.params["word_embeddings_size"]
        )
        self.linear_z = nn.Linear(
            self.params["encoder_dim"], self.params["word_embeddings_size"]
        )

    def init_hidden_states(self, encoder_out):
        """
        Create the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, shape: (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)

        states = [h, c]
        return states

    def forward_step(self, encoder_output, prev_word_embeddings, states):
        """Perform a single decoding step."""
        decoder_hidden_state, decoder_cell_state = states

        attention_weighted_encoding, alpha = self.attention(
            encoder_output, decoder_hidden_state
        )
        gating_scalars = self.sigmoid(self.f_beta(decoder_hidden_state))
        attention_weighted_encoding = gating_scalars * attention_weighted_encoding

        decoder_input = torch.cat(
            (prev_word_embeddings, attention_weighted_encoding), dim=1
        )
        decoder_hidden_state, decoder_cell_state = self.decode_step(
            decoder_input, (decoder_hidden_state, decoder_cell_state)
        )

        decoder_hidden_state_embedded = self.linear_h(decoder_hidden_state)
        attention_weighted_encoding_embedded = self.linear_z(
            attention_weighted_encoding
        )
        scores = self.linear_o(
            self.dropout(
                prev_word_embeddings
                + decoder_hidden_state_embedded
                + attention_weighted_encoding_embedded
            )
        )

        states = [decoder_hidden_state, decoder_cell_state]
        return scores, states, alpha

    def loss(self, scores, target_captions, decode_lengths, alphas):
        loss = self.loss_cross_entropy(scores, target_captions, decode_lengths)

        # Add doubly stochastic attention regularization
        loss += self.params["alpha_c"] * ((1.0 - alphas.sum(dim=1)) ** 2).mean()
        return loss


class AttentionModule(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(AttentionModule, self).__init__()

        # Linear layer to transform encoded image
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)

        # Linear layer to transform decoder's output
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)

        # Linear layer to calculate values to be softmax-ed
        self.full_att = nn.Linear(attention_dim, 1)

        # ReLU layer
        self.relu = nn.ReLU()

        # Softmax layer to calculate attention weights
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, shape: (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, shape: (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(
            encoder_out
        )  # output shape: (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(
            decoder_hidden
        )  # output shape: (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(
            2
        )  # output shape: (batch_size, num_pixels)
        alpha = self.softmax(att)  # output shape: (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(
            dim=1
        )  # output shape: (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha
