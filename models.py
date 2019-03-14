import torch
from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet101(pretrained=True)

        # Remove linear and pool layers, these are only used for classification
        modules = list(resnet.children())[:-2]
        self.model = nn.Sequential(*modules)

        # Resize input image to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size)
        )

        # Disable calculation of all gradients
        for p in self.model.parameters():
            p.requires_grad = False

        # Enable calculation of some gradients for fine tuning
        self.set_fine_tuning_enabled()

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

    def set_fine_tuning_enabled(self, enable_fine_tuning=True):
        """
        Enable or disable the computation of gradients for the convolutional blocks 2-4 of the encoder.

        :param enable_fine_tuning: Set to True to enable fine tuning
        """
        # The convolutional blocks 2-4 are found at position 5-7 in the model
        for c in list(self.model.children())[5:]:
            for p in c.parameters():
                p.requires_grad = enable_fine_tuning


class AttentionModule(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(AttentionModule, self).__init__()
        self.encoder_att = nn.Linear(
            encoder_dim, attention_dim
        )  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(
            decoder_dim, attention_dim
        )  # linear layer to transform decoder's output
        self.full_att = nn.Linear(
            attention_dim, 1
        )  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

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


class DecoderWithAttention(nn.Module):
    def __init__(
        self,
        attention_dim,
        embed_dim,
        decoder_dim,
        vocab_size,
        start_token,
        end_token,
        padding_token,
        max_caption_len,
        encoder_dim=2048,
        dropout=0.5,
    ):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param start_token: word map value of the sentence <start> token
        :param end_token: word map value of the sentence <end> token
        :param padding_token: word map value of the sentence <pad> token
        :param max_caption_len: maximum caption length (for decoding in evaluation mode)
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout rate
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.start_token = start_token
        self.end_token = end_token
        self.padding_token = padding_token
        self.max_caption_len = max_caption_len
        self.dropout = dropout

        self.attention = AttentionModule(encoder_dim, decoder_dim, attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(
            embed_dim + encoder_dim, decoder_dim, bias=True
        )  # decoding LSTMCell
        self.init_h = nn.Linear(
            encoder_dim, decoder_dim
        )  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(
            encoder_dim, decoder_dim
        )  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(
            decoder_dim, encoder_dim
        )  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(
            decoder_dim, vocab_size
        )  # linear layer to find scores over vocabulary

        self.init_weights()

    def init_weights(self):
        """
        Initialize some parameters with values from a uniform distribution
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Load an embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of the embedding layer.

        :param fine_tune: Set to True to allow fine tuning
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Create the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, shape: (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward_step(
        self,
        encoder_out,
        decoder_hidden_state,
        decoder_cell_state,
        prev_word_embeddings,
    ):
        """Perform a single decoding step."""

        attention_weighted_encoding, alpha = self.attention(
            encoder_out, decoder_hidden_state
        )
        gate = self.sigmoid(
            self.f_beta(decoder_hidden_state)
        )  # gating scalar, (batch_size_t, encoder_dim)
        attention_weighted_encoding = gate * attention_weighted_encoding

        decoder_input = torch.cat(
            (prev_word_embeddings, attention_weighted_encoding), dim=1
        )
        decoder_hidden_state, decoder_cell_state = self.decode_step(
            decoder_input, (decoder_hidden_state, decoder_cell_state)
        )  # (batch_size, decoder_dim)

        scores = self.fc(self.dropout(decoder_hidden_state))  # (batch_size, vocab_size)

        return scores, alpha, decoder_hidden_state, decoder_cell_state

    def forward(self, encoder_out, target_captions, decode_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, shape: (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param target_captions: encoded target captions, shape: (batch_size, max_caption_length)
        :param decode_lengths: caption lengths, shape: (batch_size, 1)
        :return: scores for vocabulary, decode lengths, weights
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)

        if self.training:
            # Embed the target captions (output shape: (batch_size, max_caption_length, embed_dim))
            embedded_target_captions = self.embedding(target_captions)

        else:
            decode_lengths = torch.full(
                (batch_size,), self.max_caption_len, dtype=torch.int64, device=device
            )

        # Initialize LSTM state (output shape: (batch_size, decoder_dim))
        decoder_hidden_state, decoder_cell_state = self.init_hidden_state(encoder_out)

        # Tensors to hold word prediction scores and alphas
        scores = torch.zeros(
            (batch_size, max(decode_lengths), vocab_size), device=device
        )
        alphas = torch.zeros(
            batch_size, max(decode_lengths), encoder_out.size(1), device=device
        )

        # At the start, all 'previous words' are the <start> token
        prev_predicted_words = torch.full(
            (batch_size,), self.start_token, dtype=torch.int64, device=device
        )

        for t in range(max(decode_lengths)):
            if self.training:
                prev_word_embeddings = embedded_target_captions[:, t, :]
            else:
                # Find all sequences where an <end> token has been produced in the last timestep
                ind_end_token = (
                    torch.nonzero(prev_predicted_words == self.end_token)
                    .view(-1)
                    .tolist()
                )

                # Update the decode lengths accordingly
                decode_lengths[ind_end_token] = torch.min(
                    decode_lengths[ind_end_token],
                    torch.full_like(decode_lengths[ind_end_token], t, device=device),
                )

                prev_word_embeddings = self.embedding(prev_predicted_words)

            # Check if all sequences are finished:
            indices_incomplete_sequences = torch.nonzero(decode_lengths > t).view(-1)
            if len(indices_incomplete_sequences) == 0:
                break

            scores_for_timestep, alphas_for_timestep, decoder_hidden_state, decoder_cell_state = self.forward_step(
                encoder_out,
                decoder_hidden_state,
                decoder_cell_state,
                prev_word_embeddings,
            )

            # Update the previously predicted words
            prev_predicted_words = torch.max(scores_for_timestep, dim=1)[1]

            scores[indices_incomplete_sequences, t, :] = scores_for_timestep[
                indices_incomplete_sequences
            ]
            alphas[indices_incomplete_sequences, t, :] = alphas_for_timestep[
                indices_incomplete_sequences
            ]

        return scores, alphas, decode_lengths
