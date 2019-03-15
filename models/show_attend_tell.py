import torch
from torch import nn
import torchvision
import torch.nn.functional as F

from utils import TOKEN_START, TOKEN_END, decode_caption

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_current_beam(top_k_sequences, top_k_scores, word_map):
    print("\n")
    for sequence, score in zip(top_k_sequences, top_k_scores):
        print(
            "{} \t\t\t\t Score: {}".format(
                decode_caption(sequence.numpy(), word_map), score
            )
        )


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


class SATDecoder(nn.Module):
    def __init__(
        self,
        attention_dim,
        embed_dim,
        decoder_dim,
        word_map,
        max_caption_len,
        encoder_dim=2048,
        dropout=0.5,
    ):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param word_map: vocabulary word map
        :param max_caption_len: maximum caption length (for decoding in evaluation mode)
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout rate
        """
        super(SATDecoder, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = len(word_map)
        self.word_map = word_map
        self.max_caption_len = max_caption_len
        self.dropout = dropout

        self.attention = AttentionModule(encoder_dim, decoder_dim, attention_dim)

        self.embedding = nn.Embedding(self.vocab_size, embed_dim)
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
            decoder_dim, self.vocab_size
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

    def forward(self, encoder_out, target_captions=None, decode_lengths=None):
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
            (batch_size,), self.word_map[TOKEN_START], dtype=torch.int64, device=device
        )

        for t in range(max(decode_lengths)):
            if self.training:
                prev_word_embeddings = embedded_target_captions[:, t, :]
            else:
                # Find all sequences where an <end> token has been produced in the last timestep
                ind_end_token = (
                    torch.nonzero(prev_predicted_words == self.word_map[TOKEN_END])
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

    def beam_search(
        self,
        encoder_out,
        decoder,
        word_map,
        beam_size=1,
        max_caption_len=50,
        store_alphas=False,
        print_beam=False,
    ):
        """Generate and return the top k sequences using beam search."""

        current_beam_width = beam_size

        # Encode
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(
            1, -1, encoder_dim
        )  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(
            beam_size, num_pixels, encoder_dim
        )  # (k, num_pixels, encoder_dim)

        # Tensor to store top k sequences; now they're just <start>
        top_k_sequences = torch.full(
            (beam_size, 1), word_map[TOKEN_START], dtype=torch.int64, device=device
        )

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(beam_size).to(device)  # (k)

        if store_alphas:
            # Tensor to store top k sequences' alphas; now they're just 1s
            seqs_alpha = torch.ones(beam_size, 1, enc_image_size, enc_image_size).to(
                device
            )  # (k, 1, enc_image_size, enc_image_size)

        # Lists to store completed sequences, scores, and alphas
        complete_seqs = []
        complete_seqs_alpha = []
        complete_seqs_scores = []

        # Start decoding
        decoder_hidden_state, decoder_cell_state = decoder.init_hidden_state(
            encoder_out
        )

        for step in range(0, max_caption_len - 1):
            embeddings = decoder.embedding(top_k_sequences[:, step]).squeeze(
                1
            )  # (k, embed_dim)

            predictions, alpha, decoder_hidden_state, decoder_cell_state = decoder.forward_step(
                encoder_out, decoder_hidden_state, decoder_cell_state, embeddings
            )
            scores = F.log_softmax(predictions, dim=1)

            # Add the new scores
            scores = (
                top_k_scores.unsqueeze(1).expand_as(scores) + scores
            )  # (k, vocab_size)

            # For the first timestep, the scores from previous decoding are all the same, so in order to create 5 different
            # sequences, we should only look at one branch
            if step == 0:
                scores = scores[0]

            # Find the top k of the flattened scores
            top_k_scores, top_k_words = scores.view(-1).topk(
                current_beam_width, 0, largest=True, sorted=True
            )  # (k)

            # Convert flattened indices to actual indices of scores
            prev_seq_inds = top_k_words / self.vocab_size  # (k)
            next_words = top_k_words % self.vocab_size  # (k)

            # Add new words to sequences
            top_k_sequences = torch.cat(
                (top_k_sequences[prev_seq_inds], next_words.unsqueeze(1)), dim=1
            )  # (k, step+2)

            if print_beam:
                print_current_beam(top_k_sequences, top_k_scores, word_map)

            # Store the new alphas
            if store_alphas:
                alpha = alpha.view(
                    -1, enc_image_size, enc_image_size
                )  # (k, enc_image_size, enc_image_size)
                seqs_alpha = torch.cat(
                    (seqs_alpha[prev_seq_inds], alpha[prev_seq_inds].unsqueeze(1)),
                    dim=1,
                )  # (k, step+2, enc_image_size, enc_image_size)

            # Check for complete and incomplete sequences (based on the <end> token)
            incomplete_inds = (
                torch.nonzero(next_words != word_map[TOKEN_END]).view(-1).tolist()
            )
            complete_inds = (
                torch.nonzero(next_words == word_map[TOKEN_END]).view(-1).tolist()
            )

            # Set aside complete sequences and reduce beam size accordingly
            if len(complete_inds) > 0:
                complete_seqs.extend(top_k_sequences[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
                if store_alphas:
                    complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())

            # Stop if k captions have been completely generated
            current_beam_width = len(incomplete_inds)
            if current_beam_width == 0:
                break

            # Proceed with incomplete sequences
            top_k_sequences = top_k_sequences[incomplete_inds]
            decoder_hidden_state = decoder_hidden_state[prev_seq_inds[incomplete_inds]]
            decoder_cell_state = decoder_cell_state[prev_seq_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_seq_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds]
            if store_alphas:
                seqs_alpha = seqs_alpha[incomplete_inds]

        sorted_sequences = [
            sequence
            for _, sequence in sorted(
                zip(complete_seqs_scores, complete_seqs), reverse=True
            )
        ]
        if not store_alphas:
            return sorted_sequences
        else:
            sorted_alphas = [
                alpha
                for _, alpha in sorted(
                    zip(complete_seqs_scores, complete_seqs_alpha), reverse=True
                )
            ]
            return sorted_sequences, sorted_alphas
