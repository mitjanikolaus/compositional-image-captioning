import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from utils import TOKEN_START, decode_caption, TOKEN_END

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CaptioningModelDecoder(nn.Module):
    DEFAULT_MODEL_PARAMS = {}

    def __init__(self, word_map, params, pretrained_embeddings=None):
        super(CaptioningModelDecoder, self).__init__()
        self.params = update_params(self.DEFAULT_MODEL_PARAMS, params)

        self.vocab_size = len(word_map)
        self.word_map = word_map

        self.word_embedding = nn.Embedding(
            self.vocab_size, self.params["word_embeddings_size"]
        )

        if pretrained_embeddings is not None:
            self.word_embedding.weight = nn.Parameter(pretrained_embeddings)

        self.set_fine_tune_embeddings(self.params["fine_tune_decoder_word_embeddings"])

        self.loss_function = nn.CrossEntropyLoss().to(device)

    def set_fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of the embedding layer.

        :param fine_tune: Set to True to allow fine tuning
        """
        for p in self.word_embedding.parameters():
            p.requires_grad = fine_tune

    def update_previous_word(self, scores, target_words, t):
        if self.training:
            if random.random() < self.params["teacher_forcing_ratio"]:
                use_teacher_forcing = True
            else:
                use_teacher_forcing = False
        else:
            use_teacher_forcing = False

        if use_teacher_forcing:
            next_words = target_words[:, t + 1]
        else:
            next_words = torch.argmax(scores, dim=1)

        return next_words

    def forward(self, encoder_output, target_captions=None, decode_lengths=None):
        """
        Forward propagation.

        :param encoder_output: output features of the encoder
        :param target_captions: encoded target captions, shape: (batch_size, max_caption_length)
        :param decode_lengths: caption lengths, shape: (batch_size, 1)
        :return: scores for vocabulary, decode lengths, weights
        """

        batch_size = encoder_output.size(0)

        # Flatten image
        encoder_output = encoder_output.view(batch_size, -1, encoder_output.size(-1))

        if not self.training:
            decode_lengths = torch.full(
                (batch_size,),
                self.params["max_caption_len"],
                dtype=torch.int64,
                device=device,
            )

        # Initialize LSTM state
        states = self.init_hidden_states(encoder_output)

        # Tensors to hold word prediction scores and alphas
        scores = torch.zeros(
            (batch_size, max(decode_lengths), self.vocab_size), device=device
        )
        alphas = torch.zeros(
            batch_size, max(decode_lengths), encoder_output.size(1), device=device
        )

        # At the start, all 'previous words' are the <start> token
        prev_words = torch.full(
            (batch_size,), self.word_map[TOKEN_START], dtype=torch.int64, device=device
        )

        for t in range(max(decode_lengths)):
            if not self.training:
                # Find all sequences where an <end> token has been produced in the last timestep
                ind_end_token = (
                    torch.nonzero(prev_words == self.word_map[TOKEN_END])
                    .view(-1)
                    .tolist()
                )

                # Update the decode lengths accordingly
                decode_lengths[ind_end_token] = torch.min(
                    decode_lengths[ind_end_token],
                    torch.full_like(decode_lengths[ind_end_token], t, device=device),
                )

            # Check if all sequences are finished:
            indices_incomplete_sequences = torch.nonzero(decode_lengths > t).view(-1)
            if len(indices_incomplete_sequences) == 0:
                break

            prev_words_embedded = self.word_embedding(prev_words)
            scores_for_timestep, states, alphas_for_timestep = self.forward_step(
                encoder_output, prev_words_embedded, states
            )

            # Update the previously predicted words
            prev_words = self.update_previous_word(
                scores_for_timestep, target_captions, t
            )

            scores[indices_incomplete_sequences, t, :] = scores_for_timestep[
                indices_incomplete_sequences
            ]
            if alphas_for_timestep is not None:
                alphas[indices_incomplete_sequences, t, :] = alphas_for_timestep[
                    indices_incomplete_sequences
                ]

        return scores, decode_lengths, alphas

    def loss_cross_entropy(self, scores, target_captions, decode_lengths):
        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        target_captions = target_captions[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        decode_lengths, sort_ind = decode_lengths.sort(dim=0, descending=True)
        packed_scores, _ = pack_padded_sequence(
            scores[sort_ind], decode_lengths, batch_first=True
        )
        packed_targets, _ = pack_padded_sequence(
            target_captions[sort_ind], decode_lengths, batch_first=True
        )

        return self.loss_function(packed_scores, packed_targets)

    def beam_search(
        self,
        encoder_output,
        beam_size=1,
        store_alphas=False,
        store_beam=False,
        print_beam=False,
    ):
        """Generate and return the top k sequences using beam search."""

        current_beam_width = beam_size

        enc_image_size = encoder_output.size(1)
        encoder_dim = encoder_output.size()[-1]

        # Flatten encoding
        encoder_output = encoder_output.view(1, -1, encoder_dim)

        # We'll treat the problem as having a batch size of k
        encoder_output = encoder_output.expand(
            beam_size, encoder_output.size(1), encoder_dim
        )

        # Tensor to store top k sequences; now they're just <start>
        top_k_sequences = torch.full(
            (beam_size, 1), self.word_map[TOKEN_START], dtype=torch.int64, device=device
        )

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(beam_size, device=device)

        if store_alphas:
            # Tensor to store top k sequences' alphas; now they're just 1s
            seqs_alpha = torch.ones(beam_size, 1, enc_image_size, enc_image_size).to(
                device
            )

        # Lists to store completed sequences, scores, and alphas and the full decoding beam
        complete_seqs = []
        complete_seqs_alpha = []
        complete_seqs_scores = []
        beam = []

        # Initialize hidden states
        states = self.init_hidden_states(encoder_output)

        # Start decoding
        for step in range(0, self.params["max_caption_len"] - 1):
            prev_words = top_k_sequences[:, step]

            prev_word_embeddings = self.word_embedding(prev_words)
            predictions, states, alpha = self.forward_step(
                encoder_output, prev_word_embeddings, states
            )
            scores = F.log_softmax(predictions, dim=1)

            # Add the new scores
            scores = top_k_scores.unsqueeze(1).expand_as(scores) + scores

            # For the first timestep, the scores from previous decoding are all the same, so in order to create 5
            # different sequences, we should only look at one branch
            if step == 0:
                scores = scores[0]

            # Find the top k of the flattened scores
            top_k_scores, top_k_words = scores.view(-1).topk(
                current_beam_width, 0, largest=True, sorted=True
            )

            # Convert flattened indices to actual indices of scores
            prev_seq_inds = top_k_words / self.vocab_size  # (k)
            next_words = top_k_words % self.vocab_size  # (k)

            # Add new words to sequences
            top_k_sequences = torch.cat(
                (top_k_sequences[prev_seq_inds], next_words.unsqueeze(1)), dim=1
            )

            if print_beam:
                print_current_beam(top_k_sequences, top_k_scores, self.word_map)
            if store_beam:
                beam.append(top_k_sequences)

            # Store the new alphas
            if store_alphas:
                alpha = alpha.view(-1, enc_image_size, enc_image_size)
                seqs_alpha = torch.cat(
                    (seqs_alpha[prev_seq_inds], alpha[prev_seq_inds].unsqueeze(1)),
                    dim=1,
                )

            # Check for complete and incomplete sequences (based on the <end> token)
            incomplete_inds = (
                torch.nonzero(next_words != self.word_map[TOKEN_END]).view(-1).tolist()
            )
            complete_inds = (
                torch.nonzero(next_words == self.word_map[TOKEN_END]).view(-1).tolist()
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
            for i in range(len(states)):
                states[i] = states[i][prev_seq_inds[incomplete_inds]]
            encoder_output = encoder_output[prev_seq_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds]
            if store_alphas:
                seqs_alpha = seqs_alpha[incomplete_inds]

        if len(complete_seqs) < beam_size:
            complete_seqs.extend(top_k_sequences[incomplete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[incomplete_inds])
            if store_alphas:
                complete_seqs_alpha.extend(seqs_alpha[incomplete_inds])

        sorted_sequences = [
            sequence
            for _, sequence in sorted(
                zip(complete_seqs_scores, complete_seqs), reverse=True
            )
        ]
        sorted_alphas = None
        if store_alphas:
            sorted_alphas = [
                alpha
                for _, alpha in sorted(
                    zip(complete_seqs_scores, complete_seqs_alpha), reverse=True
                )
            ]
        return sorted_sequences, sorted_alphas, beam


def create_encoder_optimizer(encoder, params):
    optimizer_params = update_params(encoder.DEFAULT_OPTIMIZER_PARAMS, params)
    optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, encoder.parameters()),
        lr=optimizer_params["encoder_learning_rate"],
    )
    return optimizer


def create_decoder_optimizer(decoder, params):
    optimizer_params = update_params(decoder.DEFAULT_OPTIMIZER_PARAMS, params)
    optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, decoder.parameters()),
        lr=optimizer_params["decoder_learning_rate"],
    )
    return optimizer


def update_params(defaults, params):
    updated = defaults.copy()
    for key, value in defaults.items():
        if key in params and params[key] is not None:
            updated[key] = params[key]
    return updated


def print_current_beam(top_k_sequences, top_k_scores, word_map):
    print("\n")
    for sequence, score in zip(top_k_sequences, top_k_scores):
        print(
            "{} \t\t\t\t Score: {}".format(
                decode_caption(sequence.cpu().numpy(), word_map), score
            )
        )
