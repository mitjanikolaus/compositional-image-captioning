import random

import torch
from torch import nn
import torch.nn.functional as F


from utils import TOKEN_START, TOKEN_END, decode_caption, update_params

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_MODEL_PARAMS = {
    "teacher_forcing_ratio": 1,
    "image_features_size": 2048,
    "embeddings_size": 1000,
    "attention_lstm_size": 1000,
    "attention_layer_size": 512,
    "language_lstm_size": 1000,
    "max_caption_len": 50,
}

DEFAULT_OPTIMIZER_PARAMS = {"decoder_learning_rate": 1e-4}


def one_hot(num_classes, index):
    return torch.eye(num_classes, device=device, dtype=torch.long)[index]


def create_top_down_decoder_optimizer(decoder, params):
    optimizer_params = update_params(DEFAULT_OPTIMIZER_PARAMS, params)
    optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, decoder.parameters()),
        lr=optimizer_params["decoder_learning_rate"],
    )
    return optimizer


class TopDownDecoder(nn.Module):
    def __init__(self, word_map, params):
        super(TopDownDecoder, self).__init__()
        self.params = update_params(DEFAULT_MODEL_PARAMS, params)
        self.vocab_size = len(word_map)
        self.word_map = word_map

        self.embed_word = nn.Embedding(self.vocab_size, self.params["embeddings_size"])
        self.attention_lstm = AttentionLSTM(
            self.params["embeddings_size"],
            self.params["language_lstm_size"],
            self.params["image_features_size"],
            self.params["attention_lstm_size"],
        )
        self.language_lstm = LanguageLSTM(
            self.params["attention_lstm_size"],
            self.params["image_features_size"],
            self.params["language_lstm_size"],
        )
        self.attention = VisualAttention(
            self.params["image_features_size"],
            self.params["attention_lstm_size"],
            self.params["attention_layer_size"],
        )
        self.predict_word = PredictWord(
            self.params["language_lstm_size"], self.vocab_size
        )

        # linear layers to find initial states of LSTMs
        self.init_h1 = nn.Linear(
            self.params["image_features_size"],
            self.attention_lstm.lstm_cell.hidden_size,
        )
        self.init_c1 = nn.Linear(
            self.params["image_features_size"],
            self.attention_lstm.lstm_cell.hidden_size,
        )
        self.init_h2 = nn.Linear(
            self.params["image_features_size"], self.language_lstm.lstm_cell.hidden_size
        )
        self.init_c2 = nn.Linear(
            self.params["image_features_size"], self.language_lstm.lstm_cell.hidden_size
        )

    def init_inference(self, batch_size, v_mean):
        start_word = torch.full(
            (batch_size,), self.word_map[TOKEN_START], device=device, dtype=torch.long
        )

        h1 = self.init_h1(v_mean)
        c1 = self.init_c1(v_mean)
        h2 = self.init_h2(v_mean)
        c2 = self.init_c2(v_mean)
        state = [h1, c1, h2, c2]

        return state, start_word

    def forward(self, image_features, target_sequences=None, decode_lengths=None):
        batch_size = image_features.size(0)

        if not self.training:
            decode_lengths = torch.full(
                (batch_size,),
                self.params["max_caption_len"],
                dtype=torch.int64,
                device=device,
            )

        v_mean = image_features.mean(dim=1)

        state, prev_words = self.init_inference(batch_size, v_mean)
        scores = torch.zeros(
            (batch_size, max(decode_lengths), self.vocab_size), device=device
        )

        for t in range(max(decode_lengths)):
            # Find all sequences where an <end> token has been produced in the last timestep
            ind_end_token = (
                torch.nonzero(prev_words == self.word_map[TOKEN_END]).view(-1).tolist()
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

            scores_for_timestep, state = self.forward_step(
                state, prev_words, v_mean, image_features
            )

            prev_words = self.update_previous_word(
                scores_for_timestep, target_sequences, t
            )

            scores[indices_incomplete_sequences, t, :] = scores_for_timestep[
                indices_incomplete_sequences
            ]

        # TODO return alphas
        return scores, None, decode_lengths

    def forward_step(self, state, prev_words, v_mean, image_feats):
        h1, c1, h2, c2 = state
        prev_words_embedded = self.embed_word(prev_words)
        h1, c1 = self.attention_lstm(h1, c1, h2, v_mean, prev_words_embedded)
        v_hat = self.attention(image_feats, h1)
        h2, c2 = self.language_lstm(h2, c2, h1, v_hat)
        scores = self.predict_word(h2)
        state = [h1, c1, h2, c2]
        return scores, state

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

    def beam_search(
        self,
        image_features,
        beam_size,
        max_caption_len=50,
        store_alphas=False,
        print_beam=False,
    ):
        """Generate and return the top k sequences using beam search."""

        if store_alphas:
            raise NotImplementedError(
                "Storage of alphas for this model is not supported"
            )

        current_beam_width = beam_size

        # Encode
        encoder_dim = image_features.size(2)

        # We'll treat the problem as having a batch size of k
        image_features = image_features.expand(
            beam_size, image_features.size(1), encoder_dim
        )  # (k, num_pixels, encoder_dim)

        # Tensor to store top k sequences; now they're just <start>
        top_k_sequences = torch.full(
            (beam_size, 1), self.word_map[TOKEN_START], dtype=torch.int64, device=device
        )

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(beam_size).to(device)  # (k)

        # Lists to store completed sequences, scores, and alphas
        complete_seqs = []
        complete_seqs_scores = []

        # Start decoding
        v_mean = image_features.mean(dim=1)
        states, prev_words = self.init_inference(beam_size, v_mean)

        for step in range(0, max_caption_len - 1):
            prev_words = top_k_sequences[:, step]
            scores, states = self.forward_step(
                states, prev_words, v_mean, image_features
            )
            scores = F.log_softmax(scores, dim=1)

            # Add the new scores
            scores = (
                top_k_scores.unsqueeze(1).expand_as(scores) + scores
            )  # (k, vocab_size)

            # For the first timestep, the scores from previous decoding are all the same, so in order to create 5
            # different sequences, we should only look at one branch
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
                print_current_beam(top_k_sequences, top_k_scores, self.word_map)

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

            # Stop if k captions have been completely generated
            current_beam_width = len(incomplete_inds)
            if current_beam_width == 0:
                break

            # Proceed with incomplete sequences
            top_k_sequences = top_k_sequences[incomplete_inds]
            for i in range(len(states)):
                states[i] = states[i][prev_seq_inds[incomplete_inds]]
            v_mean = v_mean[prev_seq_inds[incomplete_inds]]
            image_features = image_features[prev_seq_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds]

        if len(complete_seqs) < beam_size:
            complete_seqs.extend(top_k_sequences[incomplete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[incomplete_inds])

        sorted_sequences = [
            sequence
            for _, sequence in sorted(
                zip(complete_seqs_scores, complete_seqs), reverse=True
            )
        ]
        return sorted_sequences


class AttentionLSTM(nn.Module):
    def __init__(self, dim_word_emb, dim_lang_lstm, dim_image_feats, hidden_size):
        super(AttentionLSTM, self).__init__()
        self.lstm_cell = nn.LSTMCell(
            dim_lang_lstm + dim_image_feats + dim_word_emb, hidden_size, bias=True
        )

    def forward(self, h1, c1, h2, v_mean, prev_words_embedded):
        input_features = torch.cat((h2, v_mean, prev_words_embedded), dim=1)
        h_out, c_out = self.lstm_cell(input_features, (h1, c1))
        return h_out, c_out


class LanguageLSTM(nn.Module):
    def __init__(self, dim_att_lstm, dim_visual_att, hidden_size):
        super(LanguageLSTM, self).__init__()
        self.lstm_cell = nn.LSTMCell(
            dim_att_lstm + dim_visual_att, hidden_size, bias=True
        )

    def forward(self, h2, c2, h1, v_hat):
        input_features = torch.cat((h1, v_hat), dim=1)
        h_out, c_out = self.lstm_cell(input_features, (h2, c2))
        return h_out, c_out


class VisualAttention(nn.Module):
    def __init__(self, dim_image_features, dim_att_lstm, hidden_layer_size):
        super(VisualAttention, self).__init__()
        self.linear_image_features = nn.Linear(
            dim_image_features, hidden_layer_size, bias=False
        )
        self.linear_att_lstm = nn.Linear(dim_att_lstm, hidden_layer_size, bias=False)
        self.tanh = nn.Tanh()
        self.linear_attention = nn.Linear(hidden_layer_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image_features, h1):
        image_features_embedded = self.linear_image_features(image_features)
        att_lstm_embedded = self.linear_att_lstm(h1).unsqueeze(1)

        all_feats_emb = image_features_embedded + att_lstm_embedded.repeat(
            1, image_features.size()[1], 1
        )

        activate_feats = self.tanh(all_feats_emb)
        attention = self.linear_attention(activate_feats)
        normalized_attention = self.softmax(attention)

        weighted_feats = normalized_attention * image_features
        attention_weighted_image_features = weighted_feats.sum(dim=1)
        return attention_weighted_image_features


class PredictWord(nn.Module):
    def __init__(self, dim_language_lstm, vocab_size):
        super(PredictWord, self).__init__()
        self.fc = nn.Linear(dim_language_lstm, vocab_size, bias=True)

    def forward(self, h2):
        y = self.fc(h2)
        return y


def print_current_beam(top_k_sequences, top_k_scores, word_map):
    print("\n")
    for sequence, score in zip(top_k_sequences, top_k_scores):
        print(
            "{} \t\t\t\t Score: {}".format(
                decode_caption(sequence.numpy(), word_map), score
            )
        )
