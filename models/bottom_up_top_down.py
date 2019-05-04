import torch
from torch import nn

from models.captioning_model import CaptioningModelDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TopDownDecoder(CaptioningModelDecoder):
    DEFAULT_MODEL_PARAMS = {
        "teacher_forcing_ratio": 1,
        "dropout_ratio": 0.0,
        "image_features_size": 2048,
        "word_embeddings_size": 1000,
        "attention_lstm_size": 1000,
        "attention_layer_size": 512,
        "language_lstm_size": 1000,
        "max_caption_len": 20,
        "fine_tune_decoder_word_embeddings": True,
    }
    DEFAULT_OPTIMIZER_PARAMS = {"decoder_learning_rate": 1e-4}

    def __init__(self, word_map, params, pretrained_embeddings=None):
        super(TopDownDecoder, self).__init__(word_map, params, pretrained_embeddings)

        self.attention_lstm = AttentionLSTM(
            self.params["word_embeddings_size"],
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

        # Dropout layer
        self.dropout = nn.Dropout(p=self.params["dropout_ratio"])

        # Linear layer to find scores over vocabulary
        self.fully_connected = nn.Linear(
            self.params["language_lstm_size"], self.vocab_size, bias=True
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

    def init_hidden_states(self, encoder_output):
        v_mean = encoder_output.mean(dim=1)

        h1 = self.init_h1(v_mean)
        c1 = self.init_c1(v_mean)
        h2 = self.init_h2(v_mean)
        c2 = self.init_c2(v_mean)
        states = [h1, c1, h2, c2]

        return states

    def forward_step(self, encoder_output, prev_words_embedded, states):
        v_mean = encoder_output.mean(dim=1)
        h1, c1, h2, c2 = states
        h1, c1 = self.attention_lstm(h1, c1, h2, v_mean, prev_words_embedded)
        v_hat = self.attention(encoder_output, h1)
        h2, c2 = self.language_lstm(h2, c2, h1, v_hat)
        scores = self.fully_connected(self.dropout(h2))
        states = [h1, c1, h2, c2]
        return scores, states, None

    def loss(self, scores, target_captions, decode_lengths, alphas):
        return self.loss_cross_entropy(scores, target_captions, decode_lengths)

    def beam_search(
        self,
        encoder_output,
        beam_size,
        stochastic_beam_search=False,
        diverse_beam_search=False,
        store_alphas=False,
        store_beam=False,
        print_beam=False,
    ):
        """Generate and return the top k sequences using beam search."""

        if store_alphas:
            raise NotImplementedError(
                "Storage of alphas for this model is not supported"
            )

        return super(TopDownDecoder, self).beam_search(
            encoder_output,
            beam_size,
            stochastic_beam_search,
            diverse_beam_search,
            store_alphas,
            store_beam,
            print_beam,
        )


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
