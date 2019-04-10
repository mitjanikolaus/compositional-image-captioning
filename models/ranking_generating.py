import torch
from torch import nn
from torch.autograd import Variable

from models.captioning_model import CaptioningModelDecoder, update_params

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0.2, max_violation=True):
        super(ContrastiveLoss, self).__init__()

        self.margin = margin
        self.max_violation = max_violation

    def forward(self, images_embedded, captions_embedded):
        # compute image-caption score matrix
        scores = cosine_sim(images_embedded, captions_embedded)
        diagonal = scores.diag().view(images_embedded.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > 0.5
        I = Variable(mask).to(device)
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


def l2_norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


def cosine_sim(images_embedded, captions_embedded):
    """Cosine similarity between all the image and sentence pairs
    """
    return images_embedded.mm(captions_embedded.t())


class RankGenEncoder(nn.Module):
    DEFAULT_MODEL_PARAMS = {"image_features_size": 2048, "joint_embeddings_size": 1024}
    DEFAULT_OPTIMIZER_PARAMS = {"encoder_learning_rate": 1e-4}

    def __init__(self, params):
        super(RankGenEncoder, self).__init__()
        self.params = update_params(self.DEFAULT_MODEL_PARAMS, params)

        self.image_emb = nn.Linear(
            self.params["image_features_size"], self.params["joint_embeddings_size"]
        )

    def forward(self, image_features):
        features_embedded = self.image_emb(image_features)

        return features_embedded


class RankGenDecoder(CaptioningModelDecoder):
    DEFAULT_MODEL_PARAMS = {
        "teacher_forcing_ratio": 1,
        "dropout_ratio": 0.0,
        "joint_embeddings_size": 1024,
        "word_embeddings_size": 1000,
        "attention_lstm_size": 1000,
        "attention_layer_size": 512,
        "language_generation_lstm_size": 1000,
        "max_caption_len": 50,
        "fine_tune_decoder_embeddings": True,
    }
    DEFAULT_OPTIMIZER_PARAMS = {"decoder_learning_rate": 1e-4}

    def __init__(self, word_map, params, pretrained_embeddings=None):
        super(RankGenDecoder, self).__init__(word_map, params, pretrained_embeddings)

        self.attention_lstm = AttentionLSTM(
            self.params["joint_embeddings_size"],
            self.params["language_generation_lstm_size"],
            self.params["attention_lstm_size"],
        )
        self.language_encoding_lstm = LanguageEncodingLSTM(
            self.params["word_embeddings_size"], self.params["joint_embeddings_size"]
        )
        self.language_generation_lstm = LanguageGenerationLSTM(
            self.params["attention_lstm_size"],
            self.params["joint_embeddings_size"],
            self.params["language_generation_lstm_size"],
        )
        self.attention = VisualAttention(
            self.params["joint_embeddings_size"],
            self.params["attention_lstm_size"],
            self.params["attention_layer_size"],
        )

        # Dropout layer
        self.dropout = nn.Dropout(p=self.params["dropout_ratio"])

        # Linear layer to find scores over vocabulary
        self.fully_connected = nn.Linear(
            self.params["language_generation_lstm_size"], self.vocab_size, bias=True
        )

        # linear layers to find initial states of LSTMs
        self.init_h_attention = nn.Linear(
            self.params["joint_embeddings_size"],
            self.attention_lstm.lstm_cell.hidden_size,
        )
        self.init_c_attention = nn.Linear(
            self.params["joint_embeddings_size"],
            self.attention_lstm.lstm_cell.hidden_size,
        )
        self.init_h_lan_gen = nn.Linear(
            self.params["joint_embeddings_size"],
            self.language_generation_lstm.lstm_cell.hidden_size,
        )
        self.init_c_lan_gen = nn.Linear(
            self.params["joint_embeddings_size"],
            self.language_generation_lstm.lstm_cell.hidden_size,
        )

        self.criterion = ContrastiveLoss()

    def init_hidden_states(self, encoder_output):
        v_mean = encoder_output.mean(dim=1)

        h_lan_enc, c_lan_enc = self.language_encoding_lstm.init_state(
            encoder_output.size(0)
        )
        h_attention = self.init_h_attention(v_mean)
        c_attention = self.init_c_attention(v_mean)
        h_lan_gen = self.init_h_lan_gen(v_mean)
        c_lan_gen = self.init_c_lan_gen(v_mean)
        states = [h_lan_enc, c_lan_enc, h_attention, c_attention, h_lan_gen, c_lan_gen]

        return states

    def forward_step(self, encoder_output, prev_words_embedded, states):
        v_mean = encoder_output.mean(dim=1)
        h_lan_enc, c_lan_enc, h_attention, c_attention, h_lan_gen, c_lan_gen = states

        h_lan_enc, c_lan_enc = self.language_encoding_lstm(
            h_lan_enc, c_lan_enc, prev_words_embedded
        )

        h_attention, c_attention = self.attention_lstm(
            h_attention, c_attention, h_lan_gen, v_mean, h_lan_enc
        )
        v_hat = self.attention(encoder_output, h_attention)
        h_lan_gen, c_lan_gen = self.language_generation_lstm(
            h_lan_gen, c_lan_gen, h_attention, v_hat
        )
        scores = self.fully_connected(self.dropout(h_lan_gen))
        states = [h_lan_enc, c_lan_enc, h_attention, c_attention, h_lan_gen, c_lan_gen]
        return scores, states, None

    def forward_step_ranking(self, prev_words_embedded, states):
        h_lan_enc, c_lan_enc = states

        h_lan_enc, c_lan_enc = self.language_encoding_lstm(
            h_lan_enc, c_lan_enc, prev_words_embedded
        )

        states = [h_lan_enc, c_lan_enc]

        return states

    def embed_images(self, encoder_output):
        images_embedded = encoder_output.mean(dim=1)
        images_embedded = l2_norm(images_embedded)
        return images_embedded

    def embed_captions(self, captions, decode_lengths):
        # Initialize LSTM state
        batch_size = captions.size(0)
        states = self.language_encoding_lstm.init_state(batch_size)

        # Tensor to store hidden activations
        hidden_activations = torch.zeros(
            (batch_size, max(decode_lengths), self.params["joint_embeddings_size"]),
            device=device,
        )

        for t in range(max(decode_lengths)):
            prev_words_embedded = self.word_embedding(captions[:, t])
            states = self.forward_step_ranking(prev_words_embedded, states)

            hidden_activations[:, t, :] = states[0]

        captions_embedded = torch.zeros(
            (batch_size, self.params["joint_embeddings_size"]),
            device=device,
            requires_grad=False,
        )
        for i, length in enumerate(decode_lengths):
            captions_embedded[i] = hidden_activations[i][length - 1]

        captions_embedded = l2_norm(captions_embedded)
        return captions_embedded

    def forward_ranking(self, encoder_output, captions, decode_lengths):
        """
        Forward propagation for the ranking task.

        """
        images_embedded = self.embed_images(encoder_output)
        captions_embedded = self.embed_captions(captions, decode_lengths)

        return images_embedded, captions_embedded

    def beam_search(
        self,
        encoder_output,
        beam_size=1,
        store_alphas=False,
        store_beam=False,
        print_beam=False,
    ):
        """Generate and return the top k sequences using beam search."""

        if store_alphas:
            raise NotImplementedError(
                "Storage of alphas for this model is not supported"
            )

        return super(RankGenDecoder, self).beam_search(
            encoder_output, beam_size, store_alphas, store_beam, print_beam
        )


class AttentionLSTM(nn.Module):
    def __init__(self, joint_embeddings_size, dim_lang_lstm, hidden_size):
        super(AttentionLSTM, self).__init__()
        self.lstm_cell = nn.LSTMCell(
            2 * joint_embeddings_size + dim_lang_lstm, hidden_size, bias=True
        )

    def forward(self, h1, c1, h2, v_mean, h_lan_enc):
        input_features = torch.cat((h2, v_mean, h_lan_enc), dim=1)
        h_out, c_out = self.lstm_cell(input_features, (h1, c1))
        return h_out, c_out


class LanguageEncodingLSTM(nn.Module):
    def __init__(self, word_embeddings_size, hidden_size):
        super(LanguageEncodingLSTM, self).__init__()
        self.lstm_cell = nn.LSTMCell(word_embeddings_size, hidden_size, bias=True)

    def forward(self, h, c, prev_words_embedded):
        h_out, c_out = self.lstm_cell(prev_words_embedded, (h, c))
        return h_out, c_out

    def init_state(self, batch_size):
        h = torch.zeros((batch_size, self.lstm_cell.hidden_size), device=device)
        c = torch.zeros((batch_size, self.lstm_cell.hidden_size), device=device)
        return [h, c]


class LanguageGenerationLSTM(nn.Module):
    def __init__(self, dim_att_lstm, dim_visual_att, hidden_size):
        super(LanguageGenerationLSTM, self).__init__()
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
