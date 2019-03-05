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
    self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

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
    out = self.model(images)  # output shape: (batch_size, 2048, image_size/32, image_size/32)
    out = self.adaptive_pool(out)  # output shape: (batch_size, 2048, encoded_image_size, encoded_image_size)
    out = out.permute(0, 2, 3, 1)  # output shape: (batch_size, encoded_image_size, encoded_image_size, 2048)
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
    self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
    self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
    self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
    self.relu = nn.ReLU()
    self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

  def forward(self, encoder_out, decoder_hidden):
    """
    Forward propagation.

    :param encoder_out: encoded images, shape: (batch_size, num_pixels, encoder_dim)
    :param decoder_hidden: previous decoder output, shape: (batch_size, decoder_dim)
    :return: attention weighted encoding, weights
    """
    att1 = self.encoder_att(encoder_out)  # output shape: (batch_size, num_pixels, attention_dim)
    att2 = self.decoder_att(decoder_hidden)  # output shape: (batch_size, attention_dim)
    att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # output shape: (batch_size, num_pixels)
    alpha = self.softmax(att)  # output shape: (batch_size, num_pixels)
    attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # output shape: (batch_size, encoder_dim)

    return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):

  def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, start_token, encoder_dim=2048, dropout=0.5):
    """
    :param attention_dim: size of attention network
    :param embed_dim: embedding size
    :param decoder_dim: size of decoder's RNN
    :param vocab_size: size of vocabulary
    :param start_token: word map value of the sentence <start> token
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
    self.dropout = dropout

    self.attention = AttentionModule(encoder_dim, decoder_dim, attention_dim)

    self.embedding = nn.Embedding(vocab_size, embed_dim)
    self.dropout = nn.Dropout(p=self.dropout)
    self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
    self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
    self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
    self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
    self.sigmoid = nn.Sigmoid()
    self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary

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

  def forward(self, encoder_out, encoded_captions, caption_lengths):
    """
    Forward propagation.

    :param encoder_out: encoded images, shape: (batch_size, enc_image_size, enc_image_size, encoder_dim)
    :param encoded_captions: encoded captions, shape: (batch_size, max_caption_length)
    :param caption_lengths: caption lengths, shape: (batch_size, 1)
    :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
    """

    batch_size = encoder_out.size(0)
    encoder_dim = encoder_out.size(-1)
    vocab_size = self.vocab_size

    # Flatten image
    encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # Sort input data by decreasing lengths to allow for decrease of batch size when sequences are complete
    caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
    encoder_out = encoder_out[sort_ind]
    encoded_captions = encoded_captions[sort_ind]

    # Embedding
    embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

    # Initialize LSTM state
    h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

    # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
    # So, decoding lengths are actual lengths - 1
    decode_lengths = (caption_lengths - 1).tolist()

    # Tensors to hold word prediction scores and alphas
    predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
    alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

    # At each time-step, decode by
    # attention-weighing the encoder's output based on the decoder's previous hidden state output
    # then generate a new word in the decoder with the previous word and the attention weighted encoding
    for t in range(max(decode_lengths)):
      batch_size_t = sum([l > t for l in decode_lengths])
      attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
      gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
      attention_weighted_encoding = gate * attention_weighted_encoding

      if self.training:
        prev_word_embeddings = embeddings[:batch_size_t, t, :]
      else:
        # In evaluation mode, we do not feed back the target tokens into the decoder but instead use its own output
        # from the previous time step.
        if t == 0:
          # At the start, all 'previous words' are the start token
          prev_predicted_words = torch.full((batch_size_t,), self.start_token, dtype=torch.int64, device=device)
        else:
          prev_predicted_words = torch.max(predictions[:batch_size_t, t-1, :],dim=1)[1]
        prev_word_embeddings = self.embedding(prev_predicted_words)

      h, c = self.decode_step(
        torch.cat((prev_word_embeddings, attention_weighted_encoding), dim=1),
        (h[:batch_size_t], c[:batch_size_t])
      )  # (batch_size_t, decoder_dim)

      preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
      predictions[:batch_size_t, t, :] = preds
      alphas[:batch_size_t, t, :] = alpha

    return predictions, encoded_captions, decode_lengths, alphas, sort_ind