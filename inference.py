import torch
import torch.nn.functional as F

from utils import TOKEN_START, TOKEN_END

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_caption(encoder, decoder, img, word_map, beam_size=1, max_caption_len=50, store_alphas=False):
  k = beam_size
  vocab_size = len(word_map)

  # Move image to GPU device, if available
  image = img.to(device)

  # Encode
  encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
  enc_image_size = encoder_out.size(1)
  encoder_dim = encoder_out.size(3)

  # Flatten encoding
  encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
  num_pixels = encoder_out.size(1)

  # We'll treat the problem as having a batch size of k
  encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

  # Tensor to store top k previous words at each step; now they're just <start>
  k_prev_words = torch.full((k,1), word_map[TOKEN_START], dtype=torch.int64, device=device)

  # Tensor to store top k sequences; now they're just <start>
  top_k_sequences = k_prev_words  # (k, 1)

  # Tensor to store top k sequences' scores; now they're just 0
  top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

  if store_alphas:
    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

  # Lists to store completed sequences, scores, and alphas
  complete_seqs = []
  complete_seqs_alpha = []
  complete_seqs_scores = []

  # Start decoding
  decoder_hidden_state, decoder_cell_state = decoder.init_hidden_state(encoder_out)

  for step in range(0, max_caption_len-1):
    embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (k, embed_dim)

    predictions, alpha, decoder_hidden_state, decoder_cell_state = decoder.forward_step(
      encoder_out, decoder_hidden_state, decoder_cell_state, embeddings
    )

    scores = F.log_softmax(predictions, dim=1)

    # Add the new scores
    scores = top_k_scores.expand_as(scores) + scores  # (k, vocab_size)

    if step == 0:
      # For the first step, all k points will have the same scores
      top_k_scores, top_k_words = scores[0].topk(k, 0, largest=True, sorted=True)  # (k)
    else:
      # Unroll and find top scores, and their unrolled indices
      top_k_scores, top_k_words = scores.view(-1).topk(k, 0, largest=True, sorted=True)  # (k)

    # Convert unrolled indices to actual indices of scores
    prev_word_inds = top_k_words / vocab_size  # (k)
    next_word_inds = top_k_words % vocab_size  # (k)

    # Add new words to sequences, alphas
    top_k_sequences = torch.cat((top_k_sequences[prev_word_inds], next_word_inds.unsqueeze(1)), dim=1)  # (k, step+2)
    if store_alphas:
      alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (k, enc_image_size, enc_image_size)
      seqs_alpha = torch.cat(
        (seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)), dim=1
      )  # (k, step+2, enc_image_size, enc_image_size)

    # Check for complete and incomplete sequences (based on the <end> token)
    incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map[TOKEN_END]]
    complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

    # Set aside complete sequences and reduce beam size accordingly
    if len(complete_inds) > 0:
      complete_seqs.extend(top_k_sequences[complete_inds].tolist())
      complete_seqs_scores.extend(top_k_scores[complete_inds])
      if store_alphas:
        complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
      k -= len(complete_inds)

    # Stop if k captions have been completely generated
    if k == 0:
      break

    # Proceed with incomplete sequences
    top_k_sequences = top_k_sequences[incomplete_inds]
    decoder_hidden_state = decoder_hidden_state[prev_word_inds[incomplete_inds]]
    decoder_cell_state = decoder_cell_state[prev_word_inds[incomplete_inds]]
    encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
    top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
    k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
    if store_alphas:
      seqs_alpha = seqs_alpha[incomplete_inds]

  index_of_best_sequence = complete_seqs_scores.index(max(complete_seqs_scores))
  best_generated_sequence = complete_seqs[index_of_best_sequence]

  if not store_alphas:
    return best_generated_sequence
  else:
    alphas = complete_seqs_alpha[index_of_best_sequence]
    return best_generated_sequence, alphas

