import torch
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


def beam_search_decode(
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
    vocab_size = len(word_map)

    # Encode
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
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
    decoder_hidden_state, decoder_cell_state = decoder.init_hidden_state(encoder_out)

    for step in range(0, max_caption_len - 1):
        embeddings = decoder.embedding(top_k_sequences[:, step]).squeeze(
            1
        )  # (k, embed_dim)

        predictions, alpha, decoder_hidden_state, decoder_cell_state = decoder.forward_step(
            encoder_out, decoder_hidden_state, decoder_cell_state, embeddings
        )
        scores = F.log_softmax(predictions, dim=1)

        # Add the new scores
        scores = top_k_scores.unsqueeze(1).expand_as(scores) + scores  # (k, vocab_size)

        # For the first timestep, the scores from previous decoding are all the same, so in order to create 5 different
        # sequences, we should only look at one branch
        if step == 0:
            scores = scores[0]

        # Find the top k of the flattened scores
        top_k_scores, top_k_words = scores.view(-1).topk(
            current_beam_width, 0, largest=True, sorted=True
        )  # (k)

        # Convert flattened indices to actual indices of scores
        prev_seq_inds = top_k_words / vocab_size  # (k)
        next_words = top_k_words % vocab_size  # (k)

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
                (seqs_alpha[prev_seq_inds], alpha[prev_seq_inds].unsqueeze(1)), dim=1
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
