import argparse
import sys

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

def evaluate(data_folder, test_set_image_coco_ids_file, checkpoint, beam_size=1, max_caption_len=50):
  # Load model
  checkpoint = torch.load(checkpoint, map_location=device)
  decoder = checkpoint['decoder']
  decoder = decoder.to(device)
  decoder.eval()
  encoder = checkpoint['encoder']
  encoder = encoder.to(device)
  encoder.eval()

  # Load word map
  word_map_path = os.path.join(data_folder, getWordMapFilename())
  with open(word_map_path, 'r') as json_file:
    word_map = json.load(json_file)
  vocab_size = len(word_map)

  # Normalization
  normalize = transforms.Normalize(mean=IMAGENET_IMAGES_MEAN, std=IMAGENET_IMAGES_STD)

  # DataLoader
  _, _, test_images_split = getImageIndicesSplitsFromFile(data_folder, test_set_image_coco_ids_file)
  loader = torch.utils.data.DataLoader(
    CaptionDataset(data_folder, test_images_split, SPLIT_TEST, transform=transforms.Compose([normalize])),
    batch_size=1, shuffle=True, num_workers=1, pin_memory=True
  )

  # Lists for target captions and generated captions for each image
  target_captions = []
  generated_captions = []

  for i, (image, _, _, all_captions_for_image) in enumerate(
      tqdm(loader, desc="Evaluate with beam size " + str(beam_size))):

    k = beam_size

    # Move to GPU device, if available
    image = image.to(device)  # (1, 3, 256, 256)

    # Encode
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map[TOKEN_START]]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    top_k_sequences = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Lists to store completed sequences and scores
    complete_seqs = []
    complete_seqs_scores = []

    # Start decoding
    decoder_hidden_state, decoder_cell_state = decoder.init_hidden_state(encoder_out)

    for step in range(0, max_caption_len):
      embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (k, embed_dim)

      attention_weighted_encoding, _ = decoder.attention(encoder_out, decoder_hidden_state)  # (k, encoder_dim)

      gate = decoder.sigmoid(decoder.f_beta(decoder_hidden_state))  # (k, encoder_dim)
      attention_weighted_encoding = gate * attention_weighted_encoding

      decoder_hidden_state, decoder_cell_state = decoder.decode_step(
        torch.cat([embeddings, attention_weighted_encoding], dim=1), (decoder_hidden_state, decoder_cell_state)
      )  # (k, decoder_dim)

      # Tensor containing the score for each word in the vocabulary, for each branch in the beam
      scores = decoder.fc(decoder_hidden_state)  # (k, vocab_size)
      scores = F.log_softmax(scores, dim=1)

      # Add the new scores
      scores = top_k_scores.expand_as(scores) + scores  # (k, vocab_size)

      # For the first step, all k points will have the same scores (since same k previous words, h, c)
      if step == 0:
        top_k_scores, top_k_words = scores[0].topk(k, 0, largest=True, sorted=True)  # (k)
      else:
        # Unroll and find top scores, and their unrolled indices
        top_k_scores, top_k_words = scores.view(-1).topk(k, 0, largest=True, sorted=True)  # (k)

      # Convert unrolled indices to actual indices of scores
      prev_word_inds = top_k_words / vocab_size  # (k)
      next_word_inds = top_k_words % vocab_size  # (k)

      # Add new words to sequences
      top_k_sequences = torch.cat([top_k_sequences[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (k, step+2)

      # Check for complete and incomplete sequences (based on the <end> token)
      incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map[TOKEN_END]]
      complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

      # Set aside complete sequences and reduce beam size accordingly
      if len(complete_inds) > 0:
        complete_seqs.extend(top_k_sequences[complete_inds].tolist())
        complete_seqs_scores.extend(top_k_scores[complete_inds])
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

    # Target captions
    target_captions.append(
      [getCaptionWithoutSpecialTokens(caption, word_map) for caption in all_captions_for_image[0].tolist()]
    )

    # Generated caption
    best_generated_sequence = complete_seqs[complete_seqs_scores.index(max(complete_seqs_scores))]
    generated_captions.append(getCaptionWithoutSpecialTokens(best_generated_sequence, word_map))

    # print(decodeCaption(generated_captions[0], word_map))
    # showImg(image.squeeze(0).numpy())
    assert len(target_captions) == len(generated_captions)

  # Calculate BLEU-4 scores
  bleu4 = corpus_bleu(target_captions, generated_captions)

  print("\nBLEU-4 score @ beam size of {} is {}".format(beam_size, bleu4))
  return bleu4

def check_args(args):
  parser = argparse.ArgumentParser()
  parser.add_argument('-D', '--data-folder',
                      help='Folder where the preprocessed data is located',
                      default=os.path.expanduser('~/datasets/coco2014_preprocessed/'))
  parser.add_argument('-T', '--test-set-image-coco-ids-file',
                      help='File containing JSON-serialized list of image IDs for the test set',
                      default='data/white_cars.json')
  parser.add_argument('-C', '--checkpoint',
                      help='Path to checkpoint of trained model',
                      default='best_checkpoint.pth.tar')
  parser.add_argument('-B', '--beam-size',
                      help='Size of the decoding beam',
                      type=int, default=1)
  parser.add_argument('-L', '--max-caption-len',
                      help='Maximum caption length',
                      type=int, default=50)

  parsed_args = parser.parse_args(args)
  print(parsed_args)
  return parsed_args


if __name__ == '__main__':
  parsed_args = check_args(sys.argv[1:])
  evaluate(
    data_folder=parsed_args.data_folder,
    test_set_image_coco_ids_file=parsed_args.test_set_image_coco_ids_file,
    checkpoint=parsed_args.checkpoint,
    beam_size=parsed_args.beam_size,
    max_caption_len=parsed_args.max_caption_len
  )
