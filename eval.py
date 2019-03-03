import argparse
import sys

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from inference import generate_caption
from utils import *
from nltk.translate.bleu_score import corpus_bleu
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
  _, _, test_images_split = get_image_indices_splits_from_file(data_folder, test_set_image_coco_ids_file)
  loader = torch.utils.data.DataLoader(
    CaptionDataset(data_folder, test_images_split, SPLIT_TEST, transform=transforms.Compose([normalize])),
    batch_size=1, shuffle=True, num_workers=1, pin_memory=True
  )

  # Lists for target captions and generated captions for each image
  target_captions = []
  generated_captions = []

  for i, (image, _, _, all_captions_for_image) in enumerate(
      tqdm(loader, desc="Evaluate with beam size " + str(beam_size))):

    # Target captions
    target_captions.append(
      [get_caption_without_special_tokens(caption, word_map) for caption in all_captions_for_image[0].tolist()]
    )

    # Generated caption
    generated_caption = generate_caption(encoder, decoder, image, word_map, beam_size, max_caption_len, store_alphas=False)
    generated_captions.append(get_caption_without_special_tokens(generated_caption, word_map))

    # print(decodeCaption(generated_caption, word_map))
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
