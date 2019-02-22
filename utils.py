import os

def getBaseFilename(train_val_test_splits, captions_per_image, vocabulary_size, max_caption_len):
  return (os.path.basename(train_val_test_splits).split('.')[0] + '_' + str(captions_per_image) + '_captions_per_img_'
          + str(vocabulary_size) + '_vocabulary_size_'
          + str(max_caption_len) + '_max_caption_len')

def getWordMapFilename(train_val_test_splits, captions_per_image, vocabulary_size, max_caption_len):
  base_filename = getBaseFilename(train_val_test_splits, captions_per_image, vocabulary_size, max_caption_len)
  return 'word_map' + base_filename + '.json'

def getImagesFilename(split, train_val_test_splits, captions_per_image, vocabulary_size, max_caption_len):
  base_filename = getBaseFilename(train_val_test_splits, captions_per_image, vocabulary_size, max_caption_len)
  return split + '_images_' + base_filename + '.hdf5'

def getCaptionsFilename(split, train_val_test_splits, captions_per_image, vocabulary_size, max_caption_len):
  base_filename = getBaseFilename(train_val_test_splits, captions_per_image, vocabulary_size, max_caption_len)
  return split + '_captions_' + base_filename + '.json'

def getCaptionLengthsFilename(split, train_val_test_splits, captions_per_image, vocabulary_size, max_caption_len):
  base_filename = getBaseFilename(train_val_test_splits, captions_per_image, vocabulary_size, max_caption_len)
  return split + '_caption_lengths' + base_filename + '.json'



