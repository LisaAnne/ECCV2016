#!/usr/bin/env python

from hashlib import sha1
import os
import random
random.seed(3)
import re
import sys
import h5py
import numpy as np
import pickle as pkl
import pdb 

eightyK = False

sys.path.append('.')
sys.path.append('../../data/coco/coco')
from init import *
home_dir = caffe_dir

MAX_HASH = 100000

from pycocotools.coco import COCO

from hdf5_sequence_generator import SequenceGenerator, HDF5SequenceWriter

# UNK_IDENTIFIER is the word used to identify unknown words
UNK_IDENTIFIER = '<unk>'

SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
def split_sentence(sentence):
  # break sentence into a list of words and punctuation
  sentence = [s.lower() for s in SENTENCE_SPLIT_REGEX.split(sentence.strip()) if len(s.strip()) > 0]
  # remove the '.' from the end of the sentence
  if sentence[-1] != '.':
    # print "Warning: sentence doesn't end with '.'; ends with: %s" % sentence[-1]
    return sentence
  return sentence[:-1]

MAX_WORDS = 20

class CocoSequenceGenerator(SequenceGenerator):
  def __init__(self, coco, batch_num_streams, image_root, vocab=None,
               max_words=MAX_WORDS, align=True, shuffle=True, gt_captions=True,
               pad=True, truncate=True, split_ids=None):
    self.max_words = max_words
    num_empty_lines = 0
    self.images = []
    num_total = 0
    num_missing = 0
    num_captions = 0
    known_images = {}
    self.coco = coco
    if split_ids is None:
      split_ids = coco.imgs.keys()
    self.image_path_to_id = {}
    for image_id in split_ids:
      image_info = coco.imgs[image_id]
      image_path = '%s/%s' % (image_root, image_info['file_name'])
      self.image_path_to_id[image_path] = image_id
      if os.path.isfile(image_path):
        assert image_id not in known_images  # no duplicates allowed
        known_images[image_id] = {}
        known_images[image_id]['path'] = image_path
        if gt_captions:
          known_images[image_id]['sentences'] = [split_sentence(anno['caption'])
              for anno in coco.imgToAnns[image_id]]
          num_captions += len(known_images[image_id]['sentences'])
        else:
          known_images[image_id]['sentences'] = []
      else:
        num_missing += 1
        print 'Warning (#%d): image not found: %s' % (num_missing, image_path)
      num_total += 1
    print '%d/%d images missing' % (num_missing, num_total)
    if vocab is None:
      self.init_vocabulary(known_images)
    else:
      self.vocabulary_inverted = vocab
      self.vocabulary = {}
      for index, word in enumerate(self.vocabulary_inverted):
        self.vocabulary[word] = index
    self.image_sentence_pairs = []
    num_no_sentences = 0
    for image_filename, metadata in known_images.iteritems():
      if not metadata['sentences']:
        num_no_sentences += 1
        print 'Warning (#%d): image with no sentences: %s' % (num_no_sentences, image_filename)
      for sentence in metadata['sentences']:
        self.image_sentence_pairs.append((metadata['path'], sentence))
    self.index = 0
    self.num_resets = 0
    self.num_truncates = 0
    self.num_pads = 0
    self.num_outs = 0
    self.image_list = []
    SequenceGenerator.__init__(self)
    self.batch_num_streams = batch_num_streams
    # make the number of image/sentence pairs a multiple of the buffer size
    # so each timestep of each batch is useful and we can align the images
    if align:
      num_pairs = len(self.image_sentence_pairs)
      remainder = num_pairs % batch_num_streams
      if remainder > 0:
        num_needed = batch_num_streams - remainder
        for i in range(num_needed):
          choice = random.randint(0, num_pairs - 1)
          self.image_sentence_pairs.append(self.image_sentence_pairs[choice])
      assert len(self.image_sentence_pairs) % batch_num_streams == 0
    if shuffle:
      random.shuffle(self.image_sentence_pairs)
    self.pad = pad
    self.truncate = truncate
    self.negative_one_padded_streams = frozenset(('input_sentence', 'target_sentence'))

  def streams_exhausted(self):
    return self.num_resets > 0

  def init_vocabulary(self, image_annotations, min_count=5):
    words_to_count = {}
    for image_id, annotations in image_annotations.iteritems():
      for annotation in annotations['sentences']:
        for word in annotation:
          word = word.strip()
          if word not in words_to_count:
            words_to_count[word] = 0
          words_to_count[word] += 1
    # Sort words by count, then alphabetically
    words_by_count = sorted(words_to_count.keys(), key=lambda w: (-words_to_count[w], w))
    print 'Initialized vocabulary with %d words; top 10 words:' % len(words_by_count)
    for word in words_by_count[:10]:
      print '\t%s (%d)' % (word, words_to_count[word])
    # Add words to vocabulary
    self.vocabulary = {UNK_IDENTIFIER: 0}
    self.vocabulary_inverted = [UNK_IDENTIFIER]
    for index, word in enumerate(words_by_count):
      word = word.strip()
      if words_to_count[word] < min_count:
        break
      self.vocabulary_inverted.append(word)
      self.vocabulary[word] = index + 1
    print 'Final vocabulary (restricted to words with counts of %d+) has %d words' % \
        (min_count, len(self.vocabulary))

  def dump_vocabulary(self, vocab_filename):
    print 'Dumping vocabulary to file: %s' % vocab_filename
    with open(vocab_filename, 'wb') as vocab_file:
      for word in self.vocabulary_inverted:
        vocab_file.write('%s\n' % word)
    print 'Done.'

  def dump_image_file(self, image_filename, dummy_image_filename=None):
    print 'Dumping image list to file: %s' % image_filename
    with open(image_filename, 'wb') as image_file:
      for image_path, _ in self.image_list:
        image_file.write('%s\n' % image_path)
    if dummy_image_filename is not None:
      print 'Dumping image list with dummy labels to file: %s' % dummy_image_filename
      with open(dummy_image_filename, 'wb') as image_file:
        for path_and_hash in self.image_list:
          image_file.write('%s %d\n' % path_and_hash)
    print 'Done.'

  def next_line(self):
    num_lines = float(len(self.image_sentence_pairs))
    self.index += 1
    if self.index == 1 or self.index == num_lines or self.index % 10000 == 0:
      print 'Processed %d/%d (%f%%) lines' % (self.index, num_lines,
                                              100 * self.index / num_lines)
    if self.index == num_lines:
      self.index = 0
      self.num_resets += 1

  def line_to_stream(self, sentence):
    stream = []
    for word in sentence:
      word = word.strip()
      if word in self.vocabulary:
        stream.append(self.vocabulary[word]) 
      else:  # unknown word; append UNK
        if word in self.vocabulary:
          stream.append(self.vocabulary[UNK_IDENTIFIER])        
    # increment the stream -- 0 will be the EOS character
    stream = [s + 1 for s in stream]
    return stream

  def get_pad_value(self, stream_name):
    pdb.set_trace()
    return -1 if stream_name in self.negative_one_padded_streams else 0

