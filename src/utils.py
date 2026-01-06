# src/utils.py
import os
import yaml
import numpy as np
from PIL import Image
import tensorflow as tf


def load_config(path="config.yaml"):
    """Load YAML config from path."""
    with open(path) as f:
        return yaml.safe_load(f)


def make_image_processor(image_size):
    """
    Return a function that loads/resizes an image and normalizes it to [-1, +1].
    Accepts either a file path (str) or a NumPy array / PIL Image.
    """
    def proc(img):
        if isinstance(img, str):
            im = Image.open(img).convert("RGB")
        else:
            im = Image.fromarray(img) if not isinstance(img, Image.Image) else img
            im = im.convert("RGB")
        im = im.resize((image_size, image_size))
        arr = np.array(im).astype(np.float32) / 255.0
        # Normalize to [-1, +1] which is suitable for the lightweight CNN encoder
        arr = (arr - 0.5) * 2.0
        return arr
    return proc


def prepare_dataset(cfg, split="train", keep_small=False):
    """
    Load the HuggingFace dataset specified in cfg['dataset']['hf_name'],
    tokenize captions with a BERT tokenizer, and process images.
    Returns:
      processed: list of dicts with keys 'images' (seq_len, H, W, C) and 'input_ids' (seq_len, T)
      tokenizer: the HF tokenizer (AutoTokenizer)
    """
    # Lazy imports to avoid heavy import at module load time
    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds = load_dataset(cfg['dataset']['hf_name'], split=split)
    if keep_small:
        ds = ds.select(range(min(64, len(ds))))
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    image_size = cfg['dataset']['image_size']
    seq_len = cfg['dataset']['seq_len']
    max_cap_len = cfg['dataset']['max_caption_len']
    image_proc = make_image_processor(image_size)

    def map_example(example):
        # try several possible keys for frames and captions (robust to dataset schema)
        frames = None
        captions = None
        for k in ["frames", "images", "frames_paths", "image_paths", "imgs"]:
            if k in example:
                frames = example[k]
                break
        for k in ["captions", "descriptions", "texts", "story"]:
            if k in example:
                captions = example[k]
                break
        if frames is None:
            frames = []
        if captions is None:
            captions = []

        frames = frames[:seq_len]
        captions = captions[:seq_len]

        imgs = []
        for f in frames:
            try:
                imgs.append(image_proc(f))
            except Exception:
                # if loading fails, append a zero image
                imgs.append(np.zeros((image_size, image_size, 3), dtype=np.float32))
        # pad frames if fewer than seq_len
        while len(imgs) < seq_len:
            imgs.append(np.zeros((image_size, image_size, 3), dtype=np.float32))

        tok_out = tokenizer(
            captions,
            padding='max_length',
            truncation=True,
            max_length=max_cap_len,
            return_tensors="np"
        )
        input_ids = tok_out['input_ids']
        # pad token rows if fewer than seq_len
        if input_ids.shape[0] < seq_len:
            pad_rows = np.zeros((seq_len - input_ids.shape[0], max_cap_len), dtype=np.int32)
            input_ids = np.vstack([input_ids, pad_rows])

        return {
            "images": np.stack(imgs).astype(np.float32),   # (seq_len, H, W, C)
            "input_ids": input_ids.astype(np.int32)       # (seq_len, T)
        }

    processed = []
    for ex in ds:
        processed.append(map_example(ex))
    return processed, tokenizer


def generator_from_processed(processed_list, cfg):
    """Yield tuples (images, input_ids) for tf.data.Dataset.from_generator."""
    def gen():
        for ex in processed_list:
            yield ex['images'], ex['input_ids']
    return gen


def make_tf_dataset(processed_list, cfg, shuffle=True):
    """
    Build a batched tf.data.Dataset from processed_list.
    Yields: (batch_images, batch_input_ids) with shapes:
      batch_images: (B, seq_len, H, W, C)
      batch_input_ids: (B, seq_len, T)
    """
    seq_len = cfg['dataset']['seq_len']
    max_cap_len = cfg['dataset']['max_caption_len']
    batch_size = cfg['dataset']['batch_size']
    image_size = cfg['dataset']['image_size']
    out_types = (tf.float32, tf.int32)
    out_shapes = ((seq_len, image_size, image_size, 3), (seq_len, max_cap_len))
    ds = tf.data.Dataset.from_generator(
        generator_from_processed(processed_list, cfg),
        output_types=out_types,
        output_shapes=out_shapes
    )
    if shuffle:
        ds = ds.shuffle(1024)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

# ===== Evaluation Metrics =====
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

# IMPORTANT:
# Download nltk resources ONCE externally:
# >>> import nltk
# >>> nltk.download('wordnet')
# >>> nltk.download('omw-1.4')

def distinct_n(sentences, n):
    """
    Compute corpus-level Distinct-n.
    """
    ngrams = []
    for sent in sentences:
        tokens = sent.split()
        if len(tokens) < n:
            continue
        ngrams.extend(zip(*[tokens[i:] for i in range(n)]))
    if len(ngrams) == 0:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def compute_bleu(reference, hypothesis):
    """
    Sentence-level BLEU with smoothing.
    """
    if len(hypothesis.strip()) == 0:
        return 0.0
    smoothie = SmoothingFunction().method4
    return sentence_bleu(
        [reference.split()],
        hypothesis.split(),
        smoothing_function=smoothie
    )


def compute_meteor(reference, hypothesis):
    """
    Sentence-level METEOR.
    """
    if len(reference.strip()) == 0 or len(hypothesis.strip()) == 0:
        return 0.0
    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    return meteor_score([ref_tokens], hyp_tokens)


