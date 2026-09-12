"""
train_lm.py
-----------
Train a character-level n-gram language model for CTC beam search fusion.

Pure Python implementation (no KenLM dependency required).
Uses add-k smoothing for simplicity.

Usage:
    # From Alice In Wonderland MP3 transcriptions via Whisper:
    python train_lm.py --mp3_dir /path/to/mp3s --out lm.pkl

    # From a text file:
    python train_lm.py --text_file corpus.txt --out lm.pkl
"""

import os, sys, argparse, pickle, json
import numpy as np
from collections import defaultdict

# Add parent for Whisper access
SCRATCH = '<REPO_ROOT>'
MP3_DIR = os.path.join(SCRATCH, 'Alice_In_Wonderland_mp3')


class CharNgramLM:
    """
    Character-level n-gram language model with add-k smoothing.

    Provides .score(text) method returning log probability for
    use with CTC beam search shallow fusion.
    """

    def __init__(self, order=4, smoothing=0.01):
        self.order = order
        self.smoothing = smoothing
        self.ngram_counts = {}       # {n: {context: {char: count}}}
        self.context_counts = {}     # {n: {context: total_count}}
        self.vocab = set()

    def train(self, texts):
        """Train on a list of text strings."""
        # Build vocabulary
        for text in texts:
            for c in text.lower():
                self.vocab.add(c)
        self.vocab = sorted(self.vocab)
        self.vocab_size = len(self.vocab)

        print(f"  Vocab size: {self.vocab_size}")
        print(f"  Vocab: {''.join(self.vocab[:50])}...")

        # Count n-grams for orders 1..self.order
        for n in range(1, self.order + 1):
            self.ngram_counts[n] = defaultdict(lambda: defaultdict(int))
            self.context_counts[n] = defaultdict(int)

            for text in texts:
                text = text.lower()
                # Pad with start-of-sequence markers
                padded = '^' * (n - 1) + text
                for i in range(len(padded) - n + 1):
                    context = padded[i:i + n - 1] if n > 1 else ''
                    char = padded[i + n - 1]
                    self.ngram_counts[n][context][char] += 1
                    self.context_counts[n][context] += 1

        total_chars = sum(self.context_counts[1].values())
        print(f"  Total characters: {total_chars:,}")
        for n in range(1, self.order + 1):
            n_contexts = len(self.context_counts[n])
            print(f"  {n}-gram contexts: {n_contexts:,}")

    def _log_prob_ngram(self, context, char, n):
        """Log probability of char given context at order n with smoothing."""
        counts = self.ngram_counts.get(n, {})
        ctx_counts = self.context_counts.get(n, {})

        count = counts.get(context, {}).get(char, 0)
        total = ctx_counts.get(context, 0)

        # Add-k smoothing
        prob = (count + self.smoothing) / (total + self.smoothing * self.vocab_size)
        return np.log(prob + 1e-30)

    def score(self, text):
        """
        Return log probability of text under the n-gram model.
        Uses interpolation (backoff) across all orders.
        """
        if not text:
            return 0.0

        text = text.lower()
        log_prob = 0.0
        padded = '^' * (self.order - 1) + text

        for i in range(self.order - 1, len(padded)):
            char = padded[i]
            # Interpolate across orders with equal weights
            char_log_prob = -float('inf')
            for n in range(1, self.order + 1):
                context = padded[max(0, i - n + 1):i] if n > 1 else ''
                lp = self._log_prob_ngram(context, char, n)
                # Simple average in log space via log-sum-exp
                if char_log_prob == -float('inf'):
                    char_log_prob = lp
                else:
                    char_log_prob = np.logaddexp(char_log_prob, lp)
            # Subtract log(order) for average
            char_log_prob -= np.log(self.order)
            log_prob += char_log_prob

        return log_prob

    def save(self, path):
        """Save LM to pickle file."""
        data = {
            'order': self.order,
            'smoothing': self.smoothing,
            'vocab': self.vocab,
            'vocab_size': self.vocab_size,
            'ngram_counts': {n: dict(d) for n, d in self.ngram_counts.items()},
            'context_counts': {n: dict(d) for n, d in self.context_counts.items()},
        }
        # Convert defaultdicts to regular dicts for pickling
        for n in data['ngram_counts']:
            data['ngram_counts'][n] = {
                k: dict(v) for k, v in data['ngram_counts'][n].items()
            }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"  LM saved to: {path}")

    @classmethod
    def load(cls, path):
        """Load LM from pickle file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        lm = cls(order=data['order'], smoothing=data['smoothing'])
        lm.vocab = data['vocab']
        lm.vocab_size = data['vocab_size']
        lm.ngram_counts = data['ngram_counts']
        lm.context_counts = data['context_counts']
        return lm


def transcribe_mp3s(mp3_dir, whisper_model_name='base.en'):
    """Transcribe all MP3 files using Whisper to build text corpus."""
    import whisper
    import glob

    print(f"Loading Whisper model: {whisper_model_name}")
    model = whisper.load_model(whisper_model_name)

    mp3_files = sorted(glob.glob(os.path.join(mp3_dir, '*.mp3')))
    print(f"Found {len(mp3_files)} MP3 files")

    texts = []
    for mp3_path in mp3_files:
        print(f"  Transcribing: {os.path.basename(mp3_path)}")
        result = model.transcribe(mp3_path, language='en')
        text = result['text'].strip().lower()
        texts.append(text)
        print(f"    {len(text)} chars")

    return texts


IDX_TO_CHAR_LM = {0: '', 1: ' ', 28: "'"}
for _i, _c in enumerate('abcdefghijklmnopqrstuvwxyz'):
    IDX_TO_CHAR_LM[_i + 2] = _c


def extract_text_from_npz(npz_path):
    """Extract training text corpus from a prepared transcribe_data.npz file."""
    import numpy as np
    print(f"  Loading text labels from: {npz_path}")
    d = np.load(npz_path)
    texts = []
    text_arr  = d['text_train']
    tlen_arr  = d['text_len_train']
    for t, tl in zip(text_arr, tlen_arr):
        chars = [IDX_TO_CHAR_LM.get(int(c), '') for c in t[:int(tl)]]
        text = ''.join(chars).strip()
        if text:
            texts.append(text)
    print(f"  Extracted {len(texts):,} text samples")
    return texts


def main(args):
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f"\nCharacter-level {args.order}-gram LM Training")

    # Get training text — prefer npz > text_file > MP3 transcription
    if args.npz_file:
        print(f"Extracting text from npz: {args.npz_file}")
        texts = extract_text_from_npz(args.npz_file)
    elif args.text_file:
        print(f"Loading text from: {args.text_file}")
        with open(args.text_file, 'r') as f:
            texts = [line.strip().lower() for line in f if line.strip()]
    else:
        print(f"Transcribing MP3s from: {args.mp3_dir}")
        texts = transcribe_mp3s(args.mp3_dir, args.whisper_model)

        # Save transcriptions for reuse
        corpus_path = os.path.join(os.path.dirname(args.out), 'corpus.txt')
        with open(corpus_path, 'w') as f:
            for t in texts:
                f.write(t + '\n')
        print(f"  Corpus saved to: {corpus_path}")

    total_chars = sum(len(t) for t in texts)
    print(f"\nCorpus: {len(texts)} texts, {total_chars:,} total characters")

    # Train LM
    print(f"\nTraining {args.order}-gram LM (smoothing={args.smoothing})...")
    lm = CharNgramLM(order=args.order, smoothing=args.smoothing)
    lm.train(texts)

    # Save
    lm.save(args.out)

    # Quick test
    test_texts = [
        "alice was beginning",
        "down the rabbit hole",
        "zzzzz xxxxx",
    ]
    print(f"\nTest scores:")
    for t in test_texts:
        score = lm.score(t)
        print(f"  '{t}' -> {score:.2f}")

    print("\nDone.")


if __name__ == '__main__':
    _HERE = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--mp3_dir', default=MP3_DIR)
    parser.add_argument('--text_file', default=None,
                        help='Optional: path to text corpus file')
    parser.add_argument('--npz_file', default=None,
                        help='Optional: path to .npz from prepare_data.py to extract text labels')
    parser.add_argument('--out', default=os.path.join(_HERE, 'data', 'lm.pkl'))
    parser.add_argument('--order', type=int, default=4)
    parser.add_argument('--smoothing', type=float, default=0.01)
    parser.add_argument('--whisper_model', default='base.en')
    args = parser.parse_args()
    main(args)
