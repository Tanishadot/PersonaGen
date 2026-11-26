"""
Build backend/data/empathetic_dialogues.json from the EmpatheticDialogues dataset.

Usage:
    python prepare_dataset.py                      # downloads the archive
    python prepare_dataset.py path/to/empatheticdialogues.tar.gz

Each record is one (speaker utterance -> listener reply) pair:
    {"situation", "prompt", "response", "emotion", "fine_emotion"}

`fine_emotion` is the dataset's own label (one of 32, e.g. "lonely").
`emotion` maps it onto the 7 labels the RoBERTa classifier predicts
(anger, disgust, fear, joy, neutral, sadness, surprise), so retrieval can
filter by the emotion detected in the user's message.
"""

import csv
import io
import json
import os
import sys
import tarfile
import urllib.request

DATASET_URL = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"
HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(HERE, "data", "empathetic_dialogues.json")

FINE_TO_COARSE = {
    # anger
    "angry": "anger", "annoyed": "anger", "furious": "anger", "jealous": "anger",
    # disgust
    "disgusted": "disgust",
    # fear
    "afraid": "fear", "terrified": "fear", "anxious": "fear", "apprehensive": "fear",
    # sadness
    "sad": "sadness", "lonely": "sadness", "devastated": "sadness", "disappointed": "sadness",
    "guilty": "sadness", "ashamed": "sadness", "embarrassed": "sadness",
    "nostalgic": "sadness", "sentimental": "sadness",
    # joy
    "excited": "joy", "proud": "joy", "grateful": "joy", "confident": "joy", "hopeful": "joy",
    "joyful": "joy", "content": "joy", "caring": "joy", "trusting": "joy", "faithful": "joy",
    "impressed": "joy", "prepared": "joy", "anticipating": "joy",
    # surprise
    "surprised": "surprise",
}


def clean(text: str) -> str:
    return text.replace("_comma_", ",").strip()


def read_rows(archive_path: str):
    with tarfile.open(archive_path, "r:gz") as tar:
        for split in ("train", "valid", "test"):
            member = tar.extractfile(f"empatheticdialogues/{split}.csv")
            text = io.TextIOWrapper(member, encoding="utf-8")
            # Some rows contain stray commas in the trailing "tags" column,
            # so read positionally and only use the first 6 fields.
            reader = csv.reader(text)
            next(reader)  # header
            for row in reader:
                if len(row) >= 6:
                    yield row


def build_pairs(rows):
    conversations = {}
    for conv_id, utt_idx, context, situation, _speaker, utterance, *_ in rows:
        try:
            idx = int(utt_idx)
        except ValueError:
            continue
        conv = conversations.setdefault(conv_id, {"context": context, "situation": situation, "turns": {}})
        conv["turns"][idx] = utterance

    pairs = []
    for conv in conversations.values():
        fine = conv["context"].strip()
        coarse = FINE_TO_COARSE.get(fine)
        if coarse is None:
            continue
        turns = conv["turns"]
        # Odd turns are the person describing their feelings, even turns the
        # empathetic listener. Pair each speaker turn with the listener reply.
        for idx in sorted(turns):
            if idx % 2 == 1 and idx + 1 in turns:
                prompt, response = clean(turns[idx]), clean(turns[idx + 1])
                if prompt and response:
                    pairs.append({
                        "situation": clean(conv["situation"]),
                        "prompt": prompt,
                        "response": response,
                        "emotion": coarse,
                        "fine_emotion": fine,
                    })
    return pairs


def main():
    if len(sys.argv) > 1:
        archive = sys.argv[1]
    else:
        archive = os.path.join(HERE, "data", "empatheticdialogues.tar.gz")
        if not os.path.exists(archive):
            os.makedirs(os.path.dirname(archive), exist_ok=True)
            print(f"Downloading {DATASET_URL} ...")
            urllib.request.urlretrieve(DATASET_URL, archive)

    pairs = build_pairs(read_rows(archive))
    if len(pairs) < 1000:
        raise SystemExit(f"Only {len(pairs)} pairs parsed - archive looks wrong")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False)

    counts = {}
    for p in pairs:
        counts[p["emotion"]] = counts.get(p["emotion"], 0) + 1
    print(f"Wrote {len(pairs)} pairs to {OUTPUT_PATH}")
    for emotion, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {emotion:9s} {n}")


if __name__ == "__main__":
    main()
