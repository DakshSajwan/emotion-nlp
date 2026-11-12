"""Preset sentence pool management and randomized sampling for emotion detection app.

Builds high-confidence curated sample pools from test predictions with rich
fallbacks and stateful queue sampling.
"""

from __future__ import annotations

import csv
from pathlib import Path
import random
import time
from typing import Any, MutableMapping

from src.config import ACTIVE_DISTILBERT_PREDS, LABELS, PREDICTIONS_DIR

# ---------------------------------------------------------------------------
# Fallback Pools (at least 15 clearly emotional, hand-written sentences per label)
# ---------------------------------------------------------------------------
FALLBACK_POOLS: dict[str, list[str]] = {
    "sadness": [
        "I feel completely hollow inside after hearing the heartbreaking news about my childhood friend today.",
        "Everything feels so heavy and pointless right now, and I cannot seem to stop crying.",
        "It hurts so much knowing that some relationships can never be repaired no matter how hard you try.",
        "I spent the entire weekend staring at old photographs feeling overwhelmed with grief and loneliness.",
        "Walking through my empty apartment reminds me of everything I have lost over the past year.",
        "I feel so discouraged and exhausted that getting out of bed feels like an impossible task.",
        "The persistent gloom outside perfectly mirrors the cold emptiness I have been carrying in my chest.",
        "I received another rejection letter today and it feels like all my hard work was completely wasted.",
        "Nobody seemed to notice that I was struggling silently in the corner of the crowded room.",
        "I feel crushed by the sudden disappointment after having my hopes raised so high last week.",
        "The quiet silence of this empty house makes the overwhelming sense of loss almost unbearable.",
        "I feel utterly defeated and cannot find any motivation to keep pushing forward right now.",
        "Saying goodbye at the train station was one of the saddest moments of my entire life.",
        "I feel so forgotten and isolated as if the world has moved on without me.",
        "A wave of profound sorrow washed over me when I visited our favorite old park bench.",
    ],
    "joy": [
        "I just found out that I got accepted into my dream university and I am absolutely thrilled!",
        "We won the championship game after months of relentless practice and the team is celebrating ecstatically!",
        "Waking up to bright warm sunshine and good news from family filled my heart with pure happiness.",
        "I finally finished paying off all my student loans and I feel completely liberated and overjoyed!",
        "Our research paper was accepted with outstanding reviewer praises and I could not be happier!",
        "Holding our healthy newborn baby in my arms for the first time was sheer magical bliss.",
        "Everything aligned wonderfully today and I am walking on air with a huge smile on my face.",
        "We laughed so hard until our stomachs ached during our reunion dinner with old college friends.",
        "I received a wonderful promotion and salary raise that recognized all my dedication and effort.",
        "Dancing barefoot in the gentle summer rain brought back the pure unadulterated joy of childhood.",
        "I am so immensely grateful for this beautiful life and all the kind people around me.",
        "The concert was an absolute triumph and singing along with thousands of fans felt truly exhilarating!",
        "Cooking a delicious feast with family on Sunday afternoon brought such warmth and delightful cheer.",
        "I finally conquered my fear of public speaking and delivered a presentation that received standing applause!",
        "Today was one of those rare, perfect days where everything felt wonderfully peaceful and bright.",
    ],
    "love": [
        "Looking into your eyes reminds me how blessed I am to share this journey with you.",
        "I cherish every gentle moment we spend together talking softly into the late evening hours.",
        "My grandmother held my hand tenderly and told me how proud she has always been of me.",
        "Being surrounded by true lifelong friends who accept you unconditionally is the greatest gift of all.",
        "I feel deeply affectionate and thankful for my partner who supports my dreams without hesitation.",
        "A warm embrace from someone you love can melt away even the most difficult hardships.",
        "I adore watching the sunset with you while sharing our deepest hopes for the future.",
        "The compassion and caring kindness shown by my mother has shaped everything good in my life.",
        "I am so endlessly in love with your sweet smile and your generous, beautiful soul.",
        "Holding hands on an autumn walk reminds me that true devotion grows deeper with every year.",
        "I feel so safe, treasured, and completely understood whenever I am sitting right next to you.",
        "Writing this heartfelt anniversary letter reminds me of how deeply our bond has flourished over time.",
        "Seeing my children play together with such genuine affection brings tears of tenderness to my eyes.",
        "Your sweet handwritten note tucked into my lunchbox made my entire afternoon feel filled with romance.",
        "True intimacy is knowing someone's vulnerabilities and loving them even more fiercely because of them.",
    ],
    "anger": [
        "I am furious that customer service lied to me and completely refused to refund my stolen package!",
        "It makes my blood boil when colleagues take credit for the grueling work that I completed alone.",
        "I am so disgusted by their blatant hypocrisy and arrogant refusal to admit their obvious mistakes.",
        "They vandalized public property without any remorse and it makes me absolutely livid with rage.",
        "I cannot stand how disrespectful and rude that supervisor was to the hardworking junior staff.",
        "Finding out that someone I trusted spread malicious rumors behind my back makes me intensely angry.",
        "The airline canceled our flight without warning and stranded hundreds of families with zero compensation.",
        "I am fed up with these greedy corporations cutting safety standards while racking up record profits.",
        "His arrogant condescending tone during the meeting pushed my patience past the breaking point today.",
        "It enrages me to see innocent people mistreated by people who abuse their authority.",
        "I worked eighty hours this week only to be insulted by an ungrateful, unreasonable client.",
        "They cheated openly on the final exam while the proctor pretended not to notice anything.",
        "I have never felt so betrayed and insulted by someone I considered a close personal friend.",
        "Their total negligence resulted in severe property damage and they still deny any legal responsibility.",
        "I am sick and tired of having my reasonable boundaries ignored and continually trampled upon.",
    ],
    "fear": [
        "I woke up in the dead of night hearing sudden strange scratching noises inside the bedroom wall.",
        "My hands were shaking uncontrollably and my throat went dry as the tornado sirens started wailing.",
        "I am terrified about the upcoming medical biopsy results and I cannot calm my racing heart.",
        "Driving on the icy mountain highway during a blinding blizzard was terrifying beyond all imagination.",
        "A sickening dread washed over me when I realized my young child had wandered off alone.",
        "I feel so anxious and petrified about losing my job in the upcoming corporate layoffs.",
        "Walking down the dark alleyway alone at midnight made my pulse race with sudden panic.",
        "The sudden violent turbulence caused the entire airplane to shudder and passengers screamed in fright.",
        "I froze in absolute horror when I saw the aggressive pitbull charging across the unfenced yard.",
        "An overwhelming sense of impending doom kept me awake staring into the darkness all night.",
        "I am so afraid that my chronic illness might worsen before doctors find an effective treatment.",
        "Standing at the edge of the tall crumbling cliff made my knees buckle with dizzying vertigo.",
        "Hearing footsteps slowly creeping up the wooden staircase gave me chills of cold creeping panic.",
        "I felt cornered and helpless when the masked intruder demanded all the store's cash at gunpoint.",
        "The eerie creaking sounds coming from the abandoned basement filled me with dreadful apprehension.",
    ],
    "surprise": [
        "I opened the front door to find all my distant relatives shouting happy birthday in unison!",
        "I was utterly stunned when the judge announced that our small underdog team won first prize!",
        "Never in my wildest dreams did I anticipate receiving an unexpected inheritance from an estranged aunt.",
        "I had no idea you were secretly planning a destination proposal in Paris for months!",
        "It was an astonishing revelation to discover that my quiet neighbor was once an Olympic gold medalist.",
        "I gasped in utter disbelief when the scratch lottery ticket revealed a ten thousand dollar jackpot!",
        "The sudden fireworks burst across the night sky in a dazzling array of unexpected colors.",
        "I was completely caught off guard by their sudden marriage announcement during Thanksgiving dinner.",
        "It came as a shocking surprise when the company announced a four-day workweek for everyone.",
        "I could not believe my eyes when our lost cat strolled back home after six long months!",
        "The magician made a massive grand piano vanish before our eyes in an astonishing fraction of a second!",
        "I was speechless with pleasant shock when they awarded me employee of the year today.",
        "Finding an authentic vintage record signed by the artist in a thrift shop was totally mind-blowing!",
        "I never expected the quietest student in class to deliver the most captivating, electrifying speech.",
        "The movie's plot twist at the very end left the entire theater audience gasping in amazement.",
    ],
}


def build_pools(csv_path: Path | str | None = None) -> dict[str, list[str]]:
    """Build emotion sentence pools from test predictions with fallback safety.

    Reads predictions from ACTIVE_DISTILBERT_PREDS (or specified csv_path), falling
    back to distilbert/baseline prediction files or curated fallback pools if missing.
    For each label in config.LABELS:
      - keeps rows where true_label == pred_label
      - confidence >= 0.90
      - text has between 6 and 40 words
      - drops duplicates

    If the CSV is missing or a label has fewer than 15 sentences, FALLBACK_POOLS
    is used to ensure at least 15 valid samples per class.

    Args:
        csv_path: Optional path to prediction CSV. Defaults to ACTIVE_DISTILBERT_PREDS.

    Returns:
        dict mapping label -> list of unique text strings.
    """
    if csv_path is None:
        target_path = ACTIVE_DISTILBERT_PREDS
        if not target_path.exists():
            v1_path = PREDICTIONS_DIR / "distilbert_test_preds.csv"
            base_path = PREDICTIONS_DIR / "baseline_test_preds.csv"
            if v1_path.exists():
                target_path = v1_path
            elif base_path.exists():
                target_path = base_path
    else:
        target_path = Path(csv_path)

    pools: dict[str, list[str]] = {label: [] for label in LABELS}

    if not target_path.exists():
        # Fallback when file is missing
        return {label: list(FALLBACK_POOLS[label]) for label in LABELS}

    seen_texts: dict[str, set[str]] = {label: set() for label in LABELS}

    try:
        with open(target_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                true_label = row.get("true_label", "").strip()
                pred_label = row.get("pred_label", "").strip()
                text = row.get("text", "").strip()
                conf_str = row.get("confidence", "0.0")

                if true_label not in pools or true_label != pred_label:
                    continue

                try:
                    conf = float(conf_str)
                except (ValueError, TypeError):
                    continue

                if conf < 0.90:
                    continue

                words = len(text.split())
                if not (6 <= words <= 40):
                    continue

                if text not in seen_texts[true_label]:
                    seen_texts[true_label].add(text)
                    pools[true_label].append(text)
    except Exception:
        # On any I/O or parsing failure, return fallbacks
        return {label: list(FALLBACK_POOLS[label]) for label in LABELS}

    # Ensure every label has at least 15 sentences using fallbacks if needed
    for label in LABELS:
        if len(pools[label]) < 15:
            for fallback_sample in FALLBACK_POOLS.get(label, []):
                if fallback_sample not in seen_texts[label]:
                    seen_texts[label].add(fallback_sample)
                    pools[label].append(fallback_sample)

    return pools


def next_sample(
    label: str,
    state: MutableMapping[str, Any] | dict[str, Any],
    pools: dict[str, list[str]] | None = None,
) -> str:
    """Return the next unique sentence from a shuffled queue for the specified emotion.

    Uses a shuffled queue stored in state[f"queue_{label}"]. When the queue is
    empty, rebuilds it by shuffling the full pool, ensuring the first item of
    the new queue is not the last sentence shown.

    Args:
        label: Emotion label string (e.g. 'joy', 'sadness').
        state: Mutable mapping (such as st.session_state or dict) tracking session queues.
        pools: Optional pre-built pools dictionary. If None, calls build_pools().

    Returns:
        The next sentence string from the emotion pool.
    """
    if pools is None:
        pools = build_pools()

    if label not in pools or not pools[label]:
        # Fallback to FALLBACK_POOLS if label is unknown or pool is empty
        pool = list(FALLBACK_POOLS.get(label, ["I am feeling emotional today."]))
    else:
        pool = list(pools[label])

    queue_key = f"queue_{label}"
    last_key = f"last_{label}"

    queue: list[str] = state.get(queue_key, [])

    if not queue:
        # Rebuild queue by shuffling the full pool
        new_queue = list(pool)
        rng = random.Random(time.time_ns())
        rng.shuffle(new_queue)

        last_shown = state.get(last_key)
        # Ensure the first item of the new queue is not the last sentence that was shown
        if len(new_queue) > 1 and last_shown is not None and new_queue[0] == last_shown:
            swap_idx = rng.randint(1, len(new_queue) - 1)
            new_queue[0], new_queue[swap_idx] = new_queue[swap_idx], new_queue[0]

        queue = new_queue

    sample = queue.pop(0)
    state[queue_key] = queue
    state[last_key] = sample
    return sample
