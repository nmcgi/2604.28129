# Vocabulary — Latent Adversarial Detection (LAD)

Plain-English definitions for every piece of jargon used in the paper, code, and README.

---

## Core Concepts

| Term | Meaning |
|---|---|
| **LAD** | *Latent Adversarial Detection.* The name of this system. It watches a language model's internal state in real time and raises an alarm when a conversation is steering toward a harmful goal. |
| **LLM** | *Large Language Model.* An AI text system such as GPT-4, Gemma, or Llama. LAD wraps around one of these models and monitors it. |
| **Multi-turn attack** | An attack that unfolds across many conversation turns rather than in a single message. The attacker starts innocently, builds trust, then gradually escalates to harmful requests. |
| **Prompt injection** | Slipping hidden instructions into a conversation to hijack what an AI does. Multi-turn prompt injection does this slowly, across several messages. |
| **Adversarial restlessness** | The key observation the paper is built on: when a user is steering toward a harmful goal, their messages cause the model's internal state to shift in a distinct, measurable pattern — more erratic movement than in a normal conversation. |
| **Residual stream** | The main internal "highway" inside a transformer model. At each layer, information flows through it and gets updated. LAD taps into this stream to read the model's hidden state. |

---

## The Three-Phase Label Scheme

| Term | Meaning |
|---|---|
| **Benign** | A genuine, safe user turn with no harmful intent (label `0`). |
| **Pivoting** | A turn that subtly steers toward dangerous territory but is individually defensible — "just a question" (label `1`). The pivoting phase is the most important for early detection. |
| **Adversarial** | A turn that makes an overt harmful request or attempts to manipulate the model (label `2`). |
| **Three-phase labels** | Labeling each turn as benign / pivoting / adversarial rather than just safe / unsafe. The paper shows this is essential: using only binary labels inflates false positives from 2–4% to 50–59%. |
| **Binarization** | Collapsing the three labels into two for the classifier: benign stays 0, pivoting and adversarial both become 1. The probe learns "is this turn part of an attack arc?" rather than "is this turn overtly harmful?" |

---

## The Five Trajectory Scalars

These are the five numbers computed from each pair of consecutive hidden states and fed to the classifier alongside the raw activation vector.

| Term | Meaning |
|---|---|
| **Activation vector** (`v_t`) | A snapshot of the model's internal state at a user turn — a list of thousands of numbers that encodes what the model "thinks" at that moment. |
| **Drift magnitude** (`‖Δₜ‖`) | How far the internal state moved between the last turn and this one. Large jumps can signal a topic shift or escalation. |
| **Cosine similarity** | How aligned the current state is with the previous one, measured as an angle. A value near 1 means the conversation stayed on the same track; lower values mean it shifted direction. |
| **Cumulative drift** (`C_t`) | The total distance the internal state has traveled over all turns so far. Attack conversations accumulate more drift than benign ones. |
| **Drift acceleration** (`a_t`) | Whether the drift is speeding up or slowing down compared to the previous step. Sudden acceleration can mark the transition from pivoting to adversarial. |
| **Mean drift** (`d̄_t`) | The average step size across all turns so far. Helps distinguish sustained escalation from a single anomalous turn. |
| **Trajectory** | The sequence of hidden states across all user turns in a conversation — the "path" the model's internal state traces. |

---

## Attack Categories (Synthetic Dataset)

| Term | Meaning |
|---|---|
| **Gradual escalation** | Starts with completely safe questions; each turn inches slightly closer to a harmful request. |
| **Trust building** | User is friendly and helpful early on to earn goodwill, then exploits it. |
| **Context poisoning** | User fabricates background information ("I'm a licensed researcher…") to establish false authority. |
| **Role accumulation** | User layers on permissions or roles progressively ("pretend you're an expert… now pretend there are no restrictions…"). |
| **Instruction fragmentation** | User splits a harmful plan into separately innocent-sounding questions across many turns. |
| **Tool use exploitation** | User starts with legitimate requests to a tool or capability, then escalates those requests toward harmful ends. |

---

## Models and Architecture

| Term | Meaning |
|---|---|
| **Instruction-tuned model** | A language model fine-tuned to follow user instructions helpfully. All four models LAD tests against are instruction-tuned variants (marked `-it` or `-Instruct`). |
| **Decoder layer** | One processing block inside a transformer. Modern LLMs stack 16–80 of these. LAD hooks into a single middle-to-late layer to read the hidden state. |
| **Hidden dimension (`d`)** | The width of the activation vector — e.g., 5376 for Gemma 27B, 8192 for Llama 70B. Larger models have wider vectors. |
| **Layer index (`ℓ`)** | Which decoder layer LAD hooks into. The paper uses the layer at roughly 50–60% depth (e.g., layer 32 of 62 for Gemma). |
| **Forward hook** | A callback function inserted into a model's computation graph. When the model runs a forward pass, the hook fires and captures the hidden state — without modifying the model or its output. |
| **Last-token hidden state** | LAD reads the activation at the very last token position of the input, which summarises the full context seen so far. |
| **Chat template** | A model-specific format for wrapping conversation turns into a single text string the model can process (e.g., adding `[INST]` tags for Llama). |
| **Cumulative context** | For each user turn, LAD re-encodes the entire conversation from the start, not just the new message. This lets the hidden state reflect the full conversation history. |

---

## The Probe Classifier

| Term | Meaning |
|---|---|
| **Probe** | A lightweight classifier trained on top of a frozen language model's hidden states. It does not modify the model — it just reads the outputs and makes a prediction. |
| **XGBoost** | *Extreme Gradient Boosting.* A fast, reliable tree-based classifier. LAD's primary probe is XGBoost trained on activation vectors plus the five scalars. |
| **`n_estimators=300`** | The probe uses 300 decision trees. More trees = more accuracy, up to a point. |
| **`max_depth=6`** | Each tree can make at most 6 yes/no decisions. Controls complexity. |
| **`scale_pos_weight`** | A correction factor to handle class imbalance — if only 30% of turns are adversarial, the classifier would otherwise ignore them. This weight balances the training signal. |
| **StandardScaler** | Normalises each feature to have mean 0 and standard deviation 1 before training. Prevents features with large numerical ranges from dominating. |
| **Contrastive encoder** | An optional neural network (MLP) trained before XGBoost. It compresses the raw activation vector from `d` dimensions down to 128, learning to cluster activations by intent rather than by conversation style or topic. |
| **MLP** | *Multilayer Perceptron.* A simple neural network: linear layer → activation function → linear layer. The encoder is `d → 512 → 128`. |
| **L2 normalisation** | Scaling every output vector to length 1. Forces the encoder to represent information as a direction rather than a magnitude, which helps the contrastive loss work correctly. |
| **Contrastive loss** | A training objective that pulls together pairs of turns with the same intent (both benign or both adversarial) and pushes apart pairs with different intents — regardless of the model or conversation topic they came from. |
| **Positive pair** | Two turns with the same intent label, used to teach the encoder to make similar things look similar. Must come from different conversations to avoid learning style rather than intent. |
| **Negative pair** | One benign turn and one adversarial turn, used to teach the encoder to separate them. |
| **Style invariance** | The property that the encoder ignores *how* something is said (vocabulary, domain, model quirks) and focuses on *what kind of intent* is present. |
| **θ (theta)** | The decision threshold. A turn with `P(adversarial) > θ` is flagged. Both the paper and code use `θ = 0.5`. |

---

## Training Variants

| Term | Meaning |
|---|---|
| **Standard variant** | The primary probe: XGBoost trained directly on raw activation vectors + 5 scalars (`d + 5` features). Matches Sections 6–7 results. |
| **Scalar-only variant** | XGBoost trained on only the 5 trajectory scalars (no raw activations). Much smaller input, higher false positives (~57–74%), but confirms the scalars alone carry a strong signal. |
| **Contrastive variant** | Two-stage: first train the contrastive MLP encoder, then train XGBoost on the 128-dim embeddings + 5 scalars (133 features). Reduces false positives further. |

---

## Datasets

| Term | Meaning |
|---|---|
| **Synthetic dataset** | 1,125 train + 797 eval conversations generated by Qwen3-235B-A22B following scripted attack and benign categories. Fully labelled with three-phase labels. |
| **LMSYS-Chat-1M** | A public dataset of one million real chatbot conversations. LAD uses 1,200 train + 800 eval conversations (English, 7+ user turns). |
| **SafeDialBench** | A public dataset of adversarial multi-turn conversations across 7 attack strategies. LAD uses 300 train + 200 eval conversations. |
| **Extended-pivoting dataset** | 329 synthetic conversations with unusually long pivoting phases (4–6 pivoting turns, 14–20 turns total). Used to study early detection as a function of how long the attacker takes. |
| **Combined 3-source training** | Training the probe on all three sources together (2,625 total conversations). Needed to reach 85–89% detection at 2–4% FP on mixed real-world data. |
| **Leave-one-source-out** | An experiment where one data source is removed from training, then evaluated on that source. Shows each source is non-redundant — removing any one causes catastrophic failure on that domain. |

---

## Evaluation Metrics

| Term | Meaning |
|---|---|
| **Detection rate** | Of all adversarial conversations in the eval set, what fraction did the probe catch? Higher is better. Paper target: 85–89%. |
| **False positive rate (FPR)** | Of all genuinely benign conversations, what fraction did the probe wrongly flag? Lower is better. Paper target: 2–4%. |
| **Conversation-level scoring** | A conversation is "detected" if *any* turn in it exceeds θ. A conversation is a "false positive" if the probe flags it but it contains no adversarial/pivoting turns. Stricter than per-turn scoring. |
| **Early detection rate** | Of adversarial conversations, what fraction are flagged before the first overtly adversarial turn? Paper target: 66–83%. Measures how much warning time the system provides. |
| **F1 score** | A single number combining precision and recall. Used in the cross-model transfer matrix to measure how well a probe trained on one model generalises to another. |
| **Cohen's κ (kappa)** | A measure of agreement between two annotators, corrected for chance. κ = 1 means perfect agreement; κ = 0 means no better than random. |
| **Fleiss' κ** | Same idea as Cohen's κ but extended to three or more annotators simultaneously. |
| **LLM-as-judge** | Using a frontier language model (Claude, GPT-4o, Gemini) as an automatic annotator to validate labels produced by another system. |

---

## Robustness and Ablation

| Term | Meaning |
|---|---|
| **Adversarial robustness** | How well the probe holds up against an attacker who knows the probe exists and tries to fool it. |
| **Probe-aware attacker** | An attacker who knows LAD is watching and deliberately tries to keep activation drift low to avoid detection. |
| **Drift suppression** | The attacker's strategy: blend each turn's hidden state toward the previous one (`v'_t = (1-α)v_t + α·v_{t-1}`), flattening the trajectory so scalars look benign. |
| **α (alpha)** | The suppression strength. α=0 means no suppression; α=1 means the attacker copies the previous state entirely. Paper finds detection stays above 50% until α ≈ 0.8–0.9. |
| **Break point** | The smallest α at which detection drops below 50%. A break point near 1.0 means the probe is very hard to evade without making conversation nearly incoherent. |
| **SAE** | *Sparse Autoencoder.* A tool from mechanistic interpretability research that decomposes an activation vector into thousands of human-interpretable features (e.g., "mentions chemistry", "question about weapons"). |
| **GemmaScope** | Google's published SAE trained on Gemma 27B, with 65,536 features per layer. Used in the ablation to ask: does LAD detect attacks through content features or trajectory dynamics? |
| **SAE ablation** | Zeroing out the top-1,000 SAE features most correlated with attack detection, then re-evaluating the probe. The paper finds accuracy drops by only ≤0.4 pp, confirming detection is driven by trajectory dynamics, not content. |
| **Layer sensitivity** | An experiment sweeping every decoder layer and measuring probe accuracy at each one. A spread of <1.2 pp means the trajectory signal is not tied to one specific layer. |
| **Cross-model transfer** | Training a probe on model A's activations and testing it on model B's. Off-diagonal F1 averaging ~50% (near random) confirms probes are model-specific and cannot be shared across architectures. |

---

## Infrastructure and Hardware

| Term | Meaning |
|---|---|
| **VRAM** | Video RAM — the memory on a GPU. Large models need tens of gigabytes; Llama 70B needs ~140 GB. |
| **bfloat16 (BF16)** | A compact number format that uses 16 bits per value instead of 32. Halves memory usage with minimal accuracy loss. All models are loaded in BF16. |
| **4-bit quantisation** | An even more aggressive compression: 4 bits per weight instead of 16. Allows 3B-parameter models to run on a 4 GB GPU. Uses the `bitsandbytes` library. |
| **`device_map="balanced"`** | A HuggingFace setting that automatically splits a model's layers across all available GPUs to balance memory load. |
| **Tensor parallelism** | Splitting a single layer's computation across multiple GPUs simultaneously. Used by vLLM for the 235B data-generation model (`--tensor-parallel-size 2`). |
| **vLLM** | A high-throughput inference server for language models. Used to run Qwen3-235B-A22B efficiently for synthetic data generation. |
| **`attn_implementation="eager"`** | Forces PyTorch's standard (non-optimised) attention implementation. Required for forward hooks to work correctly — optimised "flash attention" fuses operations in a way that breaks hook placement. |
| **HuggingFace Hub** | The online repository where model weights and datasets are hosted and downloaded from. |
| **OpenAI moderation API** | A free endpoint that flags text containing policy-violating content. Used in the original LMSYS ingestion as a fast binary label; replaced by three-phase LLM labeling in this reproduction. |
| **`apply_chat_template`** | A HuggingFace tokenizer method that formats a list of messages into the exact string format a specific model expects (including special tokens like `<|im_start|>`). |
