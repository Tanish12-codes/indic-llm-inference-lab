# CONCEPTS — explained simply

One short section per idea, in the order the project meets them. Each experiment ends
with three interview questions and answers you can say out loud.

---

## 1. Token
A **token** is the unit of text a language model actually reads — a "word-piece".
Think of LEGO: the model can't hold a whole sentence, it holds bricks. A common
English word like `the` is one brick; a rare word like `unbelievable` might be
three bricks: `un` + `believ` + `able`.
Everything an LLM costs — memory, time, money on an API — is counted **per token**,
not per word.

## 2. Tokenizer
The **tokenizer** is the machine that cuts text into tokens and gives each one an ID
number (`"the"` → 262). It is trained *before* the model by looking at lots of text
and gluing together the pieces that appear most often (this recipe is called
**BPE — Byte Pair Encoding**). So a tokenizer is good at the languages it saw a lot of.
- **Vocabulary** = the list of all bricks it knows (GPT-2: 50,257; Qwen2.5: ~151,000).
- **Byte-level**: GPT-2 and Qwen start from raw bytes, so they can *always* write any
  text — even if they have to spell a Hindi letter byte by byte.
- **Encode** = text → IDs. **Decode** = IDs → text. A good tokenizer gives back exactly
  the text you put in (`decode(encode(text)) == text`).

## 3. UTF-8 bytes (why Hindi starts at a disadvantage)
Computers store letters as bytes. English letters are 1 byte each. Every Devanagari
letter or matra (क, ि, ं …) is **3 bytes**. A byte-level tokenizer that never learned
Hindi merges may need up to 3 tokens for ONE Hindi character.

## 4. Fertility
**Fertility = tokens ÷ words.** "How many bricks does one word break into?"
English with an English tokenizer ≈ 1.3. If Hindi is 5, the same sentence costs about
4× more tokens — so 4× more memory for the KV cache, ~4× more decode steps, and on an
API, ~4× the bill. Its friend is **characters per token** (higher = each token carries
more text = cheaper).
*Caveat:* "word" = split on spaces, and Marathi packs more into one word than English,
so fertility alone exaggerates a little; tokens for the *same meaning* (parallel
sentences) is the fairest comparison, which is why we use FLORES+.

## 5. Inference
**Training** = teaching the model (changing its weights). **Inference** = *using* it:
weights are frozen and we only run the maths forward to get the next token.
This whole project is about inference cost.

## 6. Forward pass, logits, greedy decoding
One **forward pass** = push token IDs through the network once. Out come **logits**:
one score per vocabulary word for "what comes next". **Greedy decoding** = always pick
the highest score (argmax). It's deterministic, which makes it perfect for testing.

## 7. Prefill and decode
Generating text has two phases — like reading a question, then writing the answer:
- **Prefill**: read the whole prompt in ONE forward pass (all tokens in parallel).
  Fast per token, because the GPU gets a big chunk of work at once.
- **Decode**: write the answer **one token at a time**; each new token needs another
  forward pass. You can't parallelise this — token 5 depends on token 4.
"Time to first token" ≈ prefill; "tokens per second" ≈ decode.

## 8. Attention, Q/K/V
In each layer every token asks a **question** (Query), every earlier token holds up a
**label** (Key) and a **content card** (Value). The question is compared with all
labels; the best matches' content cards are blended together. A token may only look at
itself and earlier tokens (**causal mask**) — no peeking at the future.

## 9. KV cache
During decode, the Keys and Values of old tokens **never change**. Recomputing them for
every new token is like re-reading a whole book from page 1 each time you add a line.
The **KV cache** saves them: each step we compute K, V only for the newest token and
append it. Work per step drops from "whole sequence" to "one token".
Price: memory. Per token it costs
**2 (K and V) × layers × KV-heads × head_dim × bytes per number**, multiplied by tokens
and batch. More tokens (e.g. Hindi under a bad tokenizer) → bigger cache.
*Classic bug:* the new token's **position id = current cache length** (not 0, not the
step number from the prompt start minus one).

## 10. Preallocation
Appending with `torch.cat` builds a **brand-new, bigger** tensor every step and copies
everything into it — like buying a bigger bookshelf each time you buy a book.
**Preallocation** buys one shelf big enough for the max length up front and just puts
each new K/V into the next empty slot (in-place write). Less copying, no memory churn.

## 11. Quantization
Store numbers with fewer bits. FP32 = 4 bytes, FP16 = 2, **INT8 = 1**.
To squeeze a vector of decimals into integers −127…127 we keep one **scale** per vector:
`int8 = round(x / scale)`, `scale = max|x| / 127`; to use it, `x ≈ int8 × scale`
(**dequantize**). Like writing prices in "units of ₹0.37" instead of exact rupees:
much shorter, tiny rounding error. For the KV cache: ~4× smaller than FP32, ~2× smaller
than FP16, at the cost of small errors that *can* flip a later token.

## 12. MHA, GQA, MQA
Attention has several **heads** (parallel "question askers").
- **MHA (multi-head)**: every query head has its own K and V head. Biggest cache.
- **MQA (multi-query)**: all query heads share ONE K/V head. Smallest cache.
- **GQA (grouped-query)**: in between — e.g. 12 query heads share 4 K/V heads (groups of 3).
KV memory scales with the number of **KV heads**, so GQA/MQA shrink the cache 3×/12×.
That's why modern models (Qwen2.5-0.5B: 14 query heads, 2 KV heads) use GQA.

## 13. Memory-bound vs compute-bound
A GPU has two speeds: how fast it does maths (FLOP/s) and how fast it moves data from
memory (bandwidth, GB/s).
- **Compute-bound**: lots of maths per byte loaded (prefill — big matrix × big matrix).
- **Memory-bound**: little maths per byte (decode with batch 1 — every step must stream
  ALL weights and the whole KV cache from VRAM to do a tiny vector × matrix).
Decode is a chef who spends more time walking to the fridge than cooking.
So decode speed ≈ bytes moved ÷ bandwidth — and a smaller KV cache helps directly.

## 14. Kernel-launch overhead
Every GPU operation (a matmul, an add, a softmax) is a **kernel** that the CPU must
*launch*. Each launch costs a few microseconds no matter how small the job. GPT-2 does
hundreds of small kernels per token, so for short sequences the GPU is mostly
**waiting for the CPU to hand it work** — like a courier making 300 trips with one
envelope each. That's why the KV cache's speed-up looks small at 50 tokens and large
at 500: at short lengths both versions are paying the same fixed launch cost.

## 15. Weight memory vs KV memory
**Weights** = the model itself; fixed: parameters × bytes (0.5 B params × 2 bytes ≈ 1 GB
in FP16). **KV cache** = grows with tokens × batch. For long prompts or many users the
KV cache can become bigger than the weights. For **MoE** models (like Param2) all
experts must sit in memory even though only a few are *used* per token, so weight
memory follows *total* parameters while compute follows *active* parameters.

---

# Interview questions per experiment

## Experiment 01 — token counts

**Q1. Why does the same sentence cost more tokens in Hindi than in English?**
"Tokenizers learn their pieces from training text. GPT-2 saw mostly English, so it has whole-word
tokens for English but almost nothing for Devanagari. Each Hindi letter is 3 bytes in UTF-8 and GPT-2
often spends one token per byte or two. In my test, the same 200 sentences cost 7.3× more tokens in
Hindi than in English with GPT-2, 4.3× with Qwen2.5, and only 1.15× with Param2, which was built for
Indian languages."

**Q2. What is fertility, and what's wrong with it as a metric?**
"Fertility is tokens per word — how many pieces a word breaks into. The problem is that a 'word'
isn't the same unit in every language: Marathi glues suffixes onto words, so it has fewer, longer
words. Per word, Marathi looked much worse than Hindi; per sentence they cost almost the same.
For cost comparisons I use tokens for the same meaning, using parallel sentences like FLORES+."

**Q3. Can a better tokenizer for Indian languages hurt anything?**
"Yes — vocabulary is a fixed budget. Param2 cut Hindi tokens by 5× compared with Qwen2.5, but its
English got about 20 % more expensive than GPT-2's: 33.7 versus 27.1 tokens per sentence. Also, a
bigger vocabulary makes the embedding and output layers bigger. And decode(encode(text)) isn't
always identical: Qwen2.5 failed to round-trip 13.5 % of Hindi sentences. I traced all 27 failures to
its NFC Unicode normaliser: letters like फ़ stored as one code point get split into फ plus a nukta
sign. It looks identical on screen, but the bytes differ, so exact string matching breaks."

## Experiment 02 — GPT-2 from scratch

**Q1. What are the gotchas when loading Hugging Face GPT-2 weights into your own code?**
"Three things. HF stores the linear layers as Conv1D with weights shaped input-by-output, so you
transpose them for nn.Linear. GPT-2 uses the tanh approximation of GELU, not exact GELU. And there's
no separate output matrix — the output layer reuses the token-embedding matrix, which is called weight tying."

**Q2. Your logits differ from Hugging Face by 1e-4. How do you know it isn't a bug?**
"I measured the noise floor: Hugging Face's own model differs from itself between CPU and GPU by up
to 2.3e-4, and mine was always closer than that. At logits around 100, 1e-4 is about 15 float32 steps —
0.0001 percent. The predicted next token was identical at every position. A real bug like the wrong
GELU shows up around 1e-3 or much more."

**Q3. Why do GPU and CPU give slightly different answers for the same maths?**
"Floating-point addition isn't associative: (a+b)+c can differ from a+(b+c) in the last bits. GPU
kernels split a big sum into chunks and add them in a different order from the CPU. So you compare with
a tolerance chosen from a measured noise floor, never with exact equality."

## Experiment 03 — KV cache correctness

**Q1. What is a KV cache, and does it change the model's output?**
"During generation, the keys and values of earlier tokens never change, so we store them instead of
recomputing them every step. It changes only the amount of work, not the result: my cached and uncached
versions and Hugging Face's generate produced exactly the same 200 tokens."

**Q2. What's the classic KV-cache bug?**
"Giving the new token the wrong position id. With a cache you feed in only one token, so if you build
positions from zero, every new token thinks it's the first word. In my demo the output diverged at the
very first decode step and collapsed into repeated commas — only 4 percent of tokens matched."

**Q3. How would you test a KV-cache implementation?**
"Three checks: greedy output equal to the uncached version and to a reference library for a few
hundred tokens; the cached logits equal a full re-run at each step; and a deliberately broken version must
fail the test — otherwise the test can't catch the bug."

## Experiment 04 — speed and memory, cached vs uncached

**Q1. How much did the KV cache speed up generation, and why does the speed-up depend on length?**
"On GPT-2 small on my laptop GPU: no gain at 50 new tokens, 1.5 times at 200, 2.4 times at 500.
Without a cache, each step reprocesses the whole sequence, so the cost per token grows — from 8.7 to
30 milliseconds per token. With a cache it stays around 9 to 12. At short lengths both are dominated
by the same fixed overhead per step, so the cache can't show its advantage yet."

**Q2. What is kernel-launch overhead, and how did you see it?**
"Every GPU operation is a kernel the CPU has to launch, and each launch costs microseconds regardless of
size. GPT-2 launches a few hundred small kernels per token. At 50 tokens, uncached and cached both cost
about 9 milliseconds per token even though uncached does far more maths — so the time was launch and
Python overhead, not computation. Engines fix this with fused kernels, torch.compile or CUDA graphs."

**Q3. How do you calculate KV-cache memory, and did it match?**
"2 for K and V, times layers, times KV heads, times head dimension, times tokens, times batch, times
bytes per number. For GPT-2 in FP32 that's 72 KiB per token. My measured cache tensors matched the formula
exactly at every length — 36.14 MiB at 514 tokens. One detail: the cache holds prompt plus n minus one tokens,
because the last generated token is never fed back in."

## Experiment 06 — int8 KV cache

**Q1. How does int8 KV-cache quantization work?**
"For each key or value vector, store one scale — the largest absolute value divided by 127 — and store
the vector as integers from −127 to 127. Before attention, multiply back by the scale. With 64-number
vectors and an FP16 scale, that's 66 bytes instead of 256 in FP32."

**Q2. What did you gain and what did it cost?**
"Memory: exactly 74 percent less than FP32 and 48 percent less than FP16. Quality: across 5 prompts and
200 greedy tokens each, every token was identical to the FP32 cache, with about 0.6 percent reconstruction
error. But my version was 1.5 to 2.5 times slower, because it dequantizes the whole cache with separate
kernels every step."

**Q3. Then why do real systems use quantized KV caches?**
"Decode is memory-bound: each step must read the whole cache. If dequantization is fused into the attention
kernel, you read 2 to 4 times fewer bytes, so you get both less memory and more speed, and you can fit
longer contexts or more users. My experiment shows the memory half; the speed half needs a fused kernel."

_(more experiments are added as they run)_
