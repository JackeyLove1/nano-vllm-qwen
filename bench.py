import os
import time
from random import randint, seed
#from nanovllm import LLM, SamplingParams
from vllm import LLM, SamplingParams


def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    path = os.path.expanduser("./models/Qwen3-0.6B/")
    llm = LLM(
        path,
        enforce_eager=False,
        max_model_len=4096,
        gpu_memory_utilization=0.6,
    )

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()

# (nano-vllm) root@HackJacky:/mnt/c/Software/Codes/py/nano-vllm-qwen# python3 bench.py 
# `torch_dtype` is deprecated! Use `dtype` instead!
# Generating: 100%|█████████████████████████████████████████| 1/1 [00:04<00:00,  4.13s/it, Prefill=1tok/s, Decode=41tok/s]
# Total: 133966tok, Time: 190.42s, Throughput: 703.52tok/s

# vllm 0.16.0
# Adding requests: 100%|██████████████████████████████████████████████████| 1/1 [00:00<00:00, 310.80it/s]
# Processed prompts: 100%|█| 1/1 [00:48<00:00, 48.25s/it, est. speed input: 0.06 toks/s, output: 0.33 tok
# Total: 133966tok, Time: 194.62s, Throughput: 688.35tok/s