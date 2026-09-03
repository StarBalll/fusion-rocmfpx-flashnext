#!/usr/bin/env python3
# 长 prompt prefill 微基准: 测 prompt_per_second (首 token 延迟主导)
# 用法: prefill-bench.py <port> <rounds>
# 每轮生成内容唯一的 ~N-token 文本, 避免 KV 缓存命中导致测量失真
import json, random, sys, time, urllib.request

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8132
rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 2

WORDS = ["kernel","buffer","stream","pipeline","tensor","cache","quantize","embed",
         "attention","moe","router","hipcub","radix","scan","prefill","decode",
         "register","wavefront","barrier","fence","slice","stride","reduce"]

def make_prompt(n_lines):
    rng = random.Random(f"rocmfpx-fusion-{port}-{n_lines}-{rounds}-{time.time()}")
    lines = []
    for i in range(n_lines):
        ws = " ".join(rng.choice(WORDS) for _ in range(10))
        lines.append(f"{i:05d}: {ws}")
    body = "\n".join(lines)
    return (f"以下是一份系统日志摘录, 共{n_lines}行. 请仔细阅读全部内容, "
            f"然后只回答: 最后一行的编号是多少? 不要输出其他内容.\n\n{body}\n\n最后一行的编号是:")

def run(tag, n_lines):
    prompt = make_prompt(n_lines)
    req = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    r = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=req, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(r, timeout=600) as resp:
        d = json.load(resp)
    wall = time.time() - t0
    t = d.get("timings", {})
    pt = t.get("prompt_tokens") or d.get("usage", {}).get("prompt_tokens", 0)
    pps = t.get("prompt_per_second", 0)
    print(f"  [{tag}] prompt_tokens={pt:6d} | prefill {pps:8.0f} t/s | 首 token 墙钟 {wall:6.2f}s")
    return pt, pps

print(f"══ prefill 微基准 :{port} ({rounds} 轮 × 2 档长度) ══")
for n_lines in (600, 1200):  # ≈12k / ≈23.5k tokens (19.6 tokens/line, 须 < 32k ctx)
    for i in range(rounds):
        run(f"{'12k' if n_lines==600 else '23k'} r{i+1}", n_lines)
