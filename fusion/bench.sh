#!/usr/bin/env bash
# =============================================================
# 融合构建 vs qwen38fn2 (drluoto) 的对比基准
# 前提: 两个模型服务都在跑
#   qwen38fn2 (drluoto)      → http://127.0.0.1:8131
#   融合构建 (ROCmFPX+drluoto) → http://127.0.0.1:8132
# 注意防 OOM: 两个 74GB 模型不能同时驻留!
#   方法: 测完一个立刻 stop, 再 start 另一个
# =============================================================
set -euo pipefail

PROMPT_FILE="/tmp/fn2-prompt.json"
if [ ! -f "$PROMPT_FILE" ]; then
  python3 - <<'PY'
import json
d = {"messages":[{"role":"user","content":
"请审查这段 Rust 异步缓存代码的并发 bug(锁内 await、compute 应放锁外、多个任务同时计算同一 key 的竞态), 输出修复后的完整代码:\n\nuse std::sync::{Arc, Mutex};\nuse std::collections::HashMap;\nstruct Cache { data: Mutex<HashMap<String,String>> }\nimpl Cache {\n  fn new() -> Self { Self{data:Mutex::new(HashMap::new())} }\n  async fn get_or_insert(&self, key:String, compute: impl FnOnce()->String + Send + 'static) -> String {\n    let mut g = self.data.lock().unwrap();\n    if let Some(v)=g.get(&key){return v.clone();}\n    let v=compute(); g.insert(key.clone(),v.clone()); v\n  }\n}\n"}],
"max_tokens":256,"temperature":0.0,"chat_template_kwargs":{"enable_thinking":false}}
json.dump(d, open("/tmp/fn2-prompt.json","w"))
PY
fi

bench_one() {
  local name="$1" port="$2"
  echo "--- $name (:${port}) ---"
  if ! curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
    echo "  ✗ 服务未启动, 跳过"
    return
  fi
  for i in 1 2; do
    curl -s "http://127.0.0.1:${port}/v1/chat/completions" \
      -H "Content-Type: application/json" -d @"$PROMPT_FILE" \
      | python3 -c "
import sys,json
d=json.load(sys.stdin); t=d.get('timings',{})
acc = t.get('draft_n_accepted',0)/max(t.get('draft_n',1),1) if t.get('draft_n') else None
acc_s = f'{acc:.2f}' if acc is not None else '-'
print(f\"  解码 {t.get('predicted_per_second',0):6.1f} t/s | 接受率 {acc_s} | prompt {t.get('prompt_per_second',0):6.0f} t/s\")"
  done
}

echo "════════ 融合构建 vs drluoto ════════"
echo
echo "■ 冷启轮: 先测 8132(融合), 再测 8131(drluoto)"
bench_one "融合构建" 8132
bench_one "drluoto" 8131
echo
echo "■ 热启轮 (KV 缓存命中)"
bench_one "融合构建" 8132
bench_one "drluoto" 8131

echo
echo "提示: 两个 74GB 模型若无法同时驻留 (防 OOM 铁律), 请:"
echo "  测完一个 → ./run-model.sh stop qwen38fn2 → 启动另一个 → 重跑本脚本"
