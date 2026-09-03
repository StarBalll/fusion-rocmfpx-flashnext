#!/usr/bin/env bash
# 融合构建服务器自检: 健康 / 模型信息 / MTP 回答质量
set -uo pipefail
PORT="${1:-8132}"
BASE="http://127.0.0.1:${PORT}"

echo "═══ 1) /health ═══"
curl -sf "$BASE/health" && echo || { echo "✗ health 未就绪"; exit 1; }

echo
echo "═══ 2) /v1/models (alias 确认) ═══"
curl -sf "$BASE/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print('  id:', d['data'][0]['id'])"

echo
echo "═══ 3) 回答质量自检 (非流式, 应答'17'且格式正常) ═══"
REQ_FILE=/tmp/verify-fusion-req.json
cat > "$REQ_FILE" <<'EOF'
{"messages":[{"role":"user","content":"9 + 8 等于多少? 只回答数字本身, 不要思考过程."}],
 "max_tokens":4096,"temperature":0.0,"chat_template_kwargs":{"enable_thinking":false}}
EOF
curl -s "$BASE/v1/chat/completions" -H "Content-Type: application/json" -d @"$REQ_FILE" > /tmp/verify-fusion-resp.json
python3 - <<'PY'
import json
d = json.load(open("/tmp/verify-fusion-resp.json"))
if "error" in d:
    print("  ✗ API 错误:", d["error"]); raise SystemExit(1)
msg = d["choices"][0]["message"]["content"]
t = d.get("timings", {})
print("  回答:", repr(msg[:120]))
print(f"  prompt_tokens={t.get('prompt_tokens')}  predicted_tokens={t.get('predicted_tokens')}")
print(f"  decode: {t.get('predicted_per_second',0):.1f} t/s | prefill: {t.get('prompt_per_second',0):.0f} t/s")
acc = None
if t.get("draft_n"):
    acc = t.get("draft_n_accepted",0)/t["draft_n"]
    print(f"  MTP 草稿: n={t['draft_n']} accepted={t.get('draft_n_accepted')} 接受率={acc:.2f}")
else:
    print("  MTP 草稿: (timings 无 draft 字段)")
ok_content = any(x in msg for x in ("17", "十七"))
print("  ✓ 内容检查通过" if ok_content else "  ✗ 内容异常 (期望 17)")
raise SystemExit(0 if ok_content else 2)
PY
