# Vera Bot — magicpin AI Challenge Submission

## Quick Start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
uvicorn bot:app --host 0.0.0.0 --port 8080
```

The bot is now live at `http://localhost:8080`.

## Deploy Publicly (Railway — free tier)

1. Push this folder to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Add `ANTHROPIC_API_KEY` as an environment variable
4. Railway auto-deploys; your URL will be `https://your-app.up.railway.app`

Alternatively use Render, Fly.io, or any VPS.

## Test with judge simulator

```bash
# Edit judge_simulator.py:
BOT_URL = "http://localhost:8080"
LLM_PROVIDER = "anthropic"
LLM_API_KEY = "your-key"

python judge_simulator.py
```

---

## Architecture

### 4-Context Composition Framework

Every Vera message = `compose(category, merchant, trigger, customer?)`

```
CategoryContext   ─►
MerchantContext   ─►  Claude (Sonnet 4)  ─►  WhatsApp message
TriggerContext    ─►
CustomerContext?  ─►
```

### Key Design Decisions

**1. Trigger-specific guidance**  
Each trigger kind gets bespoke instructions injected into the prompt:
- `research_digest` → lead with trial n, source, segment; offer to share
- `perf_spike` → celebrate; suggest capitalizing on momentum  
- `perf_dip` → name the gap vs peer median; suggest ONE specific fix
- `recall_due` → real slots, real price, hi-en mix CTA
- `dormant_with_vera` → re-engage with insight, not guilt

**2. Auto-reply detection**  
Reply composer checks for "Thank you for contacting..." patterns → `wait` action with 3600s backoff. Avoids burning turns on canned WhatsApp auto-replies.

**3. Intent-routing**  
On YES/go-ahead/do-it replies → action mode immediately. Composer knows to draft/deliver what was promised, not re-qualify.

**4. Suppression**  
`suppression_key` tracked in memory. Same key never sent twice in a session. Prevents duplicate nudges.

**5. Category voice enforcement**  
System prompt bakes in clinical/peer tone for dentists/doctors. Taboos ("guaranteed", "cure") listed explicitly. Generic discount language penalized.

### Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/context` | POST | Receive category/merchant/customer/trigger context pushes |
| `/v1/tick` | POST | Periodic wake-up; compose proactive messages |
| `/v1/reply` | POST | Handle merchant/customer replies |
| `/v1/healthz` | GET | Liveness probe |
| `/v1/metadata` | GET | Bot identity |
| `/v1/teardown` | POST | Wipe state at test end |

### Scoring targets

| Dimension | Approach |
|---|---|
| Specificity | Force-feed trial_n, CTR numbers, source citations via trigger guidance |
| Category Fit | Per-category voice in system prompt + taboo enforcement |
| Merchant Fit | Signals, offers, customer_aggregate injected in every composition |
| Decision Quality | Trigger-kind routing; suppression; expiry checks |
| Engagement | Compulsion levers: specificity, social proof, curiosity CTA |
