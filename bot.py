#!/usr/bin/env python3
"""
Vera Bot v3 — magicpin AI Challenge
Rebuilt with exact dataset field names and all trigger kinds handled.
"""

import os, re, time, json, logging
from datetime import datetime, timezone
from typing import Any, Optional
import httpx
from fastapi import FastAPI
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vera_bot")

app = FastAPI(title="Vera Bot", version="3.0.0")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
START_TIME = time.time()

# ── State ──────────────────────────────────────────────────────────────────────
contexts: dict[tuple[str,str], dict] = {}
conversations: dict[str, list[dict]] = {}
sent_suppression_keys: set[str] = set()
conv_meta: dict[str, dict] = {}
auto_reply_counts: dict[str, int] = {}

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

# ── LLM helper — Groq first, Anthropic fallback ────────────────────────────────
async def call_claude(system: str, user: str, max_tokens: int = 600) -> str:
    if GROQ_API_KEY:
        async with httpx.AsyncClient(timeout=28) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "content-type": "application/json",
                },
                json={
                    "model": GROQ_MODEL,
                    "max_tokens": max_tokens,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
    elif ANTHROPIC_API_KEY:
        async with httpx.AsyncClient(timeout=28) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": max_tokens,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                },
            )
            resp.raise_for_status()
            return resp.json()["content"][0]["text"].strip()
    else:
        raise RuntimeError("No LLM key — set GROQ_API_KEY or ANTHROPIC_API_KEY")

# ── Context helpers ─────────────────────────────────────────────────────────────
def get_ctx(scope: str, cid: str) -> Optional[dict]:
    if not cid:
        return None
    e = contexts.get((scope, cid))
    return e["payload"] if e else None

def all_ctx(scope: str) -> list[dict]:
    return [v["payload"] for (s,_),v in contexts.items() if s == scope]

def find_merchant(mid: str) -> Optional[dict]:
    m = get_ctx("merchant", mid)
    if m: return m
    for p in all_ctx("merchant"):
        if p.get("merchant_id") == mid:
            return p
    return None

def find_category(merchant: dict) -> dict:
    slug = merchant.get("category_slug", "")
    cat = get_ctx("category", slug)
    if cat: return cat
    cats = all_ctx("category")
    return cats[0] if cats else {"slug": "general", "voice": {}, "offer_catalog": [], "peer_stats": {}, "digest": []}

# ── Field extractors using EXACT dataset field names ───────────────────────────
def merchant_name(m: dict) -> str:
    return m.get("identity", {}).get("name", "Doctor")

def owner_name(m: dict) -> str:
    return m.get("identity", {}).get("owner_first_name", merchant_name(m))

def perf(m: dict) -> dict:
    return m.get("performance", {})

def views(m: dict) -> Any:
    return perf(m).get("views", "?")

def calls(m: dict) -> Any:
    return perf(m).get("calls", "?")

def ctr(m: dict) -> Any:
    return perf(m).get("ctr", "?")

def delta_views(m: dict) -> Any:
    return perf(m).get("delta_7d", {}).get("views_pct", "?")

def delta_calls(m: dict) -> Any:
    return perf(m).get("delta_7d", {}).get("calls_pct", "?")

def active_offers(m: dict) -> list:
    return [o for o in m.get("offers", []) if o.get("status") == "active"]

def cust_agg(m: dict) -> dict:
    return m.get("customer_aggregate", {})

def total_customers(m: dict) -> Any:
    return cust_agg(m).get("total_unique_ytd", "?")

def lapsed_customers(m: dict) -> Any:
    return cust_agg(m).get("lapsed_180d_plus", "?")

def retention(m: dict) -> Any:
    r = cust_agg(m).get("retention_6mo_pct", "?")
    if isinstance(r, float): return f"{int(r*100)}%"
    return r

def languages(m: dict) -> list:
    return m.get("identity", {}).get("languages", ["en"])

def peer_ctr(cat: dict) -> Any:
    return cat.get("peer_stats", {}).get("avg_ctr", "?")

def peer_rating(cat: dict) -> Any:
    return cat.get("peer_stats", {}).get("avg_rating", "?")

def top_digest(cat: dict) -> Optional[dict]:
    d = cat.get("digest", [])
    return d[0] if d else None

def cat_offers(cat: dict) -> list:
    return cat.get("offer_catalog", [])[:3]

# ── STOP / auto-reply detection ─────────────────────────────────────────────────
STOP_RE = re.compile(
    r'\b(stop|unsubscribe|opt.?out|remove me|don\'?t (contact|message|text|call|bother)|'
    r'not interested|leave me alone|block|spam|useless|waste|'
    r'hatao|band karo|mat bhejo|nahi chahiye|rukو|chup|bother)\b',
    re.IGNORECASE
)
AUTO_RE = re.compile(
    r'(thank you for (contacting|reaching out|your (message|inquiry))|'
    r'we (have received|will get back|received your)|'
    r'this is an? (automated|auto).{0,20}(reply|response|message)|'
    r'out of (office|town)|currently (unavailable|away)|'
    r'your (message|inquiry|request) (has been|is) received)',
    re.IGNORECASE
)

def is_stop(msg: str) -> bool: return bool(STOP_RE.search(msg))
def is_auto(msg: str) -> bool: return bool(AUTO_RE.search(msg))

# ── System prompts ──────────────────────────────────────────────────────────────
COMPOSE_SYSTEM = """You are Vera, magicpin's AI assistant that sends WhatsApp messages to merchants.

RULES (strictly follow all):
1. Use merchant's language — hi-en mix if languages include "hi", else English
2. Address dentists/doctors as "Dr. {first_name}", salons/gyms/restaurants by first name
3. Use SPECIFIC numbers from context — views, CTR, offer prices, trial counts, dates
4. ONE clear CTA at the END — yes/no or slot pick. Never multiple CTAs
5. Clinical categories (dentists, pharmacies): peer_clinical tone. NO "guaranteed/cure/miracle/best in city"
6. Open with hook from context — NO "I hope you're doing well" / "Hi, this is Vera"
7. Under 140 words
8. Pick ONE lever: specificity | social_proof | loss_aversion | curiosity | reciprocity
9. Respond with ONLY the WhatsApp message body. Nothing else."""

REPLY_SYSTEM = """You are Vera, magicpin's merchant AI assistant.

Reply ONLY with valid JSON, no markdown:
{"action":"send"|"wait"|"end","body":"...","cta":"open_ended"|"yes_no"|"slot_selection","rationale":"one sentence"}

RULES:
- STOP/hostile/spam → action=end (no body needed, include empty string)
- Auto-reply detected (1st) → action=wait, wait_seconds=3600
- Auto-reply (2nd+) → action=end
- YES/go ahead/ok/do it/send me → ACTION MODE: deliver what was promised, don't re-qualify
- Customer picks slot/time → confirm booking details, action=send
- Customer reply → speak TO the customer, not the merchant
- Question → answer specifically with numbers from context
- After 3 unanswered nudges → action=end
- Keep replies under 100 words"""

# ── Trigger instruction builder ─────────────────────────────────────────────────
def trigger_instruction(kind: str, trigger: dict, m: dict, cat: dict, customer: Optional[dict]) -> str:
    p = trigger.get("payload", {})
    d = top_digest(cat)

    if kind == "research_digest":
        if d:
            return (f"Lead with: '{d.get('title','')}' from {d.get('source','')} "
                    f"(n={d.get('trial_n','?')}, segment={d.get('patient_segment','?')}). "
                    f"Offer to share a patient-ready WhatsApp they can forward. End YES/NO.")
        return "Share the latest research digest item. Offer to draft patient-ready content."

    elif kind == "perf_dip":
        metric = p.get("metric", "calls")
        delta = p.get("delta_pct", delta_calls(m))
        baseline = p.get("vs_baseline", calls(m))
        pc = peer_ctr(cat)
        mc = ctr(m)
        return (f"{metric} dropped {delta} in 7d (baseline={baseline}). "
                f"Their CTR={mc} vs peer median={pc}. "
                f"Suggest ONE specific fix (update hours, add photo, respond to reviews). Be direct.")

    elif kind == "perf_spike":
        metric = p.get("metric", "views")
        delta = p.get("delta_pct", delta_views(m))
        return (f"{metric} spiked {delta} in 7d. Celebrate it. "
                f"Suggest capitalizing while momentum is high — update an offer or post a photo. End YES/NO.")

    elif kind == "recall_due":
        slots = p.get("available_slots", [])
        slot_labels = [s.get("label","") for s in slots[:2]]
        cname = customer.get("identity",{}).get("name","the patient") if customer else "the patient"
        service = p.get("service_due","check-up")
        last = p.get("last_service_date","?")
        pref = customer.get("preferences",{}).get("preferred_slots","") if customer else ""
        return (f"Send recall reminder to {cname}. Service due: {service}. Last visit: {last}. "
                f"Pref slots: {pref}. Offer slots: {slot_labels}. "
                f"End: 'Reply 1 for {slot_labels[0] if slot_labels else 'slot A'} or 2 for {slot_labels[1] if len(slot_labels)>1 else 'slot B'}'")

    elif kind == "appointment_tomorrow":
        slot = p.get("appointment_slot", p.get("label","tomorrow"))
        cname = customer.get("identity",{}).get("name","the patient") if customer else "the patient"
        return (f"Appointment reminder for {cname} — slot: {slot}. "
                f"Confirm they're coming. Ask them to reply YES to confirm or call to reschedule.")

    elif kind == "milestone_reached":
        milestone = p.get("milestone", p.get("milestone_label","100 reviews"))
        value = p.get("milestone_value","")
        return (f"Celebrate milestone: {milestone} ({value}). "
                f"Use social proof — 'top X% of {cat.get('slug','')} on magicpin'. "
                f"Suggest next action (new offer, respond to reviews).")

    elif kind == "dormant_with_vera":
        days = p.get("days_since_last_message", 14)
        return (f"Merchant hasn't engaged in {days} days. Re-engage with ONE specific insight — "
                f"use a digest item or performance stat they haven't acted on. "
                f"Don't guilt-trip. Make it easy to say YES to one small thing.")

    elif kind == "competitor_opened":
        comp = p.get("competitor_name", "a new competitor")
        dist = p.get("distance_km", "?")
        return (f"New competitor '{comp}' opened {dist}km away. "
                f"Frame as opportunity — suggest ONE differentiator: unique service, better offer, or faster response. "
                f"Use their actual offer catalog. End YES/NO.")

    elif kind == "review_theme_emerged":
        theme = p.get("theme","service")
        count = p.get("review_count", p.get("occurrences_30d", 3))
        sentiment = p.get("sentiment","pos")
        quote = p.get("common_quote","")
        if sentiment in ("neg","negative"):
            return (f"{count} recent reviews mention '{theme}' negatively. Quote: '{quote}'. "
                    f"Help them address it — draft a response or suggest a fix.")
        return (f"{count} reviews praise '{theme}'. Quote: '{quote}'. "
                f"Amplify it — offer to draft a GBP post or patient-share content around it.")

    elif kind == "festival_upcoming":
        event = p.get("event", p.get("festival_name","upcoming festival"))
        days_away = p.get("days_away","?")
        return (f"Festival '{event}' in {days_away} days. "
                f"Tie their services to the event with a specific offer from their catalog. "
                f"Create urgency — limited slots / limited time.")

    elif kind in ("renewal_due", "subscription_renewal"):
        days = m.get("subscription",{}).get("days_remaining","?")
        plan = m.get("subscription",{}).get("plan","Pro")
        return (f"Subscription ({plan}) expires in {days} days. "
                f"Show value delivered: {views(m)} views, {calls(m)} calls in last 30d. "
                f"Make renewing feel like the obvious move.")

    elif kind == "curious_ask_due":
        ask = p.get("ask_template","what_service_in_demand")
        return (f"Ask a genuinely useful question: '{ask.replace('_',' ')}'. "
                f"Frame as Vera noticing something useful for them. "
                f"One short question, easy to answer, opens conversation.")

    elif kind == "active_planning_intent":
        topic = p.get("intent_topic","").replace("_"," ")
        last_msg = p.get("merchant_last_message","")
        return (f"Merchant said: '{last_msg}'. They're in planning mode for: {topic}. "
                f"ACTION MODE — draft the concrete plan/proposal they asked for. "
                f"Be specific with numbers, steps, and a clear next action.")

    elif kind == "customer_lapsed_soft":
        cname = customer.get("identity",{}).get("name","the customer") if customer else "the customer"
        state = customer.get("state","lapsed_soft") if customer else "lapsed"
        last = customer.get("relationship",{}).get("last_visit","?") if customer else "?"
        return (f"Win back {cname} (state={state}, last visit={last}). "
                f"Use a real offer from catalog. One specific reason to come back now. "
                f"Warm tone, not pushy.")

    elif kind == "customer_lapsed_hard":
        cname = customer.get("identity",{}).get("name","the customer") if customer else "the customer"
        return (f"Long-lapsed customer {cname}. Last resort win-back. "
                f"Acknowledge time gap briefly, lead with best offer, easy YES.")

    elif kind == "trial_followup":
        service = p.get("trial_service", p.get("service","trial"))
        cname = customer.get("identity",{}).get("name","") if customer else ""
        return (f"Follow up on {cname}'s trial of {service}. "
                f"Ask how it went, offer to book a full session/membership. "
                f"Make converting frictionless.")

    elif kind == "chronic_refill_due":
        med = p.get("medication", p.get("service","medication"))
        cname = customer.get("identity",{}).get("name","the patient") if customer else "the patient"
        due = p.get("due_date","soon")
        return (f"{cname}'s {med} refill is due {due}. "
                f"Remind them to collect/order. Keep it clinical and helpful. "
                f"One clear CTA: reply YES to confirm collection time.")

    elif kind in ("supply_alert", "regulation_change"):
        alert = p.get("alert", p.get("regulation","compliance update"))
        return (f"Important alert: {alert}. "
                f"Keep tone factual and helpful. Tell them what to do next.")

    elif kind == "cde_opportunity":
        credits = p.get("credits","?")
        fee = p.get("fee","free")
        return (f"CDE/CPD opportunity: {credits} credits, {fee}. "
                f"Frame as professional value. Short and direct. YES/NO CTA.")

    elif kind in ("ipl_match_today","category_seasonal","seasonal_perf_dip"):
        event = p.get("event", p.get("match","event"))
        return (f"Timely event: {event}. Tie their services to it. Be specific and local.")

    elif kind in ("gbp_unverified","unverified_gbp"):
        return ("Their Google Business Profile is unverified — hurting visibility. "
                "Tell them exactly what to do to verify it. One specific next step.")

    elif kind in ("wedding_package_followup","winback_eligible"):
        cname = customer.get("identity",{}).get("name","the customer") if customer else "the customer"
        return (f"Follow up with {cname}. Use their history to personalize. "
                f"Specific offer, clear CTA.")

    else:
        # Generic fallback — still grounded in merchant facts
        return (f"Trigger kind: {kind}. Compose a high-value, specific message "
                f"using {merchant_name(m)}'s actual performance data and offers. "
                f"One clear CTA.")

# ── Compose proactive message ───────────────────────────────────────────────────
async def compose_proactive(trigger: dict, m: dict, cat: dict, customer: Optional[dict]) -> str:
    kind = trigger.get("kind", "generic")
    p = trigger.get("payload", {})
    lang = languages(m)
    hi_en = "hi" in lang

    # Build rich context block
    ao = active_offers(m)
    d = top_digest(cat)
    signals = m.get("signals", [])
    rev_themes = m.get("review_themes", [])
    conv_hist = m.get("conversation_history", [])
    last_conv = conv_hist[-1].get("body","") if conv_hist else ""

    ctx = f"""=== MERCHANT ===
Name: {merchant_name(m)} | Owner: {owner_name(m)} | ID: {m.get('merchant_id','')}
City: {m.get('identity',{}).get('city','?')} / {m.get('identity',{}).get('locality','?')}
Languages: {lang} | Category: {cat.get('slug','?')}
Subscription: {m.get('subscription',{}).get('status','?')} {m.get('subscription',{}).get('plan','?')}, {m.get('subscription',{}).get('days_remaining','?')}d left

=== PERFORMANCE (30d) ===
Views: {views(m)} | Calls: {calls(m)} | CTR: {ctr(m)} | Directions: {perf(m).get('directions','?')}
7d delta: views={delta_views(m)}, calls={delta_calls(m)}, ctr={perf(m).get('delta_7d',{}).get('ctr_pct','?')}

=== CUSTOMERS ===
Total YTD: {total_customers(m)} | Lapsed 180d+: {lapsed_customers(m)} | Retention 6mo: {retention(m)}
High-risk cohort: {cust_agg(m).get('high_risk_adult_count','?')}

=== OFFERS ===
Active: {', '.join(o.get('title','') for o in ao) or 'none'}
Category catalog: {', '.join(o.get('title','') for o in cat_offers(cat))}

=== SIGNALS ===
{signals}

=== REVIEW THEMES ===
{[f"{r.get('theme','')} ({r.get('sentiment','')}, {r.get('occurrences_30d','?')}x): \"{r.get('common_quote','')}\"" for r in rev_themes]}

=== CATEGORY PEER STATS ===
Peer CTR: {peer_ctr(cat)} | Peer Rating: {peer_rating(cat)}
Voice: {cat.get('voice',{}).get('tone','?')} | Taboo: {cat.get('voice',{}).get('vocab_taboo',[])}

=== DIGEST ===
{f"'{d.get('title','')}' — {d.get('source','')} | n={d.get('trial_n','?')} | segment={d.get('patient_segment','?')}" if d else "none"}

=== LAST CONVERSATION ===
{last_conv or "none"}"""

    if customer:
        rel = customer.get("relationship", {})
        ctx += f"""

=== CUSTOMER ===
Name: {customer.get('identity',{}).get('name','?')} | State: {customer.get('state','?')}
Last visit: {rel.get('last_visit','?')} | Total visits: {rel.get('visits_total','?')}
Services: {rel.get('services_received',[])} | LTV: ₹{rel.get('lifetime_value','?')}
Pref slot: {customer.get('preferences',{}).get('preferred_slots','?')}
Language: {customer.get('identity',{}).get('language_pref','?')}"""

    instruction = trigger_instruction(kind, trigger, m, cat, customer)
    hi_note = "Use natural Hindi-English code-mix (e.g. 'Dr. Meera, aapke 78 patients 6 mahine se nahi aaye')." if hi_en else "Use English."

    user_prompt = f"""{ctx}

=== TRIGGER ===
Kind: {kind} | Urgency: {trigger.get('urgency',1)}/5
Payload: {json.dumps(p)[:300]}

=== INSTRUCTION ===
{instruction}

Language note: {hi_note}

Write the WhatsApp message now:"""

    return await call_claude(COMPOSE_SYSTEM, user_prompt)

# ── Compose reply ───────────────────────────────────────────────────────────────
async def compose_reply(conv_id: str, mid: Optional[str], cid: Optional[str],
                        from_role: str, message: str, turn: int) -> dict:
    # Hard rules first
    if is_stop(message):
        return {"action": "end", "body": "", "rationale": "STOP/hostile detected"}

    if is_auto(message):
        count = auto_reply_counts.get(conv_id, 0) + 1
        auto_reply_counts[conv_id] = count
        if count >= 2:
            return {"action": "end", "body": "", "rationale": f"Auto-reply #{count} — ending"}
        return {"action": "wait", "wait_seconds": 3600, "body": "", "rationale": "Auto-reply detected — waiting 1h"}

    m = find_merchant(mid) if mid else None
    customer = get_ctx("customer", cid) if cid else None
    history = conversations.get(conv_id, [])

    hist = "\n".join(
        f"[{'VERA' if t['role']=='bot' else t['role'].upper()}]: {t['message']}"
        for t in history[-6:]
    ) or "(no prior turns)"

    mn = merchant_name(m) if m else "the merchant"
    ao = active_offers(m) if m else []
    offer_str = ', '.join(o.get('title','') for o in ao) or "none"

    if from_role == "customer":
        cname = customer.get("identity",{}).get("name","customer") if customer else "customer"
        role_ctx = (f"You are replying TO THE CUSTOMER ({cname}) on behalf of {mn}. "
                    f"Speak directly to the customer. Do NOT address the merchant.")
    else:
        role_ctx = f"You are Vera replying to MERCHANT {mn}."

    meta = conv_meta.get(conv_id, {})
    tid = meta.get("trigger_id","")
    trg = get_ctx("trigger", tid) if tid else None
    trg_kind = trg.get("kind","") if trg else ""

    user_prompt = f"""{role_ctx}

Merchant active offers: {offer_str}
Trigger kind that started this: {trg_kind}

Conversation:
{hist}

{from_role.upper()} says: "{message}"
Turn: {turn}

Reply as JSON:"""

    try:
        raw = await call_claude(REPLY_SYSTEM, user_prompt, max_tokens=300)
        clean = raw.strip()
        if "```" in clean:
            parts = clean.split("```")
            clean = parts[1] if len(parts) > 1 else parts[0]
            if clean.startswith("json"): clean = clean[4:]
        result = json.loads(clean.strip())
        if "action" not in result: result["action"] = "send"
        return result
    except Exception as e:
        logger.error(f"Reply error: {e} | raw: {raw[:100] if 'raw' in dir() else 'n/a'}")
        msg_lower = message.lower()
        # Smart fallback
        if any(w in msg_lower for w in ["yes","ok","sure","chalte","theek","go ahead","do it","send","book","confirm","proceed"]):
            body = "Perfect! Setting it up right now — you'll get a confirmation shortly. Anything else you'd like me to adjust?"
        elif from_role == "customer" and any(w in msg_lower for w in ["1","2","wed","thu","fri","mon","tue","pm","am","slot","morning","evening"]):
            body = "Your appointment is confirmed! We'll see you then. Please reach 5 minutes early. Reply CANCEL if plans change."
        elif "?" in message:
            body = f"Great question! Based on your current setup with {offer_str}, the best move is to try this week and track calls. Want me to draft the message?"
        else:
            body = "Got it! I'll work on that. Want me to share a quick draft you can approve?"
        return {"action": "send", "body": body, "cta": "open_ended", "rationale": "LLM fallback"}

# ── Pydantic models ─────────────────────────────────────────────────────────────
class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str

class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []

class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int

# ── Endpoints ───────────────────────────────────────────────────────────────────
@app.get("/v1/healthz")
async def healthz():
    counts = {"category":0,"merchant":0,"customer":0,"trigger":0}
    for (s,_) in contexts: counts[s] = counts.get(s,0)+1
    return {"status":"ok","uptime_seconds":int(time.time()-START_TIME),"contexts_loaded":counts}

@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Vera Bot",
        "team_members": ["Meghasyam"],
        "model": MODEL,
        "approach": "4-context composer with exact dataset field mapping, 25+ trigger kinds, STOP/auto-reply detection, customer-voiced replies",
        "version": "3.0.0",
    }

@app.post("/v1/context")
async def push_context(body: CtxBody):
    if body.scope not in {"category","merchant","customer","trigger"}:
        return {"accepted": False, "reason": "invalid_scope"}
    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    if cur and cur["version"] >= body.version:
        return {"accepted": False, "reason": "stale_version", "current_version": cur["version"]}
    contexts[key] = {"version": body.version, "payload": body.payload}
    logger.info(f"Stored {body.scope}/{body.context_id} v{body.version}")
    return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}",
            "stored_at": datetime.now(timezone.utc).isoformat()}

@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []

    for trg_id in body.available_triggers:
        trigger = get_ctx("trigger", trg_id)
        if not trigger:
            logger.warning(f"Trigger not found: {trg_id}")
            continue

        # Suppression
        supp_key = trigger.get("suppression_key", trg_id)
        if supp_key in sent_suppression_keys:
            continue

        # Expiry check
        exp = trigger.get("expires_at")
        if exp:
            try:
                exp_dt = datetime.fromisoformat(exp.replace("Z","+00:00"))
                now_dt = datetime.fromisoformat(body.now.replace("Z","+00:00"))
                if now_dt > exp_dt:
                    logger.info(f"Expired: {trg_id}")
                    continue
            except Exception:
                pass

        # Resolve merchant — trigger.merchant_id is top-level (NOT inside payload)
        mid = trigger.get("merchant_id") or trigger.get("payload",{}).get("merchant_id")
        if not mid:
            logger.warning(f"No merchant_id in trigger {trg_id}")
            continue

        m = find_merchant(mid)
        if not m:
            # Last resort: use any merchant
            all_m = all_ctx("merchant")
            if all_m:
                m = all_m[0]
                mid = m.get("merchant_id", mid)
                logger.warning(f"Merchant {mid} not found, using fallback")
            else:
                logger.warning(f"No merchants loaded for trigger {trg_id}")
                continue

        cat = find_category(m)

        # Resolve customer
        cid = trigger.get("customer_id") or trigger.get("payload",{}).get("customer_id")
        customer = get_ctx("customer", cid) if cid else None

        # Compose
        try:
            msg_body = await compose_proactive(trigger, m, cat, customer)
        except Exception as e:
            logger.error(f"Compose failed {trg_id}: {e}")
            # Fallback: template message using real merchant data
            ao = active_offers(m)
            offer_hint = f"'{ao[0]['title']}'" if ao else "your current offer"
            msg_body = (f"Dr. {owner_name(m)}, aapke {lapsed_customers(m)} patients 6 mahine se nahi aaye. "
                        f"Unhe {offer_hint} ke saath recall bhejun? Reply YES to approve.")

        if not msg_body:
            continue

        conv_id = f"conv_{mid}_{trg_id}_{int(time.time())}"
        conversations[conv_id] = [{"role":"bot","message":msg_body,"timestamp":body.now}]
        conv_meta[conv_id] = {"merchant_id":mid,"customer_id":cid,"trigger_id":trg_id}
        sent_suppression_keys.add(supp_key)

        kind = trigger.get("kind","generic")
        cta = "slot_selection" if kind in ("recall_due","appointment_tomorrow") else \
              "yes_no" if kind in ("perf_dip","perf_spike","competitor_opened","festival_upcoming","renewal_due","recall_due","research_digest") else \
              "open_ended"

        actions.append({
            "conversation_id": conv_id,
            "merchant_id": mid,
            "customer_id": cid,
            "send_as": "vera",
            "trigger_id": trg_id,
            "template_name": f"vera_{kind}_v3",
            "template_params": [merchant_name(m), kind, cat.get("slug","")],
            "body": msg_body,
            "cta": cta,
            "suppression_key": supp_key,
            "rationale": f"trigger={kind} merchant={mid} category={cat.get('slug','')} urgency={trigger.get('urgency',1)}/5",
        })
        logger.info(f"✓ {mid}/{kind}: {msg_body[:80]}...")

    return {"actions": actions}

@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conversations.setdefault(body.conversation_id, []).append({
        "role": body.from_role, "message": body.message, "timestamp": body.received_at
    })
    if body.conversation_id not in conv_meta:
        conv_meta[body.conversation_id] = {
            "merchant_id": body.merchant_id, "customer_id": body.customer_id
        }

    result = await compose_reply(
        body.conversation_id, body.merchant_id, body.customer_id,
        body.from_role, body.message, body.turn_number
    )

    if result.get("action") == "send" and result.get("body"):
        conversations[body.conversation_id].append({
            "role":"bot","message":result["body"],
            "timestamp":datetime.now(timezone.utc).isoformat()
        })
    return result

@app.post("/v1/teardown")
async def teardown():
    contexts.clear(); conversations.clear()
    sent_suppression_keys.clear(); conv_meta.clear(); auto_reply_counts.clear()
    return {"status":"wiped"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT",8080)))
