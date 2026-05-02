#!/usr/bin/env python3
"""
Vera Bot — magicpin AI Challenge Submission
============================================
A production-quality merchant AI assistant implementing the 4-context composition framework.

Endpoints:
  POST /v1/context   — receive context pushes (category, merchant, customer, trigger)
  POST /v1/tick      — periodic wake-up; compose proactive messages
  POST /v1/reply     — handle merchant/customer replies
  GET  /v1/healthz   — liveness probe
  GET  /v1/metadata  — bot identity
"""

import os
import time
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Optional
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vera_bot")

app = FastAPI(title="Vera Bot", version="1.0.0")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
START_TIME = time.time()

# ─────────────────────────────────────────────
# In-memory state
# ─────────────────────────────────────────────
contexts: dict[tuple[str, str], dict] = {}          # (scope, context_id) -> {version, payload}
conversations: dict[str, list[dict]] = {}            # conv_id -> [turns]
sent_suppression_keys: set[str] = set()             # prevent duplicate sends
conv_meta: dict[str, dict] = {}                      # conv_id -> {merchant_id, trigger_id, ...}

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-20250514"

# ─────────────────────────────────────────────
# Claude composition helper
# ─────────────────────────────────────────────

async def call_claude(system: str, user: str, max_tokens: int = 600) -> str:
    """Call Claude API with the given system + user prompt."""
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    async with httpx.AsyncClient(timeout=25) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": MODEL,
                "max_tokens": max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"].strip()


# ─────────────────────────────────────────────
# Context helpers
# ─────────────────────────────────────────────

def get_ctx(scope: str, context_id: str) -> Optional[dict]:
    entry = contexts.get((scope, context_id))
    return entry["payload"] if entry else None


def format_context_for_prompt(category: dict, merchant: dict, trigger: dict, customer: Optional[dict] = None) -> str:
    """Serialize all 4 contexts into a structured prompt block."""
    lines = []

    # ── Category ──
    cat_slug = category.get("slug", "unknown")
    voice = category.get("voice", {})
    peer = category.get("peer_stats", {})
    digest = category.get("digest", [])
    seasonal = category.get("seasonal_beats", [])
    trends = category.get("trend_signals", [])
    offers_cat = category.get("offer_catalog", [])
    patient_content = category.get("patient_content_library", [])

    lines.append(f"=== CATEGORY CONTEXT: {cat_slug} ===")
    lines.append(f"Voice/Tone: {voice.get('tone','neutral')} | Taboos: {voice.get('taboos',[])} | Allowed vocab: {voice.get('vocab_allowed',[])}")
    lines.append(f"Peer stats: avg_rating={peer.get('avg_rating','?')}, avg_reviews={peer.get('avg_reviews','?')}, avg_ctr={peer.get('avg_ctr','?')}, scope={peer.get('scope','?')}")
    if offers_cat:
        titles = [o.get("title","") for o in offers_cat[:5]]
        lines.append(f"Canonical offers: {', '.join(titles)}")
    if digest:
        lines.append(f"Latest digest ({len(digest)} items):")
        for d in digest[:3]:
            src = d.get("source", "")
            lines.append(f"  • [{d.get('id','')}] {d.get('title','')} — {src} | n={d.get('trial_n','?')} | segment: {d.get('patient_segment','?')}")
    if seasonal:
        lines.append(f"Seasonal beats: {'; '.join(s.get('note','') + ' (' + s.get('month_range','') + ')' for s in seasonal[:3])}")
    if trends:
        lines.append(f"Trend signals: {'; '.join(t.get('query','') + ' +' + str(int(t.get('delta_yoy',0)*100)) + '% YoY' for t in trends[:3])}")
    if patient_content:
        lines.append(f"Patient content items available: {len(patient_content)} (can offer to draft/share)")

    # ── Merchant ──
    identity = merchant.get("identity", {})
    sub = merchant.get("subscription", {})
    perf = merchant.get("performance", {})
    offers_m = merchant.get("offers", [])
    conv_hist = merchant.get("conversation_history", {})
    cust_agg = merchant.get("customer_aggregate", {})
    signals = merchant.get("signals", [])

    lines.append(f"\n=== MERCHANT CONTEXT ===")
    lines.append(f"Name: {identity.get('name','?')} | ID: {merchant.get('merchant_id','?')}")
    lines.append(f"Location: {identity.get('locality','?')}, {identity.get('city','?')} | Verified: {identity.get('verified','?')}")
    lines.append(f"Languages: {identity.get('languages',['en'])}")
    lines.append(f"Subscription: {sub.get('status','?')} | Plan: {sub.get('plan','?')} | Days remaining: {sub.get('days_remaining','?')}")
    lines.append(f"Performance (30d): views={perf.get('views_30d','?')}, calls={perf.get('calls_30d','?')}, ctr={perf.get('ctr_30d','?')}, directions={perf.get('directions_30d','?')}")
    lines.append(f"  7d delta: views={perf.get('views_7d_delta','?')}, calls={perf.get('calls_7d_delta','?')}")
    if offers_m:
        active = [o for o in offers_m if o.get("status") == "active"]
        lines.append(f"Active offers: {', '.join(o.get('title','?') for o in active[:3]) or 'none'}")
    if cust_agg:
        lines.append(f"Customer aggregate: active={cust_agg.get('active_count','?')}, lapsed={cust_agg.get('lapsed_count','?')}, retention_6mo={cust_agg.get('retention_6mo','?')}")
    if signals:
        sig_list = signals if isinstance(signals, list) else []
        lines.append(f"Signals: {', '.join(str(s) for s in sig_list[:6])}")
    recent_turns = conv_hist.get("turns", []) if isinstance(conv_hist, dict) else []
    if recent_turns:
        last_turn = recent_turns[-1] if recent_turns else {}
        lines.append(f"Last interaction: {last_turn.get('timestamp','?')} — role={last_turn.get('role','?')} tag={last_turn.get('tag','?')}")

    # ── Trigger ──
    payload = trigger.get("payload", {})
    lines.append(f"\n=== TRIGGER CONTEXT ===")
    lines.append(f"Kind: {trigger.get('kind','?')} | Scope: {trigger.get('scope','?')} | Urgency: {trigger.get('urgency','?')}/5 | Source: {trigger.get('source','?')}")
    lines.append(f"Payload: {json.dumps(payload, ensure_ascii=False)[:400]}")
    lines.append(f"Suppression key: {trigger.get('suppression_key','?')}")

    # ── Customer (optional) ──
    if customer:
        cust_id = customer.get("customer_id", "?")
        c_ident = customer.get("identity", {})
        rel = customer.get("relationship", {})
        lines.append(f"\n=== CUSTOMER CONTEXT ===")
        lines.append(f"Customer: {c_ident.get('name','?')} | Phone: {c_ident.get('phone','?')}")
        lines.append(f"Languages: {c_ident.get('languages',['en'])} | Preferences: {c_ident.get('preferences',{})}")
        lines.append(f"Relationship: last_visit={rel.get('last_visit_date','?')}, visits_ytd={rel.get('visits_ytd','?')}, lapse_state={rel.get('lapse_state','?')}, avg_spend={rel.get('avg_spend_inr','?')}")
        lines.append(f"Upcoming: {customer.get('upcoming',{})}")

    return "\n".join(lines)


SYSTEM_PROMPT = """You are Vera, magicpin's AI assistant that sends WhatsApp messages to merchants and their customers on behalf of merchants.

Your job: given 4 context layers (category, merchant, trigger, customer?), compose ONE high-quality, concise WhatsApp message.

Rules:
1. Match the merchant's language preference (hi-en code-mix for Hindi-preferring merchants; English otherwise)
2. Use service+price offers ("Dental Cleaning @ ₹299") NOT generic discounts ("Flat 20% off")
3. One clear CTA at the END (not buried in middle)
4. For clinical categories (dentists, doctors, lawyers): use peer/clinical tone, NO hype words, NO "guaranteed" or "cure"
5. Use SPECIFIC numbers, dates, source citations — never vague
6. Open with the merchant's name or directly relevant hook — NO "I hope you're doing well" openers
7. Keep it under 160 words
8. Use ONE compulsion lever: specificity, loss-aversion, social proof, curiosity, or reciprocity
9. NEVER re-introduce yourself after the first message
10. Respond ONLY with the message body. No explanation. No quotes around the output.

Anti-patterns to avoid:
- Multiple CTAs ("Reply YES for X, NO for Y")
- Generic "AMAZING DEAL!" language for clinical verticals
- Hallucinated data (only cite what's in the context)
- Long preambles
- Sending same message verbatim again"""


async def compose_proactive(category: dict, merchant: dict, trigger: dict, customer: Optional[dict] = None) -> dict:
    """Use Claude to compose a proactive outbound message."""
    context_block = format_context_for_prompt(category, merchant, trigger, customer)

    trigger_kind = trigger.get("kind", "generic")
    merchant_name = merchant.get("identity", {}).get("name", "Merchant")
    category_slug = category.get("slug", "")
    peer_ctr = category.get("peer_stats", {}).get("avg_ctr", 0)
    merch_ctr = merchant.get("performance", {}).get("ctr_30d", 0)

    user_prompt = f"""Compose a WhatsApp message for this context:

{context_block}

Trigger kind: {trigger_kind}
Merchant name: {merchant_name}
Category: {category_slug}

Additional guidance based on trigger:
{_trigger_guidance(trigger_kind, trigger, merchant, category, customer)}

Respond with ONLY the message body."""

    body = await call_claude(SYSTEM_PROMPT, user_prompt)
    cta = _determine_cta(trigger_kind)
    send_as = "vera" if trigger.get("scope") == "merchant" else "merchant_on_behalf"

    return {
        "body": body,
        "cta": cta,
        "send_as": send_as,
        "template_name": f"vera_{trigger_kind}_v1",
    }


def _trigger_guidance(kind: str, trigger: dict, merchant: dict, category: dict, customer: Optional[dict]) -> str:
    payload = trigger.get("payload", {})
    merchant_name = merchant.get("identity", {}).get("name", "the merchant")
    peer = category.get("peer_stats", {})
    merch_perf = merchant.get("performance", {})

    if kind == "research_digest":
        digest = category.get("digest", [])
        if digest:
            top = digest[0]
            return (f"Lead with the digest item: '{top.get('title','')}' from {top.get('source','')}. "
                    f"Trial n={top.get('trial_n','?')}, segment: {top.get('patient_segment','?')}. "
                    "Offer to pull the abstract + draft a patient-ed WhatsApp they can share.")

    elif kind == "perf_spike":
        delta = merch_perf.get("views_7d_delta", "?")
        return f"Celebrate the spike (views up {delta}). Suggest capitalizing — update an offer or post a new photo while momentum is high."

    elif kind == "perf_dip":
        delta = merch_perf.get("calls_7d_delta", "?")
        peer_ctr = peer.get("avg_ctr", "?")
        merch_ctr = merch_perf.get("ctr_30d", "?")
        return (f"Calls dipped {delta} WoW. Their CTR is {merch_ctr} vs peer median {peer_ctr}. "
                "Suggest one specific fix (update hours, add photo, respond to reviews).")

    elif kind == "recall_due" and customer:
        c_name = customer.get("identity", {}).get("name", "the customer")
        rel = customer.get("relationship", {})
        last_visit = rel.get("last_visit_date", "?")
        prefs = customer.get("identity", {}).get("preferences", {})
        time_pref = prefs.get("appointment_time", "")
        return (f"Send recall reminder to {c_name}. Last visit: {last_visit}. "
                f"Time preference: {time_pref}. Include real offer from merchant catalog. "
                "Offer 2 specific slots. End with '1 for slot A, 2 for slot B' style CTA.")

    elif kind == "milestone_reached":
        milestone = payload.get("milestone", "100 reviews")
        return f"Congratulate on reaching {milestone}. Use social proof — 'you're now in the top X% of {category.get('slug','')} on magicpin'. Suggest next milestone action."

    elif kind == "dormant_with_vera":
        days = payload.get("days_since_last_message", 14)
        return (f"Merchant hasn't replied in {days} days. Re-engage with ONE specific, high-value insight from the digest "
                "or a performance stat they haven't seen. Don't remind them they've been quiet.")

    elif kind == "competitor_opened":
        comp_name = payload.get("competitor_name", "a new competitor")
        distance = payload.get("distance_km", "?")
        return (f"New competitor '{comp_name}' opened {distance}km away. Frame as opportunity: "
                "suggest updating their offer or adding a unique service to differentiate.")

    elif kind == "review_theme_emerged":
        theme = payload.get("theme", "service quality")
        count = payload.get("review_count", 3)
        return (f"{count} recent reviews mention '{theme}'. Either help address it (if negative) or "
                "amplify it in a new GBP post (if positive).")

    elif kind == "scheduled_recurring":
        return ("This is a weekly cadence message. Pick the most interesting insight from the category digest "
                "or a trend signal. Be curious and conversational, not promotional.")

    elif kind in ("festival_upcoming", "weather_heatwave", "local_news_event"):
        event = payload.get("event", kind)
        return f"Event: {event}. Tie the merchant's services to the event. Be timely and relevant. Suggest a quick offer or post."

    elif kind == "ctr_below_peer":
        peer_ctr = peer.get("avg_ctr", "?")
        merch_ctr = merch_perf.get("ctr_30d", "?")
        return f"CTR is {merch_ctr} vs peer median {peer_ctr}. Suggest ONE specific fix with expected impact."

    elif kind == "subscription_renewal":
        days = merchant.get("subscription", {}).get("days_remaining", "?")
        return f"Subscription expires in {days} days. Remind them of value delivered. Use concrete numbers from their performance."

    return f"Compose a relevant, high-value message for a {kind} trigger."


def _determine_cta(trigger_kind: str) -> str:
    if trigger_kind in ("recall_due", "appointment_tomorrow"):
        return "slot_selection"
    if trigger_kind in ("research_digest", "category_trend_movement"):
        return "open_ended"
    if trigger_kind in ("perf_spike", "perf_dip", "ctr_below_peer"):
        return "yes_no"
    if trigger_kind in ("festival_upcoming", "weather_heatwave"):
        return "yes_no"
    return "open_ended"


# ─────────────────────────────────────────────
# Reply composition
# ─────────────────────────────────────────────

REPLY_SYSTEM_PROMPT = """You are Vera, magicpin's merchant AI assistant.

A merchant or customer has just replied to your previous message. Analyze their reply and decide the BEST next action.

Respond ONLY with valid JSON in this exact format (no other text):
{
  "action": "send" | "wait" | "end",
  "body": "...",  (only if action=send)
  "wait_seconds": 1800,  (only if action=wait)
  "cta": "open_ended",  (only if action=send)
  "rationale": "one sentence explaining your decision"
}

Decision rules:
- "end" if: hostile/angry, explicit "stop/unsubscribe/not interested", or after 3+ unanswered nudges
- "wait" if: merchant says "call later", "busy", "will check", needs time — wait 1800s (30 min)
- "send" for everything else — advance the conversation

If the merchant said YES / "go ahead" / "do it" / "send me" → action mode immediately. DRAFT or DELIVER what was promised. Don't re-qualify.
If they asked a question → answer it specifically.
If they gave their auto-reply (generic "Thank you for contacting..." style) → treat as no-reply; send a brief re-engagement.

Auto-reply detection: if message looks like "Thank you for contacting [Business Name]" or generic canned WhatsApp reply → action = "wait", wait_seconds=3600, rationale explains auto-reply detection.

Keep responses under 120 words. Be specific, not generic."""


async def compose_reply(
    conversation_id: str,
    merchant_id: Optional[str],
    customer_id: Optional[str],
    from_role: str,
    message: str,
    turn_number: int,
) -> dict:
    """Compose a reply to an incoming message."""
    history = conversations.get(conversation_id, [])
    meta = conv_meta.get(conversation_id, {})

    # Gather context
    merchant = get_ctx("merchant", merchant_id) if merchant_id else None
    customer = get_ctx("customer", customer_id) if customer_id else None
    trigger_id = meta.get("trigger_id")
    trigger = get_ctx("trigger", trigger_id) if trigger_id else None
    category = None
    if merchant:
        cat_slug = merchant.get("category_slug") or merchant.get("identity", {}).get("category_slug", "")
        category = get_ctx("category", cat_slug)

    # Build history snippet
    hist_lines = []
    for t in history[-6:]:
        role_label = "VERA" if t["role"] == "bot" else t["role"].upper()
        hist_lines.append(f"[{role_label}]: {t['message']}")
    history_str = "\n".join(hist_lines) if hist_lines else "(no prior turns)"

    merchant_name = merchant.get("identity", {}).get("name", "Merchant") if merchant else "Merchant"
    category_slug = category.get("slug", "unknown") if category else "unknown"

    user_prompt = f"""Merchant: {merchant_name} | Category: {category_slug} | Turn: {turn_number}

Conversation so far:
{history_str}

{from_role.upper()} just replied: "{message}"

Compose the next Vera response. Return JSON only."""

    raw = await call_claude(REPLY_SYSTEM_PROMPT, user_prompt, max_tokens=400)

    # Parse JSON
    try:
        # Strip any markdown fences if present
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        result = json.loads(clean.strip())
    except Exception:
        # Fallback graceful response
        result = {
            "action": "send",
            "body": "Got it! Let me help you with that. What would you prefer — I can send you more details or set this up right away?",
            "cta": "open_ended",
            "rationale": "JSON parse fallback — advancing conversation"
        }

    return result


# ─────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        counts[scope] = counts.get(scope, 0) + 1
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts,
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Vera Bot",
        "team_members": ["Challenge Participant"],
        "model": MODEL,
        "approach": "4-context LLM composer with trigger-specific guidance, auto-reply detection, and intent-routing",
        "contact_email": "participant@example.com",
        "version": "1.0.0",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/v1/context")
async def push_context(body: CtxBody):
    valid_scopes = {"category", "merchant", "customer", "trigger"}
    if body.scope not in valid_scopes:
        return {"accepted": False, "reason": "invalid_scope", "details": f"scope must be one of {valid_scopes}"}

    key = (body.scope, body.context_id)
    current = contexts.get(key)
    if current and current["version"] >= body.version:
        return {"accepted": False, "reason": "stale_version", "current_version": current["version"]}

    contexts[key] = {"version": body.version, "payload": body.payload}
    logger.info(f"Context stored: {body.scope}/{body.context_id} v{body.version}")

    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []

    for trg_id in body.available_triggers:
        trigger = get_ctx("trigger", trg_id)
        if not trigger:
            continue

        # Skip if already sent (suppression)
        supp_key = trigger.get("suppression_key", trg_id)
        if supp_key in sent_suppression_keys:
            logger.info(f"Suppressed trigger {trg_id} (key={supp_key})")
            continue

        # Validate trigger not expired
        expires_at = trigger.get("expires_at")
        if expires_at:
            try:
                exp_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                now_dt = datetime.fromisoformat(body.now.replace("Z", "+00:00"))
                if now_dt > exp_dt:
                    logger.info(f"Trigger {trg_id} expired; skipping")
                    continue
            except Exception:
                pass

        # Resolve merchant
        merchant_id = trigger.get("payload", {}).get("merchant_id") or trigger.get("merchant_id")
        if not merchant_id:
            continue

        merchant = get_ctx("merchant", merchant_id)
        if not merchant:
            continue

        # Resolve category
        cat_slug = (
            merchant.get("category_slug")
            or merchant.get("identity", {}).get("category_slug", "")
        )
        category = get_ctx("category", cat_slug)
        if not category:
            # Try to find any category as fallback
            cat_keys = [(s, cid) for (s, cid) in contexts if s == "category"]
            if cat_keys:
                category = contexts[cat_keys[0]]["payload"]
            else:
                continue

        # Resolve optional customer
        customer_id = trigger.get("payload", {}).get("customer_id")
        customer = get_ctx("customer", customer_id) if customer_id else None

        try:
            composed = await compose_proactive(category, merchant, trigger, customer)
        except Exception as e:
            logger.error(f"Composition failed for {trg_id}: {e}")
            continue

        body_text = composed.get("body", "")
        if not body_text:
            continue

        conv_id = f"conv_{merchant_id}_{trg_id}_{int(time.time())}"

        # Store conversation + metadata
        conversations[conv_id] = [{
            "role": "bot",
            "message": body_text,
            "timestamp": body.now,
        }]
        conv_meta[conv_id] = {
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "trigger_id": trg_id,
        }
        sent_suppression_keys.add(supp_key)

        merchant_name = merchant.get("identity", {}).get("name", "merchant")
        trigger_kind = trigger.get("kind", "generic")

        action_obj = {
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": composed.get("send_as", "vera"),
            "trigger_id": trg_id,
            "template_name": composed.get("template_name", f"vera_{trigger_kind}_v1"),
            "template_params": [merchant_name, trigger_kind, cat_slug],
            "body": body_text,
            "cta": composed.get("cta", "open_ended"),
            "suppression_key": supp_key,
            "rationale": f"Trigger={trigger_kind}; category={cat_slug}; merchant={merchant_name}; urgency={trigger.get('urgency',1)}/5",
        }
        actions.append(action_obj)
        logger.info(f"Composed action for {merchant_id}/{trg_id}: {body_text[:80]}...")

    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    # Store incoming turn
    conversations.setdefault(body.conversation_id, []).append({
        "role": body.from_role,
        "message": body.message,
        "timestamp": body.received_at,
    })

    # Update conv_meta if needed
    if body.conversation_id not in conv_meta:
        conv_meta[body.conversation_id] = {
            "merchant_id": body.merchant_id,
            "customer_id": body.customer_id,
        }

    try:
        result = await compose_reply(
            conversation_id=body.conversation_id,
            merchant_id=body.merchant_id,
            customer_id=body.customer_id,
            from_role=body.from_role,
            message=body.message,
            turn_number=body.turn_number,
        )
    except Exception as e:
        logger.error(f"Reply composition failed: {e}")
        result = {
            "action": "send",
            "body": "Thanks for your message! Let me check and get back to you right away.",
            "cta": "open_ended",
            "rationale": "Fallback on composition error",
        }

    # Store bot reply if sending
    if result.get("action") == "send" and result.get("body"):
        conversations[body.conversation_id].append({
            "role": "bot",
            "message": result["body"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    return result


# ─────────────────────────────────────────────
# Optional teardown
# ─────────────────────────────────────────────

@app.post("/v1/teardown")
async def teardown():
    contexts.clear()
    conversations.clear()
    sent_suppression_keys.clear()
    conv_meta.clear()
    return {"status": "wiped"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
