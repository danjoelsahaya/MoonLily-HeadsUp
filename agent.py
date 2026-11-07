"""
agent.py - YOUR poker bot. This one file is a complete web server.

Run it:
    python agent.py                  # listens on port 5008
    PORT=5010 python agent.py        # or pick a port

Play a match against a baseline (three windows):
    python agent.py                                 # your bot on port 5008
    python baseline-bots/baseline_random.py         # opponent on port 5009
    python judge_engine.py                          # judge: both halves (uses those two ports)

Check that the tournament can run it:
    python -m pytest -q              # tests in test_agent.py
    ./tools/check_build.sh           # builds and smoke-tests it like the tournament

TODO:
    1. Assign your agent a unique PARTICIPANT and AGENT_NAME below.
    2. Implement the three functions between "YOUR CODE HERE" and "END OF YOUR CODE".

Everything under "SERVER PLUMBING" talks to the judge. You normally never
need to change it, and if you break it the judge can't talk to your bot.
"""

import os
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from evaluator import card_rank, card_suit, cards_to_str, omaha_score, warmup  # handy helpers, see evaluator.py
from game import Action, AgentConfig, HandResult, Observation, Street

# Build omaha_score's lookup table now, while the server starts, instead of
# during your first decision (where it would eat into your time bank).
# Do any other slow setup (loading a model, ...) here too: the server must
# answer within 15 seconds of starting.
warmup()

# rename as your team/bot's name
PARTICIPANT = "Moon Lily"
AGENT_NAME = "HandRatingBot"


# ===== YOUR CODE HERE =====
from collections import Counter
from itertools import combinations
from evaluator import category_name, get_table

# Edit these defaults before submitting. Environment overrides help local testing.
RISK_LEVEL = 0.5  # 0.0 = cautious, 0.5 = balanced, 1.0 = aggressive
SHOW_RATINGS = False  # True prints every decision in the bot's terminal.
CATEGORY_RATINGS = (5, 20, 35, 50, 60, 75, 85, 95, 100)


def rate_hand(hole_cards, board):
    """Return an absolute heuristic rating, never a probability of winning."""
    if not board:
        ranks = Counter(card_rank(card) for card in hole_cards)
        suits = Counter(card_suit(card) for card in hole_cards)
        high_cards = sum(card_rank(card) for card in hole_cards) / 40
        pairs = sum(1 + rank / 8 for rank, count in ranks.items() if count >= 2)
        suited_pairs = sum(count >= 2 for count in suits.values())
        connected = sum(rank + 1 in ranks for rank in ranks)
        rating = min(80.0, 10 + 25 * high_cards + 10 * pairs
                     + 7 * suited_pairs + 3 * connected)
        return {"rating": round(rating, 1), "category": "preflop potential",
                "score": None, "best_hand": ""}

    # This also works on the flop and turn, where omaha_score requires 5 board cards.
    table = get_table()
    candidates = (
        pair + triple
        for pair in combinations(hole_cards, 2)
        for triple in combinations(board, 3)
    )
    best_hand = max(candidates, key=lambda cards: table[sum(1 << c for c in cards)])
    score = table[sum(1 << c for c in best_hand)]
    category = score >> 20
    # The official packed score supplies exact tie breaks within each category.
    kicker_fraction = (score & ((1 << 20) - 1)) / 0x88888
    rating = min(100.0, CATEGORY_RATINGS[category] + 4 * kicker_fraction)
    return {"rating": round(rating, 1), "category": category_name(score),
            "score": score, "best_hand": cards_to_str(best_hand)}


def select_action(obs, rating, risk_level):
    """Higher risk lowers thresholds. Bet price affects decisions, not ratings."""
    call_cost = min(obs.to_call, obs.my_stack)
    price = call_cost / max(1, obs.pot + call_cost)
    call_threshold = 38 - 20 * risk_level + 25 * price
    half_threshold = 68 - 20 * risk_level
    pot_threshold = 86 - 20 * risk_level
    if rating >= pot_threshold and Action.RAISE_POT in obs.legal_actions:
        return Action.RAISE_POT
    if rating >= half_threshold and Action.RAISE_HALF in obs.legal_actions:
        return Action.RAISE_HALF
    if obs.to_call == 0 or rating >= call_threshold:
        return Action.CHECK_CALL
    if Action.FOLD in obs.legal_actions:
        return Action.FOLD
    return Action.CHECK_CALL


def new_half(config: AgentConfig, memory: dict) -> None:
    risk_level = float(os.environ.get("RISK_LEVEL", RISK_LEVEL))
    if not 0 <= risk_level <= 1:
        raise ValueError("RISK_LEVEL must be between 0 and 1")
    memory["risk_level"] = risk_level
    memory["show_ratings"] = os.environ.get("SHOW_RATINGS", str(int(SHOW_RATINGS))) == "1"
    memory["hands_played"] = 0


def choose_action(obs: Observation, memory: dict) -> Action:
    # Cache only this hand/street, keeping memory bounded across a half.
    key = (obs.hand_index, tuple(obs.hole_cards), tuple(obs.board))
    if memory.get("rating_key") != key:
        memory["rating_key"] = key
        memory["hand_rating"] = rate_hand(obs.hole_cards, obs.board)
    report = dict(memory["hand_rating"])
    rating = report["rating"]
    risk = "high" if rating < 40 else "moderate" if rating < 70 else "lower"
    action = select_action(obs, rating, memory["risk_level"])
    report.update(hand_index=obs.hand_index, street=obs.street.name,
                  hand_risk=risk, risk_level=memory["risk_level"], action=action.name)
    memory["last_decision"] = report
    if memory["show_ratings"]:
        print(f"hand={obs.hand_index} street={obs.street.name} "
              f"rating={rating}/100 category={report['category']} "
              f"hand_risk={risk} aggression={memory['risk_level']} "
              f"action={action.name} best={report['best_hand']}", flush=True)
    return action


def on_hand_end(result: HandResult, memory: dict) -> None:
    memory["hands_played"] += 1

# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
#
# Keep your per-half state in `memory`. In the tournament every half runs in
# a fresh server process, so everything starts over. When you test locally,
# though, one process can play both halves and only `memory` is wiped, NOT
# other global variables. Using `memory` makes your bot behave the same way
# in both places.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5008 for your own testing convenience
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
