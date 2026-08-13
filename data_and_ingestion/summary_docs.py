"""
Summary/aggregate documents for the FIFA World Cup 2026 knowledge base.

RAG retrieves only the top-k documents per query, so it struggles with
tournament-wide "aggregation" questions (most goals, top scorer, group
standings, etc.) whose answer requires scanning the whole dataset.

This module pre-computes those aggregates and emits them as their own
documents, so a single retrieved summary doc can answer the whole class
of questions. Called from load_documents().
"""

import os
import csv
from collections import defaultdict


def _read_csv(data_dir, name):
    path = os.path.join(data_dir, name)
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _int(v, default=0):
    try:
        return int(v)
    except (ValueError, TypeError):
        return default


def _float(v, default=0.0):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def build_summary_docs(data_dir):
    teams = {t["team_id"]: t for t in _read_csv(data_dir, "teams.csv")}
    matches = _read_csv(data_dir, "matches.csv")
    players = _read_csv(data_dir, "player_stats.csv")
    stages = {s["stage_id"]: s for s in _read_csv(data_dir, "tournament_stages.csv")}
    detailed = _read_csv(data_dir, "matches_detailed.csv")

    def tname(tid):
        return teams.get(tid, {}).get("team_name", f"team {tid}")

    docs = []

    # ---- Per-team goals for/against, split by group vs total ----------------
    team_agg = defaultdict(lambda: {
        "gf": 0, "ga": 0, "gf_group": 0, "ga_group": 0,
        "p": 0, "w": 0, "d": 0, "l": 0, "pts": 0,
    })
    for m in matches:
        if m["status"] != "Completed":
            continue
        hs, as_ = _int(m["home_score"]), _int(m["away_score"])
        is_group = stages.get(m["stage_id"], {}).get("stage_name") == "Group Stage"
        for tid, gf, ga in ((m["home_team_id"], hs, as_), (m["away_team_id"], as_, hs)):
            a = team_agg[tid]
            a["gf"] += gf
            a["ga"] += ga
            if is_group:
                a["gf_group"] += gf
                a["ga_group"] += ga
                a["p"] += 1
                if gf > ga:
                    a["w"] += 1; a["pts"] += 3
                elif gf == ga:
                    a["d"] += 1; a["pts"] += 1
                else:
                    a["l"] += 1

    # Doc: teams by goals scored (total, with group subtotal)
    ranked = sorted(team_agg.items(), key=lambda kv: kv[1]["gf"], reverse=True)
    lines = ["Teams ranked by total goals scored in the 2026 World Cup "
             "(group-stage goals in parentheses):"]
    for i, (tid, a) in enumerate(ranked, 1):
        lines.append(f"{i}. {tname(tid)} - {a['gf']} goals ({a['gf_group']} in group stage)")
    docs.append({"doc_id": "summary-team-goals-scored", "doc_type": "summary",
                 "content": "\n".join(lines)})

    # Doc: teams by goals conceded (fewest = best defense)
    ranked = sorted(team_agg.items(), key=lambda kv: kv[1]["ga"])
    lines = ["Teams ranked by fewest goals conceded in the 2026 World Cup "
             "(best defenses first; group-stage conceded in parentheses):"]
    for i, (tid, a) in enumerate(ranked, 1):
        lines.append(f"{i}. {tname(tid)} - {a['ga']} conceded ({a['ga_group']} in group stage)")
    docs.append({"doc_id": "summary-team-goals-conceded", "doc_type": "summary",
                 "content": "\n".join(lines)})

    # ---- Group standings ----------------------------------------------------
    groups = defaultdict(list)
    for tid, t in teams.items():
        groups[t["group_letter"]].append(tid)
    for g in sorted(groups):
        rows = []
        for tid in groups[g]:
            a = team_agg[tid]
            rows.append((tid, a["pts"], a["w"], a["d"], a["l"],
                         a["gf_group"], a["ga_group"], a["gf_group"] - a["ga_group"]))
        rows.sort(key=lambda r: (r[1], r[7], r[5]), reverse=True)  # pts, GD, GF
        lines = [f"Group {g} final standings (2026 World Cup group stage):"]
        for pos, r in enumerate(rows, 1):
            lines.append(f"{pos}. {tname(r[0])} - {r[1]} pts "
                         f"({r[2]}W {r[3]}D {r[4]}L, GF {r[5]}, GA {r[6]}, GD {r[7]:+d})")
        docs.append({"doc_id": f"summary-group-{g}-standings", "doc_type": "summary",
                     "content": "\n".join(lines)})

    # ---- Player leaderboards -----------------------------------------------
    def player_board(metric, title, doc_id, top=20):
        ranked = sorted(players, key=lambda p: _int(p[metric]), reverse=True)
        ranked = [p for p in ranked if _int(p[metric]) > 0][:top]
        lines = [title]
        for i, p in enumerate(ranked, 1):
            lines.append(f"{i}. {p['player_name']} ({tname(p['team_id'])}) - "
                         f"{_int(p[metric])}")
        return {"doc_id": doc_id, "doc_type": "summary", "content": "\n".join(lines)}

    docs.append(player_board("goals",
                "Top goalscorers of the 2026 World Cup (player, team, goals):",
                "summary-top-scorers"))
    docs.append(player_board("assists",
                "Top assist providers of the 2026 World Cup (player, team, assists):",
                "summary-top-assists"))
    docs.append(player_board("yellow_cards",
                "Players with the most yellow cards in the 2026 World Cup:",
                "summary-most-yellow-cards"))

    # Clean sheets (goalkeepers)
    gks = sorted(players, key=lambda p: _int(p["clean_sheets"]), reverse=True)
    gks = [p for p in gks if _int(p["clean_sheets"]) > 0][:15]
    lines = ["Goalkeepers with the most clean sheets in the 2026 World Cup:"]
    for i, p in enumerate(gks, 1):
        lines.append(f"{i}. {p['player_name']} ({tname(p['team_id'])}) - "
                     f"{_int(p['clean_sheets'])} clean sheets")
    docs.append({"doc_id": "summary-clean-sheets", "doc_type": "summary",
                 "content": "\n".join(lines)})

    # ---- Highest-scoring matches & biggest wins -----------------------------
    completed = [m for m in detailed if m["status"] == "Completed"]
    by_total = sorted(completed,
                      key=lambda m: _int(m["home_score"]) + _int(m["away_score"]),
                      reverse=True)[:15]
    lines = ["Highest-scoring matches of the 2026 World Cup (by total goals):"]
    for i, m in enumerate(by_total, 1):
        tot = _int(m["home_score"]) + _int(m["away_score"])
        lines.append(f"{i}. {m['home_team_name']} {m['home_score']}-{m['away_score']} "
                     f"{m['away_team_name']} ({m['stage_name']}, {m['date']}) - {tot} goals")
    docs.append({"doc_id": "summary-highest-scoring-matches", "doc_type": "summary",
                 "content": "\n".join(lines)})

    by_margin = sorted(completed,
                       key=lambda m: abs(_int(m["home_score"]) - _int(m["away_score"])),
                       reverse=True)[:15]
    lines = ["Biggest wins (largest margins) of the 2026 World Cup:"]
    for i, m in enumerate(by_margin, 1):
        margin = abs(_int(m["home_score"]) - _int(m["away_score"]))
        lines.append(f"{i}. {m['home_team_name']} {m['home_score']}-{m['away_score']} "
                     f"{m['away_team_name']} ({m['stage_name']}, {m['date']}) - margin {margin}")
    docs.append({"doc_id": "summary-biggest-wins", "doc_type": "summary",
                 "content": "\n".join(lines)})

    # ---- Venues by matches hosted ------------------------------------------
    venue_count = defaultdict(int)
    for m in detailed:
        venue_count[m["stadium_name"]] += 1
    ranked_v = sorted(venue_count.items(), key=lambda kv: kv[1], reverse=True)
    lines = ["Venues ranked by number of 2026 World Cup matches hosted:"]
    for i, (v, c) in enumerate(ranked_v, 1):
        lines.append(f"{i}. {v} - {c} matches")
    docs.append({"doc_id": "summary-venues-by-matches", "doc_type": "summary",
                 "content": "\n".join(lines)})

    # ---- Tournament champion (answers "who won the World Cup / the final") --
    def _winner_loser(m):
        hs, as_ = _int(m["home_score"]), _int(m["away_score"])
        if hs != as_:
            return (m["home_team_name"], m["away_team_name"]) if hs > as_ \
                else (m["away_team_name"], m["home_team_name"])
        hp, ap = _int(m["home_penalty_score"]), _int(m["away_penalty_score"])
        return (m["home_team_name"], m["away_team_name"]) if hp >= ap \
            else (m["away_team_name"], m["home_team_name"])

    finals = [m for m in detailed
              if m["stage_name"] == "Final" and m["status"] == "Completed"]
    if finals:
        f = finals[0]
        champion, runner_up = _winner_loser(f)
        lines = [
            f"{champion} won the 2026 FIFA World Cup.",
            f"{champion} are the champions and winners of the tournament. "
            f"{runner_up} finished as runners-up.",
            f"Final: {f['home_team_name']} {f['home_score']}-{f['away_score']} "
            f"{f['away_team_name']} ({f['result_type']}) on {f['date']} at "
            f"{f['stadium_name']}, {f['city']}.",
            f"Player of the match in the final: {f['player_of_the_match_name']}.",
        ]
        third = [m for m in detailed
                 if m["stage_name"] == "Third-place match"
                 and m["status"] == "Completed"]
        if third:
            tw, _ = _winner_loser(third[0])
            lines.append(f"Third place: {tw}.")
        docs.append({"doc_id": "summary-tournament-champion", "doc_type": "summary",
                     "content": "\n".join(lines)})

    # ---- Knockout stage results (bracket) ----------------------------------
    knockout_order = ["Round of 32", "Round of 16", "Quarter-finals",
                      "Semi-finals", "Third-place match", "Final"]
    ko = [m for m in detailed
          if m["stage_name"] in knockout_order and m["status"] == "Completed"]
    if ko:
        ko.sort(key=lambda m: (knockout_order.index(m["stage_name"]), m["date"]))
        lines = ["2026 FIFA World Cup knockout stage results (Round of 32 "
                 "through the Final):"]
        current = None
        for m in ko:
            if m["stage_name"] != current:
                current = m["stage_name"]
                lines.append(f"\n{current}:")
            extra = f" ({m['result_type']})" if m["result_type"] != "Regular" else ""
            lines.append(f"  {m['home_team_name']} {m['home_score']}-"
                         f"{m['away_score']} {m['away_team_name']}{extra} "
                         f"[{m['date']}]")
        docs.append({"doc_id": "summary-knockout-results", "doc_type": "summary",
                     "content": "\n".join(lines)})

    return docs


if __name__ == "__main__":
    docs = build_summary_docs("source_data")
    print(f"Built {len(docs)} summary documents:\n")
    for d in docs:
        print(f"### {d['doc_id']}")
        print(d["content"])
        print()