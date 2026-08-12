"""
Ingestion for the FIFA World Cup 2026 RAG project.

Reads the Kaggle CSVs from source_data/ and builds a flat list of text
"documents", each a dict of {"doc_id", "doc_type", "content"} — the same
shape minsearch.Index and RAGBase expect.

Document types built:
    - match  (104)   : one per match, with score, stats, and timed events
    - player (~1248) : one per player, tournament totals + bio
    - team   (48)    : one per team, with aggregated tournament results
    - venue  (16)    : one per stadium, with matches hosted

Usage:
    from ingest import load_documents
    documents = load_documents()          # reads ./source_data by default
    documents = load_documents("path/to/source_data")
"""

import os
import csv
from collections import defaultdict

from data_and_ingestion.summary_docs import build_summary_docs


# ---------------------------------------------------------------------------
# CSV loading helpers
# ---------------------------------------------------------------------------

def _read_csv(data_dir, name):
    """Read a CSV into a list of dicts. Returns [] if the file is missing."""
    path = os.path.join(data_dir, name)
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _index_by(rows, key):
    """Build {key_value: row} for quick lookups."""
    return {row[key]: row for row in rows}


def _num(value):
    """Format a numeric-ish string nicely, blank-safe."""
    if value is None or value == "":
        return None
    return value


# ---------------------------------------------------------------------------
# Document builders
# ---------------------------------------------------------------------------

def _build_match_docs(data_dir):
    """One document per match, from matches_detailed + team stats + events."""
    detailed = _read_csv(data_dir, "matches_detailed.csv")
    matches = _read_csv(data_dir, "matches.csv")
    team_stats = _read_csv(data_dir, "match_team_stats.csv")
    events = _read_csv(data_dir, "match_events.csv")
    teams = _index_by(_read_csv(data_dir, "teams.csv"), "team_id")
    players = _index_by(_read_csv(data_dir, "player_stats.csv"), "player_id")

    # match_id -> {team_id -> stats row}
    stats_by_match = defaultdict(dict)
    for row in team_stats:
        stats_by_match[row["match_id"]][row["team_id"]] = row

    # match_id -> list of event rows (sorted by minute)
    events_by_match = defaultdict(list)
    for row in events:
        events_by_match[row["match_id"]].append(row)
    for mid in events_by_match:
        events_by_match[mid].sort(key=lambda r: int(r["minute"]) if r["minute"] else 0)

    # map matches.csv team ids by match, so we can attach team-level stats
    match_core = _index_by(matches, "match_id")

    docs = []
    for m in detailed:
        mid = m["match_id"]
        lines = []

        # Headline
        stage = m["stage_name"]
        home, away = m["home_team_name"], m["away_team_name"]
        hs, as_ = m["home_score"], m["away_score"]
        lines.append(f"{stage}: {home} {hs}-{as_} {away} ({m['date']})")
        lines.append(
            f"{home} ({m['home_fifa_code']}) vs {away} ({m['away_fifa_code']}) "
            f"at {m['stadium_name']}, {m['city']}, {m['country']}."
        )
        lines.append(f"Final score: {home} {hs}, {away} {as_}.")

        # Penalties
        if _num(m.get("home_penalty_score")) or _num(m.get("away_penalty_score")):
            lines.append(
                f"Penalty shootout: {home} {m['home_penalty_score']}, "
                f"{away} {m['away_penalty_score']}."
            )

        lines.append(f"Result type: {m['result_type']}. Status: {m['status']}.")
        lines.append(f"Expected goals (xG): {home} {m['home_xg']}, {away} {m['away_xg']}.")
        lines.append(
            f"Player of the match: {m['player_of_the_match_name']}. "
            f"Referee: {m['referee_name']}."
        )
        lines.append(
            f"Goalkeepers: {home} {m['home_goalkeeper']}, {away} {m['away_goalkeeper']}."
        )

        # Team stats (possession/shots/etc.)
        core = match_core.get(mid, {})
        home_id, away_id = core.get("home_team_id"), core.get("away_team_id")
        for side, tid in (("Home/" + home, home_id), ("Away/" + away, away_id)):
            st = stats_by_match.get(mid, {}).get(tid)
            if st:
                lines.append(
                    f"{side} stats: possession {st['possession_pct']}%, "
                    f"shots {st['total_shots']} ({st['shots_on_target']} on target), "
                    f"corners {st['corners']}, fouls {st['fouls']}, "
                    f"offsides {st['offsides']}, saves {st['saves']}."
                )

        # Timed events
        evs = events_by_match.get(mid, [])
        if evs:
            lines.append("Key events:")
            for e in evs:
                pname = players.get(e["player_id"], {}).get("player_name", f"player {e['player_id']}")
                tname = teams.get(e["team_id"], {}).get("team_name", f"team {e['team_id']}")
                lines.append(f"  {e['minute']}' {e['event_type']} - {pname} ({tname})")

        docs.append({
            "doc_id": f"match-{mid}",
            "doc_type": "match",
            "content": "\n".join(lines),
        })
    return docs


def _build_player_docs(data_dir):
    """One document per player: tournament totals joined with bio."""
    stats = _read_csv(data_dir, "player_stats.csv")
    bios = _index_by(_read_csv(data_dir, "squads_and_players.csv"), "player_id")
    teams = _index_by(_read_csv(data_dir, "teams.csv"), "team_id")

    docs = []
    for p in stats:
        pid = p["player_id"]
        team = teams.get(p["team_id"], {})
        team_name = team.get("team_name", f"team {p['team_id']}")
        bio = bios.get(pid, {})

        def g(key, default="0"):
            v = p.get(key)
            return v if v not in (None, "") else default

        lines = []
        lines.append(f"{p['player_name']} - {team_name}, position {p['position']}.")
        lines.append(
            f"Tournament stats: {g('matches_played')} matches "
            f"({g('matches_started')} started), {g('minutes_played')} minutes."
        )
        goals_line = f"Goals: {g('goals')}, assists: {g('assists')}"
        if g("shots", "") != "":
            goals_line += (
                f", shots: {g('shots')} ({g('shots_on_target')} on target)"
            )
        goals_line += (
            f", penalty goals: {g('penalty_goals')}, own goals: {g('own_goals')}."
        )
        lines.append(goals_line)
        lines.append(
            f"Cards: {g('yellow_cards')} yellow, {g('red_cards')} red."
        )
        # Goalkeeper-relevant fields
        if _num(p.get("clean_sheets")) or _num(p.get("saves")) or p["position"] == "GK":
            lines.append(
                f"Goalkeeping: {p['clean_sheets']} clean sheets, {p['saves']} saves, "
                f"{p['goals_conceded']} conceded."
            )
        if _num(p.get("average_rating")):
            lines.append(f"Average rating: {p['average_rating']}.")

        # Bio
        if bio:
            bits = []
            if bio.get("club_team"):
                bits.append(f"club {bio['club_team']}")
            if bio.get("market_value_eur"):
                bits.append(f"market value EUR {bio['market_value_eur']}")
            if bio.get("caps"):
                bits.append(f"{bio['caps']} caps")
            if bio.get("date_of_birth"):
                bits.append(f"born {bio['date_of_birth']}")
            if bio.get("height_cm"):
                bits.append(f"height {bio['height_cm']}cm")
            if bits:
                lines.append("Bio: " + ", ".join(bits) + ".")

        docs.append({
            "doc_id": f"player-{pid}",
            "doc_type": "player",
            "content": "\n".join(lines),
        })
    return docs


def _build_team_docs(data_dir):
    """One document per team, enriched with aggregated match results."""
    teams = _read_csv(data_dir, "teams.csv")
    matches = _read_csv(data_dir, "matches.csv")
    stages = _index_by(_read_csv(data_dir, "tournament_stages.csv"), "stage_id")

    # Aggregate per team from matches.csv
    agg = defaultdict(lambda: {"p": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "stages": set()})
    for m in matches:
        if m["status"] != "Completed":
            continue
        h, a = m["home_team_id"], m["away_team_id"]
        try:
            hs, as_ = int(m["home_score"]), int(m["away_score"])
        except (ValueError, TypeError):
            continue
        stage_name = stages.get(m["stage_id"], {}).get("stage_name", "")
        for tid, gf, ga in ((h, hs, as_), (a, as_, hs)):
            agg[tid]["p"] += 1
            agg[tid]["gf"] += gf
            agg[tid]["ga"] += ga
            agg[tid]["stages"].add(stage_name)
            if gf > ga:
                agg[tid]["w"] += 1
            elif gf == ga:
                agg[tid]["d"] += 1
            else:
                agg[tid]["l"] += 1

    docs = []
    for t in teams:
        tid = t["team_id"]
        a = agg.get(tid)
        lines = []
        lines.append(
            f"{t['team_name']} ({t['fifa_code']}) - Group {t['group_letter']}, "
            f"{t['confederation']}."
        )
        lines.append(
            f"Pre-tournament FIFA ranking: {t['fifa_ranking_pre_tournament']}, "
            f"Elo rating: {t['elo_rating']}. Manager: {t['manager_name']}."
        )
        if a:
            lines.append(
                f"Tournament record: {a['p']} played, {a['w']} won, "
                f"{a['d']} drawn, {a['l']} lost."
            )
            lines.append(f"Goals: {a['gf']} scored, {a['ga']} conceded.")
            reached = [s for s in a["stages"] if s]
            if reached:
                lines.append("Stages played: " + ", ".join(sorted(reached)) + ".")

        docs.append({
            "doc_id": f"team-{tid}",
            "doc_type": "team",
            "content": "\n".join(lines),
        })
    return docs


def _build_venue_docs(data_dir):
    """One document per venue, listing matches hosted."""
    venues = _read_csv(data_dir, "venues.csv")
    detailed = _read_csv(data_dir, "matches_detailed.csv")

    # venue stadium_name -> matches (detailed has stadium_name text)
    hosted = defaultdict(list)
    for m in detailed:
        hosted[m["stadium_name"]].append(m)

    docs = []
    for v in venues:
        lines = []
        lines.append(
            f"{v['stadium_name']} - {v['city']}, {v['country']}."
        )
        lines.append(
            f"Capacity: {v['capacity']}, elevation: {v['elevation_meters']}m."
        )
        games = hosted.get(v["stadium_name"], [])
        if games:
            lines.append(f"Hosted {len(games)} matches:")
            for m in games:
                lines.append(
                    f"  {m['date']} ({m['stage_name']}): "
                    f"{m['home_team_name']} {m['home_score']}-{m['away_score']} "
                    f"{m['away_team_name']}"
                )
        docs.append({
            "doc_id": f"venue-{v['venue_id']}",
            "doc_type": "venue",
            "content": "\n".join(lines),
        })
    return docs


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_documents(data_dir="source_data"):
    """Build and return the full list of documents from the CSVs."""
    documents = []
    documents += _build_match_docs(data_dir)
    documents += _build_player_docs(data_dir)
    documents += _build_team_docs(data_dir)
    documents += _build_venue_docs(data_dir)
    documents += build_summary_docs(data_dir)
    return documents


if __name__ == "__main__":
    docs = load_documents()
    from collections import Counter
    counts = Counter(d["doc_type"] for d in docs)
    print(f"Built {len(docs)} documents:")
    for dtype, n in counts.items():
        print(f"  {dtype}: {n}")
    print("\n--- sample match document ---")
    print(next(d["content"] for d in docs if d["doc_type"] == "match"))