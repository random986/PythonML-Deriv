"""Analyze the trade log to find consecutive loss streaks per side."""
import re, sys

LOG_PATH = r"C:\Users\User\.gemini\antigravity\brain\8926759b-4bfa-462e-9315-5d03ca8a8d76\.system_generated\tasks\task-3074.log"

settle_re = re.compile(r"\[SETTLE\] side=(\w+) .* profit=\$([\-\d.]+) .* won=(True|False)")
fire_re = re.compile(r"\[TRADE\] Firing on (\S+) \| OVER=(\d+)% UNDER=(\d+)%")

over_trades = []
under_trades = []
fires = []

with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        m = settle_re.search(line)
        if m:
            side = m.group(1)
            profit = float(m.group(2))
            won = m.group(3) == "True"
            entry = {"side": side, "profit": profit, "won": won, "line": line.strip()[:120]}
            if side == "over":
                over_trades.append(entry)
            else:
                under_trades.append(entry)
        
        m2 = fire_re.search(line)
        if m2:
            fires.append({"market": m2.group(1), "over_conf": int(m2.group(2)), "under_conf": int(m2.group(3))})

# Find longest consecutive loss streak for each side
def find_streaks(trades, side_name):
    max_streak = 0
    current_streak = 0
    streak_start = 0
    worst_start = 0
    worst_end = 0
    
    for i, t in enumerate(trades):
        if not t["won"]:
            if current_streak == 0:
                streak_start = i
            current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak
                worst_start = streak_start
                worst_end = i
        else:
            current_streak = 0
    
    print(f"\n{'='*80}")
    print(f"  {side_name.upper()} SIDE ANALYSIS")
    print(f"{'='*80}")
    print(f"  Total trades: {len(trades)}")
    wins = sum(1 for t in trades if t['won'])
    losses = len(trades) - wins
    print(f"  Wins: {wins}  Losses: {losses}  Win rate: {wins/len(trades)*100:.1f}%")
    print(f"  Longest consecutive LOSS streak: {max_streak}")
    
    if max_streak > 0:
        print(f"\n  Worst streak (trades #{worst_start+1} to #{worst_end+1}):")
        for i in range(worst_start, worst_end + 1):
            t = trades[i]
            print(f"    #{i+1}: profit=${t['profit']:>10.2f} | won={t['won']}")
    
    # Show ALL loss streaks >= 3
    print(f"\n  All loss streaks >= 3:")
    current_streak = 0
    streak_start = 0
    for i, t in enumerate(trades):
        if not t["won"]:
            if current_streak == 0:
                streak_start = i
            current_streak += 1
        else:
            if current_streak >= 3:
                total_loss = sum(trades[j]["profit"] for j in range(streak_start, i))
                print(f"    Streak of {current_streak} losses (trades #{streak_start+1}-#{i}): total loss = ${total_loss:.2f}")
            current_streak = 0
    if current_streak >= 3:
        total_loss = sum(trades[j]["profit"] for j in range(streak_start, len(trades)))
        print(f"    Streak of {current_streak} losses (trades #{streak_start+1}-#{len(trades)}): total loss = ${total_loss:.2f}")

find_streaks(over_trades, "over")
find_streaks(under_trades, "under")

# Show the fire confidence values during the worst streaks
print(f"\n{'='*80}")
print(f"  FIRE CONFIDENCE ANALYSIS")
print(f"{'='*80}")
print(f"  Total fires: {len(fires)}")

# Check how often the confidence is wildly skewed
skewed = sum(1 for f in fires if f["over_conf"] > 90 or f["under_conf"] > 90)
print(f"  Fires with >90% confidence on one side: {skewed} ({skewed/len(fires)*100:.1f}%)")
print(f"  Fires with >100% confidence (bug!): {sum(1 for f in fires if f['over_conf'] > 100 or f['under_conf'] > 100)}")

# Show the last 20 fires
print(f"\n  Last 20 fires:")
for f in fires[-20:]:
    print(f"    {f['market']:>10} | OVER={f['over_conf']:>3}% | UNDER={f['under_conf']:>3}%")
