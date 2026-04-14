"""
Match simulation via point-level decomposition.

Build order (from spec):
  1. Core pipeline must be working first
  2. Point decomposition: hold/break probability per service game
  3. Then this simulation layer

Placeholder — implement after core pipeline is confirmed working.
"""

# TODO: implement after point decomposition is working
# Steps:
#   1. Use hold_pct and break_pct from serve_return.py to estimate
#      P(hold | service game) for each player
#   2. Simulate games -> sets -> match via Markov chain or Monte Carlo
#   3. Return P(player_a wins match) and distribution of set scores
