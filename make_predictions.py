"""Walk forward through every game and save the model's pre-game probability for each (G_pre.pkl).
Several research scripts (meta_review.py, experience_test.py) read this file."""
import pandas as pd
from ratings import load_games, run, data_files
G = load_games(data_files())
_, G["p"] = run(G)
G["p_pre"] = G.p
G.to_pickle("G_pre.pkl")
print(f"G_pre.pkl: {len(G):,} games")
