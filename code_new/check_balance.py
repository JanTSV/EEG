import numpy as np
from pathlib import Path

def check_balance():
    # Pfad zu den Features
    feat_dir = Path("/Volumes/HardDiskYF/ACADEMIA/EEG/ds006761-download/derivatives_new/02_features")
    
    # Wir prüfen Pair 1, beide Spieler
    for player in [1, 2]:
        print(f"--- Checking Player {player} ---")
        y_path = feat_dir / f"pair-01_player-{player}_features_y.npy"
        
        if not y_path.exists():
            print("File not found.")
            continue
            
        y = np.load(y_path)
        
        # Zähle Vorkommen (1=Rock, 2=Paper, 3=Scissors - Annahme)
        classes, counts = np.unique(y, return_counts=True)
        
        print(f"Total Trials: {len(y)}")
        for cls, count in zip(classes, counts):
            print(f"  Class {cls}: {count} trials")
            
        # Check für Pseudo-Trials
        # Wir brauchen mindestens 4 Trials pro Klasse für EINEN Pseudo-Trial
        if any(counts < 4):
            print("  [CRITICAL] Not enough trials for Pseudo-Averaging (Need >= 4)!")
        else:
            print("  [OK] Sufficient data for decoding.")
        print("")

if __name__ == "__main__":
    check_balance()