import mne
import yaml
from pathlib import Path

def check_channels():
    cfg_path = "code_new/config.yaml"
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    data_root = Path(cfg['paths']['data_root'])
    pair_id = 1
    
    sub_str = f"sub-{pair_id:02d}"
    eeg_dir = data_root / sub_str / 'eeg'
    
    # Alle .bdf files holen
    all_files = list(eeg_dir.glob("*.bdf"))
    
    # FILTER: Ignoriere Dateien, die mit '._' beginnen (macOS Müll)
    valid_files = [f for f in all_files if not f.name.startswith("._")]
    
    if not valid_files:
        print(f"KEINE gültigen .bdf DATEIEN GEFUNDEN IN {eeg_dir}")
        print(f"(Gefunden wurden nur: {[f.name for f in all_files]})")
        return

    raw_path = valid_files[0]
    print(f"Lade echte Datei: {raw_path.name}")
    
    try:
        # read_raw_bdf nutzen
        raw = mne.io.read_raw_bdf(raw_path, preload=False, verbose='error')
        ch_names = raw.ch_names
        
        print(f"\n--- KANAL-ANALYSE ({len(ch_names)} Kanäle) ---")
        print("Erste 10 Kanäle:", ch_names[:10])
        print("Letzte 10 Kanäle:", ch_names[-10:])
        
        # Check auf typische BioSemi Muster
        print("\nStichproben:")
        if 'A1' in ch_names: print("  -> Enthält 'A1' (BioSemi Style)")
        if 'Fp1' in ch_names: print("  -> Enthält 'Fp1' (Standard Style)")
        
        # Zähle Prefixes aus der Config
        p1_pre = cfg['subjects']['channel_prefix_p1'] 
        p2_pre = cfg['subjects']['channel_prefix_p2'] 
        
        c1 = sum(1 for ch in ch_names if ch.startswith(p1_pre))
        c2 = sum(1 for ch in ch_names if ch.startswith(p2_pre))
        
        print(f"\nConfig Check:")
        print(f"  Prefix '{p1_pre}' (Player 1): {c1} Treffer")
        print(f"  Prefix '{p2_pre}' (Player 2): {c2} Treffer")
            
    except Exception as e:
        print(f"Fehler beim Laden: {e}")

if __name__ == "__main__":
    check_channels()