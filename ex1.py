import sys
import click
from pathlib import Path
from mne_bids import (BIDSPath,read_raw_bids)
from ccs_eeg_utils import read_annotations_core

# import ccs_eeg_utils.py
sys.path.insert(0, '.')

@click.command()
@click.option("--bids", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path), help="Path to bids/ dir.")
def main(bids):
    subject_id = '030'
    
    # Path to data set and its format
    bids_path = BIDSPath(subject=subject_id, task="P3", session="P3", datatype='eeg', suffix='eeg', root=bids)

    # read the file
    raw = read_raw_bids(bids_path)

    # fix the annotations readin
    read_annotations_core(bids_path,raw)

if __name__ == "__main__":
    main()