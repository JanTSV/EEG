# EEG

Repository for this [course](https://www.s-ccs.de/course_EEG/course.html).

## Project

### Get started
- Step 1: Download data: [link to example data, code and figures](https://osf.io/yjxkn/files) or follow instructions on [full data set (~80gb)](https://openneuro.org/datasets/ds006761/versions/1.0.0/download).
- Step 2: Ensure structure of project folder (for example data, it should look like this): 
```
EEG/
│  README.md
│  requirements.txt
│
└─data/
    └─example_data/
        │  participants.json
        │  participants.tsv
        │  dataset_description.json
        |  CHANGES
        │  README.txt
        └─derivatives/
        └─sub-01/
            └─eeg/
                │  sub-01_task-RPS_eeg.bdf
                │  sub-01_task-RPS_eeg.json
                │  sub-01_task-RPS_events.json
                │  sub-01_task-RPS_events.tsv
```




## Exercises

### Get started

```bash
# Install used modules (please use a venv)
pip install -r requirements.txt

# Run exercise N
python ex<N>.py <ARGS>  # for example: python ex1.py
```

### Exercise 1

Please download [this](https://osf.io/9cnmx/) data set.

The path to the `/bids` dir is passed as a command-line argument (via `--bids`).
```bash
python ex1.py --bids path/to/bids
```

You can always run
```bash
python ex1.py --help
```
to get more information.