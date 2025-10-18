# EEG

Repository for this [course](https://www.s-ccs.de/course_EEG/course.html).

## Get started

```bash
# Install used modules (please use a venv)
pip install -r requirements.txt

# Run exercise N
python ex<N>.py <ARGS>  # for example: python ex1.py
```

## Exercise 1

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