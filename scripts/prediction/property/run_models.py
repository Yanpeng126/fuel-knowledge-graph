"""Run the nested five-fold recipe-group property-model workflow."""
import argparse
import subprocess
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for options in ([], ['--extended'], ['--enriched'], ['--selected']):
        subprocess.run([sys.executable, str(HERE / 'block_models.py'), *options], check=True)
    subprocess.run([sys.executable, str(HERE / 'graph_models.py'), '--numeric-context'], check=True)
    root = REPO / 'results/property'
    subprocess.run([sys.executable, str(HERE / 'select_predictions.py'), '--runs-root', str(root)], check=True)
if __name__ == '__main__':
    main()
