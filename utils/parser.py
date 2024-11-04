import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='BioNeuralNet Pipeline')
    parser.add_argument('--start', type=int, required=True, help='Starting component index (1-5)')
    parser.add_argument('--end', type=int, required=True, help='Ending component index (1-5)')
    return parser
