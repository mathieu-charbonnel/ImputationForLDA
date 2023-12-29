import src.plots.printing1graph as pr
import argparse

def validate_probs_missingness(value):
    value = float(value)
    if 0 <= value <= 1:
        return value
    else:
        raise argparse.ArgumentTypeError("probs_missingness must be a float between 0 and 1")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate a graph comparing different imputation methods"
    )

    # Add command line arguments with types and validation
    parser.add_argument("--dimensions",
                        type=int, 
                        help="Integer value representing the generated data dimension")
    parser.add_argument("--cov_matrice", 
                        choices=["random",
                                 "normal",
                                 "str_correlation_higherIndex",
                                 "str_correlation+high_diagonal"], 
                        help="Defines the relationships between the different features")
    parser.add_argument("--probs_missingness", 
                        type=validate_probs_missingness, 
                        help="Value between 0 and 1 defining the missingness ratio")
    parser.add_argument("--type_missingness", 
                        choices=["MCAR", "MAR", "MNAR"],
                        help="Define the type of missingness")

    # Parse the command line arguments
    args = parser.parse_args()

    # Convert the Namespace object to a dictionary
    arguments = vars(args)

    return arguments

def main():
    # Parse command line arguments into a dictionary
    arguments = parse_arguments()
    pr.one_graph(
        arguments['cov_matrice'],
        arguments['dimensions'],
        arguments['type_missingness'],
        arguments['probs_missingness']
    )

if __name__ == "__main__":
    main()


