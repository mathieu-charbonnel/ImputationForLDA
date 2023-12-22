import src.plots.printing1graph as pr
import sys

def load_config(file_path):
    try:
        # Dynamically import the module from the specified file path
        config_module = __import__(file_path.replace('.py', ''))
        config_data = config_module.config_data
        return config_data
    except Exception as e:
        print(f"Error loading configuration from {file_path}: {e}")
        sys.exit(1)

def main(config_file_path):
    # Load the config file as a Python dictionary
    config_data = load_config(config_file_path)

    # Extract parameters from the config file
    dimensions = config_data['dimensions']
    cov_matrice = config_data['cov_matrice']
    probs_missingness = config_data['probs_missingness']
    type_missingness = config_data['type_missingness']
    for c in cov_matrice:
        for d in dimensions:
            for p in probs_missingness:
                for t in type_missingness:
                    pr.one_graph(c,d,t,p)


if __name__ == "__main__":
    # Check if the config file path is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python main_script.py <config_file_path>")
        sys.exit(1)

    # Get the config file path from the command-line arguments
    config_file_path = sys.argv[1]

    # Call the main function with the provided config file path
    main(config_file_path)

