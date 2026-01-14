import torch
import json
import re
import string
import sys

def get_gpu_info_str():
    """
    Checks for GPU availability and returns a formatted string with the info.
    """
    lines = ["--- GPU Hardware Check ---"]
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        lines.append(f"Success: Found {device_count} GPU(s) available.")
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            lines.append(f"  - GPU {i}: {gpu_name}")
    else:
        lines.append("Error: PyTorch cannot find or access any GPU.")
    lines.append("--------------------------")
    return "\n".join(lines)

def parse_json_answer(text: str) -> str | None:
    # Use a regular expression to find the JSON blob, even if it's surrounded by text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        # If no JSON-like structure is found, return None
        return None

    json_str = match.group(0)
    try:
        # Try to parse the found string as JSON
        data = json.loads(json_str)
        # Return the value associated with the "answer" key
        # .get() is safer than [] as it returns None if the key doesn't exist
        return data.get("answer")
    except json.JSONDecodeError:
        # If the string is not valid JSON, return None
        return None


def load_population_from_file(filename, individual_creator):
    """
    Parses a file to load a population and their fitness scores.
    The file should contain lines in the format:
      - Vector: [0, 1, ...]  =>  Accuracy: 0.1234
    
    Args:
        filename (str): The path to the file to load.
        individual_creator (class): The DEAP creator class used to instantiate individuals
                                    (e.g., creator.Individual).
    """
    population = []
    print(f"--- Attempting to resume from file: {filename} ---")
    try:
        with open(filename, 'r') as f:
            for line in f:
                match = re.search(r"- Vector: \[(.*?)\]\s*=>\s*Accuracy: ([\d.]+)", line)
                if match:
                    vector_str = match.group(1)
                    accuracy_str = match.group(2)
                    
                    vector = [int(x.strip()) for x in vector_str.split(',')]
                    accuracy = float(accuracy_str)
                    
                    # Use the passed-in creator, not a hardcoded one
                    ind = individual_creator(vector)
                    ind.fitness.values = (accuracy,)
                    population.append(ind)
        
        if not population:
            print(f"[ERROR] Could not find any valid population data in '{filename}'. Exiting.")
            # Use sys.exit() because exit() is not always available by default
            sys.exit()
            
        print(f"Successfully loaded {len(population)} individuals from {filename}.")
        return population
    except IOError as e:
        print(f"\n[ERROR] Could not read the resume file '{filename}'. Reason: {e}. Exiting.")
        sys.exit()