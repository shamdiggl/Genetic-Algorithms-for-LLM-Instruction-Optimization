# --- Imports ---
import os

# Set CUDA to use the PCI bus ID for device ordering
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# You can add a print statement to verify
print("CUDA_DEVICE_ORDER:", os.environ.get("CUDA_DEVICE_ORDER"))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

from vllm import LLM, SamplingParams
import random
import numpy as np
import torch
from tqdm import tqdm
import re
import string
from deap import base, creator, tools, algorithms
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import time 
import argparse
from utils import get_gpu_info_str, parse_json_answer
from attributes import apply_attributes, ATTRIBUTES, BASE_PROMPT

script_start_time = time.perf_counter()
parser = argparse.ArgumentParser(description="Run a Genetic Algorithm on Prompts.")


# Add arguments
parser.add_argument('--eval-set-size', type=int, help='Size of the fixed set for consistent evaluation.')
parser.add_argument('--population-size', type=int, help='How many individuals (vectors) in each generation.')
parser.add_argument('--crossover-prob', type=float, help='Probability of mating two individuals.')
parser.add_argument('--mutation-prob', type=float, help='Probability of an individual mutating.')
parser.add_argument('--crossover-prob-indpb', type=float, help='Independent probability for each attribute to be exchanged during crossover.')
parser.add_argument('--mutation-prob-indpb', type=float, help='Independent probability for each bit to be flipped during mutation.')
parser.add_argument('--n-generations', type=int, help='How many generations to run.')
parser.add_argument('--n-elites', type=int, help='Number of best individuals to carry over to the next generation.')
parser.add_argument('--patience', type=int, help='Number of generations with no improvement to wait before stopping.')
parser.add_argument('--seed', type=int, help='Random seed for reproducibility.')
parser.add_argument('--log-filename', type=str, help='File to save the progress and results.')
parser.add_argument('--output-filename', type=str, help='File to save the final results and summary.')

# Parse the arguments
args = parser.parse_args()

# 2. --- Use the Arguments in Your Code ---
# Instead of hard-coded variables, use the values from 'args'
EVAL_SET_SIZE = args.eval_set_size
POPULATION_SIZE = args.population_size
CROSSOVER_PROB = args.crossover_prob
MUTATION_PROB = args.mutation_prob
CROSSOVER_PROB_INDPB = args.crossover_prob_indpb
MUTATION_PROB_INDPB = args.mutation_prob_indpb
N_GENERATIONS = args.n_generations
N_ELITES = args.n_elites
PATIENCE = args.patience
SEED = args.seed
PROGRESS_LOG_FILENAME = args.log_filename
OUTPUT_FILENAME = args.output_filename


MODEL_ID = "Qwen/Qwen3-8B-AWQ" 

llm = LLM(  
    model=MODEL_ID,
    enforce_eager=True,       # This is the key fix for the torch.compile error
    trust_remote_code=True,    # This is required for most Qwen models
    max_model_len=8192,       # Limit the context window to a realistic size for your task
    gpu_memory_utilization=0.95 # Allow vLLM to use up to 95% of the GPU's VRAM
    )
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


config_lines = ["--- Configuration for this run ---"]
for arg, value in vars(args).items():
    config_lines.append(f"{arg}: {value}")
config_lines.append("----------------------------------")
config_str = "\n".join(config_lines)
print(config_str)

gpu_info_str = get_gpu_info_str()
print(gpu_info_str)

LOG_HEADER_INFO = f"{config_str}\n{gpu_info_str}"


mmlu_dataset = load_dataset("TIGER-Lab/MMLU-Pro")
mmlu_dataset_train = mmlu_dataset["test"]
torch.random.manual_seed(SEED)
fixed_eval_set = mmlu_dataset_train.shuffle(seed=SEED).select(range(EVAL_SET_SIZE))

def evaluate_individual(thinking=False):
    correct_predictions = 0
    total_predictions = len(fixed_eval_set)
    reasoning_batch_of_messages = []
    for item in fixed_eval_set:
        question = item['question']
        option_strings = []
        for i, option in enumerate(item['options']):
            # Get the correct letter from the alphabet
            letter = string.ascii_uppercase[i]
            # Create the formatted string for a single option and add it to our list
            option_strings.append(f"{letter}. {option}")
        all_options_string = "\n".join(option_strings)
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant. You will be given a multiple-choice question. First follow the instructions and then please include a JSON object containing the key 'answer' with the letter of the correct choice. For example: {\"answer\": \"C\"}"},
            {"role": "user", "content": f"Question: {question}\n"
                                        f"Options:\n{all_options_string}\n"}    
        ]
        reasoning_batch_of_messages.append(messages)

    # VLLM: Convert the list of message dicts into a list of prompt strings
    reasoning_prompts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        for messages in reasoning_batch_of_messages
    ]

    # VLLM: Define sampling parameters using the SamplingParams class
    reasoning_params = SamplingParams(
        n=1, # Number of output sequences to return for each prompt
        max_tokens=1024,
        temperature=0.0, # Using 0.0 for deterministic output
        top_k=-1, # Disable top-k sampling
    )

    final_outputs = llm.generate(reasoning_prompts, reasoning_params, use_tqdm=False)

    for i in range(len(fixed_eval_set)):
        item = fixed_eval_set[i]
        generated_text = final_outputs[i].outputs[0].text        
        ground_truth_letter = chr(ord('A') + item['answer_index'])
        
        # MODIFIED: Use the new JSON parser
        model_choice = parse_json_answer(generated_text)
        
        if model_choice == ground_truth_letter:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
        
    return (accuracy,)

# ==============================================================================
# PART 3: GENETIC ALGORITHM SETUP (using DEAP)
# ==============================================================================
print("\n--- Configuring Genetic Algorithm with DEAP ---")



print(f"GA Parameters: Population={POPULATION_SIZE}, Generations={N_GENERATIONS}, CXPB={CROSSOVER_PROB}, MUTPB={MUTATION_PROB}")

#Base-line Score
baseline_score = evaluate_individual([0] * len(ATTRIBUTES))[0]
print(f"Baseline F1 Score (no attributes): {baseline_score:.4f}")
LOG_HEADER_INFO += f"\nBaseline Score (no attributes): {baseline_score:.4f}\n"



# ============================================
# === NEW CODE FOR DETAILED LOGGING STARTS ===
# ============================================

# 1. Create a global log to store evaluation results
evaluation_log = []
generation_counter = 0 # To track which generation an evaluation belongs to

def evaluate_and_log(individual_vector):
    """
    A wrapper function that calls the original evaluation function
    and logs the result to our global list.
    """
    # This is the actual call to the expensive LLM evaluation
    fitness = evaluate_individual(individual_vector)

    # Log the data we want to track
    log_entry = {
        'generation': generation_counter,
        'vector': list(individual_vector), # Convert to list for clean logging
        'f1_score': fitness[0]
    }
    evaluation_log.append(log_entry)

    # Return the fitness tuple as DEAP expects
    return fitness

# ============================================
# === NEW CODE FOR DETAILED LOGGING ENDS   ===
# ============================================


# --- DEAP Toolbox Setup ---
# We are maximizing the F1 score, so weight is 1.0
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# An individual is a list of bits, with the FitnessMax property
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator: creates a 0 or 1
toolbox.register("attr_bool", random.randint, 0, 1)

# Individual generator: a list of 0s/1s of length NUM_ATTRIBUTES
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(ATTRIBUTES))

# Population generator: a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register the core genetic operators
# !!! IMPORTANT CHANGE HERE: Register our new wrapper function !!!
toolbox.register("evaluate", evaluate_and_log)    # Use the logging wrapper
toolbox.register("mate", tools.cxUniform, indpb=CROSSOVER_PROB_INDPB)        # Two-point crossover
toolbox.register("mutate", tools.mutFlipBit, indpb=MUTATION_PROB_INDPB) # Flip-bit mutation
toolbox.register("select", tools.selTournament, tournsize=3) # Tournament selection



# ==============================================================================
# === NEW FUNCTION FOR PROGRESSIVE LOGGING (ADD THIS BLOCK) ====================
# ==============================================================================
def format_deap_log(log_object):
    if not log_object:
        return "The log object is empty."

    header = "{:<5} {:<8} {:<10} {:<10} {:<8} {:<8}\n".format(
        "gen", "nevals", "avg", "std", "min", "max"
    )
    separator = "-" * len(header) + "\n"
    
    body = ""
    for entry in log_object:
        body += "{:<5} {:<8} {:<10.6f} {:<10.6f} {:<8.6f} {:<8.6f}\n".format(
            entry.get('gen', 'N/A'),
            entry.get('nevals', 'N/A'),
            entry.get('avg', 'N/A'),
            entry.get('std', 'N/A'),
            entry.get('min', 'N/A'),
            entry.get('max', 'N/A')
        )
    
    return header + separator + body

def save_progress_to_file(header_info, log_data, stats_stream, populations, filename):
    """
    Writes the complete run history to a file, including a snapshot of the
    entire population at the current generation.
    """
    output_lines = [header_info]  # Start with the config/GPU header

    # --- Section 1: Generation Statistics ---
    output_lines.append("\n===================================")
    output_lines.append("     GENERATION STATISTICS (Logbook) ")
    output_lines.append("===================================")
    output_lines.append(format_deap_log(stats_stream))

    # --- Section 2: Population Snapshot (THE NEW PART) ---
    output_lines.append("\n===================================")
    output_lines.append("        POPULATIONS          ")
    output_lines.append("===================================")

    
    for i, pop in enumerate(populations):
        sorted_pop = sorted(pop, key=lambda ind: ind.fitness.values[0], reverse=True)
        output_lines.append(f"\n--- Population for Generation {i} ({len(sorted_pop)}) ---")
        for ind in sorted_pop:
            # Check if fitness is valid before trying to access it
            if ind.fitness.valid:
                score = f"{ind.fitness.values[0]:.4f}"
            else:
                score = "N/A (Not yet evaluated)"
            output_lines.append(f"  - Vector: {list(ind)}  =>  Accuracy: {score}")


    # Write everything to the file
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("\n".join(output_lines))
    except IOError as e:
        print(f"\n[ERROR] Could not write progress to file '{filename}'. Reason: {e}")

# ==============================================================================
# PART 4: RUNNING THE GENETIC ALGORITHM
# ==============================================================================
print("\n--- Starting Genetic Algorithm Evolution ---")
print("NOTE: Each generation requires evaluating the population. This will take time.\n")

# Initialize the population
pop = toolbox.population(n=POPULATION_SIZE)
# Keep track of the best individual ever seen
hof = tools.HallOfFame(1)

# Setup statistics tracking
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

# This is the manual implementation of the eaSimple loop to gain control
# over the generation counter.

print("Evaluating initial population...")
# Evaluate the entire population
fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

# Update Hall of Fame and statistics with the initial population
pops = []
pops.append(list(map(toolbox.clone, pop)))  # Store the initial population for logging
hof.update(pop)
record = stats.compile(pop)
logbook = tools.Logbook()
logbook.record(gen=0, **record)
print(logbook) # Print initial stats


save_progress_to_file(LOG_HEADER_INFO, evaluation_log, logbook, pops, PROGRESS_LOG_FILENAME)
print(f"Initial progress saved to {PROGRESS_LOG_FILENAME}\n")

last_best_score = hof[0].fitness.values[0]
stagnation_counter = 0
print(f"Initial best score: {last_best_score}")

# Begin the evolution
for g in tqdm(range(1, N_GENERATIONS + 1)):
    # Increment our global generation counter
    generation_counter = g

    elites = tools.selBest(pop, k=N_ELITES)
    elites = list(map(toolbox.clone, elites))
    # Select the next generation individuals

    offspring = toolbox.select(pop, len(pop) - N_ELITES)
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CROSSOVER_PROB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < MUTATION_PROB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    print(f"\n--- Generation {g}: Evaluating {len(invalid_ind)} new individuals ---")
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # The population is now replaced by the offspring
    pop[:] = elites + offspring
    
    # Update the hall of fame and the statistics with the new population
    pops.append(list(map(toolbox.clone, pop)))  # Store the current population for logging
    hof.update(pop)
    record = stats.compile(pop)
    logbook.record(gen=g, **record)
    print(logbook.stream) # Print stats for the current generation

    save_progress_to_file(LOG_HEADER_INFO, evaluation_log, logbook, pops, PROGRESS_LOG_FILENAME)
    print(f"Progress for generation {g} saved to {PROGRESS_LOG_FILENAME}")

    current_best_score = hof[0].fitness.values[0]
    
    if current_best_score > last_best_score:
        last_best_score = current_best_score
        stagnation_counter = 0 # Reset counter on improvement
        print(f"New best score found: {last_best_score}. Resetting stagnation counter.")
    else:
        stagnation_counter += 1
        print(f"No improvement. Stagnation counter: {stagnation_counter}/{PATIENCE}")

    if stagnation_counter >= PATIENCE:
        print(f"\n--- Stopping early: Best score has not improved for {PATIENCE} generations. ---")
        break # Exit the loop

# ==============================================================================
# PART 5: DISPLAYING THE RESULTS
# ==============================================================================
print("\n\n--- Genetic Algorithm Finished ---")

# Get the best individual from the Hall of Fame
best_individual = hof[0]
best_fitness = best_individual.fitness.values[0]
active_attributes = [ATTRIBUTES[i]['name'] for i, bit in enumerate(best_individual) if bit == 1]

# --- Create Final Results Summary ---
final_results_filename = OUTPUT_FILENAME
output_lines = [] # 1. An empty list is created to hold all lines for the final report.

# 2. All the summary details are appended to this list.
output_lines.append("===================================")
output_lines.append("         OPTIMAL PROMPT FOUND      ")
output_lines.append("===================================")
output_lines.append(f"Best Individual (Vector): {best_individual}")
output_lines.append(f"Best Fitness (F1 Score):  {best_fitness:.4f}")
output_lines.append("Active Attributes for Best Prompt:")
if active_attributes:
    for attr in active_attributes:
        output_lines.append(f"  - {attr}")
else:
    output_lines.append("  - None (Base prompt was optimal)")
output_lines.append("===================================")
output_lines.append("\nExample of the optimal prompt structure:")
example_prompt = apply_attributes("Q: What is the capital of France?", best_individual)
output_lines.append(f'"{example_prompt}"')

# 3. The total runtime is calculated and also appended to the SAME list.
script_end_time = time.perf_counter()
total_runtime_seconds = script_end_time - script_start_time
run_minutes = int(total_runtime_seconds // 60)
run_seconds = total_runtime_seconds % 60
output_lines.append("\n===================================")
output_lines.append(f"Total script runtime: {run_minutes} minutes and {run_seconds:.2f} seconds.")

# 4. The list is joined into a single string.
final_output_str = "\n".join(output_lines)

# 5. This complete string is written to the final results file.
try:
    with open(final_results_filename, 'w', encoding='utf-8') as f:
        f.write(final_output_str)
except IOError as e:
    print(f"\nError: Could not write final summary to file '{PROGRESS_LOG_FILENAME}'. Reason: {e}")

# 6. The same complete string is printed to the console.
print("\n" + final_output_str)