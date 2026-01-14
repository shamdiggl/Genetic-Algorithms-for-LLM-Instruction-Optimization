# evaluation_utils.py

import string
import json
import re
import random
from tqdm import tqdm
from vllm import SamplingParams

# --- Globals that will be set by the main script ---
llm = None
tokenizer = None
DATASET = "" # Will be set to "PRO" or "BBH" by the main pipeline
SEED = 42
import json
ATTRIBUTES = [
    # --- Group 1: Core Persona & Role-Playing ---
    # --- Index 0:
    {"name": "expert", "type": "instruction", "text": "Act as an expert on the following context."},
    # --- Index 1:
    {"name": "tutor_persona", "type": "instruction", "text": "Act like a friendly and patient tutor."},
    # --- Index 2:
    {"name": "skeptic_persona", "type": "instruction", "text": "Assume the role of a skeptic. Provide a critical analysis."},

    # --- Group 2: Reasoning & Cognitive Process ---
    # --- Index 3:
    {"name": "cot", "type": "instruction", "text": "Let’s think step by step to answer the question."},
    # --- Index 4:
    {"name": "meta_prompting", "type": "instruction", "text": "Before providing the answer, describe your reasoning process."},
    # --- Index 5:
    {"name": "critique_and_improve", "type": "instruction", "text": "After providing the initial answer, take a step back, critique it for flaws or omissions, and then provide an improved, final version."},
    # --- Index 6:
    {"name": "uses_self_ask", "type": "instruction", "text": "Before answering, generate a series of clarifying questions you would ask yourself to fully understand and address the query. Then, answer the question."},
    # --- Index 7:
    {"name": "query_reformulation", "type": "instruction", "text": "If the query is ambiguous, rephrase it for clarity before answering."},
    # --- Index 8:
    {"name": "reflective_prompting", "type": "instruction", "text": "After providing your answer, reflect on its potential biases or alternative interpretations."},
    # --- Index 9:
    {"name": "state_uncertainty", "type": "instruction", "text": "If you are unsure about any part of the answer, you must state your uncertainty or the limitations of your knowledge."},

    # --- Group 3: Output Style, Verbosity, Motivation, Stakes & Tone ---
    # --- Index 10:
    {"name": "no_fluff", "type": "instruction", "text": "Get straight to the point. No conversational filler."},
    # --- Index 11:
    {"name": "optimize_token_quantity", "type": "instruction", "text": "Be concise."},
    # --- Index 12:
    {"name": "clarity_seeking", "type": "instruction", "text": "Explain to me like I'm 11 years old."},
    # --- Index 13:
    {"name": "analogy_reasoning", "type": "instruction", "text": "Explain this concept using a helpful analogy."},
    # --- Index 14:
    {"name": "empathetic_tone", "type": "instruction", "text": "Adopt a compassionate and understanding tone in your response."},
    # --- Index 15:
    {"name": "emotion_prompting", "type": "instruction", "text": "Respond with a reassuring and calm tone."},
    # --- Index 16:
    {"name": "high_stakes_context", "type": "instruction", "text": "This is very important for my career. A great answer is critical."},
    # --- Index 17:
    {"name": "establishes_rewards", "type": "instruction", "text": "You will be rewarded for following the instructions."},
    # --- Index 18:
    {"name": "tipping", "type": "instruction", "text": "I will tip $10 for an answer that follows instructions and is accurate."},
    # --- Index 19:
    {"name": "penalty_enforcement", "type": "instruction", "text": "Failure to follow these instructions will result in a penalty."},
    # --- Index 20:
    {"name": "low_balling_challenge", "type": "instruction", "text": "This task is notoriously difficult for AI models. I bet you can't do it correctly, but prove me wrong."},
    # --- Index 21:
    {"name": "uses_positive_tone", "type": "instruction", "text": "Let's find an excellent solution together."},
    # --- Index 22:
    {"name": "humorous_tone", "type": "instruction", "text": "Add a touch of humor to your response."},
    # --- Index 23:
    {"name": "target_audience_expert_output", "type": "instruction", "text": "Tailor your response to an expert audience, using technical terminology where appropriate."},
    # --- Index 24:
    {"name": "system2_attention", "type": "instruction", "text": "Access your deep knowledge base on this topic before answering."},

    # --- Group 4: Specific Content & Formatting Requirements ---
    # --- Index 25:
    {"name": "summarize_context", "type": "instruction", "text": "Before answering your question, summarize the context first."},
    # --- Index 26:
    {"name": "cite_sources", "type": "instruction", "text": "For every claim you make, cite the source."},
    # --- Index 27:
    {"name": "highlight_keywords", "type": "instruction", "text": "**Bold** the most important keywords in your response."},
    # --- Index 28:
    {"name": "output_scoping_by_count", "type": "instruction", "text": "Provide a list of items."},
    # --- Index 29:
    {"name": "correction_prompting", "type": "instruction", "text": "Review your answer for factual errors or inconsistencies and correct them."},
    # --- Index 30:
    {"name": "enforces_explicit_constraints", "type": "instruction", "text": "Adhere strictly to all specified constraints."},
    # --- Index 31:
    {"name": "numbered_steps_output", "type": "instruction", "text": "Provide your answer in numbered steps."},
    # --- Index 32:
    {"name": "bullet_points_output", "type": "instruction", "text": "List the key features in bullet points."},

    # --- Group 5: Responsible AI & Safety ---
    # --- Index 33:
    {"name": "mitigate_bias", "type": "instruction", "text": "Ensure your answer is free from any gender, racial, or cultural biases and stereotypes."},
    # --- Index 34:
    {"name": "ensure_safety", "type": "instruction", "text": "Generate safe and appropriate content, avoiding any harmful or offensive language."},

    # --- Group 6: Prefixes ---
    # --- Index 35:
    {"name": "first_command_prefix", "type": "prefix", "text": "My first command is: "},
    # --- Index 36:
    {"name": "follow_instructions_prefix", "type": "prefix", "text": "Carefully follow every instruction below: "},

    # --- Group 7: Suffixes ---
    # --- Index 37:
    {"name": "please", "type": "suffix", "text": " Please!"},
    # --- Index 38:
    {"name": "thanks", "type": "suffix", "text": " Thank you!"},
    # --- Index 39:
    {"name": "polite_closure", "type": "suffix", "text": " I appreciate your effort."},
    # --- Index 40:
    {"name": "praise", "type": "suffix", "text": " You are the best!"},
    # --- Index 41:
    {"name": "inspirational_tone", "type": "suffix", "text": " Let's make it happen."},
    # --- Index 42:
    {"name": "directive", "type": "suffix", "text": " Now start working hard!"},
    # --- Index 43:
    {"name": "think_longer", "type": "suffix", "text": " Think longer and harder!"},
    # --- Index 44:
    {"name": "add_hashtags_instructive", "type": "suffix", "text": " #concise #accuracy #factual"},
    # --- Index 45:
    {"name": "add_hashtags_social", "type": "suffix", "text": " #topic"},

    # --- Group 8: Global Transforms ---
    # --- Index 46:
    {"name": "enumerate_instructions", "type": "transform"},
    # --- Index 47:
    {"name": "json_format", "type": "transform"},
]

NUM_ATTRIBUTES = len(ATTRIBUTES)

def apply_attributes(vector: list[int]) -> str:
    if len(vector) != len(ATTRIBUTES):
        raise ValueError(f"Vector must have length {len(ATTRIBUTES)}.")

    # Step 1: Collect all active components
    prefixes, instructions, suffixes, active_transforms = [], [], [], set()
    for i, attr in enumerate(ATTRIBUTES):
        if vector[i] == 1:
            if attr['type'] == 'prefix':
                prefixes.append(attr['text'])
            elif attr['type'] == 'instruction':
                instructions.append(attr['text'])
            elif attr['type'] == 'suffix':
                suffixes.append(attr['text'])
            elif attr['type'] == 'transform':
                active_transforms.add(attr['name'])

    ### MODIFIED: Added back the JSON format override logic ###
    # Step 2: Handle the JSON format transform first, as it's a complete override
    if 'json_format' in active_transforms:
        # Combine prefixes and instructions into a single block
        prefix_str = " ".join(p.strip() for p in prefixes)
        
        if 'enumerate_instructions' in active_transforms:
            instruction_str = "\n".join(f"{i+1}. {inst.strip()}" for i, inst in enumerate(instructions))
            # Place prefix before the enumerated list
            processed_instructions = f"{prefix_str}\n{instruction_str}".strip()
        else:
            instruction_str = " ".join(inst.strip() for inst in instructions)
            # Join prefix and instructions with a space
            processed_instructions = f"{prefix_str} {instruction_str}".strip()
            
        # The JSON payload includes the BASE_PROMPT, all instructions, and suffixes.
        full_instruction_text = (processed_instructions + "".join(suffixes)).strip()
        
        prompt_dict = {
            "instructions": full_instruction_text,
        }
        return json.dumps(prompt_dict, indent=2)

    # --- Step 3: Default logic for creating a standalone instruction string ---
    
    # Join all active prefixes into a single string
    prefix_str = " ".join(p.strip() for p in prefixes)

    # Process the main instructions
    instruction_str = ""
    if instructions:
        if 'enumerate_instructions' in active_transforms:
            instruction_str = "\n".join(f"{i+1}. {inst.strip()}" for i, inst in enumerate(instructions))
        else:
            instruction_str = " ".join(inst.strip() for inst in instructions)

    # Concatenate all active suffixes.
    suffix_str = "".join(suffixes)

    # Combine all parts into the final string
    main_components = []
    if prefix_str:
        main_components.append(prefix_str)
    if instruction_str:
        main_components.append(instruction_str)
        
    main_content = "\n".join(main_components)
    final_string = main_content + suffix_str
    
    return final_string

def extract_cot_answer_from_text(text: str) -> str | None:
    """
    Extracts the final letter choice from the model's CoT output text.
    This is the standard parsing method from the MMLU-Pro script.
    """
    # Pattern 1: "answer is (A)" or "answer is A"
    match = re.search(r"answer is \(?([A-J])\)?" , text)
    if match:
        return match.group(1)
        
    # Pattern 2: "... answer: A"
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)

    # Pattern 3 (Fallback): The last single uppercase letter in the generation.
    # This is a robust fallback for when the model concludes its reasoning with the letter.
    match = re.search(r"\b([A-J])\b(?!.*\b[A-J]\b)", text, re.DOTALL)
    if match:
        return match.group(0)
        
    return None # Return None if no answer can be reliably extracted

def format_cot_example_standard(example, including_answer=True):
    """
    Formats a single few-shot example for standard CoT, ending in natural language.
    """
    choices = string.ascii_uppercase
    
    prompt = "Question:\n"
    prompt += example["question"] + "\n"
    
    option_key = 'options' if 'options' in example else 'choices'
    options = example[option_key]
    
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += f"{choices[i]}. {opt}\n"
        
    if including_answer:
        cot_content = example.get("cot_content", "No reasoning provided.")
        # The MMLU-Pro script cleans the start of the CoT. We'll ensure it's correct.
        if "Let's think step by step." not in cot_content:
             cot_content = "Answer: Let's think step by step.\n" + cot_content
        else:
             cot_content = "Answer: " + cot_content.split("A: ")[-1]

        prompt += cot_content + "\n\n"
    else:
        # This is the trigger for the model to start generating its own reasoning.
        prompt += "Answer: Let's think step by step."
        

# --- FUSED Evaluation Function 1: Direct Evaluation ---

def evaluate_direct(validation_set, manuel_instructions=None, max_tokens=512):
    """
    Unified evaluation function for both MMLU-Pro and BBH.
    """
    all_prompts = []
    print(f"\n--- Evaluating with direct prompting on {DATASET} for {len(validation_set)} examples... ---")
    if manuel_instructions is None:
        manuel_instructions = {"initial": "", "instructions": ""}

    for item in tqdm(validation_set, desc="Building validation prompts"):
        item_category = item.get("category")
        initial = f"The following is multiple-choice question (with answers) about {item_category}.\n\n"
        
        # --- DATASET-SPECIFIC PROMPT FORMATTING ---
        if DATASET == "BBH":
            question_content = item['input']
            final_user_prompt = f"{initial}\n\nQuestion:\n{question_content}\n\nInstructions:\nLet's think step by step to answer the question."
        else: # MMLU-Pro
            question = item['question']
            option_strings = []
            for i, option in enumerate(item['options']): # MMLU-Pro uses 'options'
                letter = string.ascii_uppercase[i]
                option_strings.append(f"{letter}. {option}")
            all_options_string = "\n".join(option_strings)
            final_user_prompt = f"{initial}\n\nQuestion:\n{question}\n{all_options_string}\n\nInstructions: \n Let’s think step by step to answer the question."

        messages = [
            {"role": "system", "content":  "You are a helpful AI assistant that provides step-by-step reasoning to arrive at the correct answer."},
            {"role": "user", "content": final_user_prompt}
            ]
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        all_prompts.append(prompt_str)
        
    sampling_params = SamplingParams(n=1, max_tokens=max_tokens, temperature=0.0, top_k=-1, seed=SEED)
    final_outputs = llm.generate(all_prompts, sampling_params, use_tqdm=True)

    correct_count = 0
    total_count = len(final_outputs)
    for i, output in enumerate(final_outputs):
        eval_item = validation_set[i]
        generated_text = output.outputs[0].text.strip()
        model_choice = extract_cot_answer_from_text(generated_text)
        
        # --- DATASET-SPECIFIC GROUND TRUTH EXTRACTION ---
        if DATASET == "BBH":
            ground_truth_target = str(eval_item['target']).strip() # e.g., "(A)"
            ground_truth_letter = ground_truth_target[1:2] if ground_truth_target.startswith('(') else ground_truth_target
        else: # MMLU-Pro
            ground_truth_letter = chr(ord('A') + eval_item['answer_index'])

        if model_choice == ground_truth_letter:
            correct_count += 1
            
    accuracy = (correct_count / total_count) if total_count > 0 else 0
    return accuracy


def evaluate_5shot_cot_standard(test_set, few_shot_source, n_shots=5, max_tokens=2048):
    all_prompts = []
    print(f"\n--- Evaluating with standard {n_shots}-shot CoT on the test set of size {len(test_set)}... ---")

    # Group few-shot examples by category for better relevance
    examples_by_category = {}
    for item in few_shot_source:
        category = item.get("category")
        if category:
            if category not in examples_by_category:
                examples_by_category[category] = []
            examples_by_category[category].append(item)
    
    # Construct the batch of prompts
    for item in tqdm(test_set, desc="Building standard 5-shot CoT prompts"):
        item_category = item.get("category")
        
        # Select n_shots examples from the same category
        if item_category and item_category in examples_by_category and len(examples_by_category[item_category]) >= n_shots:
            shot_examples = random.sample(examples_by_category[item_category], n_shots)
        else:
            shot_examples = random.sample(few_shot_source, n_shots) # Fallback

        # Build the prompt content for the user message
        prompt_content = f"The following are multiple-choice questions (with answers) about {item_category}.\n\n"
        
        # Add the few-shot examples
        for example in shot_examples:
            prompt_content += format_cot_example_standard(example, including_answer=True)
            
        # Add the final test question without the answer
        prompt_content += format_cot_example_standard(item, including_answer=False)
        
        messages = [
            # The system prompt is more generic now
            {"role": "system", "content": "You are a helpful AI assistant that provides step-by-step reasoning to arrive at the correct answer."},
            {"role": "user", "content": prompt_content}
        ]
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking = False)
        all_prompts.append(prompt_str)

    # Define sampling parameters and run generation
    sampling_params = SamplingParams(
        n=1, max_tokens=max_tokens, temperature=0.0, top_k=-1)
    final_outputs = llm.generate(all_prompts, sampling_params, use_tqdm=True)

    # Process the results and calculate accuracy
    correct_count = 0
    total_count = len(final_outputs)

    for i, output in enumerate(final_outputs):
        eval_item = test_set[i]
        generated_text = output.outputs[0].text
        
        answer_key = 'answer_index' if 'answer_index' in eval_item else 'answer'
        ground_truth_letter = chr(ord('A') + eval_item[answer_key])
        
        # Use the standard CoT regex parser
        model_choice = extract_cot_answer_from_text(generated_text)

        if model_choice == ground_truth_letter:
            correct_count += 1
            
    accuracy = (correct_count / total_count) if total_count > 0 else 0
    print(f"Final Accuracy: {accuracy:.4f} ({correct_count}/{total_count})")
    return accuracy, final_outputs


# --- FUSED Evaluation Function 2: Vector-based Prompting ---

# def enhance_instructions_batch(base_instructions_list, questions_list):
#     meta_prompts = []
#     for base_instr, question in zip(base_instructions_list, questions_list):
#         meta_prompt_content = META_PROMPT_TEMPLATE.format(
#             base_instructions=base_instr,
#             question_content=question
#         )
#         messages = [{"role": "user", "content": meta_prompt_content}]
#         prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,  enable_thinking=False)
#         meta_prompts.append(prompt_str)

#     enhancement_params = SamplingParams(n=1, max_tokens=512, temperature=0.0, top_k=-1, seed=SEED)
#     enhancement_outputs = llm.generate(meta_prompts, enhancement_params, use_tqdm=True)
#     enhanced_instructions = [output.outputs[0].text.strip() for output in enhancement_outputs]
#     return enhanced_instructions

def evaluate_vector(individual, validation_set):
    """
    Unified evaluation function for vector-based prompting on MMLU-Pro and BBH.
    """
    print(f"\n--- Evaluating with dynamically enhanced prompts on {DATASET} for {len(validation_set)} examples... ---")
    
    # STEP 1: Prepare batches of questions and base instructions
    base_instructions_list = []
    questions_list = []
    for item in tqdm(validation_set, desc="Step 1: Preparing base prompts"):
        # --- DATASET-SPECIFIC LOGIC ---
        if DATASET == "BBH":
            full_question_content = item['input']
        else: # MMLU-Pro
            question = item['question']
            option_strings = []
            for i, option in enumerate(item['options']): # MMLU-Pro uses 'options'
                letter = string.ascii_uppercase[i]
                option_strings.append(f"{letter}. {option}")
            all_options_string = "\n".join(option_strings)
            full_question_content = f"{question}\n{all_options_string}"
        
        questions_list.append(full_question_content)
        base_instructions = apply_attributes(individual)
        base_instructions_list.append(base_instructions)

    # STEP 2: Enhance all instructions in a single batch call
    # enhanced_instructions_list = enhance_instructions_batch(base_instructions_list, questions_list)
    enhanced_instructions_list = base_instructions_list

    # STEP 3: Build final evaluation prompts and run generation
    all_prompts = []
    initial = 'You will be given a multiple-choice question. First follow the instructions and please show your choice at the end in a JSON object with only the choice letter, e.g., {"answer": "C"}'
    for i in tqdm(range(len(validation_set)), desc="Step 3: Building final evaluation prompts"):
        question_content = questions_list[i]
        final_instructions = enhanced_instructions_list[i]
        final_user_prompt = f"{initial}\n\nQuestion:\n{question_content}\n\nInstructions:\n{final_instructions}"
        messages = [{"role": "user", "content": final_user_prompt}]
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        all_prompts.append(prompt_str)

    sampling_params = SamplingParams(n=1, max_tokens=512, temperature=0.0, top_k=-1, seed=SEED)
    final_outputs = llm.generate(all_prompts, sampling_params, use_tqdm=True)

    # STEP 4: Process results and calculate accuracy
    correct_count = 0
    total_count = len(final_outputs)
    for i, output in enumerate(final_outputs):
        eval_item = validation_set[i]
        generated_text = output.outputs[0].text
        model_choice = parse_json_answer(generated_text)

        # --- DATASET-SPECIFIC LOGIC ---
        if DATASET == "BBH":
            ground_truth_target = str(eval_item['target']).strip()
            ground_truth_letter = ground_truth_target[1:2] if ground_truth_target.startswith('(') else ground_truth_target
        else: # MMLU-Pro
            ground_truth_letter = chr(ord('A') + eval_item['answer_index'])

        if model_choice == ground_truth_letter:
            correct_count += 1
            
    accuracy = (correct_count / total_count) if total_count > 0 else 0
    # Return all three values as per your provided function
    return accuracy