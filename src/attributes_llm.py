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
    {"name": "chain_of_verification", "type": "instruction", "text": "First, generate an answer. Second, verify its accuracy by cross-referencing with reliable information and facts. Finally, present the verified answer."},
    # --- Index 10:
    {"name": "step_back_prompting", "type": "instruction", "text": "Before you answer, take a step back and think about the abstract concepts or principles related to the question. Use that reasoning to construct your final answer."},
    # --- Index 11:
    {"name": "state_uncertainty", "type": "instruction", "text": "If you are unsure about any part of the answer, you must state your uncertainty or the limitations of your knowledge."},

    # --- Group 3: Output Style, Verbosity, Motivation, Stakes & Tone ---
    # --- Index 12:
    {"name": "no_fluff", "type": "instruction", "text": "Get straight to the point. No conversational filler."},
    # --- Index 13:
    {"name": "optimize_token_quantity", "type": "instruction", "text": "Be concise."},
    # --- Index 14:
    {"name": "clarity_seeking", "type": "instruction", "text": "Explain to me like I'm 11 years old."},
    # --- Index 15:
    {"name": "analogy_reasoning", "type": "instruction", "text": "Explain this concept using a helpful analogy."},
    # --- Index 16:
    {"name": "empathetic_tone", "type": "instruction", "text": "Adopt a compassionate and understanding tone in your response."},
    # --- Index 17:
    {"name": "emotion_prompting", "type": "instruction", "text": "Respond with a reassuring and calm tone."},
    # --- Index 18:
    {"name": "high_stakes_context", "type": "instruction", "text": "This is very important for my career. A great answer is critical."},
    # --- Index 19:
    {"name": "establishes_rewards", "type": "instruction", "text": "You will be rewarded for following the instructions."},
    # --- Index 20:
    {"name": "tipping", "type": "instruction", "text": "I will tip $10 for an answer that follows instructions and is accurate."},
    # --- Index 21:
    {"name": "penalty_enforcement", "type": "instruction", "text": "Failure to follow these instructions will result in a penalty."},
    # --- Index 22:
    {"name": "low_balling_challenge", "type": "instruction", "text": "This task is notoriously difficult for AI models. I bet you can't do it correctly, but prove me wrong."},
    # --- Index 23:
    {"name": "uses_positive_tone", "type": "instruction", "text": "Let's find an excellent solution together."},
    # --- Index 24:
    {"name": "humorous_tone", "type": "instruction", "text": "Add a touch of humor to your response."},
    # --- Index 25:
    {"name": "system2_attention", "type": "instruction", "text": "Access your deep knowledge base on this topic before answering."},

    # --- Group 4: Specific Content & Formatting Requirements ---
    # --- Index 26:
    {"name": "summarize_context", "type": "instruction", "text": "Before answering your question, summarize the context first."},
    # --- Index 27:
    {"name": "cite_sources", "type": "instruction", "text": "For every claim you make, cite the source."},
    # --- Index 28:
    {"name": "highlight_keywords", "type": "instruction", "text": "**Bold** the most important keywords in your response."},
    # --- Index 29:
    {"name": "output_scoping_by_count", "type": "instruction", "text": "Provide a list of items."},
    # --- Index 30:
    {"name": "correction_prompting", "type": "instruction", "text": "Review your answer for factual errors or inconsistencies and correct them."},
    # --- Index 31:
    {"name": "enforces_explicit_constraints", "type": "instruction", "text": "Adhere strictly to all specified constraints."},
    # --- Index 32:
    {"name": "numbered_steps_output", "type": "instruction", "text": "Provide your answer in numbered steps."},
    # --- Index 33:
    {"name": "bullet_points_output", "type": "instruction", "text": "List the key features in bullet points."},

    # --- Group 5: Responsible AI & Safety ---
    # --- Index 34:
    {"name": "mitigate_bias", "type": "instruction", "text": "Ensure your answer is free from any gender, racial, or cultural biases and stereotypes."},
    # --- Index 35:
    {"name": "ensure_safety", "type": "instruction", "text": "Generate safe and appropriate content, avoiding any harmful or offensive language."},

    # --- Group 6: Prefixes ---
    # --- Index 36:
    {"name": "first_command_prefix", "type": "prefix", "text": "My first command is: "},
    # --- Index 37:
    {"name": "follow_instructions_prefix", "type": "prefix", "text": "Carefully follow every instruction below: "},

    # --- Group 7: Suffixes ---
    # --- Index 38:
    {"name": "please", "type": "suffix", "text": " Please!"},
    # --- Index 39:
    {"name": "thanks", "type": "suffix", "text": " Thank you!"},
    # --- Index 40:
    {"name": "polite_closure", "type": "suffix", "text": " I appreciate your effort."},
    # --- Index 41:
    {"name": "praise", "type": "suffix", "text": " You are the best!"},
    # --- Index 42:
    {"name": "inspirational_tone", "type": "suffix", "text": " Let's make it happen."},
    # --- Index 43:
    {"name": "directive", "type": "suffix", "text": " Now start working hard!"},
    # --- Index 44:
    {"name": "think_longer", "type": "suffix", "text": " Think longer and harder!"},
    # --- Index 45:
    {"name": "add_hashtags_instructive", "type": "suffix", "text": " #concise #accuracy #factual"},
    # --- Index 46:
    {"name": "add_hashtags_social", "type": "suffix", "text": " #topic"},

    # --- Group 8: Global Transforms ---
    # --- Index 47:
    {"name": "enumerate_instructions", "type": "transform"},
]

NUM_ATTRIBUTES = len(ATTRIBUTES)
BASE_PROMPT = 'You will be given a multiple-choice question. First follow the instructions and please show your choice at the end in a JSON object with only the choice letter, e.g., {"answer": "C"}'

def apply_attributes(vector: list[int], question="") -> str:
    """
    Constructs a self-contained instruction string from a binary vector.

    This function processes prefixes, instructions, and suffixes based on the
    active attributes in the vector. It no longer includes the BASE_PROMPT
    or the question, and the JSON transform has been removed.

    Args:
        vector: A binary list where each index corresponds to an attribute.
        question: This parameter is ignored but kept for compatibility.

    Returns:
        A formatted string of combined instructions.
    """
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

    # Step 2: Assemble the parts of the instruction string

    # Join all active prefixes into a single string
    prefix_str = " ".join(p.strip() for p in prefixes)

    # Process the main instructions based on the enumeration transform
    instruction_str = ""
    if instructions:
        if 'enumerate_instructions' in active_transforms:
            # Format as a numbered list, which is naturally multi-line
            instruction_str = "\n".join(f"{i+1}. {inst.strip()}" for i, inst in enumerate(instructions))
        else:
            # Format as a single line of text
            instruction_str = " ".join(inst.strip() for inst in instructions)

    # Concatenate all active suffixes.
    # Note: Suffixes in the list have leading spaces, so direct concatenation is correct.
    suffix_str = "".join(suffixes)

    # Step 3: Combine all processed parts into the final string
    
    # Use a list to cleanly join the main components (prefix and instructions)
    # This prevents extra newlines if one part is missing.
    main_components = []
    if prefix_str:
        main_components.append(prefix_str)
    if instruction_str:
        main_components.append(instruction_str)
        
    # Join the main parts with a newline. This is effective because prefixes
    # often end with ":" and enumerated instructions are already multi-line.
    main_content = "\n".join(main_components)

    # Append the suffixes to the combined main content
    final_string = main_content + suffix_str
    
    return final_string



