import os
import requests
import time
import argparse

# ==== ARGUMENT PARSING ====
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run coreference resolution using Ollama model.")

    parser.add_argument("--input-file", required=True, help="Path to a single input text file.")
    parser.add_argument("--output-folder", required=True, help="Root output folder to store all resolved results.")
    parser.add_argument("--model", default="llama3370b32K", help="Model name to use in Ollama (default: llama3370b32K)")

    # Prompt arguments and retries
    parser.add_argument("--person-prompt", help="Path to person coreference prompt.")
    parser.add_argument("--routes-prompt", help="Path to routes coreference prompt.")
    parser.add_argument("--location-prompt", help="Path to location coreference prompt.")
    parser.add_argument("--mot-prompt", help="Path to means of transportation coreference prompt.")
    parser.add_argument("--moc-prompt", help="Path to means of communication coreference prompt.")
    parser.add_argument("--organization-prompt", help="Path to organization coreference prompt.")
    parser.add_argument("--smuggleditems-prompt", help="Path to smuggled items coreference prompt.")

    parser.add_argument("--person-retries", type=int, default=1)
    parser.add_argument("--routes-retries", type=int, default=1)
    parser.add_argument("--location-retries", type=int, default=1)
    parser.add_argument("--mot-retries", type=int, default=1)
    parser.add_argument("--moc-retries", type=int, default=1)
    parser.add_argument("--organization-retries", type=int, default=1)
    parser.add_argument("--smuggleditems-retries", type=int, default=1)

    return parser.parse_args()

# ==== UTILITY FUNCTIONS ====
def log_message(log_path, message):
    with open(log_path, "a") as log:
        log.write(message + "\n")

def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def save_text(file_path, content):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

def call_ollama_api(prompt, model):
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    api_url = f"http://{host}/api/generate"

    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            return response.json().get("response", "No response")
        else:
            return f"[ERROR {response.status_code}]: {response.text}"
    except requests.exceptions.RequestException as e:
        return f"[REQUEST FAILED]: {e}"


# ==== MAIN PROCESS ====
def process_file(input_file, output_root, prompt_paths, retries_map, model_name):

    start_time = time.time()
    input_filename = os.path.basename(input_file)
    input_base = os.path.splitext(input_filename)[0]
    output_dir = os.path.join(output_root, input_base)
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "log.txt")
    log_message(log_file, f"===== Processing {input_filename} =====")

    current_text_path = os.path.join(output_dir, "coref_current.txt")
    current_text = load_text(input_file)
    save_text(current_text_path, current_text)

    PROMPT_ORDER = [
        ("person", prompt_paths.get("person"), retries_map.get("person", 1)),
        ("routes", prompt_paths.get("routes"), retries_map.get("routes", 1)),
        ("location", prompt_paths.get("location"), retries_map.get("location", 1)),
        ("mot", prompt_paths.get("mot"), retries_map.get("mot", 1)),
        ("moc", prompt_paths.get("moc"), retries_map.get("moc", 1)),
        ("organization", prompt_paths.get("organization"), retries_map.get("organization", 1)),
        ("smuggleditems", prompt_paths.get("smuggleditems"), retries_map.get("smuggleditems", 1))
    ]

    for prefix, prompt_path, retries in PROMPT_ORDER:
        if prompt_path is None:
            error_msg = f"[ERROR] No prompt path provided for {prefix}."
            log_message(log_file, error_msg)
            raise RuntimeError(error_msg)
            #continue

        if not os.path.exists(prompt_path):
            error_msg = f"[ERROR] Prompt file not found: {prompt_path}"
            log_message(log_file, error_msg)
            raise FileNotFoundError(error_msg)
            #continue

        for attempt in range(1, retries + 1):
            prompt_template = load_text(prompt_path)
            current_text = load_text(current_text_path)
            final_prompt = prompt_template.replace("{input_text}", current_text)

            log_message(log_file, f"Calling model with {prefix} prompt [{attempt}/{retries}]...")
            start = time.time()
            response = call_ollama_api(final_prompt, model=model_name)
            duration = time.time() - start
            log_message(log_file, f"Response received in {duration:.2f} sec")

            if response.startswith("[ERROR") or response.startswith("[REQUEST FAILED]"):
                log_message(log_file, f"Error in response: {response}")
                raise RuntimeError(f"Ollama API failed for {prefix} on attempt {attempt}")

            step_filename = f"{prefix}{attempt}_coref_current.txt"
            step_path = os.path.join(output_dir, step_filename)
            save_text(step_path, response)
            save_text(current_text_path, response)

    # Save final resolved version
    final_output_path = os.path.join(output_dir, "coref_final.txt")
    final_output = load_text(current_text_path)
    save_text(final_output_path, final_output)
    log_message(log_file, f"Final output saved: {final_output_path}")
    log_message(log_file, f"Completed processing for: {input_filename}")
    total_time = time.time() - start_time
    log_message(log_file, f" Total Time: {total_time:.2f} seconds")

# ==== ENTRY POINT ====
if __name__ == "__main__":
    args = parse_arguments()

    prompt_paths = {}
    retries_map = {}

    if args.person_prompt:
        prompt_paths["person"] = args.person_prompt
        retries_map["person"] = args.person_retries
    if args.routes_prompt:
        prompt_paths["routes"] = args.routes_prompt
        retries_map["routes"] = args.routes_retries
    if args.location_prompt:
        prompt_paths["location"] = args.location_prompt
        retries_map["location"] = args.location_retries
    if args.mot_prompt:
        prompt_paths["mot"] = args.mot_prompt
        retries_map["mot"] = args.mot_retries
    if args.moc_prompt:
        prompt_paths["moc"] = args.moc_prompt
        retries_map["moc"] = args.moc_retries
    if args.organization_prompt:
        prompt_paths["organization"] = args.organization_prompt
        retries_map["organization"] = args.organization_retries
    if args.smuggleditems_prompt:
        prompt_paths["smuggleditems"] = args.smuggleditems_prompt
        retries_map["smuggleditems"] = args.smuggleditems_retries

    if not any(prompt_paths.values()):
        print("No prompts provided. Exiting.")
    else:
        process_file(
            input_file=args.input_file,
            output_root=args.output_folder,
            prompt_paths=prompt_paths,
            retries_map=retries_map,
            model_name=args.model
        )
