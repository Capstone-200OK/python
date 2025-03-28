import os
import time
import nltk

from file_utils import (
    display_directory_tree,
    collect_file_paths,
    separate_files_by_type,
    read_file_data
)
from data_processing_common import (
    compute_operations,
    execute_operations,
    process_files_by_date,
    process_files_by_type
)
# GPT-based text processing (unchanged)
from text_data_processing import (
    process_text_files
)
# Image processing that requires image_url_map
from image_data_processing import (
    process_image_files
)

nltk.download('averaged_perceptron_tagger_eng')

def ensure_nltk_data():
    """
    Make sure NLTK data (stopwords, punkt, etc.) is installed.
    """
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)

def get_yes_no(prompt):
    while True:
        response = input(prompt).strip().lower()
        if response in ('yes', 'y'):
            return True
        elif response in ('no', 'n'):
            return False
        elif response == '/exit':
            print("Exiting program.")
            exit()
        else:
            print("Please enter 'yes' or 'no'. To exit, type '/exit'.")

def get_mode_selection():
    while True:
        print("Please choose the mode to organize your files:")
        print("1. By Content (GPT-based text, naive images)")
        print("2. By Date")
        print("3. By Type")
        response = input("Enter 1, 2, or 3 (or type '/exit' to exit): ").strip()
        if response == '/exit':
            print("Exiting program.")
            exit()
        elif response == '1':
            return 'content'
        elif response == '2':
            return 'date'
        elif response == '3':
            return 'type'
        else:
            print("Invalid selection. Please enter 1, 2, or 3.")

def main():
    ensure_nltk_data()

    dry_run = True

    print("-" * 50)
    print("NOTE: Silent mode logs all outputs to a text file instead of displaying them.")
    silent_mode = get_yes_no("Would you like to enable silent mode? (yes/no): ")
    if silent_mode:
        log_file = 'operation_log.txt'
    else:
        log_file = None

    while True:
        if not silent_mode:
            print("-" * 50)

        input_path = input("Enter the path of the directory you want to organize: ").strip()
        while not os.path.exists(input_path):
            message = f"Input path {input_path} does not exist. Please enter a valid path."
            if silent_mode and log_file:
                with open(log_file, 'a') as f:
                    f.write(message + '\n')
            else:
                print(message)
            input_path = input("Enter the path of the directory you want to organize: ").strip()

        message = f"Input path recognized: {input_path}"
        if silent_mode and log_file:
            with open(log_file, 'a') as f:
                f.write(message + '\n')
        else:
            print(message)

        if not silent_mode:
            print("-" * 50)

        output_path = input("Enter the path to store organized files/folders (press Enter to use 'organized_folder'): ").strip()
        if not output_path:
            output_path = os.path.join(os.path.dirname(input_path), 'organized_folder')

        message = f"Output path set to: {output_path}"
        if silent_mode and log_file:
            with open(log_file, 'a') as f:
                f.write(message + '\n')
        else:
            print(message)

        if not silent_mode:
            print("-" * 50)

        start_time = time.time()
        file_paths = collect_file_paths(input_path)
        end_time = time.time()

        message = f"Time taken to load file paths: {end_time - start_time:.2f} seconds"
        if silent_mode and log_file:
            with open(log_file, 'a') as f:
                f.write(message + '\n')
        else:
            print(message)

        if not silent_mode:
            print("-" * 50)
            print("Directory tree before organizing:")
            display_directory_tree(input_path)
            print("*" * 50)

        while True:
            mode = get_mode_selection()

            if mode == 'content':
                # Separate images and text
                image_files, text_files = separate_files_by_type(file_paths)

                # Prepare text input for GPT
                text_tuples = []
                for fp in text_files:
                    text_content = read_file_data(fp)
                    if text_content is None:
                        msg = f"Unsupported or unreadable text file: {fp}"
                        if silent_mode and log_file:
                            with open(log_file, 'a') as f:
                                f.write(msg + '\n')
                        else:
                            print(msg)
                        continue
                    text_tuples.append((fp, text_content))

                # (2) Process images with GPT-based approach (requires image_url_map)
                data_images = process_image_files(
                    image_files,
                    silent=silent_mode,
                    log_file=log_file
                )

                # (3) Process text files with GPT
                data_texts = process_text_files(
                    text_tuples,
                    silent=silent_mode,
                    log_file=log_file
                )

                renamed_files = set()
                processed_files = set()
                all_data = data_images + data_texts

                operations = compute_operations(
                    all_data,
                    output_path,
                    renamed_files,
                    processed_files
                )

            elif mode == 'date':
                operations = process_files_by_date(
                    file_paths,
                    output_path,
                    dry_run=False,
                    silent=silent_mode,
                    log_file=log_file
                )
            elif mode == 'type':
                operations = process_files_by_type(
                    file_paths,
                    output_path,
                    dry_run=False,
                    silent=silent_mode,
                    log_file=log_file
                )
            else:
                print("Invalid mode selected.")
                return

            print("-" * 50)
            message = "Proposed directory structure:"
            if silent_mode and log_file:
                with open(log_file, 'a') as f:
                    f.write(message + '\n')
            else:
                print(message)
                print(os.path.abspath(output_path))

                def simulate_tree(ops, base):
                    t = {}
                    for op in ops:
                        rel_path = os.path.relpath(op['destination'], base)
                        parts = rel_path.split(os.sep)
                        node = t
                        for p in parts:
                            if p not in node:
                                node[p] = {}
                            node = node[p]
                    return t

                def print_tree(tree, prefix=''):
                    keys = list(tree.keys())
                    for i, k in enumerate(keys):
                        symbol = '└── ' if i == len(keys) - 1 else '├── '
                        print(prefix + symbol + k)
                        if tree[k]:
                            extension = '    ' if i == len(keys) - 1 else '│   '
                            print_tree(tree[k], prefix + extension)

                simulated = simulate_tree(operations, output_path)
                print_tree(simulated)
                print("-" * 50)

            proceed = get_yes_no("Would you like to proceed with these changes? (yes/no): ")
            if proceed:
                os.makedirs(output_path, exist_ok=True)
                msg = "Performing file operations..."
                if silent_mode and log_file:
                    with open(log_file, 'a') as f:
                        f.write(msg + '\n')
                else:
                    print(msg)

                execute_operations(
                    operations,
                    dry_run=False,
                    silent=silent_mode,
                    log_file=log_file
                )

                msg = "The files have been organized successfully."
                if silent_mode and log_file:
                    with open(log_file, 'a') as f:
                        f.write("-" * 50 + '\n' + msg + '\n' + "-" * 50 + '\n')
                else:
                    print("-" * 50)
                    print(msg)
                    print("-" * 50)
                break
            else:
                another_sort = get_yes_no("Would you like to choose another sorting method? (yes/no): ")
                if another_sort:
                    continue
                else:
                    print("Operation canceled by the user.")
                    break

        another_directory = get_yes_no("Would you like to organize another directory? (yes/no): ")
        if not another_directory:
            break

if __name__ == '__main__':
    main()
