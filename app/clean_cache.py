import os
import shutil

def clear_directory(directory_path):
    """
    Removes all files and subdirectories within the given directory.
    The directory itself is not removed.
    """
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return

    for item_name in os.listdir(directory_path):
        if item_name == 'keep':
            print(f"Skipping file: {os.path.join(directory_path, item_name)}")
            continue

        item_path = os.path.join(directory_path, item_name)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
                print(f"Deleted file: {item_path}")
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"Deleted directory: {item_path}")
        except Exception as e:
            print(f"Failed to delete {item_path}. Reason: {e}")

if __name__ == "__main__":
    # Assuming 'tmp' and 'outputs' are in the same directory as the script
    # or provide absolute paths if they are elsewhere.
    current_script_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Construct full paths relative to the script's location
    # Adjust these paths if your 'tmp' and 'outputs' directories are located elsewhere
    tmp_dir = os.path.join(current_script_directory, "tmp")
    outputs_dir = os.path.join(current_script_directory, "outputs")

    print(f"Attempting to clear contents of: {tmp_dir}")
    clear_directory(tmp_dir)
    
    print(f"\nAttempting to clear contents of: {outputs_dir}")
    clear_directory(outputs_dir)
    
    print("\nCleanup process finished.")