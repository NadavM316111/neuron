import logging

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_task(task_name, is_useful):
    logging.info(f"Task: {task_name}, Usefulness: {is_useful}")

# Example usage
log_task("Set up logging system", True)

print('PASS')