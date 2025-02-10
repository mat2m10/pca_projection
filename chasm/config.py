from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Retrieve values
PATH_data = os.getenv("PATH_data")
# Optional: Store in a dictionary for easy access
CONFIG = {
    "PATH_data": PATH_data
}