Cascading Hybrid Sentiment Analysis on X (Twitter): RoBERTa with VADER-based Prediction Refinement
This repository contains the refactored code for a cascading hybrid sentiment analysis model. It combines the power of a fine-tuned RoBERTa model with a VADER-based prediction refinement mechanism to improve sentiment detection on X (Twitter) data.

Project Structure
your_project_root/
├── README.md
├── main.py # Main script to run the hybrid model
├── utils/ # Directory for utility functions and model components
│ ├── **init**.py # Makes 'utils' a Python package
│ ├── preprocessing.py # Contains data cleaning and preprocessing functions
│ ├── roberta_model.py # Handles RoBERTa model loading, training, and prediction
│ └── vader_model.py # Handles VADER sentiment scoring
└── data/
└── sentiment140-dataset-800records.csv # Your dataset file

Setup Instructions
Clone the Repository (or create the structure):
Create the directory structure as shown above on your local machine.

Place Your Dataset:
Place your sentiment140-dataset-800records.csv file inside the data/ directory.

Create and Activate a Virtual Environment (Recommended):

python -m venv venv

# On Windows:

# venv\Scripts\activate

# On macOS/Linux:

# source venv/bin/activate

Install Dependencies:
While your virtual environment is active, install the required Python packages:

pip install pandas numpy scikit-learn torch transformers nltk tqdm

# For PyTorch (CPU-only):

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# If you have an NVIDIA GPU, visit https://pytorch.org/get-started/locally/ for the correct command.

Download NLTK Data:
The nltk.sentiment.vader module requires a lexicon. The main.py script will attempt to download it, but if you encounter issues, you can run this manually:

python -c "import nltk; nltk.download('vader_lexicon')"

Running the Hybrid Model
Update DATASET_PATH in main.py:
Open main.py and ensure the DATASET_PATH variable points correctly to your dataset file relative to main.py. For the recommended structure, it should be 'data/sentiment140-dataset-800records.csv'.

Execute the Main Script:
From your project's root directory (where main.py is located) and with your virtual environment activated, run:

python main.py

The script will output the performance metrics for the standalone RoBERTa, standalone VADER, and the hybrid model.
