# CareerAssistant

An intelligent interview system combining RAG (Retrieval-Augmented Generation) and fine-tuned models to provide personalized technical interviews.

## Features

- Background-aware interview question generation
- Real-time answer evaluation
- Personalized feedback
- Web-based interface

## Prerequisites

- Python 3.9+
- OpenAI API Key
- Pinecone API Key
- Internet connection for API access

## Installation

1. Clone the repository:

```bash
git clone [https://github.com/yourusername/CareerAssistant.git](https://github.com/HazelW777/CareerAssistant.git)
cd CareerAssistant
```

2. Create and activate virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create `.env` file in project root:

```plaintext
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=your_pinecone_environment
```

## Data Setup

1. Download the data folder from Google Drive: https://drive.google.com/file/d/14821YqVRg3JE1oMeWiHaG0SZWY9E_N89/view?usp=drive_link

2. Place the downloaded data folder in the project root directory. The structure should look like: 

```
CareerAssistant/
├── data/
│   ├── raw/
│   │   └── interview_questions.json
│   └── processed/
│       └── model_artifacts/
│           └── model_epoch_3/
├── src/
├── scripts/
...
```

## Usage

1. Process the data (if using new data):

```bash
python scripts/prepare_training_data.py
```

2. Train the model (if needed):

```bash
python scripts/train_model.py
```

3. Start the web interface:

```bash
streamlit run src/web/app.py
```

## Test Examples

You can test the system with these sample inputs:

1. Junior Developer:

```bash
I’m a recent computer science graduate with a focus on Python and Django for backend development, preparing for campus recruitment interviews.
```

2. Senior Engineer:

```bash
I’m a backend engineer with 5 years of experience, specializing in distributed systems and microservices architecture, using Go and Java.
```

## Model Performance

- Field Classification: 98.99%
- Tier Classification: 100.00%
- Subfield Classification: 42.23%
- Overall Average: 80.41%

## API References

The system uses:

- OpenAI API for text generation
- Pinecone for vector similarity search
- Hugging Face's transformers for model fine-tuning

## Troubleshooting

Common issues and solutions:

1. Model loading error:

   - Ensure model artifacts are properly generated after training
   - Check path in `src/config.py`

2. API connection issues:

   - Verify API keys in `.env`
   - Check internet connection
   - Confirm Pinecone environment settings

3. Memory issues during training:
   - Reduce batch size in `train_model.py`
   - Use smaller dataset for initial testing

## Contributing

Feel free to submit issues and enhancement requests.

## License

[MIT License](LICENSE)

## Contact

For questions or feedback, please contact [Your Contact Information].
