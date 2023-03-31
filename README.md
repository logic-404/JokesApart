# JokesApart
JokesApart is a Flask web application that uses machine learning to generate jokes. The app uses a pre-trained BERT model and cosine similarity to find the closest match to a user's input query from a pre-existing dataset of jokes.

## Components
The project contains the following components:

- `app.py`: the main Flask application file that defines the web endpoints and uses the BERT model to generate jokes.
- `jokes.tsv`: a dataset of jokes used by the application for joke generation.
- `sentence_embeddings.pkl`: precomputed sentence embeddings for the jokes dataset, used for efficient cosine similarity calculations.
- `templates/index.html`: a simple HTML file that contains the web form for user input and displays the generated joke.

## Usage
To run the app, simply clone the repository, install the required dependencies, and run the `app.py` file.

```bash
git clone https://github.com/logic-404/JokesApart.git
cd JokesApart
pip install -r requirements.txt
python app.py
```
The app is currently set to run in debug mode, but can be easily configured for production deployment using the gevent library.

## Contributing
Contributions are always welcome! Please feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.
