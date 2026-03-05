# prosody.ai
CS109 Final Project

Try out the classifier for yourself: https://poem-classifier-caiiris.onrender.com/

Data Sources & Acknowledgments
This project relies on a custom-built, enriched dataset. The data pipeline was made possible by the following sources:

Base Corpus: The raw poetic text and author names were sourced from the Poetry Foundation Poems dataset by John Titor on Kaggle (https://www.kaggle.com/datasets/tgdivy/poetry-foundation-poems?resource=download).

Historical Metadata: Publication eras and temporal metadata were algorithmically retrieved by scraping Poetry Foundation and Wikipedia.

Lexical Abstraction Index: The concrete-to-abstract ratio feature utilizes the concreteness ratings from: Brysbaert, M., Warriner, A. B., & Kuperman, V. (2014). Concreteness ratings for 40 thousand generally known English word lemmas. Behavior Research Methods.

NLP Processing: Part-of-Speech tagging and tokenization were performed using the Natural Language Toolkit (NLTK).
