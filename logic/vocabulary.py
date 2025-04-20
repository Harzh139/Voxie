from nltk.corpus import wordnet as wn

class VocabularyInfoFetcher:
    def __init__(self):
        pass

    def get_vocabulary_info(self, word):
        # Check if the word is not empty
        if not word:
            return {"error": "Please provide a valid word."}

        synsets = wn.synsets(word)
        if not synsets:
            return {"error": f"No information found for the word '{word}'."}

        # Get the definition and examples
        definition = synsets[0].definition()
        examples = synsets[0].examples()
        example = examples[0] if examples else "No example available."

        # Get synonyms and antonyms
        synonyms = set()
        antonyms = set()

        for syn in synsets:
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace("_", " "))
                for ant in lemma.antonyms():
                    antonyms.add(ant.name().replace("_", " "))

        # Return a more structured response
        vocab_info = {
            "word": word.capitalize(),
            "definition": definition,
            "synonyms": list(synonyms)[:5] if synonyms else ["None found"],
            "antonyms": list(antonyms)[:5] if antonyms else ["None found"],
            "example": example
        }
        
        return vocab_info
