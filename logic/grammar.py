import spacy
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from difflib import SequenceMatcher

# Load custom grammar correction model
class GrammarCorrector:
    def __init__(self):
        self.model_name = "vennify/t5-base-grammar-correction"
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)

    def correct_grammar(self, text):
        input_text = "grammar: " + text
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt", truncation=True)

        with torch.no_grad():
            outputs = self.model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)

        corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return corrected

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def get_similarity_score(original, corrected):
    return SequenceMatcher(None, original.lower(), corrected.lower()).ratio()

def get_feedback(score, analysis):
    tense_issue = False
    subject_verb_issue = False

    # Check for tense issues and subject-verb agreement issues
    for token in analysis:
        if token["pos"] == "VERB":
            if token["lemma"] in ["be", "have"]:  # Common verbs that may indicate tense issues
                tense_issue = True
        if token["dep"] == "nsubj" and token["pos"] != "VERB":
            subject_verb_issue = True

    # Adjust feedback based on analysis
    if tense_issue or subject_verb_issue:
        score -= 0.2  # Penalize if there's a tense or subject-verb agreement issue

    if score < 0.25:
        return "❌ Worst: Too many mistakes."
    elif score < 0.5:
        return "⚠️ Still Bad: Needs a lot of improvement."
    elif score < 0.75:
        return "🙂 Average: Some mistakes."
    else:
        return "✅ Good/Excellent: Well done!"

def correct_and_analyze(text):
    grammar_corrector = GrammarCorrector()
    corrected = grammar_corrector.correct_grammar(text)
    score = get_similarity_score(text, corrected)
    doc = nlp(corrected)
    
    # Collect token analysis using spaCy
    analysis = [{
        "text": token.text,
        "lemma": token.lemma_,
        "pos": token.pos_,
        "tag": token.tag_,
        "dep": token.dep_,
        "explanation": spacy.explain(token.tag_)
    } for token in doc]

    feedback = get_feedback(score, analysis)

    return corrected, analysis, score, feedback
