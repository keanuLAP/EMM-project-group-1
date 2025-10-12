import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from tqdm import tqdm
import argparse

class NLIQuestionProcessor:
    def __init__(self, model_name="microsoft/deberta-v2-xlarge-mnli"):
        """
        Use a pre-trained NLI model to determine answer probabilities.
        These models are specifically trained to understand entailment relationships.
        """
        print(f"Loading NLI model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # NLI models typically output: [contradiction, neutral, entailment]
        self.label_mapping = {0: "contradiction", 1: "neutral", 2: "entailment"}

    def get_entailment_score(self, premise, hypothesis):
        """
        Get entailment score between premise and hypothesis.
        Higher score means the hypothesis follows from the premise.
        """
        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = softmax(logits, dim=-1)

            # Return entailment probability (index 2)
            entailment_prob = probs[0][2].item()

        return entailment_prob

    def get_answer_probabilities(self, context, question, answers):
        """
        For each answer, check how well it's entailed by the context+question.
        """
        # Create premise from context and question
        premise = f"{context} {question}"

        answer_scores = {}

        for ans_key, answer_text in answers.items():
            try:
                # Create hypothesis: "The answer is [answer_text]"
                hypothesis = f"The answer is {answer_text.lower()}"

                score = self.get_entailment_score(premise, hypothesis)
                answer_scores[ans_key] = score

            except Exception as e:
                print(f"Error processing answer {ans_key}: {e}")
                answer_scores[ans_key] = 0.1

        # Normalize scores to probabilities
        total_score = sum(answer_scores.values())
        if total_score > 0:
            answer_probs = {k: v/total_score for k, v in answer_scores.items()}
        else:
            answer_probs = {k: 1.0/len(answer_scores) for k in answer_scores.keys()}

        return answer_probs

    def classify_question_category(self, context, question):
        """
        Classify the question into one of the predefined categories using NLI.
        """
        # Use more specific category descriptions for better classification
        category_definitions = {
            "Professional Roles & Workplace Competence": "This question asks about job roles, leadership positions, technical skills, or workplace abilities",
            "Academic & Intellectual Abilities": "This question asks about mathematical skills, academic performance, logical thinking, or STEM abilities",
            "Caregiving & Emotional Competence": "This question asks about empathy, compassion, teaching children, or providing emotional support",
            "Violence & Aggression": "This question asks about physical violence, domestic abuse, fighting, or aggressive behavior",
            "Emotional Regulation & Mental Health": "This question asks about rational behavior, emotional outbursts, mental health, or psychological stability",
            "Physical Strength & Athletics": "This question asks about physical strength, athletic performance, or sports abilities",
            "Family & Personal Life": "This question asks about starting families, personal relationships, or family obligations"
        }

        # Create premise from context only (more important than question)
        premise = context

        category_scores = {}

        for category, definition in category_definitions.items():
            try:
                # Use more specific hypothesis based on the definition
                hypothesis = definition

                score = self.get_entailment_score(premise, hypothesis)
                category_scores[category] = score

            except Exception as e:
                print(f"Error processing category {category}: {e}")
                category_scores[category] = 0.1

        # Return the category with highest score
        best_category = max(category_scores, key=category_scores.get)
        return best_category

    def process_jsonl_file(self, input_file, output_file, limit=None):
        """
        Process the JSONL file using NLI-based scoring.
        """
        results = []

        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if limit:
                lines = lines[:limit]

            for line in tqdm(lines, desc="Processing with NLI model"):
                data = json.loads(line.strip())

                context = data['context']
                question = data['question']
                answers = {
                    'ans0': data['ans0'],
                    'ans1': data['ans1'],
                    'ans2': data['ans2']
                }

                try:
                    probs = self.get_answer_probabilities(context, question, answers)

                    data['answer_probabilities'] = probs
                    data['predicted_answer'] = max(probs, key=probs.get)
                    data['confidence'] = max(probs.values())

                    # Calculate probability spread (higher = more decisive)
                    prob_values = list(probs.values())
                    prob_values.sort(reverse=True)
                    data['probability_spread'] = prob_values[0] - prob_values[1]

                    # Classify question category
                    data['predicted_category'] = self.classify_question_category(context, question)

                except Exception as e:
                    print(f"Error processing example {data.get('example_id', 'unknown')}: {e}")
                    data['answer_probabilities'] = {'ans0': 0.33, 'ans1': 0.33, 'ans2': 0.33}
                    data['predicted_answer'] = 'ans1'
                    data['confidence'] = 0.33
                    data['probability_spread'] = 0.0
                    data['predicted_category'] = 'Unknown'

                results.append(data)

        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')

        print(f"Processed {len(results)} questions. Results saved to {output_file}")

        # Statistics
        correct_predictions = sum(1 for r in results if r['predicted_answer'] == f"ans{r['label']}")
        accuracy = correct_predictions / len(results) if results else 0
        avg_confidence = sum(r['confidence'] for r in results) / len(results) if results else 0
        avg_spread = sum(r['probability_spread'] for r in results) / len(results) if results else 0

        print(f"Accuracy: {accuracy:.3f} ({correct_predictions}/{len(results)})")
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"Average probability spread: {avg_spread:.3f} (higher = more decisive)")

def main():
    parser = argparse.ArgumentParser(description='Process questions with NLI model')
    parser.add_argument('--input', default='Gender_identity.jsonl', help='Input JSONL file')
    parser.add_argument('--output', default='Gender_identity_nli_probabilities.jsonl', help='Output JSONL file')
    parser.add_argument('--model', default='microsoft/deberta-v2-xlarge-mnli',
                       help='NLI model name')
    parser.add_argument('--limit', type=int, help='Limit number of questions (for testing)')

    args = parser.parse_args()

    processor = NLIQuestionProcessor(model_name=args.model)
    processor.process_jsonl_file(args.input, args.output, limit=args.limit)

if __name__ == "__main__":
    main()

# python process_questions_nli.py --model "facebook/bart-large-mnli"
# python process_questions_nli.py --model "microsoft/deberta-v2-xxlarge-mnli"