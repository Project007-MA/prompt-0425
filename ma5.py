import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from deap import base, creator, tools
import random
from sklearn.metrics import accuracy_score
import time
import numpy as np
import warnings
from bayes_opt import BayesianOptimization
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

# Load data
train_data = pd.read_csv(r'D:\MAHESH\reworkcsluter\main_train.csv\main_train.csv')
test_data = pd.read_csv(r'D:\MAHESH\reworkcsluter\main_train.csv\main_test.csv')

# Model setup
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Prompt generation
def generate_prompt(question, answer):
    return f"Question: {question} Answer: {answer} Does this answer satisfy the question? [Yes/No]."

# Prediction
def make_prediction(prompt):
    try:
        prediction = classifier(prompt, truncation=True, max_length=128)[0]
        return prediction['label']
    except Exception as e:
        return "No"

# Accuracy
def evaluate_accuracy():
    correct = 0
    for _, row in test_data.iterrows():
        prompt = generate_prompt(row['question'], row['answer'])
        pred = make_prediction(prompt)
        if pred == "LABEL_1":
            correct += 1
    return correct / len(test_data)

# Optimization bounds
pbounds = {
    'length': (0, 1),
    'question_complexity': (0, 1)
}

# BO objective
def bayesian_objective(length, question_complexity):
    question = "What is AI?" if question_complexity < 0.5 else "Can you explain the principles of Artificial Intelligence in detail?"
    answer = "AI stands for Artificial Intelligence." if length < 0.5 else "Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans."
    prompt = generate_prompt(question, answer)
    sample = test_data.sample(n=5, random_state=42)
    correct = 0
    for _, row in sample.iterrows():
        prediction = make_prediction(prompt)
        if prediction == "LABEL_1":
            correct += 1
    return correct / len(sample)

# NSGA-II setup (with fixed best prompt)
def setup_nsga(best_prompt):
    try:
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
    except:
        pass  # Already created

    toolbox = base.Toolbox()
    toolbox.register("attr_prompt", lambda: best_prompt)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_prompt)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
    toolbox.register("select", tools.selNSGA2)

    def evaluate_individual(ind):
        fixed_prompt = ind[0]
        def generate_prompt_fixed(q, a): return fixed_prompt
        global generate_prompt
        original = generate_prompt
        generate_prompt = generate_prompt_fixed
        acc = evaluate_accuracy()
        generate_prompt = original
        return acc, 0, 0

    toolbox.register("evaluate", evaluate_individual)
    return toolbox

# Run model variant with options
def run_model_variant(use_bo=True, use_ga=True, use_nsga=True):
    if use_bo:
        optimizer = BayesianOptimization(f=bayesian_objective, pbounds=pbounds, verbose=0, random_state=42)
        optimizer.maximize(init_points=2, n_iter=3)
        best_params = optimizer.max['params']
    else:
        best_params = {'length': 0.5, 'question_complexity': 0.5}

    question = "What is AI?" if best_params['question_complexity'] < 0.5 else "Can you explain the principles of Artificial Intelligence in detail?"
    answer = "AI stands for Artificial Intelligence." if best_params['length'] < 0.5 else "Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans."
    best_prompt = generate_prompt(question, answer)

    if use_nsga:
        toolbox = setup_nsga(best_prompt)
        population = toolbox.population(n=4)
        for gen in range(2):
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.7:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values, child2.fitness.values
            for mutant in offspring:
                if random.random() < 0.2:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fits = map(toolbox.evaluate, invalid)
            for ind, fit in zip(invalid, fits):
                ind.fitness.values = fit
            population[:] = offspring

    return evaluate_accuracy()

# Collect multiple runs for statistical testing
proposed_accs = []
without_bo_accs = []
without_ga_accs = []
without_nsga_accs = []

print("Running multiple trials...")
for i in range(5):
    print(f"Trial {i+1}")
    proposed_accs.append(run_model_variant(True, True, True))
    without_bo_accs.append(run_model_variant(False, True, True))
    without_ga_accs.append(run_model_variant(True, False, True))
    without_nsga_accs.append(run_model_variant(True, True, False))

# Perform Wilcoxon tests
alpha = 0.05

def test_and_collect(name, group1, group2):
    stat, p = wilcoxon(group1, group2)
    return [name, stat, p, len(group1), alpha]

results = []
results.append(test_and_collect("Proposed vs Without BO", proposed_accs, without_bo_accs))
results.append(test_and_collect("Proposed vs Without GA", proposed_accs, without_ga_accs))
results.append(test_and_collect("Proposed vs Without NSGA-II", proposed_accs, without_nsga_accs))

# Display table
df = pd.DataFrame(results, columns=["Comparison", "W-statistic", "p-value", "Sample Size", "Significant level (α)"])
print("\nWilcoxon Signed-Rank Test Results:")
print(df.to_string(index=False))
