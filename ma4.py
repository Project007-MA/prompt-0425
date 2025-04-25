import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from deap import base, creator, tools
import random
from sklearn.metrics import accuracy_score
import time
import numpy as np
import warnings
from bayes_opt import BayesianOptimization

warnings.filterwarnings("ignore")

# Load datasets
train_data = pd.read_csv(r'D:\MAHESH\reworkcsluter\main_train.csv\main_train.csv')
test_data = pd.read_csv(r'D:\MAHESH\reworkcsluter\main_train.csv\main_test.csv')

# Model setup (using bert-base-uncased, which is a general model)
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create classification pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Generate prompt
def generate_prompt(question, answer):
    return f"Question: {question} Answer: {answer} Does this answer satisfy the question? [Yes/No]."

# Make prediction using the model
def make_prediction(prompt):
    try:
        prediction = classifier(prompt, truncation=True, max_length=128)[0]
        return prediction['label']
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "No"

# Evaluate accuracy
def evaluate_accuracy():
    correct_predictions = 0
    for _, row in test_data.iterrows():
        prompt = generate_prompt(row['question'], row['answer'])
        prediction = make_prediction(prompt)
        if prediction == "LABEL_1":  # Assuming LABEL_1 ~ "Yes"
            correct_predictions += 1
    return correct_predictions / len(test_data),

# Evaluate efficiency
def evaluate_efficiency():
    start_time = time.time()
    for _, row in test_data.iterrows():
        prompt = generate_prompt(row['question'], row['answer'])
        make_prediction(prompt)
    return time.time() - start_time,

# Evaluate interpretability as prompt length
def evaluate_interpretability():
    total_length = 0
    for _, row in test_data.iterrows():
        prompt = generate_prompt(row['question'], row['answer'])
        total_length += len(prompt.split())
    return total_length / len(test_data),

# ------------------------ Hybrid Optimization ------------------------ #

# Define Bayesian objective function
def bayesian_objective(length, question_complexity):
    question = "What is AI?" if question_complexity < 0.5 else "Can you explain the principles of Artificial Intelligence in detail?"
    answer = "AI stands for Artificial Intelligence." if length < 0.5 else "Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans."
    prompt = generate_prompt(question, answer)
    
    # Use a small sample for faster evaluation
    sample = test_data.sample(n=5, random_state=42)
    correct = 0
    for _, row in sample.iterrows():
        prediction = make_prediction(prompt)
        if prediction == "LABEL_1":
            correct += 1
    return correct / len(sample)

# Set parameter bounds
pbounds = {
    'length': (0, 1),
    'question_complexity': (0, 1)
}

# Run Bayesian Optimization
optimizer = BayesianOptimization(f=bayesian_objective, pbounds=pbounds, verbose=2, random_state=42)
optimizer.maximize(init_points=3, n_iter=5)

# Use best prompt to seed GA
best_params = optimizer.max['params']
best_question = "What is AI?" if best_params['question_complexity'] < 0.5 else "Can you explain the principles of Artificial Intelligence in detail?"
best_answer = "AI stands for Artificial Intelligence." if best_params['length'] < 0.5 else "Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans."
best_prompt = generate_prompt(best_question, best_answer)

# ------------------------ NSGA-II Setup ------------------------ #

creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_prompt", lambda: best_prompt)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_prompt)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
toolbox.register("select", tools.selNSGA2)

# Multi-objective evaluation function
def evaluate_individual(individual):
    def generate_prompt_fixed(q, a): return individual[0]
    global generate_prompt
    original_prompt_function = generate_prompt
    generate_prompt = generate_prompt_fixed

    acc = evaluate_accuracy()
    eff = evaluate_efficiency()
    interp = evaluate_interpretability()

    generate_prompt = original_prompt_function
    return acc[0], eff[0], interp[0]

toolbox.register("evaluate", evaluate_individual)

# Run NSGA-II
def run_nsga2():
    population = toolbox.population(n=10)
    generations = 5

    for gen in range(generations):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalids = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalids))
        for ind, fit in zip(invalids, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring
        best_individual = tools.selBest(population, 1)[0]
        print(f"Gen {gen}: Best prompt: {best_individual}, Fitness: {best_individual.fitness.values}")

run_nsga2()

# ------------------------ Evaluation Functions ------------------------ #

def generalization_test():
    generalization_data = pd.read_csv('main_train.csv\\generalization_test_data.csv')
    correct_predictions = 0
    for _, row in generalization_data.iterrows():
        prompt = generate_prompt(row['question'], row['answer'])
        prediction = make_prediction(prompt)
        if prediction == "LABEL_1":
            correct_predictions += 1
    print(f"Generalization Accuracy: {100 * correct_predictions / len(generalization_data):.2f}%")

def diagnostic_evaluation():
    diagnostic_data = pd.read_csv('main_train.csv\\diagnostic_test_data.csv')
    correct_predictions = 0
    for _, row in diagnostic_data.iterrows():
        prompt = generate_prompt(row['question'], row['answer'])
        prediction = make_prediction(prompt)
        if prediction == "LABEL_1":
            correct_predictions += 1
    print(f"Diagnostic Accuracy: {100 * correct_predictions / len(diagnostic_data):.2f}%")

# Run final evaluations
generalization_test()
diagnostic_evaluation()
