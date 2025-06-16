
import random
import numpy as np
from sklearn.impute import SimpleImputer
import streamlit as st
from utils.model_utils import preprocess_user_input, y_map_inv
from utils.model_utils import scaler, features_to_use, y_map_inv
# from utils.predict_utils import predict_user_input_knn, predict_user_input_ann


def generate_random_chromosomes(user_input, n=10):
    min_normal_values = {
        'Sleep Duration': 6,
        'Physical Activity Level': 30,
        'BMI Category': 0,
        'Daily Steps': 4200,
        'Systolic_BP': 115,
        'Diastolic_BP': 75
    }

    max_normal_values = {
        'Sleep Duration': 8.5,
        'Physical Activity Level': 90,
        'BMI Category': 2,
        'Daily Steps': 10000,
        'Systolic_BP': 140,
        'Diastolic_BP': 95
    }

    mean_normal_values = {
        'Sleep Duration': 7.358447,
        'Physical Activity Level': 57.949772,
        'BMI Category': 0,
        'Daily Steps': 6852.968037,
        'Systolic_BP': 124.045662,
        'Diastolic_BP': 81
    }

    bmi_map = {'Normal': 0, 'Underweight': 1, 'Overweight': 2, 'Obese': 3}
    reverse_bmi_map = {v: k for k, v in bmi_map.items()}

    chromosomes = []

    for _ in range(n):
        chromosome = {}
        for key, user_val in user_input.items():
            if key in min_normal_values:
                min_val = min_normal_values[key]
                max_val = max_normal_values[key]
                mean_val = mean_normal_values[key]

                if key == 'BMI Category':
                    user_val = bmi_map.get(user_val, 3)
                    if user_val < min_val:
                        val = random.randint(min_val, mean_val)
                    elif user_val > max_val:
                        val = random.randint(mean_val, max_val)
                    else:
                        val = user_val
                    chromosome[key] = reverse_bmi_map[val]
                else:
                    # Tentukan pembulatan jika key termasuk fitur yang perlu dibulatkan
                    is_integer_feature = key in [
                        'Physical Activity Level',
                        'Daily Steps',
                        'Systolic_BP',
                        'Diastolic_BP'
                    ]

                    if user_val < min_val:
                        val = random.uniform(min_val, mean_val)
                    elif user_val > max_val:
                        val = random.uniform(mean_val, max_val)
                    else:
                        lower = mean_val * 0.9
                        upper = mean_val * 1.1
                        val = random.uniform(lower, upper)

                    # Bulatkan jika termasuk fitur integer
                    chromosome[key] = int(round(val)) if is_integer_feature else round(val, 6)

        chromosomes.append(chromosome)


    completed_chromosomes = []

    for chrom in chromosomes:
        chrom_complete = chrom.copy()  # salin kromosom aslinya

        # Tambahkan fitur user yang belum ada di kromosom
        for key, val in user_input.items():
            if key not in chrom_complete:
                chrom_complete[key] = val

        completed_chromosomes.append(chrom_complete)


    return completed_chromosomes

def evaluate_chromosomes(chromosomes_list, scaler, features_to_use, knn_model, ann_model, target_class=0):
    results = []

    # Bobot fitness untuk masing-masing model
    ann_weight = 0.7
    knn_weight = 0.3

    imputer = SimpleImputer(strategy='most_frequent')  # atau 'mean' jika semua numerik

    for chrom in chromosomes_list:
        chrom_encoded = chrom.copy()
        print("Chrom keys:", chrom.keys())

        # Preprocessing user input
        X_input = preprocess_user_input(chrom, scaler, features_to_use)
        # Setelah X_input = preprocess_user_input(...)
        print("X_input shape:", X_input.shape)
        print("Expected features:", len(features_to_use))
        print("Feature names:", features_to_use)


        # Tangani missing value
        X_input = imputer.fit_transform(X_input)  # pastikan X_input jadi array 2D dan tidak mengandung NaN

        # Prediksi oleh KNN
        knn_proba = knn_model.predict_proba(X_input)[0]
        knn_fitness = knn_proba[target_class]
        knn_pred = int(np.argmax(knn_proba))

        # Prediksi oleh ANN
        ann_proba = ann_model.predict(X_input, verbose=0)[0]
        ann_fitness = ann_proba[target_class]
        ann_pred = int(np.argmax(ann_proba))

        # Simpan semua info
        results.append({
            'chromosome': chrom,
            'knn_pred': knn_pred,
            'knn_fitness': knn_fitness,
            'ann_pred': ann_pred,
            'ann_fitness': ann_fitness,
            'avg_fitness': (ann_weight * ann_fitness) + (knn_weight * knn_fitness)
        })

    return results

def roulette_wheel_selection(population, num_selected):
    total_fitness = sum(ind['avg_fitness'] for ind in population)
    if total_fitness == 0:
        # Fallback jika semua fitness = 0
        return random.sample(population, num_selected)
    
    selected = []
    for _ in range(num_selected):
        pick = random.uniform(0, total_fitness)
        current = 0
        for ind in population:
            current += ind['avg_fitness']
            if current >= pick:
                selected.append(ind)
                break
    return selected

def single_point_crossover(parent1, parent2):
    keys = list(parent1.keys())
    if len(keys) < 2:
        return parent1.copy(), parent2.copy()
    
    point = random.randint(1, len(keys) - 1)
    child1 = {}
    child2 = {}
    for i in range(len(keys)):
        if i < point:
            child1[keys[i]] = parent1[keys[i]]
            child2[keys[i]] = parent2[keys[i]]
        else:
            child1[keys[i]] = parent2[keys[i]]
            child2[keys[i]] = parent1[keys[i]]
    return child1, child2

def mutate(chromosome, mutation_rate=0.1):
    mutated = chromosome.copy()

    # Batas berdasarkan target class 0 (Normal)
    min_normal_values = {
        'Sleep Duration': 6,
        'Physical Activity Level': 30,
        'BMI Category': 0,
        'Daily Steps': 4200,
        'Systolic_BP': 115,
        'Diastolic_BP': 75
    }

    max_normal_values = {
        'Sleep Duration': 8.5,
        'Physical Activity Level': 90,
        'BMI Category': 2,
        'Daily Steps': 10000,
        'Systolic_BP': 140,
        'Diastolic_BP': 95
    }

    for key in mutated:
        if random.random() < mutation_rate:
            # Mutasi berdasarkan domain + batas target class normal
            if key == 'Age':
                mutated[key] = max(18, min(80, mutated[key] + random.randint(-5, 5)))

            elif key == 'Sleep Duration':
                mutated[key] = round(
                    min(max_normal_values['Sleep Duration'],
                        max(min_normal_values['Sleep Duration'],
                            mutated[key] + random.uniform(-0.5, 0.5))),
                    1
                )

            elif key == 'Physical Activity Level':
                mutated[key] = int(
                    min(max_normal_values['Physical Activity Level'],
                        max(min_normal_values['Physical Activity Level'],
                            mutated[key] + random.randint(-5, 5)))
                )

            elif key == 'Daily Steps':
                mutated[key] = int(
                    min(max_normal_values['Daily Steps'],
                        max(min_normal_values['Daily Steps'],
                            mutated[key] + random.randint(-500, 500)))
                )

            elif key == 'Systolic_BP':
                mutated[key] = int(
                    min(max_normal_values['Systolic_BP'],
                        max(min_normal_values['Systolic_BP'],
                            mutated[key] + random.randint(-3, 3)))
                )

            elif key == 'Diastolic_BP':
                mutated[key] = int(
                    min(max_normal_values['Diastolic_BP'],
                        max(min_normal_values['Diastolic_BP'],
                            mutated[key] + random.randint(-3, 3)))
                )

            elif key == 'Quality of Sleep':
                mutated[key] = max(1, min(10, mutated[key] + random.choice([-1, 0, 1])))

            elif key == 'Stress Level':
                mutated[key] = max(1, min(10, mutated[key] + random.choice([-1, 0, 1])))

            elif key == 'Heart Rate':
                mutated[key] = max(50, min(100, mutated[key] + random.randint(-5, 5)))

            elif key == 'BMI Category':
                # Enforce within BMI category 0 (Normal), 1 (Under), 2 (Over)
                categories = ['Underweight', 'Normal', 'Overweight']
                if mutated[key] in categories:
                    idx = categories.index(mutated[key])
                else:
                    idx = 1  # default to Normal if unknown
                mutated[key] = categories[(idx + random.choice([-1, 1])) % len(categories)]

            elif key == 'Gender':
                mutated[key] = 'Female' if mutated[key] == 'Male' else 'Male'

            elif key == 'Occupation':
                # Boleh tetap random karena tidak ada nilai optimal khusus
                occupations = [
                    'Manager', 'Engineer', 'Doctor', 'Lawyer', 'Accountant',
                    'Software Engineer', 'Scientist', 'Teacher', 'Nurse', 'Salesperson',
                    'Sales Representative'
                ]
                mutated[key] = random.choice(occupations)

    return mutated


def genetic_algorithm(
    evaluated_population,
    generations=10,
    elite_size=2,
    selection_size=2,
    mutation_rate=0.1,
    scaler=None,
    features_to_use=None,
    knn_model=None,
    ann_model=None,
    target_class=0
):
    population = evaluated_population

    for gen in range(generations):
        print(f"\n=== Generasi {gen+1} ===")

        # 1. Elitism: Ambil kromosom terbaik
        elites = sorted(population, key=lambda x: x['avg_fitness'], reverse=True)[:elite_size]
        # rest = sorted(population, key=lambda x: x['avg_fitness'], reverse=False)[:len(evaluated_population) - elite_size]
        # 2. Roulette Wheel untuk seleksi tambahan
        selected = roulette_wheel_selection(population, selection_size)

        # 3. Crossover
        offspring = []
        for i in range(0, len(selected), 2):
            parent1 = selected[i]['chromosome']
            parent2 = selected[i+1]['chromosome'] if i+1 < len(selected) else selected[0]['chromosome']
            child1, child2 = single_point_crossover(parent1, parent2)
            offspring.append(child1)
            offspring.append(child2)

        # 4. Mutasi
        offspring = [mutate(ch, mutation_rate) for ch in offspring]

        # 5. Evaluasi semua individu baru (elit + offspring)
        new_candidates = [elite['chromosome'] for elite in elites] + offspring
        population = evaluate_chromosomes(new_candidates, scaler, features_to_use, knn_model, ann_model, target_class)

        
    return population


def run_genetic_algorithm():
    knn = st.session_state['knn_model']
    ann_model = st.session_state['ann_model']

    user_input_dict = st.session_state.get('user_input_example')
    if not user_input_dict:
        user_input_dict = {
            'Gender': 0,
            'Age': 30,
            'Occupation': 1,
            'Sleep Duration': 6.0,
            'Quality of Sleep': 5,
            'Physical Activity Level': 30,
            'Stress Level': 5,
            'BMI Category': 0,
            'Heart Rate': 85,
            'Daily Steps': 6000,
            'Systolic_BP': 130,
            'Diastolic_BP': 85
        }

    generated_chromosomes = generate_random_chromosomes(user_input_dict, n=6)
    evaluated_population = evaluate_chromosomes(generated_chromosomes, scaler, features_to_use, knn, ann_model, target_class=0)
    best = max(evaluated_population, key=lambda x: min(x['knn_fitness'], x['ann_fitness']))

    final_population = genetic_algorithm(
        evaluated_population,
        generations=10,
        elite_size=2,
        selection_size=3,
        mutation_rate=0.1,
        scaler=scaler,
        features_to_use=features_to_use,
        knn_model=knn,
        ann_model=ann_model,
        target_class=0
    )
    best_final = max(final_population, key=lambda x: x['avg_fitness'])
    return best, best_final['ann_label'], {'final': best_final}
