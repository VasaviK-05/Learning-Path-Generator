"""
Genetic Algorithm for generating optimal learning paths
Enhanced with assessment-driven personalization
"""

import random
import math
import sys
import os
from rl.train_rl import train_rl_model
from rl.policy_io import load_agent, recommend_next_step

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from models.student import Student
from data.gre_modules import create_gre_quantitative_modules

class LearningPathGA:
    def __init__(self, population_size=50, generations=100, mutation_rate=0.1):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.modules = create_gre_quantitative_modules()
        
    class LearningPath:
        def __init__(self, module_sequence):
            self.module_sequence = module_sequence
            self.fitness = 0
            self.total_time = sum(module.time_estimate for module in module_sequence) if module_sequence else 0
            
        def __len__(self):
            return len(self.module_sequence)
        
        def __getitem__(self, index):
            return self.module_sequence[index]
        
        def __repr__(self):
            return f"Path(fitness={self.fitness:.3f}, modules={len(self)}, time={self.total_time}min)"
    
    def create_initial_population(self, student):
        population = []
        for _ in range(self.population_size):
            available_modules = self.get_available_modules(student)
            if not available_modules:
                available_modules = self.modules.copy()
            max_path_length = self._calculate_optimal_path_length(student, available_modules)
            path_length = min(max_path_length, len(available_modules))
            random_path = random.sample(available_modules, path_length)
            path = self.LearningPath(random_path)
            population.append(path)
        return population
    
    def _calculate_optimal_path_length(self, student, available_modules):
        if not available_modules:
            return 10
        avg_module_time = sum(module.time_estimate for module in available_modules) / len(available_modules)
        optimal_length = max(5, min(20, student.available_time_week // avg_module_time))
        return int(optimal_length)
    
    def get_available_modules(self, student):
        available_modules = []
        weak_concepts = student.get_weak_concepts(50)
        strong_concepts = student.get_strong_concepts(70)
        for module in self.modules:
            readiness = student.calculate_readiness(module)
            if readiness >= 30:
                addresses_weak_areas = any(concept in weak_concepts for concept in module.concepts)
                only_strong_areas = all(concept in strong_concepts for concept in module.concepts)
                if addresses_weak_areas:
                    available_modules.insert(0, module)
                elif not only_strong_areas:
                    available_modules.append(module)
                elif module.difficulty <= 2 and random.random() < 0.3:
                    available_modules.append(module)
        if not available_modules:
            available_modules = [module for module in self.modules if module.difficulty <= 2]
        print(f"   Available modules: {len(available_modules)} (based on {len(weak_concepts)} weak areas)")
        return available_modules
    
    def _get_all_required_concepts(self, student):
        target_score = student.target_score
        if target_score >= 165:
            all_concepts = set()
            for module in self.modules:
                all_concepts.update(module.concepts)
            return list(all_concepts)
        elif target_score >= 155:
            required_concepts = set()
            for module in self.modules:
                if module.difficulty <= 4:
                    required_concepts.update(module.concepts)
            return list(required_concepts)
        else:
            required_concepts = set()
            for module in self.modules:
                if module.difficulty <= 3:
                    required_concepts.update(module.concepts)
            return list(required_concepts)
    
    def calculate_fitness(self, path, student):
        if not path.module_sequence:
            return 0
        fitness = 0
        total_time = 0
        concepts_covered = set()
        weak_concepts_covered = set()
        strong_concepts_reviewed = set()
        weak_concepts = student.get_weak_concepts(threshold=50)
        strong_concepts = student.get_strong_concepts(threshold=70)
        all_required_concepts = self._get_all_required_concepts(student)
        learned_concepts = set(student.known_concepts.keys())
        prerequisite_violations = 0
        difficulty_changes = 0
        for i, module in enumerate(path.module_sequence):
            total_time += module.time_estimate
            for concept in module.concepts:
                concepts_covered.add(concept)
                if concept in weak_concepts:
                    weak_concepts_covered.add(concept)
                if concept in strong_concepts:
                    strong_concepts_reviewed.add(concept)
            for prereq in module.prerequisites:
                if prereq not in learned_concepts:
                    prerequisite_violations += 1
                    if prereq in student.known_concepts and student.known_concepts[prereq] < 30:
                        prerequisite_violations += 1
            learned_concepts.update(module.concepts)
            if i > 0:
                prev_difficulty = path[i-1].difficulty
                current_difficulty = module.difficulty
                difficulty_changes += abs(current_difficulty - prev_difficulty)
        if weak_concepts:
            weak_area_coverage = len(weak_concepts_covered) / len(weak_concepts)
        else:
            weak_area_coverage = 1.0
        fitness += weak_area_coverage * 0.4
        if all_required_concepts:
            required_coverage = len(concepts_covered & set(all_required_concepts)) / len(all_required_concepts)
        else:
            required_coverage = 1.0
        fitness += required_coverage * 0.2
        time_ratio = total_time / student.available_time_week if student.available_time_week > 0 else 1
        if time_ratio <= 1:
            time_fitness = 1 - (1 - time_ratio) * 0.5
        else:
            time_fitness = 1 / time_ratio
        fitness += time_fitness * 0.15
        max_possible_violations = sum(len(module.prerequisites) for module in path.module_sequence)
        if max_possible_violations > 0:
            prereq_fitness = 1 - (prerequisite_violations / max_possible_violations)
        else:
            prereq_fitness = 1.0
        fitness += prereq_fitness * 0.1
        if len(path.module_sequence) > 1:
            avg_difficulty_change = difficulty_changes / (len(path.module_sequence) - 1)
            progression_fitness = 1.0 / (1 + avg_difficulty_change)
        else:
            progression_fitness = 1.0
        fitness += progression_fitness * 0.08
        if path.module_sequence:
            review_penalty = len(strong_concepts_reviewed) / (len(concepts_covered) + 1)
            fitness += (1 - review_penalty) * 0.07
        path.fitness = max(0, min(1, fitness))
        path.total_time = total_time
        return path.fitness
    
    def select_parent(self, population):
        tournament_size = min(5, len(population))
        tournament = random.sample(population, tournament_size)
        tournament.sort(key=lambda x: x.fitness, reverse=True)
        return tournament[0]
    
    def crossover(self, parent1, parent2):
        if len(parent1) < 2 or len(parent2) < 2:
            return self.LearningPath(parent1.module_sequence.copy())
        child_sequence = self._ordered_crossover(parent1, parent2)
        return self.LearningPath(child_sequence)
    
    def _ordered_crossover(self, parent1, parent2):
        size = min(len(parent1), len(parent2))
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size
        for i in range(start, end + 1):
            child[i] = parent1[i]
        parent2_index = 0
        for i in range(size):
            if child[i] is None:
                while parent2_index < len(parent2):
                    module = parent2[parent2_index]
                    parent2_index += 1
                    if module not in child:
                        child[i] = module
                        break
        child = [module for module in child if module is not None]
        return child
    
    def mutate(self, path, student):
        new_sequence = path.module_sequence.copy()
        if not new_sequence:
            return path
        mutation_type = random.random()
        if mutation_type < 0.4 and len(new_sequence) >= 2:
            i, j = random.sample(range(len(new_sequence)), 2)
            new_sequence[i], new_sequence[j] = new_sequence[j], new_sequence[i]
        elif mutation_type < 0.65:
            available_modules = self.get_available_modules(student)
            unused_modules = [m for m in available_modules if m not in new_sequence]
            if unused_modules:
                weak_area_modules = [m for m in unused_modules if any(c in student.get_weak_concepts(50) for c in m.concepts)]
                if weak_area_modules:
                    new_module = random.choice(weak_area_modules)
                else:
                    new_module = random.choice(unused_modules)
                insert_pos = random.randint(0, len(new_sequence))
                new_sequence.insert(insert_pos, new_module)
        elif mutation_type < 0.85 and len(new_sequence) > 3:
            strong_concepts = student.get_strong_concepts(70)
            removable_modules = []
            for i, module in enumerate(new_sequence):
                if all(concept in strong_concepts for concept in module.concepts):
                    removable_modules.append(i)
            if removable_modules:
                remove_index = random.choice(removable_modules)
                new_sequence.pop(remove_index)
            else:
                new_sequence.pop(random.randint(0, len(new_sequence) - 1))
        else:
            if len(new_sequence) >= 4:
                start, end = sorted(random.sample(range(len(new_sequence)), 2))
                segment = new_sequence[start:end]
                random.shuffle(segment)
                new_sequence[start:end] = segment
        return self.LearningPath(new_sequence)
    
    def evolve(self, student, initial_path=None):
        print(f"🧬 Generating learning path for {student.name}...")
        print(f"   Population: {self.population_size}, Generations: {self.generations}")
        print(f"   Target Score: {student.target_score}, Available Time: {student.available_time_week}min")
        if initial_path:
            population = [initial_path]
            additional_paths = self.create_initial_population(student)
            population.extend(additional_paths[:self.population_size-1])
        else:
            population = self.create_initial_population(student)
        for path in population:
            self.calculate_fitness(path, student)
        best_fitness_history = []
        for generation in range(self.generations):
            new_population = []
            population.sort(key=lambda x: x.fitness, reverse=True)
            new_population.append(population[0])
            while len(new_population) < self.population_size:
                parent1 = self.select_parent(population)
                parent2 = self.select_parent(population)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child, student)
                self.calculate_fitness(child, student)
                new_population.append(child)
            population = new_population
            best_fitness = population[0].fitness
            best_fitness_history.append(best_fitness)
            if (generation + 1) % 10 == 0:
                avg_fitness = sum(p.fitness for p in population) / len(population)
                print(f"   Generation {generation + 1}: Best = {best_fitness:.3f}, Avg = {avg_fitness:.3f}")
        population.sort(key=lambda x: x.fitness, reverse=True)
        best_path = population[0]
        print(f"✅ Evolution completed! Best path fitness: {best_path.fitness:.3f}")
        print("\nInitializing Reinforcement Learning Optimization...")
        model_path = train_rl_model(episodes=5000)
        agent, env = load_agent(model_path)
        recommendation = recommend_next_step(agent, env)
        print("\n--- Reinforcement Learning Suggestion ---")
        print(f"Next Recommended Topic: {recommendation['topic']}")
        print(f"Suggested Activity Type: {recommendation['activity']}")
        print(f"Expected Reward: {recommendation['reward']:.3f}")
        return best_path
    
    def display_path(self, path, student):
        print(f"\n🎯 PERSONALIZED LEARNING PATH FOR {student.name}")
        print("=" * 70)
        print(f"Target: GRE Quantitative {student.target_score}+")
        print(f"Total modules: {len(path)}")
        print(f"Estimated time: {sum(m.time_estimate for m in path.module_sequence)} minutes")
        print(f"Path fitness score: {path.fitness:.3f}")
        weak_areas = student.get_weak_concepts(50)
        weak_areas_covered = set()
        for module in path.module_sequence:
            for concept in module.concepts:
                if concept in weak_areas:
                    weak_areas_covered.add(concept)
        if weak_areas:
            coverage_percentage = (len(weak_areas_covered) / len(weak_areas)) * 100
            print(f"Weak areas addressed: {len(weak_areas_covered)}/{len(weak_areas)} ({coverage_percentage:.1f}%)")
        print("\n📚 Learning Sequence:")
        print("-" * 70)
        total_time = 0
        learned_concepts = set(student.known_concepts.keys())
        for i, module in enumerate(path.module_sequence, 1):
            total_time += module.time_estimate
            readiness = student.calculate_readiness(module)
            addresses_weak_areas = any(concept in weak_areas for concept in module.concepts)
            weak_area_indicator = "🎯" if addresses_weak_areas else "  "
            print(f"{i:2d}. {weak_area_indicator} {module.name}")
            print(f"     ⏱️  {module.time_estimate:3d} min | 🎯 Diff: {module.difficulty}/5 | 📊 Ready: {readiness:3.0f}%")
            print(f"     📖 Concepts: {', '.join(module.concepts)}")
            missing_prereqs = [p for p in module.prerequisites if p not in learned_concepts]
            if missing_prereqs:
                print(f"     ⚠️  Missing prerequisites: {', '.join(missing_prereqs)}")
            learned_concepts.update(module.concepts)
            print()
    
    def analyze_path_quality(self, path, student):
        print(f"\n🔍 PATH QUALITY ANALYSIS")
        print("=" * 50)
        weak_areas = student.get_weak_concepts(50)
        weak_areas_covered = set()
        total_weak_module_count = 0
        for module in path.module_sequence:
            module_weak_concepts = [c for c in module.concepts if c in weak_areas]
            if module_weak_concepts:
                weak_areas_covered.update(module_weak_concepts)
                total_weak_module_count += 1
        print(f"• Modules targeting weak areas: {total_weak_module_count}/{len(path)}")
        print(f"• Weak areas covered: {len(weak_areas_covered)}/{len(weak_areas)}")
        print(f"• Total estimated time: {path.total_time} minutes")
        print(f"• Student's available time: {student.available_time_week} minutes")
        print(f"• Time utilization: {(path.total_time/student.available_time_week*100):.1f}%")
        learned_concepts = set(student.known_concepts.keys())
        prerequisite_violations = 0
        for module in path.module_sequence:
            for prereq in module.prerequisites:
                if prereq not in learned_concepts:
                    prerequisite_violations += 1
            learned_concepts.update(module.concepts)
        print(f"• Prerequisite violations: {prerequisite_violations}")

def test_genetic_algorithm():
    print("🧪 TESTING GENETIC ALGORITHM")
    print("=" * 50)
    from models.student import create_student_interactive
    student = create_student_interactive()
    ga = LearningPathGA(population_size=50, generations=100)
    learning_path = ga.evolve(student)
    ga.display_path(learning_path, student)
    ga.analyze_path_quality(learning_path, student)
    return student, learning_path

if __name__ == "__main__":
    test_genetic_algorithm()
