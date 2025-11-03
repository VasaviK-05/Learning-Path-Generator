#!/usr/bin/env python3
import uvicorn
import os
import sys
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from genetic_algorithm import LearningPathGA
from models.student import Student

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="Learning Path Generator API")

@app.post("/generate_path")
def generate_path():
    student = Student(
        name="Vasavi",
        target_score=165,
        available_time_week=420,
        known_concepts={
            "Algebra": 60,
            "Geometry": 45,
            "Probability": 30,
            "Arithmetic": 55,
            "Data Interpretation": 35
        }
    )

    ga = LearningPathGA(population_size=30, generations=50)
    best_path = ga.evolve(student)
    learning_sequence = [m.name for m in best_path.module_sequence]

    return JSONResponse({
        "student": student.name,
        "target_score": student.target_score,
        "best_fitness": best_path.fitness,
        "recommended_path": learning_sequence[:8]
    })

if __name__ == "__main__":
    print("🚀 Starting Learning Path Generator API Server...")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🛑 Press Ctrl+C to stop the server")
    uvicorn.run(
        "run_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["."],
        log_level="info"
    )
