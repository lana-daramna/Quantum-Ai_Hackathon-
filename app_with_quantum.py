from flask import Flask, request, jsonify
from flask_cors import CORS
import csv
import json
from datetime import datetime
import random
import math
import numpy as np

app = Flask(__name__)
CORS(app)

# -------------------------
# QUANTUM-INSPIRED OPTIMIZATION MODULE
# -------------------------
class QuantumOptimizer:
    """
    Quantum-inspired optimization (academically safe version)

    Achieves:
    - Probabilistic superposition (multiple states at once)
    - Measurement (amplitude²-based selection)
    - Annealing (temperature + cooling)
    - Tunneling-like non-greedy behavior

    -NOTE:
    This is quantum-inspired, not real quantum computing.
    """

    def __init__(self, num_qubits=8, iterations=100):
        self.num_qubits = num_qubits
        self.iterations = iterations

    def _measure(self, states):
        """
        Quantum-inspired measurement:
        Select a state based on amplitude² probability
        """
        probabilities = [s["amplitude"] ** 2 for s in states]
        total = sum(probabilities)

        if total == 0:
            return random.choice(states)

        probabilities = [p / total for p in probabilities]
        return np.random.choice(states, p=probabilities)

    def quantum_annealing(self, cost_function, num_solutions):
        """
        Quantum-inspired annealing with measurement
        """

        # --- Superposition initialization ---
        states = []
        for i in range(num_solutions):
            states.append({
                "solution_index": i,
                "amplitude": random.random()
            })

        temperature = 1.0
        cooling_rate = 0.97

        best_solution = None
        best_cost = float("inf")

        for _ in range(self.iterations):

            # --- Measurement step ---
            measured_state = self._measure(states)
            index = measured_state["solution_index"]
            cost = cost_function(index)

            # --- Greedy + tunneling-like acceptance ---
            if cost < best_cost:
                best_cost = cost
                best_solution = index
            elif random.random() < math.exp(-(cost - best_cost) / (temperature + 0.01)):
                best_cost = cost
                best_solution = index

            # --- Cooling ---
            temperature *= cooling_rate

            # --- Interference-like amplitude update ---
            for state in states:
                c = cost_function(state["solution_index"])
                state["amplitude"] *= math.exp(-c / (temperature + 0.1))

        return best_solution, best_cost

    def optimize_route_ambulance(self, routes, ambulances, risk_score, urgency_weight=0.6):
        """
        Optimize both route selection and ambulance assignment
        """

        combinations = []
        for route in routes:
            for ambulance in ambulances:
                combinations.append({
                    "route": route,
                    "ambulance": ambulance
                })

        def cost_function(index):
            combo = combinations[index]
            route = combo["route"]
            ambulance = combo["ambulance"]

            time_cost = route["eta_min"] / 30.0
            route_risk_cost = route["risk"]

            if risk_score >= 0.75 and ambulance["priority"] == "high":
                match_cost = 0.0
            elif risk_score >= 0.75 and ambulance["priority"] == "standard":
                match_cost = 0.5
            elif risk_score < 0.75 and ambulance["priority"] == "high":
                match_cost = 0.2
            else:
                match_cost = 0.0

            return (
                urgency_weight * time_cost +
                0.3 * route_risk_cost +
                0.1 * match_cost
            )

        best_index, best_cost = self.quantum_annealing(
            cost_function, len(combinations)
        )

        optimal_solution = combinations[best_index]

        return {
            "route": optimal_solution["route"],
            "ambulance": optimal_solution["ambulance"],
            "optimization_score": 1.0 - best_cost,
            "quantum_confidence": self._calculate_confidence(best_cost)
        }

    def _calculate_confidence(self, cost):
        confidence = 100 * (1.0 - min(cost, 1.0))
        return round(confidence, 1)


# -------------------------
# ROUTES DATABASE
# -------------------------
ROUTES_DB = {
    "Ramallah": [
        {"id": "RML-R1", "name": "Main Road", "distance_km": 7.5, "risk": 0.08, "eta_min": 8},
        {"id": "RML-R2", "name": "Bypass Road", "distance_km": 9.2, "risk": 0.05, "eta_min": 10},
    ],
    "Nablus": [
        {"id": "NBL-R1", "name": "City Center Route", "distance_km": 11.0, "risk": 0.10, "eta_min": 12},
        {"id": "NBL-R2", "name": "Mountain Road", "distance_km": 13.5, "risk": 0.12, "eta_min": 15},
    ],
    "Bethlehem": [
        {"id": "BTH-R1", "name": "Direct Route", "distance_km": 14.0, "risk": 0.07, "eta_min": 15},
        {"id": "BTH-R2", "name": "Tourist Bypass", "distance_km": 16.0, "risk": 0.06, "eta_min": 17},
    ],
    "Hebron": [
        {"id": "HBR-R1", "name": "Downtown Route", "distance_km": 17.0, "risk": 0.11, "eta_min": 18},
        {"id": "HBR-R2", "name": "Outer Ring Road", "distance_km": 20.0, "risk": 0.09, "eta_min": 21},
    ],
    "Jenin": [
        {"id": "JNN-R1", "name": "Northern Highway", "distance_km": 19.0, "risk": 0.08, "eta_min": 20},
        {"id": "JNN-R2", "name": "Agricultural Road", "distance_km": 22.0, "risk": 0.10, "eta_min": 24},
    ],
    "Jericho": [
        {"id": "JRC-R1", "name": "Valley Road", "distance_km": 21.0, "risk": 0.07, "eta_min": 22},
        {"id": "JRC-R2", "name": "Desert Route", "distance_km": 24.0, "risk": 0.09, "eta_min": 26},
    ],
    "Tulkarm": [
        {"id": "TLK-R1", "name": "Main Route", "distance_km": 13.0, "risk": 0.06, "eta_min": 14},
        {"id": "TLK-R2", "name": "Industrial Bypass", "distance_km": 15.0, "risk": 0.07, "eta_min": 16},
    ],
    "Qalqilya": [
        {"id": "QLQ-R1", "name": "Direct Route", "distance_km": 12.0, "risk": 0.06, "eta_min": 13},
        {"id": "QLQ-R2", "name": "Northern Bypass", "distance_km": 14.0, "risk": 0.07, "eta_min": 15},
    ],
}

# -------------------------
# HELPER FUNCTIONS
# -------------------------

def get_city_accident_risk(city):
    """Get accident risk level for a city"""
    try:
        with open("accident_risk_by_city.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["city"].lower() == city.lower():
                    level = row["accident_level"]
                    if level == "high":
                        return 0.08
                    elif level == "medium":
                        return 0.05
                    elif level == "low":
                        return 0.02
    except:
        pass
    return 0.0


def calculate_risk_score(age, gender, symptoms, chronic, incident_type, city):
    """AI-based risk assessment"""
    risk_score = 0.2
    risk_factors = [{"factor": "Base risk", "value": 0.2}]
    
    # Age factor
    if age >= 65:
        risk_score += 0.3
        risk_factors.append({"factor": "Age (65+)", "value": 0.3})
    elif age <= 12:
        risk_score += 0.2
        risk_factors.append({"factor": "Age (child)", "value": 0.2})
    
    # Symptoms
    if "chest" in symptoms.lower():
        risk_score += 0.3
        risk_factors.append({"factor": "Chest pain symptoms", "value": 0.3})
    
    # Chronic diseases
    if "heart" in chronic.lower():
        risk_score += 0.3
        risk_factors.append({"factor": "Heart disease", "value": 0.3})
    elif "diabetes" in chronic.lower():
        risk_score += 0.2
        risk_factors.append({"factor": "Diabetes", "value": 0.2})
    elif "asthma" in chronic.lower():
        risk_score += 0.1
        risk_factors.append({"factor": "Asthma", "value": 0.1})
    
    # Traffic accident
    if incident_type == "traffic_accident":
        risk_score += 0.4
        risk_factors.append({"factor": "Traffic accident", "value": 0.4})
    
    # City accident risk
    city_risk = get_city_accident_risk(city)
    risk_score += city_risk
    if city_risk > 0:
        risk_factors.append({"factor": f"City risk ({city})", "value": city_risk})
    
    # Cap risk score
    risk_score = min(risk_score, 1.0)
    
    return risk_score, risk_factors


# -------------------------
# API ROUTES
# -------------------------

@app.route("/")
def home():
    return jsonify({
        "service": "ReachAid Quantum-Enhanced Backend",
        "status": "running",
        "features": ["AI Risk Assessment", "Quantum Optimization", "Route Selection"]
    })


@app.route("/decision", methods=["GET"])
def decision():
    """
    Enhanced decision endpoint with quantum optimization
    """
    # -------- Input Parameters --------
    age = int(request.args.get("age", 0))
    gender = request.args.get("gender", "unknown")
    symptoms = request.args.get("symptoms", "")
    chronic = request.args.get("chronic_diseases", "none")
    incident_type = request.args.get("incident_type", "medical")
    city = request.args.get("location", "Ramallah")
    closed_road = request.args.get("closed_roads", "No")
    
    # -------- AI Risk Assessment --------
    risk_score, risk_factors = calculate_risk_score(
        age, gender, symptoms, chronic, incident_type, city
    )
    
    # -------- Available Routes --------
    available_routes = ROUTES_DB.get(city, ROUTES_DB["Ramallah"])
    
    # Adjust routes based on road closures
    if closed_road.lower() in ["yes", "true", "1"]:
        for route in available_routes:
            route['eta_min'] += 5
            route['risk'] += 0.03
    
    # -------- Available Ambulances --------
    ambulance_fleet = [
        {
            "id": "A1", 
            "name": "Standard Ambulance", 
            "priority": "standard",
            "equipment": ["Basic life support", "First aid", "Oxygen"],
            "capacity": "2 patients"
        },
        {
            "id": "A2", 
            "name": "High Priority Ambulance", 
            "priority": "high",
            "equipment": ["Advanced life support", "Cardiac monitor", "Trauma kit", "Medications"],
            "capacity": "1 critical patient"
        }
    ]
    
    # -------- QUANTUM OPTIMIZATION --------
    optimizer = QuantumOptimizer(num_qubits=8, iterations=100)
    
    # Determine urgency weight based on risk
    urgency_weight = 0.4 if risk_score < 0.5 else 0.7 if risk_score < 0.75 else 0.9
    
    optimal_solution = optimizer.optimize_route_ambulance(
        routes=available_routes,
        ambulances=ambulance_fleet,
        risk_score=risk_score,
        urgency_weight=urgency_weight
    )
    
    selected_route = optimal_solution['route']
    selected_ambulance = optimal_solution['ambulance']
    optimization_score = optimal_solution['optimization_score']
    quantum_confidence = optimal_solution['quantum_confidence']
    
    # -------- Decision Explanation --------
    decision_factors = []
    if risk_score >= 0.75:
        decision_factors.append("High risk score indicates critical emergency")
    if "chest" in symptoms.lower():
        decision_factors.append("Chest pain symptoms suggest cardiac emergency")
    if incident_type == "traffic_accident":
        decision_factors.append("Traffic accident requires specialized equipment")
    if "heart" in chronic.lower():
        decision_factors.append("Heart disease increases complication risk")
    
    decision_reason = (
        f"Quantum optimization selected {selected_ambulance['id']} via {selected_route['id']} "
        f"with {quantum_confidence}% confidence. "
        f"Factors: {', '.join(decision_factors) if decision_factors else 'Standard protocol'}. "
        f"Optimization balanced time ({selected_route['eta_min']} min), "
        f"route risk ({selected_route['risk']:.2%}), and resource allocation."
    )
    
    # -------- Response --------
    return jsonify({
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        
        # Patient & Incident Info
        "patient": {
            "age": age,
            "gender": gender,
            "symptoms": symptoms,
            "chronic_diseases": chronic
        },
        "incident": {
            "type": incident_type,
            "location": city,
            "road_closures": closed_road
        },
        
        # AI Risk Assessment
        "risk_assessment": {
            "risk_score": round(risk_score, 2),
            "risk_level": "Critical" if risk_score >= 0.75 else "High" if risk_score >= 0.5 else "Medium" if risk_score >= 0.3 else "Low",
            "risk_factors": risk_factors
        },
        
        # Quantum Optimization Results
        "quantum_optimization": {
            "optimization_score": round(optimization_score, 3),
            "confidence_level": f"{quantum_confidence}%",
            "algorithm": "Quantum-Inspired Annealing",
            "iterations": optimizer.iterations,
            "urgency_weight": urgency_weight
        },
        
        # Selected Route
        "route": {
            "id": selected_route['id'],
            "name": selected_route['name'],
            "distance_km": selected_route['distance_km'],
            "eta_minutes": selected_route['eta_min'],
            "risk_level": selected_route['risk'],
            "status": "Optimal"
        },
        
        # Selected Ambulance
        "ambulance": {
            "id": selected_ambulance['id'],
            "name": selected_ambulance['name'],
            "priority": selected_ambulance['priority'],
            "equipment": selected_ambulance['equipment'],
            "capacity": selected_ambulance['capacity']
        },
        
        # Decision Explanation
        "decision": {
            "reason": decision_reason,
            "factors": decision_factors,
            "recommendation": f"Dispatch {selected_ambulance['id']} immediately via {selected_route['name']}"
        },
        
        # Alternative Options (for transparency)
        "alternatives": {
            "routes": available_routes,
            "ambulances": ambulance_fleet
        }
    })


@app.route("/optimize", methods=["POST"])
def optimize_custom():
    """
    Custom optimization endpoint for advanced scenarios
    """
    data = request.get_json()
    
    routes = data.get('routes', [])
    ambulances = data.get('ambulances', [])
    risk_score = data.get('risk_score', 0.5)
    urgency_weight = data.get('urgency_weight', 0.6)
    
    optimizer = QuantumOptimizer(
        num_qubits=data.get('num_qubits', 8),
        iterations=data.get('iterations', 100)
    )
    
    result = optimizer.optimize_route_ambulance(
        routes=routes,
        ambulances=ambulances,
        risk_score=risk_score,
        urgency_weight=urgency_weight
    )
    
    return jsonify({
        "status": "success",
        "optimal_solution": result,
        "timestamp": datetime.now().isoformat()
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)