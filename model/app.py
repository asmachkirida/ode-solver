# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from model import load_model, loss_first_order, loss_second_order

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/solve-ode', methods=['POST'])
def solve_ode():
    try:
        data = request.get_json()
        ode_type = data.get('ode_type')
        x_start = float(data.get('x_start'))
        x_end = float(data.get('x_end'))

        # Validate inputs
        if ode_type not in ['first_order', 'second_order']:
            return jsonify({"error": "Invalid ODE type"}), 400
        if x_start >= x_end:
            return jsonify({"error": "Invalid range: x_start must be less than x_end"}), 400

        # Load the model based on the ODE type
        if ode_type == 'first_order':
            model = load_model('model_first_order.pth')  # Load the first-order model
        elif ode_type == 'second_order':
            model = load_model('model_second_order.pth')  # Load the second-order model

        # Generate x values
        x = torch.linspace(x_start, x_end, 100)[:, None]
        
        # Solve ODE 
        solution = model(x).detach().numpy()

        # Return the solution 
        return jsonify({"x": x.detach().numpy().tolist(), "solution": solution.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
