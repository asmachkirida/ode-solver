from flask import Flask, request, jsonify
import torch
from model import load_model, loss_first_order, loss_second_order

app = Flask(__name__)

# Load model
model = load_model('model.pth')

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

        # Generate x values
        x = torch.linspace(x_start, x_end, 100)[:, None]
        
        # Solve ODE 
        if ode_type == 'first_order':
            solution = model(x).detach().numpy()
        elif ode_type == 'second_order':
            solution = model(x).detach().numpy()

        # Return the solution 
        return jsonify({"x": x.detach().numpy().tolist(), "solution": solution.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
