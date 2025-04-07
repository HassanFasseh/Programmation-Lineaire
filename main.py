from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.patches import Polygon

app = FastAPI(title="Linear Programming Solver API")

# Set up Jinja2 templates (make sure you have a "templates" directory with index.html)
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OptimizationRequest(BaseModel):
    objective_coefficients: List[float]
    constraint_matrix: List[List[float]]
    rhs_values: List[float]
    variable_bounds: Optional[List[Tuple[Optional[float], Optional[float]]]] = None
    method: str  # "graphique" or "simplexe"
    maximize: bool = True  # Default to maximization problem

# Utility function to convert NumPy types to native Python types
def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    else:
        return obj

@app.post("/optimize")
async def optimize(request_data: OptimizationRequest):
    # Validate inputs
    if len(request_data.objective_coefficients) != len(request_data.constraint_matrix[0]):
        raise HTTPException(status_code=400, detail="Number of objective coefficients must match the number of variables in constraints")
    
    if len(request_data.constraint_matrix) != len(request_data.rhs_values):
        raise HTTPException(status_code=400, detail="Number of constraints must match the number of RHS values")
    
    # Set default variable bounds if not provided
    if request_data.variable_bounds is None:
        request_data.variable_bounds = [(0, None) for _ in range(len(request_data.objective_coefficients))]
    
    # Convert to numpy arrays for easier manipulation
    c = np.array(request_data.objective_coefficients)
    A = np.array(request_data.constraint_matrix)
    b = np.array(request_data.rhs_values)
    
    # If minimization problem, negate objective coefficients
    if not request_data.maximize:
        c = -c
    
    # Choose method
    if request_data.method.lower() == "graphique":
        if len(request_data.objective_coefficients) != 2:
            raise HTTPException(status_code=400, detail="Graphical method requires exactly 2 variables")
        result = solve_graphical_method(c, A, b, request_data.variable_bounds, request_data.maximize)
    elif request_data.method.lower() == "simplexe":
        result = solve_simplex_method(c, A, b, request_data.variable_bounds, request_data.maximize)
    else:
        raise HTTPException(status_code=400, detail="Method must be either 'graphique' or 'simplexe'")
    
    # Convert NumPy types to native Python types before returning the result
    return convert_numpy_types(result)

def solve_graphical_method(c, A, b, bounds, maximize):
    # Only works for 2 variables
    if len(c) != 2:
        raise HTTPException(status_code=400, detail="Graphical method requires exactly 2 variables")
    
    # Define plotting range
    x_min, x_max = 0, max(max(b) * 2, 20)
    y_min, y_max = 0, max(max(b) * 2, 20)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot each constraint
    constraint_lines = []
    for i in range(len(A)):
        if A[i, 0] == 0:  # Vertical line
            y = np.linspace(y_min, y_max, 100)
            x = np.ones_like(y) * (b[i] / A[i, 1])
            ax.plot(x, y, label=f"Constraint {i+1}")
            constraint_lines.append((A[i], b[i]))
        elif A[i, 1] == 0:  # Horizontal line
            x = np.linspace(x_min, x_max, 100)
            y = np.ones_like(x) * (b[i] / A[i, 0])
            ax.plot(x, y, label=f"Constraint {i+1}")
            constraint_lines.append((A[i], b[i]))
        else:
            x = np.linspace(x_min, x_max, 100)
            y = (b[i] - A[i, 0] * x) / A[i, 1]
            ax.plot(x, y, label=f"Constraint {i+1}")
            constraint_lines.append((A[i], b[i]))
    
    # Consider variable bounds
    if bounds[0][0] is not None:
        x = np.ones(100) * bounds[0][0]
        y = np.linspace(y_min, y_max, 100)
        ax.plot(x, y, 'r--', label=f"x ≥ {bounds[0][0]}")
        constraint_lines.append((np.array([1, 0]), bounds[0][0]))
    
    if bounds[0][1] is not None:
        x = np.ones(100) * bounds[0][1]
        y = np.linspace(y_min, y_max, 100)
        ax.plot(x, y, 'r--', label=f"x ≤ {bounds[0][1]}")
        constraint_lines.append((np.array([-1, 0]), -bounds[0][1]))
    
    if bounds[1][0] is not None:
        y = np.ones(100) * bounds[1][0]
        x = np.linspace(x_min, x_max, 100)
        ax.plot(x, y, 'r--', label=f"y ≥ {bounds[1][0]}")
        constraint_lines.append((np.array([0, 1]), bounds[1][0]))
    
    if bounds[1][1] is not None:
        y = np.ones(100) * bounds[1][1]
        x = np.linspace(x_min, x_max, 100)
        ax.plot(x, y, 'r--', label=f"y ≤ {bounds[1][1]}")
        constraint_lines.append((np.array([0, -1]), -bounds[1][1]))
    
    # Find vertices by computing intersections of constraint lines
    vertices = []
    for i in range(len(constraint_lines)):
        for j in range(i + 1, len(constraint_lines)):
            A1, b1 = constraint_lines[i]
            A2, b2 = constraint_lines[j]
            try:
                A_matrix = np.vstack((A1, A2))
                b_vector = np.array([b1, b2])
                vertex = np.linalg.solve(A_matrix, b_vector)
                if all(np.dot(A, vertex) <= b + 1e-10):
                    vertices.append(vertex)
            except:
                continue  # Skip if lines are parallel or unsolvable
    
    # Remove duplicate vertices
    unique_vertices = []
    for vertex in vertices:
        if not any(np.allclose(vertex, u, atol=1e-10) for u in unique_vertices):
            unique_vertices.append(vertex)
    
    if not unique_vertices:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Linear Programming - Graphical Method (No Feasible Solution)')
        ax.legend()
        ax.grid(True)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close()
        return {
            "status": "No feasible solution or unbounded",
            "plot": img_str,
            "vertices": [],
            "optimal_point": None,
            "optimal_value": None
        }
    
    # Compute objective function value at each vertex
    objective_values = [np.dot(c, vertex) for vertex in unique_vertices]
    
    optimal_index = np.argmax(objective_values) if maximize else np.argmin(objective_values)
    optimal_vertex = unique_vertices[optimal_index]
    optimal_value = objective_values[optimal_index]
    
    # Plot feasible region (using convex hull if possible)
    vertices_array = np.array(unique_vertices)
    if len(vertices_array) >= 3:
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(vertices_array)
            polygon = Polygon(vertices_array[hull.vertices], alpha=0.2, color='green')
            ax.add_patch(polygon)
        except:
            ax.fill(vertices_array[:, 0], vertices_array[:, 1], alpha=0.2, color='green')
    
    # Plot optimal point
    ax.scatter(optimal_vertex[0], optimal_vertex[1], color='red', s=100, zorder=5, label='Optimal Point')
    
    # Plot objective function contour
    if any(c):
        x_vals = np.linspace(x_min, x_max, 100)
        if c[1] != 0:
            for z in np.linspace(0, optimal_value * 1.5, 5):
                y_vals = (z - c[0] * x_vals) / c[1]
                ax.plot(x_vals, y_vals, 'k--', alpha=0.3)
        mid_x = (x_min + x_max) / 2
        mid_y = (y_min + y_max) / 2
        arrow_len = min(x_max - x_min, y_max - y_min) / 10
        if c[1] == 0:
            dx, dy = np.sign(c[0]) * arrow_len, 0
        elif c[0] == 0:
            dx, dy = 0, np.sign(c[1]) * arrow_len
        else:
            norm = np.sqrt(c[0] ** 2 + c[1] ** 2)
            dx, dy = c[0] / norm * arrow_len, c[1] / norm * arrow_len
        ax.arrow(mid_x, mid_y, dx, dy, head_width=arrow_len / 5, head_length=arrow_len / 3,
                 fc='black', ec='black', label='Objective Direction')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Linear Programming - Graphical Method')
    ax.legend()
    ax.grid(True)
    x_range = x_max - x_min
    y_range = y_max - y_min
    ax.set_xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close()
    
    vertices_output = []
    for i, vertex in enumerate(unique_vertices):
        vertices_output.append({
            "x": vertex[0],
            "y": vertex[1],
            "objective_value": objective_values[i],
            "is_optimal": i == optimal_index
        })
    
    return {
        "status": "optimal",
        "plot": img_str,
        "vertices": vertices_output,
        "optimal_point": {"x": float(optimal_vertex[0]), "y": float(optimal_vertex[1])},
        "optimal_value": float(optimal_value)
    }

class TableauStep:
    def __init__(self, tableau, basic_vars, non_basic_vars, pivot_row=None, pivot_col=None):
        self.tableau = tableau.copy()
        self.basic_vars = basic_vars.copy()
        self.non_basic_vars = non_basic_vars.copy()
        self.pivot_row = pivot_row
        self.pivot_col = pivot_col

def solve_simplex_method(c, A, b, bounds, maximize):
    num_constraints = len(b)
    num_variables = len(c)
    
    # Add slack variables
    A_standard = np.hstack((A, np.eye(num_constraints)))
    c_standard = np.hstack((c, np.zeros(num_constraints)))
    
    # Initialize the tableau
    tableau = np.zeros((num_constraints + 1, num_variables + num_constraints + 1))
    tableau[0, :-1] = -c_standard  # Convert to minimization form
    tableau[1:, :-1] = A_standard
    tableau[1:, -1] = b
    
    basic_vars = [num_variables + i for i in range(num_constraints)]
    non_basic_vars = list(range(num_variables))
    steps = [TableauStep(tableau, basic_vars, non_basic_vars)]
    
    max_iterations = 100
    for _ in range(max_iterations):
        if maximize:
            pivot_col = np.argmin(tableau[0, :-1])
            if tableau[0, pivot_col] >= -1e-10:
                break
        else:
            pivot_col = np.argmax(tableau[0, :-1])
            if tableau[0, pivot_col] <= 1e-10:
                break
        
        ratios = []
        for i in range(1, num_constraints + 1):
            if tableau[i, pivot_col] > 1e-10:
                ratios.append((i, tableau[i, -1] / tableau[i, pivot_col]))
            else:
                ratios.append((i, float('inf')))
        
        if all(ratio[1] == float('inf') for ratio in ratios):
            return {
                "status": "unbounded",
                "message": "The problem is unbounded",
                "steps": [format_tableau_step(step) for step in steps]
            }
        
        pivot_row = min(ratios, key=lambda x: x[1])[0]
        steps.append(TableauStep(tableau.copy(), basic_vars.copy(), non_basic_vars.copy(), pivot_row, pivot_col))
        
        pivot_value = tableau[pivot_row, pivot_col]
        tableau[pivot_row] = tableau[pivot_row] / pivot_value
        
        for i in range(num_constraints + 1):
            if i != pivot_row:
                tableau[i] = tableau[i] - tableau[i, pivot_col] * tableau[pivot_row]
        
        leaving_var = basic_vars[pivot_row - 1]
        entering_var = non_basic_vars[pivot_col]
        basic_vars[pivot_row - 1] = entering_var
        non_basic_vars[pivot_col] = leaving_var
        steps.append(TableauStep(tableau.copy(), basic_vars.copy(), non_basic_vars.copy()))
    
    solution = np.zeros(num_variables + num_constraints)
    for i, var in enumerate(basic_vars):
        if var < num_variables:
            solution[var] = tableau[i + 1, -1]
    
    optimal_value = -tableau[0, -1] if maximize else tableau[0, -1]
    tableau_steps = [format_tableau_step(step) for step in steps]
    
    return {
        "status": "optimal",
        "solution": {
            "variables": solution[:num_variables].tolist(),
            "value": float(optimal_value)
        },
        "steps": tableau_steps
    }

def format_tableau_step(step):
    tableau_data = step.tableau.tolist()
    basic_vars = step.basic_vars
    non_basic_vars = step.non_basic_vars
    num_rows = len(tableau_data)
    num_cols = len(tableau_data[0])
    
    basic_vars_names = []
    for var in basic_vars:
        if var < num_cols - num_rows:
            basic_vars_names.append(f"x{var+1}")
        else:
            basic_vars_names.append(f"s{var - (num_cols - num_rows) + 1}")
    
    non_basic_vars_names = []
    for var in non_basic_vars:
        if var < num_cols - num_rows:
            non_basic_vars_names.append(f"x{var+1}")
        else:
            non_basic_vars_names.append(f"s{var - (num_cols - num_rows) + 1}")
    
    formatted_tableau = []
    header = [""]
    for j in range(num_cols - 1):
        header.append(non_basic_vars_names[j] if j < len(non_basic_vars_names) else "")
    header.append("RHS")
    formatted_tableau.append(header)
    
    obj_row = ["Z"] + [round(val, 4) for val in tableau_data[0]]
    formatted_tableau.append(obj_row)
    
    for i in range(1, num_rows):
        row = [basic_vars_names[i - 1]] + [round(val, 4) for val in tableau_data[i]]
        formatted_tableau.append(row)
    
    return {
        "tableau": formatted_tableau,
        "basic_vars": basic_vars_names,
        "non_basic_vars": non_basic_vars_names,
        "pivot_row": step.pivot_row,
        "pivot_col": step.pivot_col
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
