from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

def value_iteration(grid_size, start, goal, obstacles, gamma=0.9, theta=1e-4):
    V = np.zeros((grid_size, grid_size))
    actions = [(-1, 0, '↑'), (1, 0, '↓'), (0, -1, '←'), (0, 1, '→')]
    reward_goal = 20.0
    reward_obstacle = -1.0
    reward_default = -0.4
    iterations = []

    while True:
        delta = 0
        new_V = np.copy(V)
        policy = [['' for _ in range(grid_size)] for _ in range(grid_size)]
        #遍歷所有格子
        for i in range(grid_size):
            for j in range(grid_size):
                if (i, j) == goal:
                    new_V[i, j] = reward_goal
                    continue
                elif (i, j) in obstacles:
                    new_V[i, j] = reward_obstacle
                    continue
                elif (i, j) == start:
                    continue

                max_value = float('-inf')
                best_action = ''
                #計算每個格子的最大價值
                for di, dj, action in actions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < grid_size and 0 <= nj < grid_size:
                        value = gamma * V[ni, nj]
                    else:
                        value = gamma * V[i, j]
                    if value > max_value:
                        max_value = value #選擇當前格子的最佳價值
                        best_action = action #選擇當前格子的最佳行動
                new_V[i, j] = reward_default + max_value
                policy[i][j] = best_action
                delta = max(delta, abs(new_V[i, j] - V[i, j]))

        V = new_V
        iterations.append((V.tolist(), policy))
        if delta < theta:
            break
    return iterations

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_value_iteration', methods=['POST'])
def run_value_iteration():
    data = request.json
    iterations = value_iteration(int(data['grid_size']), tuple(data['start']), tuple(data['goal']), [tuple(o) for o in data['obstacles']])
    return jsonify({'iterations': iterations})

if __name__ == '__main__':
    app.run(debug=True)
