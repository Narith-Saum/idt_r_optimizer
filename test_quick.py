from idt_r_optimizer import IDTROptimizer

def objective(params):
    return -(params['x'] - 0.7)**2 - (params['y'] - 0.3)**2 + 1.0

space = {'x': (0.0, 1.0), 'y': (0.0, 1.0)}
optimizer = IDTROptimizer(space, max_iterations=5, n_random_init=3, verbose=False, seed=42)

best_params, best_score = optimizer.optimize(objective)

print('[SUCCESS] IDT-R Optimizer Works!')
print(f'Best Score: {best_score:.6f}')
print(f'Best Params: x={best_params["x"]:.4f}, y={best_params["y"]:.4f}')
print(f'Total Evaluations: {optimizer.get_history().size()}')
