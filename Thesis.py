import numpy as np

N = 4

# generate all possible assignments of N binary variables
def generate_assignments(n):
    if n == 0:
        return [[]]
    else:
        smaller_assignments = generate_assignments(n - 1)
        return [assignment + [0] for assignment in smaller_assignments] + [assignment + [1] for assignment in smaller_assignments]
    
assignments = generate_assignments(N*N)
min_penalty = float('inf')
best_X = None
penalty_A_solutions = []  # Store X matrices with penalty = 1
penalty_B_solutions = []  # Store X matrices with penalty = 1

for assignment in assignments:
    X = np.array(assignment).reshape((N, N))
    
    penaltyA = 0
    penaltyB = 0
    
    for i in range(N):
        for j in range(N-1):
            penaltyA += (1 - X[i, j]) * (X[i,j+1])
            
    for j in range(N):
        cnt = 0
        for i in range(N):
            cnt += X[i, j]
        penaltyB += (cnt - (N-j)) ** 2
    
    if penaltyA == 1 and penaltyB == 0:
        penalty_A_solutions.append(X.copy())
    if penaltyA == 0 and penaltyB == 1:
        penalty_B_solutions.append(X.copy())

# Remove duplicates by converting to tuples for comparison
unique_A_solutions = []
seen = set()
for sol in penalty_A_solutions:
    sol_tuple = tuple(sol.flatten())
    if sol_tuple not in seen:
        seen.add(sol_tuple)
        unique_A_solutions.append(sol)
        
unique_B_solutions = []
seen = set()
for sol in penalty_B_solutions:
    sol_tuple = tuple(sol.flatten())
    if sol_tuple not in seen:
        seen.add(sol_tuple)
        unique_B_solutions.append(sol)

# Display unique solutions
print(f"Found {len(unique_A_solutions)} unique solutions with penalty A = 1 and penalty B = 0:\n")
for idx, X in enumerate(unique_A_solutions, 1):
    print(f"Solution {idx}:")
    for i in range(N):
        for j in range(N):
            print(int(X[i, j]), end=' ')
        print()
    print("-----------------")

print(f"Found {len(unique_B_solutions)} unique solutions with penalty A = 0 and penalty B = 1:\n")
for idx, X in enumerate(unique_B_solutions, 1):
    print(f"Solution {idx}:")
    for i in range(N):
        for j in range(N):
            print(int(X[i, j]), end=' ')
        print()
    print("-----------------")
