import numpy as np

# number to thermometer encoding
def number_to_thermometer(n,k):
    first_part = np.ones(n, dtype=int)
    second_part = np.zeros(k-n, dtype=int)
    
    return np.append(first_part, second_part)

N = 5

def calculate_cost(u, k):
    cost = 0
    for i in range(1, N+1):
        if k == i:
            continue
        cost += np.sum(np.bitwise_xor(u, number_to_thermometer(i, N)))
        
    return cost
    
    
for i in range(1, N+1):
    u = number_to_thermometer(i, N)
    original_cost = calculate_cost(u, i)
    
    
    print("Original Cost: ", str(original_cost))
    print(u)
    
    print("Adjusted Cost:")
    for adjust_index in range(1, N+1):
        u[adjust_index - 1] = 1 - u[adjust_index - 1]
        print(u, end = "")
        adjust_cost = calculate_cost(u, i)
        print("\t", str(adjust_cost))
        u[adjust_index - 1] = 1 - u[adjust_index - 1]
        
        