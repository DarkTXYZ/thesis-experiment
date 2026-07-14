import pyqbpp as qbpp
import networkx as nx

# create random graph of N=100
N = 5
G = nx.erdos_renyi_graph(N, 0.5)
edges = list(G.edges())

# max degree of the graph
max_degree = max(dict(G.degree()).values())

x = qbpp.var("x", shape=(N,N))

objective = 0
for u, v in edges:
    for i in range(N):
        objective += x[u][i] + x[v][i] - 2 * x[u][i] * x[v][i]

constraint = 0

for u in range(N):
    for k in range(N-1):
        constraint += (1 - x[u][k]) * (x[u][k+1])
        
for k in range(N):
    cnt = 0
    for u in range(N):
        cnt += x[u][k]
    constraint += (N - k - cnt) ** 2 

f = max_degree * constraint + objective
f.simplify_as_binary()

solver = qbpp.ExhaustiveSolver(f)
sol = solver.search()

print(f"objective = {sol(objective)}")
print(f"constraint = {sol(constraint)}")

print("Selected nodes:")
for i in range(N):
    for j in range(N):
        if sol(x[i][j]) == 1:
            print(f"1", end="")
        else:
            print(f"0", end="")
    print()
print()
