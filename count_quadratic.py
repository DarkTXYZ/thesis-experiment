import networkx as nx
from Utils import MinLA

iterations = 10
cnt = 3

while cnt < iterations:
    G = nx.erdos_renyi_graph(cnt, 0.5)
    if nx.is_connected(G):
        cnt += 1
        
        complete_bound = (cnt - 1) * cnt * (cnt + 1) // 6
        
        n, m = G.number_of_nodes(), G.number_of_edges()
        i, f, lb = 1, 0, 0
        
        while f + i <= m:
            if n == i:
                break
            f += i
            lb += i * (n - i)
            i += 1

        lb = lb + i * (m - f)
        
        print(complete_bound)
        print(lb)