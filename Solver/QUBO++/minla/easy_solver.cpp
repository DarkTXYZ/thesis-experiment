#include "qbpp.hpp"
#include "qbpp_easy_solver.hpp"

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <algorithm>

using namespace std;

bool read_graph_file(const string& filename, size_t& n, vector<pair<int, int>>& edges)
{
    ifstream infile(filename);
    if (!infile.is_open())
    {
        cerr << "Error: Cannot open file " << filename << endl;
        return false;
    }
    
    string line;
    size_t m = 0;
    bool header_read = false;
    
    while (getline(infile, line))
    {
        if (line.empty() || line[0] == '#')
        {
            continue;
        }
        
        istringstream iss(line);
        
        if (!header_read)
        {
            if (!(iss >> n >> m))
            {
                cerr << "Error: Invalid header format" << endl;
                return false;
            }
            edges.reserve(m);
            header_read = true;
        }
        else
        {
            int u, v;
            if (!(iss >> u >> v))
            {
                cerr << "Error: Invalid edge format" << endl;
                return false;
            }
            edges.push_back({u, v});
        }
    }
    
    infile.close();
    
    if (edges.size() != m)
    {
        cerr << "Warning: Expected " << m << " edges, but read " << edges.size() << endl;
    }
    
    return true;
}

size_t param_for_general_graph(size_t n, size_t m)
{
    const double naive = m * (n - 1);
    const double complete = n * (n - 1) * (n + 1) / 6.0;
    const double k = std::ceil(n + 0.5 - 0.5 * std::sqrt(8.0 * m + 1));
    const double f = (n - k) * (n - k + 1) / 2.0;
    const double edges_method = (m - f) * (k - 1) + 
                                (n - k) * (n * n + (n + 3) * k - 2 * k * k - 1) / 6.0;
    
    return static_cast<size_t>(std::min({naive, complete, edges_method}));
}

size_t calculate_penalty_parameter(size_t n, size_t m, const string& graph_type)
{
    if (graph_type.find("random") != string::npos)
    {
        return param_for_general_graph(n, m);
    }
    else if (graph_type == "path")
    {
        return static_cast<size_t>(std::floor((n * n) / 2.0) - 1);
    }
    else if (graph_type == "cycle")
    {
        return static_cast<size_t>(std::floor((n * n) / 2.0));
    }
    else if (graph_type == "star")
    {
        return static_cast<size_t>(n * (n-1) / 2);
    }
    else if (graph_type == "complete")
    {
        return static_cast<size_t>(n * (n - 1) * (n + 1) / 6);
    }
    else if (graph_type == "grid")
    {
        return param_for_general_graph(n, m);
    }
    else
    {
        return param_for_general_graph(n, m);
    }
}

qbpp::Expr build_row_penalty(const qbpp::Vector<qbpp::Vector<qbpp::Var>>& x, size_t n, size_t k)
{
    auto penalty = qbpp::expr();
    
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < k - 1; j++)
        {
            penalty += (1 - x[i][j]) * x[i][j + 1];
        }
    }
    
    return penalty;
}

qbpp::Expr build_column_penalty(const qbpp::Vector<qbpp::Vector<qbpp::Var>>& x, size_t n, size_t k)
{
    auto penalty = qbpp::expr();
    
    for (size_t j = 0; j < k; j++)
    {
        auto position_sum = qbpp::expr();
        for (size_t i = 0; i < n; i++)
        {
            position_sum += x[i][j];
        }
        
        const int expected_count = static_cast<int>(n - j);
        penalty += (position_sum - expected_count) * (position_sum - expected_count);
    }
    
    return penalty;
}

qbpp::Expr build_objective(const qbpp::Vector<qbpp::Vector<qbpp::Var>>& x, 
                           const vector<pair<int, int>>& edges, 
                           size_t k)
{
    auto objective = qbpp::expr();
    
    for (const auto& edge : edges)
    {
        const int u = edge.first;
        const int v = edge.second;
        
        auto edge_length = qbpp::expr();
        for (size_t j = 0; j < k; j++)
        {
            edge_length += x[u][j] + x[v][j] - 2 * x[u][j] * x[v][j];
        }
        
        objective += edge_length;
    }
    
    return objective;
}

bool is_valid_permutation(const vector<int>& labels)
{
    const size_t n = labels.size();
    vector<bool> seen(n, false);
    
    for (size_t i = 0; i < n; ++i)
    {
        const int val = labels[i] - 1;
        if (val < 0 || val >= static_cast<int>(n) || seen[val])
        {
            return false;
        }
        seen[val] = true;
    }
    
    return true;
}

vector<int> extract_positions(const qbpp::Sol& sol, 
                              const qbpp::Vector<qbpp::Vector<qbpp::Var>>& x, 
                              size_t n)
{
    vector<int> labels(n);
    
    for (size_t i = 0; i < n; ++i)
    {   
        for (size_t j = 0; j < x[i].size() - 1; ++j)
        {   
            if (qbpp::toInt(sol.get(x[i][j])) == 0 and qbpp::toInt(sol.get(x[i][j + 1])) == 1)
            {
                cout << "Found invalid position" << endl;
                return vector<int>();
            }
        }

        labels[i] = qbpp::toInt(qbpp::sum(sol.get(x[i])));
    }
    
    return labels;
}

size_t calculate_arrangement_cost(const vector<int>& positions, 
                                  const vector<pair<int, int>>& edges)
{
    size_t total_cost = 0;
    
    for (const auto& edge : edges)
    {
        int u = edge.first;
        int v = edge.second;
        total_cost += abs(positions[u] - positions[v]);
    }
    
    return total_cost;
}

int main(int argc, char* argv[])
{

    qbpp::license_key("52BEE7-D6C679-46FE86-C73C0D-FD0C56-DCFF58");

    size_t n;
    vector<pair<int, int>> edges;
    string graph_file;
    string graph_type = "unknown";
    
    if (argc > 1)
    {
        string input_file = argv[1];
        
        if (input_file.find('/') == string::npos)
        {
            graph_file = "processed/" + input_file;
        }
        else
        {
            graph_file = input_file;
        }
        
        cout << "Reading graph from file: " << graph_file << endl;
        
        size_t last_slash = graph_file.find_last_of("/\\");
        string filename = (last_slash != string::npos) ? graph_file.substr(last_slash + 1) : graph_file;
        
        if (filename.find("graph_") == 0)
        {
            size_t first_underscore = filename.find('_');
            size_t last_underscore = filename.find_last_of('_');
            
            if (first_underscore != string::npos && last_underscore != string::npos && 
                first_underscore != last_underscore)
            {
                graph_type = filename.substr(first_underscore + 1, last_underscore - first_underscore - 1);
            }
        }
        
        if (!read_graph_file(graph_file, n, edges))
        {
            cerr << "Failed to read graph file. Exiting." << endl;
            return 1;
        }
    }
    else
    {
        cout << "No graph file provided. Using default graph." << endl;
        n = 3;
        edges = {
            {0, 1},
            {0, 2}
        };
        graph_type = "default";
    }
    
    const size_t m = edges.size();

    cout << "=== MINLA Problem ===" << endl;
    cout << "Graph type: " << graph_type << endl;
    cout << "Vertices: " << n << endl;
    cout << "Edges: " << m << endl;
    
    auto start_time = chrono::high_resolution_clock::now();
    
    const size_t k = n;
    const size_t penalty_param = calculate_penalty_parameter(n, m, graph_type);
    
    cout << "Penalty parameter: " << penalty_param << endl;
    
    auto x = qbpp::var("x", n, k);
    
    auto row_penalty = build_row_penalty(x, n, k);
    auto col_penalty = build_column_penalty(x, n, k);
    auto objective = build_objective(x, edges, k);
    
    auto total_penalty = row_penalty + col_penalty;
    auto qubo = objective + penalty_param * total_penalty;
    
    qubo.simplify_as_binary();
    
    cout << "\n=== Solving ===" << endl;
    
    qbpp::easy_solver::EasySolver solver(qubo);
    solver.time_limit(60.0);
    
    auto solution = solver.search();
    
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    
    vector<int> positions = extract_positions(solution, x, n);

    if (positions.empty()) {
        cout << "\nNo valid arrangement found in the solution." << endl;
        return 1;
    }

    bool is_feasible = is_valid_permutation(positions);
    
    cout << "\n=== Results ===" << endl;
    cout << "Penalty parameter: " << penalty_param << endl;
    cout << "Feasibility: " << (is_feasible ? "Feasible" : "Infeasible") << endl;
    cout << "Energy: " << solution.energy() << endl;
    cout << "Calculated arrangement cost: " << calculate_arrangement_cost(positions, edges) << endl;
    cout << "Time: " << duration.count() << " ms" << endl;
    
    return 0;
}