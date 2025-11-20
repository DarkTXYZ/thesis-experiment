// docker run --rm --platform=linux/amd64 -v "$(pwd)":/workspace -w /workspace/minla qbpp-linux make exhaustive

#include "qbpp.hpp"
#include "qbpp_exhaustive_solver.hpp"

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>

using namespace std;

/// @brief Read graph from file in edge list format
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

/// @brief Verify if solution represents a valid permutation
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

/// @brief Extract vertex positions from solution
vector<int> extract_positions(const qbpp::Sol& sol, 
                              const qbpp::Vector<qbpp::Vector<qbpp::Var>>& x, 
                              size_t n)
{
    vector<int> labels(n);
    
    for (size_t i = 0; i < n; ++i)
    {
        labels[i] = qbpp::toInt(qbpp::sum(sol.get(x[i])));
    }
    
    return labels;
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

size_t param_for_graph(size_t n, size_t m, string graph_type) {
    if (graph_type == "random")
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
    else
    {
        return param_for_general_graph(n, m);
    }
}

int main(int argc, char* argv[])
{
    qbpp::license_key("52BEE7-D6C679-46FE86-C73C0D-FD0C56-DCFF58");
    
    // ========== Problem Definition ==========
    size_t n;
    vector<pair<int, int>> edges;
    string graph_file;
    string graph_type = "unknown";
    
    if (argc > 1)
    {
        graph_file = argv[1];
        cout << "Reading graph from file: " << graph_file << endl;
        
        // Extract graph type from filename (e.g., graph_cycle_10.txt -> cycle)
        size_t last_slash = graph_file.find_last_of("/\\");
        string filename = (last_slash != string::npos) ? graph_file.substr(last_slash + 1) : graph_file;
        
        // Parse format: graph_TYPE_SIZE.txt
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
    
    // ========== QUBO Formulation ==========
    auto start_time = chrono::high_resolution_clock::now();
    
    const size_t u = n;
    const size_t k = n;
    const size_t penalty_params = param_for_graph(n, m, graph_type);

    auto x = qbpp::var("x", u, k);

    // Build row penalty
    auto penalty_rows = qbpp::expr();
    for (size_t i = 0; i < u; i++)
    {
        for (size_t j = 0; j < k - 1; j++)
        {
            penalty_rows += (1 - x[i][j]) * (x[i][j + 1]);
        }
    }

    // Build column penalty
    auto penalty_cols = qbpp::expr();
    for (size_t j = 0; j < k; j++)
    {
        auto terms = qbpp::expr();
        for (size_t i = 0; i < u; i++)
        {
            terms += x[i][j];
        }
        penalty_cols += (terms - (u - j)) * (terms - (u - j));
    }

    auto penalty = penalty_rows + penalty_cols;
    auto constraint = penalty_params * penalty;

    // Build objective
    auto objective = qbpp::expr();
    for (const auto &e : edges)
    {
        int u = e.first;
        int v = e.second;

        auto abs_diff = qbpp::expr();
        for (size_t j = 0; j < k; j++)
        {
            abs_diff += x[u][j] + x[v][j] - 2 * x[u][j] * x[v][j];
        }

        objective += abs_diff;
    }

    auto qubo = objective + penalty_params * constraint;
    qubo.simplify_as_binary();

    // ========== Solve using Exhaustive Solver ==========
    cout << "\n=== Solving ===" << endl;
    
    qbpp::exhaustive_solver::ExhaustiveSolver solver(qubo);
    solver.enable_default_callback();

    auto solution = solver.search();
    
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    
    // ========== Extract and Validate Solution ==========
    vector<int> positions = extract_positions(solution, x, n);
    bool is_feasible = is_valid_permutation(positions);
    
    // ========== Output Results ==========
    cout << "\n=== Results ===" << endl;
    cout << "Penalty parameter: " << penalty_params << endl;
    cout << "Feasibility: " << (is_feasible ? "Feasible" : "Infeasible") << endl;
    cout << "Energy: " << solution.energy() << endl;
    cout << "Time: " << duration.count() << " ms" << endl;
    cout << "Arrangement: ";
    
    for (size_t i = 0; i < n; ++i)
    {
        cout << positions[i];
        if (i < n - 1) cout << " ";
    }
    cout << endl;
    
    return 0;
}