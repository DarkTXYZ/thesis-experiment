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
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

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

void process_graph(const string& graph_file, ofstream& csv_out)
{
    size_t n;
    vector<pair<int, int>> edges;
    string graph_type = "unknown";
    
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
        cerr << "Failed to read graph file: " << graph_file << endl;
        return;
    }
    
    const size_t m = edges.size();

    cout << "\n=== Processing: " << filename << " ===" << endl;
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
    
    cout << "Number of variables used: " << qubo.size() << endl;
    cout << "Solving..." << endl;
    
    qbpp::easy_solver::EasySolver solver(qubo);
    solver.time_limit(60.0);
    
    auto solution = solver.search();
    
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    
    vector<int> positions = extract_positions(solution, x, n);

    if (positions.empty()) {
        cout << "No valid arrangement found." << endl;
        csv_out << filename << "," << n << "," << m << ",INVALID,0,0," 
                << duration.count() << endl;
        return;
    }

    bool is_feasible = is_valid_permutation(positions);
    size_t arrangement_cost = calculate_arrangement_cost(positions, edges);
    
    cout << "Feasibility: " << (is_feasible ? "Feasible" : "Infeasible") << endl;
    cout << "Energy: " << solution.energy() << endl;
    cout << "Arrangement cost: " << arrangement_cost << endl;
    cout << "Time: " << duration.count() << " ms" << endl;
    
    // Write to CSV: filename,vertices,edges,feasibility,penalty_param,cost,time_ms
    csv_out << filename << "," << n << "," << m << "," 
            << (is_feasible ? "Feasible" : "Infeasible") << ","
            << penalty_param << "," << arrangement_cost << "," 
            << duration.count() << endl;
}

int main(int argc, char* argv[])
{
    string processed_dir = "../processed";
    
    // Check if a specific file or directory is provided
    if (argc > 1)
    {
        string input_arg = argv[1];
        
        // Check if it's a directory
        if (fs::is_directory(input_arg))
        {
            processed_dir = input_arg;
        }
        else
        {
            // Single file mode
            string graph_file = input_arg;
            if (graph_file.find('/') == string::npos)
            {
                graph_file = processed_dir + "/" + graph_file;
            }
            
            cout << "Processing single file: " << graph_file << endl;
            
            ofstream csv_out("../results/easy_solver_single_result.csv");
            csv_out << "filename,vertices,edges,feasibility,penalty_param,cost,time_ms" << endl;
            
            process_graph(graph_file, csv_out);
            csv_out.close();
            
            return 0;
        }
    }
    
    // Batch processing mode
    cout << "=== MINLA Batch Processing ===" << endl;
    cout << "Processing directory: " << processed_dir << endl;
    
    // Get all .txt files from processed directory
    vector<string> graph_files;
    
    try {
        for (const auto& entry : fs::directory_iterator(processed_dir))
        {
            if (entry.is_regular_file() && entry.path().extension() == ".txt")
            {
                graph_files.push_back(entry.path().string());
            }
        }
    }
    catch (const fs::filesystem_error& e)
    {
        cerr << "Error accessing directory: " << e.what() << endl;
        return 1;
    }
    
    if (graph_files.empty())
    {
        cout << "No graph files found in " << processed_dir << endl;
        return 1;
    }
    
    sort(graph_files.begin(), graph_files.end());
    
    cout << "Found " << graph_files.size() << " graph files" << endl;
    
    // Create results directory if it doesn't exist
    fs::create_directories("../results");
    
    // Open CSV output file with timestamp
    auto now = chrono::system_clock::now();
    auto time_t_now = chrono::system_clock::to_time_t(now);
    stringstream ss;
    ss << put_time(localtime(&time_t_now), "%Y%m%d_%H%M%S");
    string timestamp = ss.str();
    
    string csv_filename = "../results/easy_solver_results_" + timestamp + ".csv";
    ofstream csv_out(csv_filename);
    
    if (!csv_out.is_open())
    {
        cerr << "Failed to create output file: " << csv_filename << endl;
        return 1;
    }
    
    // Write CSV header
    csv_out << "filename,vertices,edges,feasibility,penalty_param,cost,time_ms" << endl;
    
    // Process each graph file
    for (const auto& graph_file : graph_files)
    {
        process_graph(graph_file, csv_out);
    }
    
    csv_out.close();
    
    cout << "\n=== Batch Processing Complete ===" << endl;
    cout << "Results saved to: " << csv_filename << endl;
    
    return 0;
}