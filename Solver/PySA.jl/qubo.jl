using JuMP
using QUBO
using PySA
using CSV
using DataFrames

function read_graph_file(filename)
    lines = readlines(filename)
    n, m = parse.(Int, split(lines[1]))
    edges = Tuple{Int, Int}[]
    
    for i in 2:length(lines)
        if !isempty(strip(lines[i]))
            u, v = parse.(Int, split(lines[i]))
            push!(edges, (u + 1, v + 1))
        end
    end
    
    return n, edges
end

function param_for_general_graph(n, m)
    naive = m * (n - 1)
    complete = n * (n - 1) * (n + 1) / 6.0
    k = ceil(n + 0.5 - 0.5 * sqrt(8.0 * m + 1))
    f = (n - k) * (n - k + 1) / 2.0
    edges_method = (m - f) * (k - 1) + (n - k) * (n * n + (n + 3) * k - 2 * k * k - 1) / 6.0
    
    return Int(min(naive, complete, edges_method))
end

function create_minla_qubo(n, edges, penalty)
    model = Model(() -> ToQUBO.Optimizer(PySA.Optimizer))
    
    @variable(model, x[1:n, 1:n], Bin)
    
    objective_expr = QuadExpr()
    for (u, v) in edges
        for k in 1:n
            add_to_expression!(objective_expr, 1.0, x[u, k])
            add_to_expression!(objective_expr, 1.0, x[v, k])
            add_to_expression!(objective_expr, -2.0, x[u, k], x[v, k])
        end
    end
    
    row_penalty_expr = QuadExpr()
    for i in 1:n
        for k in 1:(n-1)
            add_to_expression!(row_penalty_expr, 1.0, x[i, k+1])
            add_to_expression!(row_penalty_expr, -1.0, x[i, k], x[i, k+1])
        end
    end
    
    col_penalty_expr = QuadExpr()
    for k in 1:n
        col_sum = sum(x[i, k] for i in 1:n)
        deviation = col_sum - (n - k + 1)
        add_to_expression!(col_penalty_expr, deviation * deviation)
    end
    
    @objective(model, Min, objective_expr + penalty * (row_penalty_expr + col_penalty_expr))
    
    return model, x
end

function is_valid_arrangement(arrangement, n)
    return length(unique(arrangement)) == n && all(x -> 1 <= x <= n, arrangement)
end

function run_qubo_on_dataset(dataset_path)
    dataset_name = replace(basename(dataset_path), "_preprocessed.txt" => "")
    println("\nProcessing $(dataset_name)...")
    
    n, edges = read_graph_file(dataset_path)
    m = length(edges)
    
    println("  Vertices: $(n), Edges: $(m)")
    
    penalty = param_for_general_graph(n, m)
    println("  Penalty parameter: $(penalty)")
    
    start_time = time()
    
    model, x = create_minla_qubo(n, edges, penalty)
    optimize!(model)
    
    end_time = time()
    duration = (end_time - start_time) * 1000
    
    arrangement = zeros(Int, n)
    for i in 1:n
        label = sum(Int(round(value(x[i, k]))) for k in 1:n)
        arrangement[i] = label
    end
    
    feasible = is_valid_arrangement(arrangement, n)
    
    cost = 0
    for (u, v) in edges
        cost += abs(arrangement[u] - arrangement[v])
    end
    
    obj_value = objective_value(model)
    
    println("  Feasible: $(feasible)")
    println("  Cost: $(cost)")
    println("  Time: $(round(duration, digits=2)) ms")
    
    return (
        dataset = dataset_name,
        vertices = n,
        edges = m,
        method = "PySA",
        penalty = penalty,
        feasible = feasible,
        cost = cost,
        energy = obj_value,
        time_ms = duration
    )
end

function main()
    dataset_dir = "../../Dataset/processed"
    results_dir = "../../Results"
    output_file = joinpath(results_dir, "pysa_jl_results.csv")
    
    mkpath(results_dir)
    
    dataset_files = filter(f -> endswith(f, "_preprocessed.txt") && contains(f, "n200"), readdir(dataset_dir))
    sort!(dataset_files)
    
    results = []
    
    for dataset_file in dataset_files
        dataset_path = joinpath(dataset_dir, dataset_file)
        try
            result = run_qubo_on_dataset(dataset_path)
            push!(results, result)
        catch e
            println("Error processing $(dataset_file): $(e)")
            continue
        end
    end
    
    df = DataFrame(results)
    CSV.write(output_file, df)
    
    println("\n" * "="^60)
    println("Results saved to: $(output_file)")
    println("Total datasets processed: $(length(results))")
    println("="^60)
end

main()
