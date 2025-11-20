#!/usr/bin/env python3
"""
Validate preprocessed graph files for:
- Duplicate edges
- Self-loops
- Graph connectivity
- Correct number of vertices and edges
"""

import os
from pathlib import Path
from collections import defaultdict

def validate_graph(filepath):
    """Validate a single graph file"""
    print(f"\n{'='*70}")
    print(f"Validating: {filepath.name}")
    print(f"{'='*70}")
    
    issues = []
    
    try:
        with open(filepath, 'r') as f:
            # Read header
            first_line = f.readline().strip()
            declared_vertices, declared_edges = map(int, first_line.split())
            
            print(f"Declared: {declared_vertices} vertices, {declared_edges} edges")
            
            # Read all edges
            edges = []
            edge_set = set()
            vertices = set()
            
            for line_num, line in enumerate(f, start=2):
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                if len(parts) >= 2:
                    u, v = int(parts[0]), int(parts[1])
                    edges.append((u, v))
                    vertices.add(u)
                    vertices.add(v)
                    
                    # Check for self-loop
                    if u == v:
                        issues.append(f"❌ Self-loop found at line {line_num}: {u} - {v}")
                    
                    # Check for duplicate edges (treat as undirected)
                    edge_normalized = tuple(sorted([u, v]))
                    if edge_normalized in edge_set:
                        issues.append(f"❌ Duplicate edge found at line {line_num}: {u} - {v}")
                    edge_set.add(edge_normalized)
            
            # Check number of edges
            actual_edges = len(edges)
            if actual_edges != declared_edges:
                issues.append(f"❌ Edge count mismatch: declared {declared_edges}, actual {actual_edges}")
            else:
                print(f"✓ Edge count correct: {actual_edges} edges")
            
            # Check number of vertices
            actual_vertices = len(vertices)
            if actual_vertices != declared_vertices:
                issues.append(f"❌ Vertex count mismatch: declared {declared_vertices}, actual {actual_vertices}")
            else:
                print(f"✓ Vertex count correct: {actual_vertices} vertices")
            
            # Check if vertices are 0-indexed and consecutive
            expected_vertices = set(range(declared_vertices))
            if vertices != expected_vertices:
                missing = expected_vertices - vertices
                extra = vertices - expected_vertices
                if missing:
                    issues.append(f"❌ Missing vertices: {sorted(missing)}")
                if extra:
                    issues.append(f"❌ Extra vertices (not in 0..{declared_vertices-1}): {sorted(extra)}")
            else:
                print(f"✓ Vertices are 0-indexed and consecutive: 0..{declared_vertices-1}")
            
            # Check graph connectivity using Union-Find
            if vertices:
                parent = {v: v for v in vertices}
                
                def find(x):
                    if parent[x] != x:
                        parent[x] = find(parent[x])
                    return parent[x]
                
                def union(x, y):
                    root_x = find(x)
                    root_y = find(y)
                    if root_x != root_y:
                        parent[root_x] = root_y
                
                # Union all edges
                for u, v in edges:
                    union(u, v)
                
                # Check if all vertices have the same root
                roots = set(find(v) for v in vertices)
                if len(roots) > 1:
                    issues.append(f"❌ Graph is NOT connected: {len(roots)} connected components")
                    
                    # Find components
                    components = defaultdict(list)
                    for v in vertices:
                        components[find(v)].append(v)
                    
                    print(f"   Components:")
                    for i, (root, component) in enumerate(sorted(components.items()), 1):
                        print(f"     Component {i}: {len(component)} vertices - {sorted(component)}")
                else:
                    print(f"✓ Graph is connected")
            
            # Check for duplicate edges (if any were found earlier)
            if not any("Duplicate edge" in issue for issue in issues):
                print(f"✓ No duplicate edges")
            
            # Check for self-loops (if any were found earlier)
            if not any("Self-loop" in issue for issue in issues):
                print(f"✓ No self-loops")
            
            # Print all issues
            if issues:
                print(f"\n{'Issues found:':^70}")
                for issue in issues:
                    print(f"  {issue}")
                return False
            else:
                print(f"\n{'✓ ALL CHECKS PASSED':^70}")
                return True
                
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

def main():
    # Get the processed directory
    script_dir = Path(__file__).parent
    processed_dir = script_dir / "processed"
    
    if not processed_dir.exists():
        print(f"Error: 'processed' directory not found at {processed_dir}")
        return
    
    # Find all preprocessed graph files
    graph_files = sorted(processed_dir.glob("*_preprocessed.txt"))
    
    if not graph_files:
        print("No preprocessed graph files found")
        return
    
    print(f"Found {len(graph_files)} preprocessed graph file(s)")
    
    # Validate each file
    results = {}
    for filepath in graph_files:
        passed = validate_graph(filepath)
        results[filepath.name] = passed
    
    # Print summary
    print(f"\n\n{'='*70}")
    print(f"{'VALIDATION SUMMARY':^70}")
    print(f"{'='*70}")
    
    passed_count = sum(results.values())
    failed_count = len(results) - passed_count
    
    for filename, passed in sorted(results.items()):
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status:8} {filename}")
    
    print(f"\n{'-'*70}")
    print(f"Total: {len(results)} files | Passed: {passed_count} | Failed: {failed_count}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
