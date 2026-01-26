README for data set alchemy

=== Usage ===

This folder is one third of the dataset alchemy. The full dataset contains the three sets dev, test and valid.
Each data set consists of a folder with following files
(replace DS by the name of the dataset):

n = total number of nodes
m = total number of edges
N = number of graphs

(1) 	DS_A.txt (m lines) 
	sparse (block diagonal) adjacency matrix for all graphs,
	each line corresponds to (row, col) resp. (node_id, node_id)

(2) 	DS_graph_indicator.txt (n lines)
	column vector of graph identifiers for all nodes of all graphs,
	the value in the i-th line is the graph_id of the node with node_id i

(3)* 	DS_graph_labels.txt (N lines) 
	class labels for all graphs in the dataset,
	the value in the i-th line is the class label of the graph with graph_id i

* The graph labels are 12 quantum mechanical properties. In data set test, no such labels are given. The labels of this data set were to be predicted and submitted in this contest https://alchemy.tencent.com/.


(4) 	DS_node_labels.txt (n lines)
	column vector of node labels,
	the value in the i-th line corresponds to the node with node_id i

There are OPTIONAL files if the respective information is available:

(5) 	DS_edge_labels.txt (m lines; same size as DS_A_sparse.txt)
	labels for the edges in DS_A_sparse.txt 

(6) 	DS_edge_attributes.txt (m lines; same size as DS_A.txt)
	attributes for the edges in DS_A.txt 

(7) 	DS_node_attributes.txt (n lines) 
	matrix of node attributes,
	the comma seperated values in the i-th line is the attribute vector of the node with node_id i

(8) 	DS_graph_attributes.txt (N lines) 
	regression values for all graphs in the dataset,
	the value in the i-th line is the attribute of the graph with graph_id i


Each data set also contains a special file:

(9)	DS_graph_heavy_atoms.txt (N lins)
	value in i-th line is the number of heavy atoms in graph with graph_id i


=== Node Label Conversion === 

Node labels were converted to integer values using this map:

Component 0:
	0	C
	1	O
	2	S
	3	N
	4	F
	5	Cl



Edge labels were converted to integer values using this map:

Component 0:
	0	-
	1	:
	2	=
	3	#




Class labels were not converted

=== References ===

Source:
Alchemy: A Quantum Chemistry Dataset for Benchmarking AI Models
https://arxiv.org/abs/1906.09427