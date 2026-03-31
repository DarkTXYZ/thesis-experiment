import networkx as nx
import scipy.sparse.linalg as sla
import numpy as np
import logging
import time
import sys

# Configure logging
logging.basicConfig(
    level=logging.CRITICAL,  # Disabled - set to CRITICAL to suppress debug/info logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('lower_bound.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

class MinLALowerBounds:
    def __init__(self, G):
        """Initialize with a NetworkX graph."""
        self.G = G
        self.n = G.number_of_nodes()
        self.m = G.number_of_edges()
        logger.info(f"Initialized MinLALowerBounds with graph: n={self.n}, m={self.m}")

    def edges_method(self):
        logger.debug("Starting edges_method...")
        start_time = time.time()
        try:
            n = self.n
            m = self.m
            i = 1
            f = 0
            lb = 0
            
            while f + n - i <= m:
                if n == i:
                    break
                f = f + n - i
                lb = lb + i * (n - i)
                i = i + 1

            result = lb + i * (m - f)
            elapsed = time.time() - start_time
            logger.debug(f"edges_method completed in {elapsed:.4f}s, result={result}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"edges_method failed after {elapsed:.4f}s: {e}", exc_info=True)
            raise

    def degree_method(self):
        logger.debug("Starting degree_method...")
        start_time = time.time()
        try:
            lb = 0
            for v in self.G.nodes():
                d = self.G.degree(v)

                if d % 2 == 0:
                    lb += d * d / 4 + d / 2
                else:
                    lb += (d * d + 2 * d + 1) / 4
            
            result = lb / 2
            elapsed = time.time() - start_time
            logger.debug(f"degree_method completed in {elapsed:.4f}s, result={result}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"degree_method failed after {elapsed:.4f}s: {e}", exc_info=True)
            raise

    def juvan_mohar_method(self):
        if self.n <= 1:
            return 0
        
        L = nx.laplacian_matrix(self.G).astype(float)
        e = np.linalg.eigvals(L.toarray())
        lambda_2 = sorted(e)[1] 

        return lambda_2 * (self.n ** 2 - 1) / 6

    def gomory_hu_method(self):
        logger.debug("Starting gomory_hu_method...")
        start_time = time.time()
        try:
            logger.debug(f"Creating graph copy...")
            G_copy = self.G.copy()
            nx.set_edge_attributes(G_copy, 1, "capacity")
            
            logger.debug(f"Computing Gomory-Hu tree...")
            T = nx.gomory_hu_tree(G_copy)
            logger.debug(f"Gomory-Hu tree computed successfully")
            
            # sum all edges weight
            sum_weights = 0
            for _, _, data in T.edges(data=True):
                sum_weights += data['weight']

            elapsed = time.time() - start_time
            logger.debug(f"gomory_hu_method completed in {elapsed:.4f}s, result={sum_weights}")
            return sum_weights
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"gomory_hu_method failed after {elapsed:.4f}s: {e}", exc_info=True)
            raise

    def path_method(self):
        """
        5. Path Method (LB_path) - Approximation
        Petit's path method uses path embeddings. A simplified valid lower bound 
        utilizes the graph diameter: the longest shortest-path in the graph.
        """
        logger.debug("Starting path_method...")
        start_time = time.time()
        try:
            if not nx.is_connected(self.G):
                logger.debug(f"Graph is disconnected, processing connected components...")
                result = sum(MinLALowerBounds(self.G.subgraph(c)).path_method() 
                           for c in nx.connected_components(self.G))
                elapsed = time.time() - start_time
                logger.debug(f"path_method (disconnected) completed in {elapsed:.4f}s, result={result}")
                return result
            try:
                logger.debug(f"Computing graph diameter...")
                # The true MinLA must be at least the diameter of the graph
                diameter = nx.diameter(self.G)
                elapsed = time.time() - start_time
                logger.debug(f"path_method completed in {elapsed:.4f}s, diameter={diameter}")
                return diameter
            except nx.NetworkXError as e:
                logger.warning(f"Failed to compute diameter: {e}")
                return 0
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"path_method failed after {elapsed:.4f}s: {e}", exc_info=True)
            raise

    def evaluate_all(self):
        """Returns a dictionary of all 6 bounds."""
        logger.info(f"evaluate_all: Starting evaluation for graph with n={self.n}, m={self.m}")
        start_time = time.time()
        try:
            results = {
                "Edges Method": self.edges_method(),
                "Degree Method": self.degree_method(),
                # "Juvan-Mohar Method": self.juvan_mohar_method(),
                "Gomory-Hu Method": self.gomory_hu_method(),
                "Path Method (Approx)": self.path_method()
            }
            elapsed = time.time() - start_time
            logger.info(f"evaluate_all: Completed in {elapsed:.4f}s. Results: {results}")
            return results
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"evaluate_all failed after {elapsed:.4f}s: {e}", exc_info=True)
            raise
    
    def return_max(self):
        logger.debug("return_max: Computing max value...")
        result = max(self.evaluate_all().values())
        logger.debug(f"return_max: result={result}")
        return result
    
    def return_min(self):
        logger.debug("return_min: Computing min value...")
        result = min(self.evaluate_all().values())
        logger.debug(f"return_min: result={result}")
        return result

def calculate_lower_obj_bound(G: nx.Graph):
    logger.info(f"calculate_lower_obj_bound: Starting for graph with n={G.number_of_nodes()}, m={G.number_of_edges()}")
    start_time = time.time()
    try:
        lower_bound = MinLALowerBounds(G)
        result = lower_bound.return_max()
        elapsed = time.time() - start_time
        logger.info(f"calculate_lower_obj_bound: Completed in {elapsed:.4f}s, result={result}")
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"calculate_lower_obj_bound failed after {elapsed:.4f}s: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    # create erdos_renyi graph until connected
    G = nx.erdos_renyi_graph(25, 0.5, seed = 42)
    j = calculate_lower_obj_bound(G)
    print("Lower Bound:", j)