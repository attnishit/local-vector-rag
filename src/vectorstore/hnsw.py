"""
HNSW (Hierarchical Navigable Small World) Data Structures and Algorithms

Author: Nishit Attrey

This module implements the HNSW algorithm, which enables fast vector search
by organizing vectors in a multi-layer navigable graph structure:

- Layer 0: Contains ALL vectors (100% coverage)
- Layer 1+: Progressively sparser layers (probabilistic inclusion)
- Higher layers: Long-range connections for fast navigation
- Lower layers: Dense connections for precision

Key Concepts:
    - Skip List Inspiration: HNSW is like a multi-layer skip list for graphs
    - Navigable Small World: Greedy routing works efficiently
    - Hierarchical: Layers provide different navigation granularities
    - Approximate: Trades 100% recall for massive speed gains (1000x+)

References:
    - "Efficient and robust approximate nearest neighbor search using
      Hierarchical Navigable Small World graphs" (Malkov & Yashunin, 2018)
    - https://arxiv.org/abs/1603.09320

Author: RAG Team
Version: 0.1.0-stage8
"""

import numpy as np
import math
import heapq
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging

# Import similarity functions
from .similarity import cosine_similarity, euclidean_distance, dot_product

logger = logging.getLogger(__name__)


@dataclass
class HNSWNode:
    """
    Represents a single node (vector) in the HNSW graph.

    Each node exists at one or more layers in the graph hierarchy.
    At each layer, the node maintains bidirectional connections to its
    nearest neighbors.

    Attributes:
        node_id: Unique identifier for this node
        vector: The embedded vector (DEPRECATED: may be stale after matrix reallocation, use HNSWIndex._get_vector instead)
        metadata: Associated metadata (chunk_id, text, etc.)
        level: Maximum layer this node exists in (0 = bottom only, higher = exists in more layers)
        neighbors: Dict mapping layer number -> list of neighbor node IDs
                  Example: {0: [1, 5, 7], 1: [3, 9], 2: [12]}

    Example:
        >>> node = HNSWNode(
        ...     node_id=42,
        ...     vector=np.array([0.1, 0.2, 0.3]),
        ...     metadata={"text": "sample"},
        ...     level=2
        ... )
        >>> node.neighbors[0] = [1, 5, 7]  # Layer 0 neighbors
        >>> node.neighbors[1] = [3, 9]     # Layer 1 neighbors
    """

    node_id: int
    vector: Optional[np.ndarray]  # DEPRECATED: may become stale, use _get_vector instead
    metadata: Dict[str, Any]
    level: int
    neighbors: Dict[int, Set[int]] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize neighbor sets for all layers this node exists in."""
        # Create empty neighbor sets for each layer from 0 to self.level
        for layer in range(self.level + 1):
            if layer not in self.neighbors:
                self.neighbors[layer] = set()

    def add_neighbor(self, layer: int, neighbor_id: int) -> None:
        """
        Add a bidirectional connection to another node at a specific layer.

        Args:
            layer: The layer number (0 = bottom)
            neighbor_id: Node ID to connect to

        Note:
            This only adds the forward connection (self -> neighbor).
            The caller must add the reverse connection (neighbor -> self).
        """
        if layer not in self.neighbors:
            self.neighbors[layer] = set()

        # Set.add() is O(1) and idempotent (no need to check existence)
        self.neighbors[layer].add(neighbor_id)

    def get_neighbors(self, layer: int) -> List[int]:
        """
        Get all neighbor node IDs at a specific layer.

        Args:
            layer: The layer number

        Returns:
            List of neighbor node IDs (empty if layer doesn't exist)

        Note:
            Internal storage is Set[int] for O(1) lookups, but returns
            List[int] for backward compatibility.
        """
        neighbor_set = self.neighbors.get(layer, set())
        return list(neighbor_set)

    def __repr__(self) -> str:
        """String representation for debugging."""
        neighbor_counts = {layer: len(neighbors) for layer, neighbors in self.neighbors.items()}
        return (
            f"HNSWNode(id={self.node_id}, level={self.level}, "
            f"neighbors={neighbor_counts}, metadata={self.metadata})"
        )


class HNSWIndex:
    """
    HNSW graph index for approximate nearest neighbor search.

    The index maintains a hierarchical graph structure where:
    - All vectors exist at layer 0 (the base layer)
    - Some vectors exist at higher layers (probabilistically assigned)
    - Higher layer nodes act as "highway" entry points for fast search
    - Lower layers provide dense connectivity for accurate results

    Parameters:
        dimension: Vector dimensionality (must match embedding dimension)
        m: Maximum number of bidirectional links per node (except layer 0)
           Typical values: 12-48. Higher = better recall, more memory.
        m0: Maximum number of connections at layer 0
            Typically 2*m for denser base layer.
        ef_construction: Size of dynamic candidate list during graph construction
                        Typical values: 100-500. Higher = better graph quality, slower build.
        ml: Normalization factor for level assignment (default: 1/ln(2) ≈ 1.44)
        similarity_metric: Distance metric ("cosine", "dot", "euclidean")
        normalized: Whether input vectors are L2 normalized
        seed: Random seed for reproducible level assignment

    Attributes:
        nodes: Dict mapping node_id -> HNSWNode
        entry_point: Node ID of the top-layer entry point (or None if empty)
        level_count: Current number of layers in the graph
        next_node_id: Counter for assigning unique node IDs

    Example:
        >>> index = HNSWIndex(dimension=384, m=16, ef_construction=200)
        >>> # Stage 7: Insert vectors to build the graph
        >>> vec = np.random.randn(384)
        >>> node_id = index.insert(vec, metadata={"text": "sample"})
        >>> # Stage 8: Search for nearest neighbors
        >>> query = np.random.randn(384)
        >>> results = index.search(query, k=5, ef_search=50)
    """

    def __init__(
        self,
        dimension: int,
        m: int = 16,
        m0: Optional[int] = None,
        ef_construction: int = 200,
        ml: Optional[float] = None,
        similarity_metric: str = "cosine",
        normalized: bool = True,
        seed: Optional[int] = None,
        use_heuristic: bool = True,  # NEW: Enable diversity-aware selection by default
    ):
        """Initialize an empty HNSW index."""
        # Vector properties
        self.dimension = dimension
        self.normalized = normalized
        self.similarity_metric = similarity_metric

        # HNSW parameters
        self.m = m  # Max connections per layer (typical: 16)
        self.m0 = m0 if m0 is not None else 2 * m  # Max connections at layer 0 (typical: 32)
        self.ef_construction = ef_construction  # Construction candidate list size
        self.use_heuristic = use_heuristic  # Use diversity-aware neighbor selection (Fix 6)

        # Level assignment parameter (normalization factor)
        # ml = 1/ln(2) means each level has ~50% as many nodes as the level below
        self.ml = ml if ml is not None else 1.0 / math.log(2.0)

        # Graph structure
        self.nodes: Dict[int, HNSWNode] = {}  # All nodes in the graph
        self.entry_point: Optional[int] = None  # Top-layer entry point node ID
        self.level_count: int = 0  # Current number of layers
        self.next_node_id: int = 0  # Counter for node IDs

        # Contiguous vector storage for performance (Fix 3)
        # Using dynamic array-like growth strategy (like ArrayList in Java)
        self._vector_capacity = 1000  # Initial capacity
        self._vector_matrix = np.zeros((self._vector_capacity, dimension), dtype=np.float32)
        self._num_vectors = 0  # Actual number of vectors stored

        # Random number generator (for reproducibility)
        # Use NumPy's Generator for better performance
        if seed is not None:
            self.rng = np.random.Generator(np.random.PCG64(seed))
        else:
            self.rng = np.random.default_rng()

        logger.info(
            f"Initialized HNSW index: dimension={dimension}, m={m}, "
            f"m0={self.m0}, ef_construction={ef_construction}, "
            f"ml={self.ml:.3f}, metric={similarity_metric}"
        )

    def _random_level(self) -> int:
        """
        Randomly assign a level (layer) for a new node.

        Uses exponentially decaying probability: P(level=l) = (1/2)^l
        This creates a skip-list-like hierarchy where:
        - ~100% of nodes at level 0
        - ~50% of nodes at level 1
        - ~25% of nodes at level 2
        - ~12.5% of nodes at level 3
        - etc.

        Returns:
            Random level (0 = base layer, higher = exists in more layers)

        Algorithm:
            level = floor(-ln(uniform(0,1)) * ml)

        Example:
            With ml=1.44:
            - uniform(0,1) = 0.9 → level = 0
            - uniform(0,1) = 0.5 → level = 1
            - uniform(0,1) = 0.1 → level = 3
        """
        # Generate uniform random number in (0, 1)
        # Note: We use (0,1) not [0,1] to avoid log(0)
        uniform_random = self.rng.uniform(0.0, 1.0)

        # Apply exponential distribution
        level = int(-math.log(uniform_random) * self.ml)

        return level

    def _ensure_vector_capacity(self, needed_capacity: int) -> None:
        """
        Grow vector matrix if needed (ArrayList-style dynamic growth).

        Args:
            needed_capacity: Minimum capacity required

        Note:
            Grows by 1.5x when capacity is exceeded to amortize allocation cost.
        """
        if needed_capacity <= self._vector_capacity:
            return

        # Grow by 1.5x or to needed capacity, whichever is larger
        new_capacity = max(int(self._vector_capacity * 1.5), needed_capacity)

        logger.debug(f"Growing vector matrix: {self._vector_capacity} → {new_capacity}")

        # Allocate new matrix
        new_matrix = np.zeros((new_capacity, self.dimension), dtype=np.float32)

        # Copy existing vectors
        new_matrix[: self._num_vectors] = self._vector_matrix[: self._num_vectors]

        # Replace
        self._vector_matrix = new_matrix
        self._vector_capacity = new_capacity

    def _add_vector(self, vector: np.ndarray) -> int:
        """
        Add vector to contiguous storage.

        Args:
            vector: Vector to add

        Returns:
            Index in vector matrix (same as node_id)
        """
        self._ensure_vector_capacity(self._num_vectors + 1)

        self._vector_matrix[self._num_vectors] = vector
        vector_idx = self._num_vectors
        self._num_vectors += 1

        return vector_idx

    def _get_vector(self, node_id: int) -> np.ndarray:
        """
        Get vector by node_id (view into matrix).

        Args:
            node_id: Node ID

        Returns:
            Vector as view into the matrix (not a copy)

        Raises:
            ValueError: If node_id is invalid
        """
        if node_id >= self._num_vectors:
            raise ValueError(f"Invalid node_id: {node_id}")
        return self._vector_matrix[node_id]

    def _get_vectors_batch(self, node_ids: List[int]) -> np.ndarray:
        """
        Get multiple vectors at once (efficient matrix slice).

        Args:
            node_ids: List of node IDs

        Returns:
            Matrix of vectors (n, dimension)

        Example:
            >>> vectors = index._get_vectors_batch([0, 5, 10])
            >>> vectors.shape
            (3, 384)
        """
        if not node_ids:
            return np.array([]).reshape(0, self.dimension)
        return self._vector_matrix[node_ids]

    def __len__(self) -> int:
        """Return number of nodes in the index."""
        return len(self.nodes)

    def statistics(self) -> Dict[str, Any]:
        """
        Compute statistics about the HNSW graph structure.

        Returns:
            Dictionary with graph statistics

        Example:
            >>> stats = index.statistics()
            >>> print(f"Total nodes: {stats['num_nodes']}")
            >>> print(f"Max level: {stats['max_level']}")
        """
        if len(self) == 0:
            return {
                "num_nodes": 0,
                "max_level": -1,
                "nodes_per_level": {},
                "avg_neighbors_per_level": {},
                "entry_point": None,
            }

        # Count nodes at each level
        nodes_per_level: Dict[int, int] = {}
        neighbors_per_level: Dict[int, List[int]] = {}

        for node in self.nodes.values():
            for layer in range(node.level + 1):
                # Count node at this layer
                nodes_per_level[layer] = nodes_per_level.get(layer, 0) + 1

                # Track neighbor counts
                if layer not in neighbors_per_level:
                    neighbors_per_level[layer] = []
                neighbors_per_level[layer].append(len(node.get_neighbors(layer)))

        # Compute average neighbors per level
        avg_neighbors_per_level = {
            layer: np.mean(counts) if counts else 0.0
            for layer, counts in neighbors_per_level.items()
        }

        # Calculate memory usage
        # Each node stores: vector + metadata + neighbor lists
        # Use actual capacity, not just used vectors, to reflect true memory usage
        vector_bytes = self._vector_capacity * self.dimension * 4  # float32 arrays
        # Rough estimate for metadata and neighbor lists
        # (metadata dict + neighbor IDs per layer)
        metadata_bytes = len(self) * 1024  # ~1KB per node for metadata
        # Each neighbor list stores node IDs (4 bytes each)
        neighbor_bytes = sum(
            sum(len(node.get_neighbors(layer)) for layer in range(node.level + 1)) * 4
            for node in self.nodes.values()
        )
        total_bytes = vector_bytes + metadata_bytes + neighbor_bytes
        memory_mb = total_bytes / (1024 * 1024)

        return {
            "num_nodes": len(self),
            "max_level": self.level_count - 1 if self.level_count > 0 else -1,
            "nodes_per_level": nodes_per_level,
            "avg_neighbors_per_level": avg_neighbors_per_level,
            "entry_point": self.entry_point,
            "dimension": self.dimension,
            "m": self.m,
            "m0": self.m0,
            "memory_mb": memory_mb,
        }

    def get_node(self, node_id: int) -> Optional[HNSWNode]:
        """
        Retrieve a node by its ID.

        Args:
            node_id: The node's unique identifier

        Returns:
            HNSWNode if found, None otherwise
        """
        return self.nodes.get(node_id)

    def _get_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute distance between two vectors based on similarity metric.

        For similarity metrics (cosine, dot), we return 1 - similarity to
        get a distance measure (smaller = closer).

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Distance value (smaller = more similar)
        """
        if self.similarity_metric == "cosine":
            sim = cosine_similarity(vec1, vec2, normalized=self.normalized)
            return 1.0 - sim  # Convert similarity to distance
        elif self.similarity_metric == "dot":
            sim = dot_product(vec1, vec2)
            return -sim  # Negate to get distance (higher dot = lower distance)
        elif self.similarity_metric == "euclidean":
            return euclidean_distance(vec1, vec2)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")

    def _get_batch_distances(self, query: np.ndarray, node_ids: List[int]) -> np.ndarray:
        """
        Compute distances between query and multiple nodes efficiently.

        This is a vectorized version of _get_distance that processes multiple
        nodes at once, providing significant speedup through NumPy optimizations.

        Uses contiguous vector matrix for maximum performance (Fix 3).

        Args:
            query: Query vector
            node_ids: List of node IDs to compute distances to

        Returns:
            Array of distances corresponding to node_ids

        Example:
            >>> dists = index._get_batch_distances(query, [0, 5, 10])
            >>> # Returns array of 3 distances
        """
        if len(node_ids) == 0:
            return np.array([])

        # Use efficient matrix slicing (Fix 3) instead of list comprehension
        vectors = self._get_vectors_batch(node_ids)

        # Batch compute based on metric using optimized functions
        if self.similarity_metric == "cosine":
            from .similarity import batch_cosine_similarity

            sims = batch_cosine_similarity(query, vectors, normalized=self.normalized)
            return 1.0 - sims  # Convert to distances
        elif self.similarity_metric == "dot":
            from .similarity import batch_dot_product

            dots = batch_dot_product(query, vectors)
            return -dots  # Negate for distance
        elif self.similarity_metric == "euclidean":
            from .similarity import batch_euclidean_distance

            return batch_euclidean_distance(query, vectors)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")

    def _search_layer(
        self, query: np.ndarray, entry_points: List[int], layer: int, num_to_return: int
    ) -> List[Tuple[float, int]]:
        """
        Greedy search for nearest neighbors at a specific layer.

        This is a key subroutine used during insertion and search.
        Starting from entry points, greedily move to closer neighbors.

        Args:
            query: Query vector
            entry_points: Starting node IDs for search
            layer: Layer number to search on
            num_to_return: Number of nearest neighbors to return

        Returns:
            List of (distance, node_id) tuples, sorted by distance (closest first)

        Algorithm:
            - Maintain visited set to avoid cycles
            - Maintain candidates (min-heap of distances)
            - Maintain dynamic list of best found (max-heap)
            - Greedily explore closer neighbors
        """
        visited = set(entry_points)

        # Candidates: min-heap of (distance, node_id)
        # We explore nodes from this heap in order of increasing distance
        candidates = []
        # Best found so far: max-heap of (-distance, node_id)
        # We keep the closest num_to_return nodes
        # Using negative distance to make it a max-heap (Python has min-heap only)
        best = []

        # BATCH 1: Compute all entry point distances at once
        entry_dists = self._get_batch_distances(query, entry_points)
        for ep, dist in zip(entry_points, entry_dists):
            if ep in self.nodes:
                heapq.heappush(candidates, (dist, ep))
                heapq.heappush(best, (-dist, ep))

        while candidates:
            current_dist, current = heapq.heappop(candidates)

            # Get the current worst distance in our best list
            # (best is a max-heap with negated distances, so -best[0][0] is the worst/farthest)
            worst_best_dist = -best[0][0] if best else float("inf")

            # If current is farther than the farthest in our best list, stop
            # But only if we have found enough neighbors
            if len(best) >= num_to_return and current_dist > worst_best_dist:
                break

            # Explore neighbors of current node at this layer
            current_node = self.nodes[current]

            # Collect unvisited neighbors
            neighbors = [nid for nid in current_node.get_neighbors(layer) if nid not in visited]

            if not neighbors:
                continue

            # Mark all as visited
            visited.update(neighbors)

            # BATCH 2: Compute all neighbor distances at once
            neighbor_dists = self._get_batch_distances(query, neighbors)

            for neighbor_id, dist in zip(neighbors, neighbor_dists):
                # Always add to candidates if we haven't found enough yet
                # Or if this neighbor could potentially improve our best list
                if len(best) < num_to_return:
                    heapq.heappush(candidates, (dist, neighbor_id))
                    heapq.heappush(best, (-dist, neighbor_id))
                elif dist < -best[0][0]:
                    # This neighbor is closer than our current worst best
                    heapq.heappush(candidates, (dist, neighbor_id))
                    heapq.heappush(best, (-dist, neighbor_id))
                    heapq.heappop(best)  # Remove the farthest to maintain size

        # Convert back to normal distances and sort
        result = [(-dist, node_id) for dist, node_id in best]
        result.sort()  # Sort by distance (ascending)

        return result

    def _select_neighbors_simple(self, candidates: List[Tuple[float, int]], m: int) -> List[int]:
        """
        Select m best neighbors from candidates (simple heuristic).

        Simple version: just take the m nearest neighbors by distance.
        Uses heapq.nsmallest for O(n + m log n) complexity instead of O(n log n).

        Args:
            candidates: List of (distance, node_id) tuples
            m: Number of neighbors to select

        Returns:
            List of selected node IDs

        Note:
            More sophisticated heuristics (like in the paper) prefer diverse
            neighbors to avoid clustering. For now, we use the simple approach.
        """
        # Edge case: if we want all or fewer candidates, return all
        if len(candidates) <= m:
            return [nid for _, nid in candidates]

        # Use heapq.nsmallest for O(n + m log n) instead of full O(n log n) sort
        m_smallest = heapq.nsmallest(m, candidates, key=lambda x: x[0])
        selected = [nid for _, nid in m_smallest]
        return selected

    def _select_neighbors_heuristic(
        self,
        candidates: List[Tuple[float, int]],
        m: int,
        layer: int,
        extend_candidates: bool = False,
        keep_pruned_connections: bool = True,
    ) -> List[int]:
        """
        Select neighbors using Algorithm 4 from HNSW paper (diversity-aware).

        Prefers diverse neighbors over just closest neighbors, improving
        graph connectivity and search quality.

        Args:
            candidates: List of (distance, node_id) tuples
            m: Number of neighbors to select
            layer: Current layer
            extend_candidates: If True, extend candidates with their neighbors (experimental)
            keep_pruned_connections: If True, add back pruned connections to fill m

        Returns:
            List of selected node IDs

        Algorithm:
            The heuristic iteratively selects candidates that are closer to the
            query than to already selected neighbors. This promotes diversity
            and prevents clustering, leading to better graph routing properties.

            Key insight: A candidate e is added if it's closer to the query point
            than to ANY already selected neighbor. This prevents selecting nodes
            that are clustered together.
        """
        if len(candidates) <= m:
            return [node_id for _, node_id in candidates]

        result = []  # List of selected node IDs
        result_dists = []  # Distances from query to selected nodes (for reference)
        working_queue = sorted(candidates, key=lambda x: x[0])  # Sort by distance (closest first)
        discarded = []

        while len(working_queue) > 0 and len(result) < m:
            dist_e, e = working_queue.pop(0)

            # First candidate is always added (it's the closest)
            if len(result) == 0:
                result.append(e)
                result_dists.append(dist_e)
                continue

            # Compute distances from candidate e to all already-selected neighbors
            # We need to check if e is closer to query than to any selected neighbor
            dists_to_selected = self._get_batch_distances(self._get_vector(e), result)

            # HNSW paper Algorithm 4 heuristic:
            # Add candidate e if for all already selected nodes r:
            #   distance(e, query) < distance(e, r)
            # This means e is closer to query than to any selected neighbor,
            # ensuring we pick diverse directions from the query point.
            #
            # Note: dist_e = distance(e, query)
            # dists_to_selected[i] = distance(e, result[i])
            min_dist_to_selected = np.min(dists_to_selected)

            if dist_e < min_dist_to_selected:
                # e is closer to query than to its nearest selected neighbor
                # This means e extends in a new direction - keep it!
                result.append(e)
                result_dists.append(dist_e)
            else:
                # e is closer to an already-selected neighbor than to query
                # This means e is redundant (covered by existing selection)
                discarded.append((dist_e, e))

        # Fill remaining slots with discarded candidates if needed
        # This ensures we always return m neighbors when possible
        if keep_pruned_connections and len(result) < m:
            discarded.sort(key=lambda x: x[0])
            while len(result) < m and len(discarded) > 0:
                _, e = discarded.pop(0)
                result.append(e)

        return result

    def insert(self, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Insert a new vector into the HNSW graph (Stage 7).

        This builds the graph by:
        1. Assigning a random level to the new node
        2. Finding nearest neighbors at each layer
        3. Connecting the node to its neighbors
        4. Updating bidirectional connections

        Args:
            vector: The vector to insert (must be dimension-dimensional)
            metadata: Optional metadata dictionary

        Returns:
            Node ID of the inserted vector

        Raises:
            ValueError: If vector dimension doesn't match index dimension

        Example:
            >>> index = create_hnsw_index(dimension=384, m=16)
            >>> vec = np.random.randn(384)
            >>> node_id = index.insert(vec, metadata={"text": "sample"})
            >>> print(f"Inserted as node {node_id}")

        Algorithm (simplified):
            - Assign random level l to new node
            - Search from top layer down to layer l+1 to find entry point
            - For each layer from l down to 0:
                - Search for ef_construction nearest neighbors
                - Select m best neighbors
                - Add bidirectional connections
            - Update entry point if new node has highest level
        """
        # Validate vector
        if vector.shape[0] != self.dimension:
            raise ValueError(
                f"Vector dimension ({vector.shape[0]}) doesn't match "
                f"index dimension ({self.dimension})"
            )

        # Assign node ID and level
        node_id = self.next_node_id
        self.next_node_id += 1
        level = self._random_level()

        # Create node
        if metadata is None:
            metadata = {}

        # Add vector to contiguous matrix storage (Fix 3)
        # This must be done before creating node to ensure node_id matches vector index
        assert node_id == self._num_vectors, "node_id must match vector index"
        self._add_vector(vector.copy())  # Copy to avoid external modifications

        node = HNSWNode(
            node_id=node_id,
            vector=None,  # Don't store reference - it becomes stale after matrix reallocation
            metadata=metadata,
            level=level,
        )

        # If this is the first node, make it the entry point
        if len(self) == 0:
            self.entry_point = node_id
            self.level_count = level + 1
            self.nodes[node_id] = node

            logger.debug(f"Inserted first node: id={node_id}, level={level}")
            return node_id

        # Add node to index NOW (before connecting to neighbors)
        # This is important because pruning logic needs to access the node
        self.nodes[node_id] = node

        # Otherwise, search for nearest neighbors and connect

        # Phase 1: Search from top layer down to layer level+1
        # This finds the entry points for insertion at layer 'level'
        current_nearest = [self.entry_point]

        # Search from top layer to layer level+1 with num_to_return=1
        # (we only need one entry point for next layer)
        for lc in range(self.level_count - 1, level, -1):
            current_nearest = self._search_layer(
                query=vector, entry_points=current_nearest, layer=lc, num_to_return=1
            )
            # Extract just the node IDs (use nid to avoid shadowing outer node_id)
            current_nearest = [nid for _, nid in current_nearest]

        # Phase 2: Insert at layers level down to 0
        for lc in range(level, -1, -1):
            # Search for nearest neighbors at this layer
            candidates = self._search_layer(
                query=vector,
                entry_points=current_nearest,
                layer=lc,
                num_to_return=self.ef_construction,
            )

            # Select m neighbors at this layer
            m = self.m0 if lc == 0 else self.m

            # Use diversity-aware heuristic if enabled (Fix 6)
            if self.use_heuristic:
                neighbors = self._select_neighbors_heuristic(
                    candidates, m, layer=lc, extend_candidates=False, keep_pruned_connections=True
                )
            else:
                neighbors = self._select_neighbors_simple(candidates, m)

            # Add bidirectional connections
            for neighbor_id in neighbors:
                # Add neighbor to new node
                node.add_neighbor(lc, neighbor_id)

                # Add new node to neighbor
                neighbor_node = self.nodes[neighbor_id]
                neighbor_node.add_neighbor(lc, node_id)

                # Prune neighbor's connections if exceeded max
                max_conn = self.m0 if lc == 0 else self.m
                if len(neighbor_node.neighbors[lc]) > max_conn:
                    # Get distances to all neighbors of this neighbor
                    neighbor_ids = list(neighbor_node.neighbors[lc])
                    # Use _get_vector to avoid stale references after matrix reallocation
                    neighbor_dists = self._get_batch_distances(
                        self._get_vector(neighbor_id), neighbor_ids
                    )
                    neighbor_candidates = list(zip(neighbor_dists, neighbor_ids))

                    # Use diversity-aware heuristic for pruning if enabled
                    if self.use_heuristic:
                        pruned_neighbors = self._select_neighbors_heuristic(
                            neighbor_candidates, max_conn, layer=lc,
                            extend_candidates=False, keep_pruned_connections=True
                        )
                    else:
                        pruned_neighbors = self._select_neighbors_simple(neighbor_candidates, max_conn)

                    # Convert list to set for internal storage
                    neighbor_node.neighbors[lc] = set(pruned_neighbors)

            # Update current_nearest for next layer
            current_nearest = neighbors

        # Update entry point if new node has higher level
        if level > self.level_count - 1:
            self.entry_point = node_id
            self.level_count = level + 1

        logger.debug(
            f"Inserted node: id={node_id}, level={level}, neighbors_L0={len(node.get_neighbors(0))}"
        )

        return node_id

    def search(
        self, query: np.ndarray, k: int = 5, ef_search: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for k nearest neighbors of query vector (Stage 8).

        Uses HNSW approximate nearest neighbor search:
        1. Start from entry point at top layer
        2. Greedily navigate down through layers
        3. At layer 0, search with ef_search candidates
        4. Return k nearest neighbors

        Args:
            query: Query vector (must match index dimension)
            k: Number of nearest neighbors to return
            ef_search: Size of dynamic candidate list during search
                      Higher = better recall, slower search
                      Default: uses self.ef_search (set during creation)

        Returns:
            List of result dictionaries, sorted by distance (closest first):
            [
                {
                    "node_id": int,
                    "score": float,  # Similarity score (higher = more similar)
                    "distance": float,  # Distance (lower = more similar)
                    "metadata": dict,
                    "vector": np.ndarray  # Optional, if available
                },
                ...
            ]

        Raises:
            ValueError: If query dimension doesn't match index dimension
            RuntimeError: If index is empty

        Example:
            >>> index = create_hnsw_index(dimension=384, ef_search=50)
            >>> # ... insert vectors ...
            >>> query = np.random.randn(384)
            >>> results = index.search(query, k=5)
            >>> for result in results:
            ...     print(f"Node {result['node_id']}: score={result['score']:.4f}")

        Algorithm:
            - Search from top layer to layer 1 with num_to_return=1
              (just need one entry point for next layer)
            - At layer 0, search with num_to_return=max(ef_search, k)
            - Return top k results
        """
        # Validate query
        if query.shape[0] != self.dimension:
            raise ValueError(
                f"Query dimension ({query.shape[0]}) doesn't match "
                f"index dimension ({self.dimension})"
            )

        # Check if index is empty
        if len(self) == 0:
            raise RuntimeError("Cannot search empty index")

        # Use default ef_search if not provided
        if ef_search is None:
            ef_search = getattr(self, "ef_search", 50)

        # Ensure ef_search is at least k
        ef_search = max(ef_search, k)

        # Phase 1: Greedy search from top layer down to layer 1 (Algorithm 5 from paper)
        # Paper uses ef=1 (greedy descent) at all upper layers for fast navigation
        current_nearest = [self.entry_point]

        for lc in range(self.level_count - 1, 0, -1):
            # Greedy search: find single nearest neighbor at each upper layer
            current_nearest = self._search_layer(
                query=query, entry_points=current_nearest, layer=lc, num_to_return=1
            )
            # Extract just the node IDs
            current_nearest = [nid for _, nid in current_nearest]

        # Phase 2: Search at layer 0 with ef_search candidates
        candidates = self._search_layer(
            query=query, entry_points=current_nearest, layer=0, num_to_return=ef_search
        )

        # Take top k results
        top_k = candidates[:k]

        # Convert to result format
        results = []
        for dist, node_id in top_k:
            node = self.nodes[node_id]

            # Convert distance back to similarity score
            if self.similarity_metric == "cosine":
                score = 1.0 - dist
            elif self.similarity_metric == "dot":
                score = -dist
            elif self.similarity_metric == "euclidean":
                score = -dist  # Lower distance = higher score
            else:
                score = -dist

            result = {
                "node_id": node_id,
                "score": score,
                "distance": dist,
                "metadata": node.metadata,
                # Use _get_vector to avoid stale references after matrix reallocation
                "vector": self._get_vector(node_id),
            }
            results.append(result)

        logger.debug(f"Search found {len(results)} results for k={k}, ef_search={ef_search}")

        return results

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"HNSWIndex(nodes={len(self)}, levels={self.level_count}, "
            f"m={self.m}, ef_construction={self.ef_construction}, "
            f"metric={self.similarity_metric})"
        )


def create_hnsw_index(
    dimension: int,
    m: int = 16,
    ef_construction: int = 200,
    ef_search: int = 50,
    similarity_metric: str = "cosine",
    normalized: bool = True,
    seed: Optional[int] = None,
    use_heuristic: bool = True,  # NEW: Enable diversity-aware selection by default
) -> HNSWIndex:
    """
    Factory function to create an HNSW index with standard parameters.

    This is a convenience wrapper around HNSWIndex() that uses sensible defaults.

    Args:
        dimension: Vector dimensionality
        m: Max bidirectional links per node (typical: 12-48)
        ef_construction: Dynamic list size during construction (typical: 100-500)
        ef_search: Dynamic list size during search (typical: 50-200)
                  Used in Stage 8 search algorithm
        similarity_metric: "cosine", "dot", or "euclidean"
        normalized: Whether vectors are L2 normalized
        seed: Random seed for reproducibility
        use_heuristic: If True, use diversity-aware neighbor selection (Algorithm 4)
                      Improves graph quality at slight construction cost

    Returns:
        Initialized HNSWIndex

    Example:
        >>> index = create_hnsw_index(dimension=384, m=16, ef_search=50)
        >>> print(len(index))
        0
    """
    index = HNSWIndex(
        dimension=dimension,
        m=m,
        ef_construction=ef_construction,
        similarity_metric=similarity_metric,
        normalized=normalized,
        seed=seed,
        use_heuristic=use_heuristic,  # NEW: Pass to HNSWIndex
    )

    # Store ef_search for future use in Stage 8
    # (We'll add it as an attribute here for completeness)
    index.ef_search = ef_search

    return index
