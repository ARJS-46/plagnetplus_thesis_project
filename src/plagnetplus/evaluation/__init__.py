"""Shared evaluation tools for PlagNetPlus representations."""

from plagnetplus.evaluation.clustering import (
    evaluate_kmeans_run,
    normalised_cluster_entropy,
    pairwise_clustering_stability,
)
from plagnetplus.evaluation.dimensionality import (
    PCAResult,
    components_for_variance_threshold,
    fit_deterministic_pca,
    l2_normalise_rows,
    summarise_pca,
)
from plagnetplus.evaluation.embedding_io import (
    EmbeddingBundle,
    load_embedding_bundle,
    load_embedding_manifest,
    sha256_file,
    validate_embedding_manifest_alignment,
)
from plagnetplus.evaluation.embedding_qc import (
    dimension_statistics,
    embedding_qc_summary,
    entropy_effective_rank,
    sampled_mean_pairwise_cosine_similarity,
)
from plagnetplus.evaluation.retrieval import (
    categorical_enrichment_at_k,
    categorical_precision_at_k,
    exact_cosine_neighbours,
    prepare_cosine_features,
)
from plagnetplus.evaluation.probes import (
    LocalityProbeResult,
    fit_locality_probe,
)
from plagnetplus.evaluation.training_history import (
    load_training_history,
    summarise_training_history,
)

__all__ = [
    "EmbeddingBundle",
    "LocalityProbeResult",
    "PCAResult",
    "categorical_enrichment_at_k",
    "categorical_precision_at_k",
    "components_for_variance_threshold",
    "dimension_statistics",
    "embedding_qc_summary",
    "entropy_effective_rank",
    "evaluate_kmeans_run",
    "exact_cosine_neighbours",
    "fit_deterministic_pca",
    "fit_locality_probe",
    "l2_normalise_rows",
    "load_embedding_bundle",
    "load_embedding_manifest",
    "load_training_history",
    "normalised_cluster_entropy",
    "pairwise_clustering_stability",
    "prepare_cosine_features",
    "sampled_mean_pairwise_cosine_similarity",
    "sha256_file",
    "summarise_pca",
    "summarise_training_history",
    "validate_embedding_manifest_alignment",
]
