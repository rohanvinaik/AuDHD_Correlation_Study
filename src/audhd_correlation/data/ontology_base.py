"""Base classes for ontology mapping with caching"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import json
import hashlib
import requests
from functools import lru_cache

import pandas as pd
from rapidfuzz import fuzz, process


class OntologyType(Enum):
    """Types of ontologies supported"""

    HPO = "human_phenotype_ontology"
    SNOMED = "snomed_ct"
    ICD10 = "icd10"
    RXNORM = "rxnorm"
    ATC = "anatomical_therapeutic_chemical"
    FNDDS = "food_nutrient_database"
    HMDB = "human_metabolome_database"
    NCBI = "ncbi_taxonomy"


class MatchConfidence(Enum):
    """Confidence levels for ontology matches"""

    EXACT = "exact"  # 100% match
    HIGH = "high"  # >90% similarity
    MEDIUM = "medium"  # 70-90% similarity
    LOW = "low"  # 50-70% similarity
    AMBIGUOUS = "ambiguous"  # Multiple similar matches
    UNMATCHED = "unmatched"  # No good match found


@dataclass
class OntologyMatch:
    """Result of an ontology mapping"""

    source_term: str
    matched_term: str
    ontology_id: str
    ontology_type: OntologyType
    confidence: MatchConfidence
    similarity_score: float
    alternative_matches: List[Tuple[str, str, float]] = field(
        default_factory=list
    )  # (term, id, score)
    requires_review: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    matched_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "source_term": self.source_term,
            "matched_term": self.matched_term,
            "ontology_id": self.ontology_id,
            "ontology_type": self.ontology_type.value,
            "confidence": self.confidence.value,
            "similarity_score": self.similarity_score,
            "alternative_matches": self.alternative_matches,
            "requires_review": self.requires_review,
            "metadata": self.metadata,
            "matched_at": self.matched_at.isoformat(),
        }


class OntologyCache:
    """Cache for ontology API calls and mappings"""

    def __init__(self, cache_dir: Path, ttl_days: int = 30):
        """
        Initialize cache

        Args:
            cache_dir: Directory for cache files
            ttl_days: Time-to-live for cache entries in days
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(days=ttl_days)
        self._memory_cache: Dict[str, OntologyMatch] = {}

    def _get_cache_key(self, term: str, ontology: OntologyType) -> str:
        """Generate cache key from term and ontology"""
        key_string = f"{ontology.value}:{term.lower()}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cache_file(self, cache_key: str) -> Path:
        """Get cache file path for key"""
        return self.cache_dir / f"{cache_key}.json"

    def get(self, term: str, ontology: OntologyType) -> Optional[OntologyMatch]:
        """
        Get cached match

        Args:
            term: Source term
            ontology: Ontology type

        Returns:
            OntologyMatch if cached and valid, None otherwise
        """
        cache_key = self._get_cache_key(term, ontology)

        # Check memory cache first
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        # Check file cache
        cache_file = self._get_cache_file(cache_key)
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)

                # Check TTL
                matched_at = datetime.fromisoformat(data["matched_at"])
                if datetime.now() - matched_at < self.ttl:
                    # Reconstruct OntologyMatch
                    match = OntologyMatch(
                        source_term=data["source_term"],
                        matched_term=data["matched_term"],
                        ontology_id=data["ontology_id"],
                        ontology_type=OntologyType(data["ontology_type"]),
                        confidence=MatchConfidence(data["confidence"]),
                        similarity_score=data["similarity_score"],
                        alternative_matches=data.get("alternative_matches", []),
                        requires_review=data.get("requires_review", False),
                        metadata=data.get("metadata", {}),
                        matched_at=matched_at,
                    )

                    # Update memory cache
                    self._memory_cache[cache_key] = match
                    return match
                else:
                    # Expired, remove from cache
                    cache_file.unlink()
            except (json.JSONDecodeError, KeyError):
                # Corrupted cache file, remove it
                cache_file.unlink()

        return None

    def set(self, match: OntologyMatch) -> None:
        """
        Cache an ontology match

        Args:
            match: OntologyMatch to cache
        """
        cache_key = self._get_cache_key(match.source_term, match.ontology_type)

        # Update memory cache
        self._memory_cache[cache_key] = match

        # Update file cache
        cache_file = self._get_cache_file(cache_key)
        with open(cache_file, "w") as f:
            json.dump(match.to_dict(), f, indent=2)

    def clear_expired(self) -> int:
        """
        Clear expired cache entries

        Returns:
            Number of entries cleared
        """
        cleared = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                matched_at = datetime.fromisoformat(data["matched_at"])
                if datetime.now() - matched_at >= self.ttl:
                    cache_file.unlink()
                    cleared += 1
            except (json.JSONDecodeError, KeyError):
                cache_file.unlink()
                cleared += 1

        return cleared


class BaseOntologyMapper(ABC):
    """Abstract base class for ontology mappers"""

    def __init__(
        self,
        ontology_type: OntologyType,
        cache_dir: Optional[Path] = None,
        similarity_threshold: float = 0.7,
        use_api: bool = True,
    ):
        """
        Initialize ontology mapper

        Args:
            ontology_type: Type of ontology
            cache_dir: Directory for caching (default: .cache/ontology)
            similarity_threshold: Minimum similarity score for matches
            use_api: Whether to use API calls (vs local data only)
        """
        self.ontology_type = ontology_type
        self.similarity_threshold = similarity_threshold
        self.use_api = use_api

        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "audhd_ontology"
        self.cache = OntologyCache(cache_dir)

        # Load local ontology data
        self._local_terms: Dict[str, str] = {}  # term -> ontology_id
        self._synonyms: Dict[str, Set[str]] = {}  # term -> set of synonyms
        self._load_local_data()

    @abstractmethod
    def _load_local_data(self) -> None:
        """Load local ontology data (terms, synonyms, mappings)"""
        pass

    @abstractmethod
    def _query_api(self, term: str) -> Optional[OntologyMatch]:
        """
        Query ontology API for term

        Args:
            term: Term to query

        Returns:
            OntologyMatch if found, None otherwise
        """
        pass

    def map_term(self, term: str, force_refresh: bool = False) -> OntologyMatch:
        """
        Map a term to ontology

        Args:
            term: Term to map
            force_refresh: Skip cache and force new lookup

        Returns:
            OntologyMatch with mapping result
        """
        # Check cache first
        if not force_refresh:
            cached = self.cache.get(term, self.ontology_type)
            if cached is not None:
                return cached

        # Try exact match in local data
        normalized_term = self._normalize_term(term)
        if normalized_term in self._local_terms:
            match = OntologyMatch(
                source_term=term,
                matched_term=normalized_term,
                ontology_id=self._local_terms[normalized_term],
                ontology_type=self.ontology_type,
                confidence=MatchConfidence.EXACT,
                similarity_score=1.0,
            )
            self.cache.set(match)
            return match

        # Try synonym matching
        synonym_match = self._match_synonym(term)
        if synonym_match:
            self.cache.set(synonym_match)
            return synonym_match

        # Try fuzzy matching
        fuzzy_match = self._fuzzy_match(term)
        if fuzzy_match:
            self.cache.set(fuzzy_match)
            return fuzzy_match

        # Try API if enabled
        if self.use_api:
            api_match = self._query_api(term)
            if api_match:
                self.cache.set(api_match)
                return api_match

        # No match found
        match = OntologyMatch(
            source_term=term,
            matched_term="",
            ontology_id="",
            ontology_type=self.ontology_type,
            confidence=MatchConfidence.UNMATCHED,
            similarity_score=0.0,
            requires_review=True,
        )
        self.cache.set(match)
        return match

    def map_terms_batch(self, terms: List[str]) -> List[OntologyMatch]:
        """
        Map multiple terms in batch

        Args:
            terms: List of terms to map

        Returns:
            List of OntologyMatch results
        """
        return [self.map_term(term) for term in terms]

    def _normalize_term(self, term: str) -> str:
        """
        Normalize term for matching

        Args:
            term: Term to normalize

        Returns:
            Normalized term
        """
        # Convert to lowercase
        normalized = term.lower().strip()

        # Remove extra whitespace
        normalized = " ".join(normalized.split())

        # Remove common suffixes/prefixes
        for remove in ["disorder", "syndrome", "disease", "condition"]:
            normalized = normalized.replace(f" {remove}", "")
            normalized = normalized.replace(f"{remove} ", "")

        return normalized

    def _match_synonym(self, term: str) -> Optional[OntologyMatch]:
        """
        Match term using synonym dictionary

        Args:
            term: Term to match

        Returns:
            OntologyMatch if synonym found, None otherwise
        """
        normalized = self._normalize_term(term)

        for main_term, synonyms in self._synonyms.items():
            if normalized in synonyms:
                if main_term in self._local_terms:
                    return OntologyMatch(
                        source_term=term,
                        matched_term=main_term,
                        ontology_id=self._local_terms[main_term],
                        ontology_type=self.ontology_type,
                        confidence=MatchConfidence.HIGH,
                        similarity_score=0.95,
                        metadata={"match_type": "synonym"},
                    )

        return None

    def _fuzzy_match(self, term: str) -> Optional[OntologyMatch]:
        """
        Match term using fuzzy string matching

        Args:
            term: Term to match

        Returns:
            OntologyMatch if good match found, None otherwise
        """
        if not self._local_terms:
            return None

        normalized = self._normalize_term(term)

        # Use rapidfuzz for fuzzy matching
        matches = process.extract(
            normalized,
            self._local_terms.keys(),
            scorer=fuzz.token_sort_ratio,
            limit=5,
        )

        if not matches:
            return None

        best_match, best_score, _ = matches[0]
        similarity = best_score / 100.0

        if similarity < self.similarity_threshold:
            return None

        # Determine confidence level
        if similarity >= 0.95:
            confidence = MatchConfidence.HIGH
        elif similarity >= 0.8:
            confidence = MatchConfidence.MEDIUM
        else:
            confidence = MatchConfidence.LOW

        # Check for ambiguous matches
        requires_review = False
        alternative_matches = []
        if len(matches) > 1:
            second_score = matches[1][1] / 100.0
            if abs(similarity - second_score) < 0.05:  # Close scores
                requires_review = True
                confidence = MatchConfidence.AMBIGUOUS
                alternative_matches = [
                    (m[0], self._local_terms[m[0]], m[1] / 100.0)
                    for m in matches[1:4]
                ]

        return OntologyMatch(
            source_term=term,
            matched_term=best_match,
            ontology_id=self._local_terms[best_match],
            ontology_type=self.ontology_type,
            confidence=confidence,
            similarity_score=similarity,
            alternative_matches=alternative_matches,
            requires_review=requires_review,
            metadata={"match_type": "fuzzy"},
        )

    def export_mappings(
        self, matches: List[OntologyMatch], output_path: Path, format: str = "csv"
    ) -> None:
        """
        Export mappings to file

        Args:
            matches: List of OntologyMatch results
            output_path: Output file path
            format: Output format (csv, json, tsv)
        """
        if format == "json":
            with open(output_path, "w") as f:
                json.dump([m.to_dict() for m in matches], f, indent=2)
        else:
            # Convert to DataFrame
            df = pd.DataFrame([m.to_dict() for m in matches])

            if format == "csv":
                df.to_csv(output_path, index=False)
            elif format == "tsv":
                df.to_csv(output_path, sep="\t", index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")

    def get_review_queue(self, matches: List[OntologyMatch]) -> List[OntologyMatch]:
        """
        Get matches that require manual review

        Args:
            matches: List of OntologyMatch results

        Returns:
            List of matches requiring review
        """
        return [m for m in matches if m.requires_review]

    def get_statistics(self, matches: List[OntologyMatch]) -> Dict[str, Any]:
        """
        Get mapping statistics

        Args:
            matches: List of OntologyMatch results

        Returns:
            Dictionary with statistics
        """
        total = len(matches)
        if total == 0:
            return {}

        confidence_counts = {}
        for conf in MatchConfidence:
            confidence_counts[conf.value] = sum(
                1 for m in matches if m.confidence == conf
            )

        review_count = sum(1 for m in matches if m.requires_review)
        avg_score = sum(m.similarity_score for m in matches) / total

        return {
            "total_terms": total,
            "confidence_breakdown": confidence_counts,
            "requires_review": review_count,
            "average_similarity": avg_score,
            "unmatched_rate": confidence_counts.get(MatchConfidence.UNMATCHED.value, 0)
            / total,
        }