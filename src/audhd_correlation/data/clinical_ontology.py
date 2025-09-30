"""Clinical ontology mappers (HPO, SNOMED, ICD-10)"""
from pathlib import Path
from typing import Dict, Optional, Set
import requests
import time

from .ontology_base import (
    BaseOntologyMapper,
    OntologyMatch,
    OntologyType,
    MatchConfidence,
)


class HPOMapper(BaseOntologyMapper):
    """Mapper for Human Phenotype Ontology"""

    def __init__(self, **kwargs):
        """Initialize HPO mapper"""
        super().__init__(ontology_type=OntologyType.HPO, **kwargs)
        self.api_base = "https://hpo.jax.org/api/hpo"

    def _load_local_data(self) -> None:
        """Load local HPO terms"""
        # Common HPO terms for ADHD/Autism
        self._local_terms = {
            "attention deficit": "HP:0007018",
            "hyperactivity": "HP:0000752",
            "impulsivity": "HP:0100710",
            "inattention": "HP:0007018",
            "autism": "HP:0000729",
            "autistic behavior": "HP:0000729",
            "social communication deficit": "HP:0000011",
            "repetitive behavior": "HP:0000733",
            "restricted interests": "HP:0000723",
            "sensory sensitivity": "HP:0000993",
            "anxiety": "HP:0000739",
            "depression": "HP:0000716",
            "sleep disturbance": "HP:0002360",
            "intellectual disability": "HP:0001249",
            "developmental delay": "HP:0001263",
            "language delay": "HP:0000750",
            "motor delay": "HP:0001270",
            "seizures": "HP:0001250",
            "aggressive behavior": "HP:0000718",
            "self-injurious behavior": "HP:0100716",
            "feeding difficulties": "HP:0011968",
            "gastrointestinal symptoms": "HP:0011024",
        }

        # Add synonyms
        self._synonyms = {
            "attention deficit": {"adhd", "add", "attention problems"},
            "hyperactivity": {
                "hyperactive",
                "hyperkinetic",
                "excessive activity",
            },
            "autism": {"asd", "autism spectrum", "pervasive developmental"},
            "anxiety": {"anxious", "worry", "nervousness"},
            "depression": {"depressed mood", "sadness", "low mood"},
            "intellectual disability": {
                "mental retardation",
                "cognitive impairment",
                "id",
            },
        }

    def _query_api(self, term: str) -> Optional[OntologyMatch]:
        """
        Query HPO API

        Args:
            term: Term to query

        Returns:
            OntologyMatch if found, None otherwise
        """
        if not self.use_api:
            return None

        try:
            # Search HPO API
            url = f"{self.api_base}/search"
            params = {"q": term, "max": 5}

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if not data.get("terms"):
                return None

            # Get best match
            best = data["terms"][0]
            hpo_id = best.get("id", "")
            name = best.get("name", "")

            # Calculate confidence based on match quality
            confidence = MatchConfidence.HIGH
            similarity = 0.9  # API matches are generally good

            # Check for multiple similar results
            alternatives = []
            if len(data["terms"]) > 1:
                alternatives = [
                    (t["name"], t["id"], 0.85) for t in data["terms"][1:4]
                ]

            return OntologyMatch(
                source_term=term,
                matched_term=name,
                ontology_id=hpo_id,
                ontology_type=self.ontology_type,
                confidence=confidence,
                similarity_score=similarity,
                alternative_matches=alternatives,
                metadata={"source": "hpo_api"},
            )

        except (requests.RequestException, KeyError) as e:
            print(f"HPO API error: {e}")
            return None


class SNOMEDMapper(BaseOntologyMapper):
    """Mapper for SNOMED CT"""

    def __init__(self, **kwargs):
        """Initialize SNOMED mapper"""
        super().__init__(ontology_type=OntologyType.SNOMED, **kwargs)
        # Note: SNOMED API requires license/authentication
        self.use_api = False  # Disable API by default

    def _load_local_data(self) -> None:
        """Load local SNOMED terms"""
        # Common SNOMED CT codes for ADHD/Autism
        self._local_terms = {
            "attention deficit hyperactivity disorder": "406506008",
            "autism spectrum disorder": "408856003",
            "adhd predominantly inattentive": "192127007",
            "adhd predominantly hyperactive": "192128002",
            "adhd combined type": "406505007",
            "asperger syndrome": "43614003",
            "anxiety disorder": "197480006",
            "major depressive disorder": "370143000",
            "obsessive compulsive disorder": "191736004",
            "oppositional defiant disorder": "25501002",
            "conduct disorder": "23907005",
            "intellectual disability": "110359009",
            "learning disability": "228156009",
            "dyslexia": "81409003",
            "motor coordination disorder": "29164008",
            "sleep disorder": "39898005",
            "epilepsy": "84757009",
        }

        self._synonyms = {
            "attention deficit hyperactivity disorder": {"adhd", "add"},
            "autism spectrum disorder": {"asd", "autism", "autistic disorder"},
            "anxiety disorder": {"anxiety", "generalized anxiety"},
            "major depressive disorder": {"depression", "mdd"},
            "intellectual disability": {"mental retardation", "cognitive disability"},
        }

    def _query_api(self, term: str) -> Optional[OntologyMatch]:
        """Query SNOMED API (requires license)"""
        # SNOMED requires authentication and license
        # Placeholder for when API access is available
        return None


class ICD10Mapper(BaseOntologyMapper):
    """Mapper for ICD-10 codes"""

    def __init__(self, **kwargs):
        """Initialize ICD-10 mapper"""
        super().__init__(ontology_type=OntologyType.ICD10, **kwargs)
        self.api_base = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"

    def _load_local_data(self) -> None:
        """Load local ICD-10 codes"""
        # Common ICD-10 codes for ADHD/Autism
        self._local_terms = {
            "attention deficit hyperactivity disorder": "F90",
            "adhd predominantly inattentive": "F90.0",
            "adhd predominantly hyperactive": "F90.1",
            "adhd combined type": "F90.2",
            "autism spectrum disorder": "F84.0",
            "asperger syndrome": "F84.5",
            "pervasive developmental disorder": "F84.9",
            "anxiety disorder": "F41.9",
            "generalized anxiety disorder": "F41.1",
            "major depressive disorder": "F32.9",
            "obsessive compulsive disorder": "F42",
            "oppositional defiant disorder": "F91.3",
            "conduct disorder": "F91.9",
            "intellectual disability": "F70-F79",
            "mild intellectual disability": "F70",
            "moderate intellectual disability": "F71",
            "specific learning disorder": "F81",
            "dyslexia": "F81.0",
            "motor coordination disorder": "F82",
            "sleep disorder": "G47",
            "epilepsy": "G40",
        }

        self._synonyms = {
            "attention deficit hyperactivity disorder": {
                "adhd",
                "add",
                "hyperkinetic disorder",
            },
            "autism spectrum disorder": {"asd", "autism", "autistic disorder"},
            "anxiety disorder": {"anxiety", "anxious"},
            "major depressive disorder": {"depression", "depressive episode"},
            "intellectual disability": {"mental retardation", "id"},
        }

    def _query_api(self, term: str) -> Optional[OntologyMatch]:
        """
        Query ICD-10 API

        Args:
            term: Term to query

        Returns:
            OntologyMatch if found, None otherwise
        """
        if not self.use_api:
            return None

        try:
            # Query NLM Clinical Tables API
            params = {"sf": "code,name", "terms": term, "maxList": 5}

            response = requests.get(self.api_base, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if not data or len(data) < 4 or not data[3]:
                return None

            # Parse results
            results = data[3]
            if not results:
                return None

            # Best match
            best = results[0]
            icd_code = best[0]
            icd_name = best[1]

            # Calculate similarity (API returns ranked results)
            similarity = 0.9 if len(results) == 1 else 0.85
            confidence = MatchConfidence.HIGH

            # Alternative matches
            alternatives = []
            if len(results) > 1:
                alternatives = [(r[1], r[0], 0.8) for r in results[1:4]]

            return OntologyMatch(
                source_term=term,
                matched_term=icd_name,
                ontology_id=icd_code,
                ontology_type=self.ontology_type,
                confidence=confidence,
                similarity_score=similarity,
                alternative_matches=alternatives,
                metadata={"source": "nlm_api"},
            )

        except (requests.RequestException, KeyError, IndexError) as e:
            print(f"ICD-10 API error: {e}")
            return None


class ClinicalOntologyMapper:
    """Unified mapper for clinical terms across multiple ontologies"""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        use_hpo: bool = True,
        use_snomed: bool = True,
        use_icd10: bool = True,
        use_api: bool = True,
    ):
        """
        Initialize clinical ontology mapper

        Args:
            cache_dir: Cache directory
            use_hpo: Enable HPO mapping
            use_snomed: Enable SNOMED mapping
            use_icd10: Enable ICD-10 mapping
            use_api: Enable API calls
        """
        self.mappers = {}

        if use_hpo:
            self.mappers["hpo"] = HPOMapper(cache_dir=cache_dir, use_api=use_api)

        if use_snomed:
            self.mappers["snomed"] = SNOMEDMapper(cache_dir=cache_dir, use_api=use_api)

        if use_icd10:
            self.mappers["icd10"] = ICD10Mapper(cache_dir=cache_dir, use_api=use_api)

    def map_term(self, term: str) -> Dict[str, OntologyMatch]:
        """
        Map term to all enabled ontologies

        Args:
            term: Term to map

        Returns:
            Dictionary mapping ontology name to OntologyMatch
        """
        results = {}
        for name, mapper in self.mappers.items():
            results[name] = mapper.map_term(term)
            # Rate limiting for API calls
            if mapper.use_api:
                time.sleep(0.1)

        return results

    def map_terms_batch(self, terms: list) -> Dict[str, Dict[str, OntologyMatch]]:
        """
        Map multiple terms to all ontologies

        Args:
            terms: List of terms to map

        Returns:
            Dictionary mapping term to ontology mappings
        """
        results = {}
        for term in terms:
            results[term] = self.map_term(term)

        return results

    def export_unified_catalog(
        self, mappings: Dict[str, Dict[str, OntologyMatch]], output_path: Path
    ) -> None:
        """
        Export unified variable catalog

        Args:
            mappings: Term mappings across ontologies
            output_path: Output file path
        """
        rows = []
        for term, ontology_mappings in mappings.items():
            row = {"source_term": term}

            for onto_name, match in ontology_mappings.items():
                row[f"{onto_name}_id"] = match.ontology_id
                row[f"{onto_name}_term"] = match.matched_term
                row[f"{onto_name}_confidence"] = match.confidence.value
                row[f"{onto_name}_score"] = match.similarity_score

            # Flag if any mapping requires review
            row["requires_review"] = any(m.requires_review for m in ontology_mappings.values())

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)