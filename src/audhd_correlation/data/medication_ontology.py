"""Medication ontology mappers (RxNorm, ATC)"""
from pathlib import Path
from typing import Optional
import requests
import time

from .ontology_base import (
    BaseOntologyMapper,
    OntologyMatch,
    OntologyType,
    MatchConfidence,
)


class RxNormMapper(BaseOntologyMapper):
    """Mapper for RxNorm medication codes"""

    def __init__(self, **kwargs):
        """Initialize RxNorm mapper"""
        super().__init__(ontology_type=OntologyType.RXNORM, **kwargs)
        self.api_base = "https://rxnav.nlm.nih.gov/REST"

    def _load_local_data(self) -> None:
        """Load local RxNorm terms"""
        # Common ADHD/Autism medications
        self._local_terms = {
            "methylphenidate": "6809",
            "ritalin": "562308",
            "concerta": "284635",
            "amphetamine": "644",
            "adderall": "197382",
            "dexedrine": "3423",
            "vyvanse": "816346",
            "lisdexamfetamine": "609132",
            "atomoxetine": "37025",
            "strattera": "352121",
            "guanfacine": "5144",
            "intuniv": "769161",
            "clonidine": "2599",
            "kapvay": "880502",
            "risperidone": "35636",
            "risperdal": "311700",
            "aripiprazole": "89013",
            "abilify": "403987",
            "fluoxetine": "4493",
            "prozac": "283403",
            "sertraline": "36437",
            "zoloft": "283741",
            "escitalopram": "321988",
            "lexapro": "352385",
            "melatonin": "6916",
            "clonazepam": "2598",
            "klonopin": "205959",
        }

        self._synonyms = {
            "methylphenidate": {"mph", "ritalin", "concerta", "metadate"},
            "amphetamine": {"adderall", "dexedrine", "vyvanse"},
            "atomoxetine": {"strattera"},
            "guanfacine": {"intuniv", "tenex"},
            "clonidine": {"kapvay", "catapres"},
            "risperidone": {"risperdal"},
            "aripiprazole": {"abilify"},
            "fluoxetine": {"prozac"},
            "sertraline": {"zoloft"},
            "escitalopram": {"lexapro"},
        }

    def _query_api(self, term: str) -> Optional[OntologyMatch]:
        """
        Query RxNorm API

        Args:
            term: Medication term to query

        Returns:
            OntologyMatch if found, None otherwise
        """
        if not self.use_api:
            return None

        try:
            # Use RxNav approximateTerm endpoint for fuzzy matching
            url = f"{self.api_base}/approximateTerm.json"
            params = {"term": term, "maxEntries": 5}

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if not data.get("approximateGroup", {}).get("candidate"):
                return None

            candidates = data["approximateGroup"]["candidate"]
            if not isinstance(candidates, list):
                candidates = [candidates]

            # Get best match
            best = candidates[0]
            rxcui = best.get("rxcui", "")
            name = best.get("name", "")
            rank = int(best.get("rank", 0))

            # Calculate confidence based on rank
            if rank == 1:
                confidence = MatchConfidence.EXACT
                similarity = 1.0
            elif rank <= 5:
                confidence = MatchConfidence.HIGH
                similarity = 0.9
            else:
                confidence = MatchConfidence.MEDIUM
                similarity = 0.75

            # Get alternatives
            alternatives = []
            if len(candidates) > 1:
                alternatives = [
                    (c["name"], c["rxcui"], 0.85 - 0.05 * i)
                    for i, c in enumerate(candidates[1:4])
                ]

            return OntologyMatch(
                source_term=term,
                matched_term=name,
                ontology_id=rxcui,
                ontology_type=self.ontology_type,
                confidence=confidence,
                similarity_score=similarity,
                alternative_matches=alternatives,
                metadata={"source": "rxnorm_api", "rank": rank},
            )

        except (requests.RequestException, KeyError) as e:
            print(f"RxNorm API error: {e}")
            return None

    def get_drug_class(self, rxcui: str) -> Optional[str]:
        """
        Get drug class for RxCUI

        Args:
            rxcui: RxNorm concept unique identifier

        Returns:
            Drug class name if found
        """
        if not self.use_api:
            return None

        try:
            url = f"{self.api_base}/rxclass/class/byRxcui.json"
            params = {"rxcui": rxcui, "relaSource": "ATC"}

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data.get("rxclassDrugInfoList", {}).get("rxclassDrugInfo"):
                classes = data["rxclassDrugInfoList"]["rxclassDrugInfo"]
                if isinstance(classes, list) and classes:
                    return classes[0].get("rxclassMinConceptItem", {}).get(
                        "className"
                    )

            return None

        except (requests.RequestException, KeyError):
            return None


class ATCMapper(BaseOntologyMapper):
    """Mapper for ATC (Anatomical Therapeutic Chemical) codes"""

    def __init__(self, **kwargs):
        """Initialize ATC mapper"""
        super().__init__(ontology_type=OntologyType.ATC, **kwargs)
        # ATC codes are often accessed through WHO or national formularies
        self.use_api = False  # No free public API available

    def _load_local_data(self) -> None:
        """Load local ATC codes"""
        # Common ATC codes for ADHD/Autism medications
        self._local_terms = {
            "methylphenidate": "N06BA04",
            "amphetamine": "N06BA01",
            "dexamphetamine": "N06BA02",
            "lisdexamfetamine": "N06BA12",
            "atomoxetine": "N06BA09",
            "guanfacine": "C02AC02",
            "clonidine": "C02AC01",
            "risperidone": "N05AX08",
            "aripiprazole": "N05AX12",
            "haloperidol": "N05AD01",
            "fluoxetine": "N06AB03",
            "sertraline": "N06AB06",
            "escitalopram": "N06AB10",
            "citalopram": "N06AB04",
            "melatonin": "N05CH01",
            "clonazepam": "N03AE01",
            "diazepam": "N05BA01",
        }

        # ATC classification hierarchy
        self._atc_levels = {
            "N06BA": "Centrally acting sympathomimetics (ADHD medications)",
            "N06BA04": "Methylphenidate",
            "N06BA09": "Atomoxetine",
            "N05AX": "Other antipsychotics",
            "N06AB": "Selective serotonin reuptake inhibitors",
            "N05CH": "Melatonin receptor agonists",
            "C02AC": "Imidazoline receptor agonists",
        }

        self._synonyms = {
            "methylphenidate": {"ritalin", "concerta", "mph"},
            "amphetamine": {"adderall", "dextroamphetamine"},
            "lisdexamfetamine": {"vyvanse"},
            "atomoxetine": {"strattera"},
            "guanfacine": {"intuniv", "tenex"},
            "clonidine": {"kapvay"},
            "risperidone": {"risperdal"},
            "aripiprazole": {"abilify"},
        }

    def _query_api(self, term: str) -> Optional[OntologyMatch]:
        """Query ATC API (not available publicly)"""
        return None

    def get_atc_hierarchy(self, atc_code: str) -> dict:
        """
        Get ATC code hierarchy

        Args:
            atc_code: ATC code

        Returns:
            Dictionary with hierarchy levels
        """
        if not atc_code or len(atc_code) < 3:
            return {}

        hierarchy = {
            "level1": atc_code[0],  # Anatomical main group
            "level2": atc_code[:3],  # Therapeutic subgroup
            "level3": atc_code[:4] if len(atc_code) >= 4 else None,  # Pharmacological subgroup
            "level4": atc_code[:5] if len(atc_code) >= 5 else None,  # Chemical subgroup
            "level5": atc_code if len(atc_code) == 7 else None,  # Chemical substance
        }

        # Add descriptions if available
        for level, code in hierarchy.items():
            if code and code in self._atc_levels:
                hierarchy[f"{level}_name"] = self._atc_levels[code]

        return hierarchy


class MedicationOntologyMapper:
    """Unified mapper for medications across RxNorm and ATC"""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        use_rxnorm: bool = True,
        use_atc: bool = True,
        use_api: bool = True,
    ):
        """
        Initialize medication ontology mapper

        Args:
            cache_dir: Cache directory
            use_rxnorm: Enable RxNorm mapping
            use_atc: Enable ATC mapping
            use_api: Enable API calls
        """
        self.mappers = {}

        if use_rxnorm:
            self.mappers["rxnorm"] = RxNormMapper(cache_dir=cache_dir, use_api=use_api)

        if use_atc:
            self.mappers["atc"] = ATCMapper(cache_dir=cache_dir, use_api=False)

    def map_medication(self, medication: str) -> dict:
        """
        Map medication to all ontologies

        Args:
            medication: Medication name

        Returns:
            Dictionary with RxNorm and ATC mappings
        """
        results = {}

        for name, mapper in self.mappers.items():
            match = mapper.map_term(medication)
            results[name] = match

            # Get additional info for RxNorm
            if name == "rxnorm" and match.ontology_id:
                drug_class = mapper.get_drug_class(match.ontology_id)
                if drug_class:
                    match.metadata["drug_class"] = drug_class

            # Get ATC hierarchy
            if name == "atc" and match.ontology_id:
                hierarchy = mapper.get_atc_hierarchy(match.ontology_id)
                match.metadata["atc_hierarchy"] = hierarchy

            # Rate limiting
            if mapper.use_api:
                time.sleep(0.1)

        return results

    def map_medications_batch(self, medications: list) -> dict:
        """
        Map multiple medications

        Args:
            medications: List of medication names

        Returns:
            Dictionary mapping medication to ontology mappings
        """
        results = {}
        for med in medications:
            results[med] = self.map_medication(med)

        return results