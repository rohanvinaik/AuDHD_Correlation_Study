"""Tests for ontology mapping system"""
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.audhd_correlation.data.ontology_base import (
    OntologyCache,
    OntologyMatch,
    OntologyType,
    MatchConfidence,
)
from src.audhd_correlation.data.clinical_ontology import (
    HPOMapper,
    SNOMEDMapper,
    ICD10Mapper,
    ClinicalOntologyMapper,
)
from src.audhd_correlation.data.medication_ontology import (
    RxNormMapper,
    ATCMapper,
    MedicationOntologyMapper,
)
from src.audhd_correlation.data.diet_ontology import FNDDSMapper
from src.audhd_correlation.data.variable_catalog import (
    VariableCatalog,
    VariableDefinition,
    VariableType,
    DataType,
)


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory"""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_ontology_cache_set_get(temp_cache_dir):
    """Test caching of ontology matches"""
    cache = OntologyCache(temp_cache_dir)

    match = OntologyMatch(
        source_term="ADHD",
        matched_term="Attention deficit hyperactivity disorder",
        ontology_id="HP:0007018",
        ontology_type=OntologyType.HPO,
        confidence=MatchConfidence.EXACT,
        similarity_score=1.0,
    )

    # Set and get from cache
    cache.set(match)
    retrieved = cache.get("ADHD", OntologyType.HPO)

    assert retrieved is not None
    assert retrieved.source_term == "ADHD"
    assert retrieved.ontology_id == "HP:0007018"
    assert retrieved.confidence == MatchConfidence.EXACT


def test_hpo_mapper_exact_match(temp_cache_dir):
    """Test HPO mapper with exact match"""
    mapper = HPOMapper(cache_dir=temp_cache_dir, use_api=False)

    match = mapper.map_term("attention deficit")

    assert match.confidence == MatchConfidence.EXACT
    assert match.ontology_id == "HP:0007018"
    assert match.ontology_type == OntologyType.HPO


def test_hpo_mapper_synonym_match(temp_cache_dir):
    """Test HPO mapper with synonym"""
    mapper = HPOMapper(cache_dir=temp_cache_dir, use_api=False)

    match = mapper.map_term("ADHD")

    assert match.confidence == MatchConfidence.HIGH
    assert "HP:" in match.ontology_id
    assert match.metadata.get("match_type") == "synonym"


def test_hpo_mapper_fuzzy_match(temp_cache_dir):
    """Test HPO mapper with fuzzy matching"""
    mapper = HPOMapper(cache_dir=temp_cache_dir, use_api=False)

    match = mapper.map_term("attention problem")

    # Should match "attention deficit" with fuzzy matching
    assert match.confidence in [
        MatchConfidence.HIGH,
        MatchConfidence.MEDIUM,
        MatchConfidence.LOW,
    ]
    assert match.similarity_score >= 0.7


def test_hpo_mapper_no_match(temp_cache_dir):
    """Test HPO mapper with no match"""
    mapper = HPOMapper(cache_dir=temp_cache_dir, use_api=False)

    match = mapper.map_term("completely_made_up_term_12345")

    assert match.confidence == MatchConfidence.UNMATCHED
    assert match.requires_review is True


def test_snomed_mapper(temp_cache_dir):
    """Test SNOMED mapper"""
    mapper = SNOMEDMapper(cache_dir=temp_cache_dir, use_api=False)

    match = mapper.map_term("attention deficit hyperactivity disorder")

    assert match.ontology_type == OntologyType.SNOMED
    assert match.ontology_id == "406506008"


def test_icd10_mapper(temp_cache_dir):
    """Test ICD-10 mapper"""
    mapper = ICD10Mapper(cache_dir=temp_cache_dir, use_api=False)

    match = mapper.map_term("ADHD")

    assert match.ontology_type == OntologyType.ICD10
    assert "F90" in match.ontology_id


def test_clinical_ontology_mapper(temp_cache_dir):
    """Test unified clinical ontology mapper"""
    mapper = ClinicalOntologyMapper(cache_dir=temp_cache_dir, use_api=False)

    mappings = mapper.map_term("ADHD")

    assert "hpo" in mappings
    assert "snomed" in mappings
    assert "icd10" in mappings

    # Each should have a valid match
    for onto_name, match in mappings.items():
        assert isinstance(match, OntologyMatch)
        assert match.ontology_type in [
            OntologyType.HPO,
            OntologyType.SNOMED,
            OntologyType.ICD10,
        ]


def test_rxnorm_mapper(temp_cache_dir):
    """Test RxNorm medication mapper"""
    mapper = RxNormMapper(cache_dir=temp_cache_dir, use_api=False)

    match = mapper.map_term("methylphenidate")

    assert match.ontology_type == OntologyType.RXNORM
    assert match.ontology_id == "6809"
    assert match.confidence == MatchConfidence.EXACT


def test_rxnorm_mapper_brand_name(temp_cache_dir):
    """Test RxNorm mapper with brand name"""
    mapper = RxNormMapper(cache_dir=temp_cache_dir, use_api=False)

    match = mapper.map_term("Ritalin")

    # Should match via synonyms
    assert match.confidence in [MatchConfidence.EXACT, MatchConfidence.HIGH]
    assert match.ontology_type == OntologyType.RXNORM


def test_atc_mapper(temp_cache_dir):
    """Test ATC medication mapper"""
    mapper = ATCMapper(cache_dir=temp_cache_dir, use_api=False)

    match = mapper.map_term("methylphenidate")

    assert match.ontology_type == OntologyType.ATC
    assert match.ontology_id == "N06BA04"


def test_atc_hierarchy(temp_cache_dir):
    """Test ATC code hierarchy extraction"""
    mapper = ATCMapper(cache_dir=temp_cache_dir, use_api=False)

    hierarchy = mapper.get_atc_hierarchy("N06BA04")

    assert hierarchy["level1"] == "N"
    assert hierarchy["level2"] == "N06"
    assert hierarchy["level3"] == "N06B"
    assert hierarchy["level4"] == "N06BA"
    assert hierarchy["level5"] == "N06BA04"


def test_medication_ontology_mapper(temp_cache_dir):
    """Test unified medication ontology mapper"""
    mapper = MedicationOntologyMapper(cache_dir=temp_cache_dir, use_api=False)

    mappings = mapper.map_medication("methylphenidate")

    assert "rxnorm" in mappings
    assert "atc" in mappings

    assert mappings["rxnorm"].ontology_id == "6809"
    assert mappings["atc"].ontology_id == "N06BA04"


def test_fndds_mapper(temp_cache_dir):
    """Test FNDDS food mapper"""
    mapper = FNDDSMapper(cache_dir=temp_cache_dir, use_api=False)

    match = mapper.map_term("apple")

    assert match.ontology_type == OntologyType.FNDDS
    assert match.ontology_id == "09003"


def test_fndds_food_classification(temp_cache_dir):
    """Test food category classification"""
    mapper = FNDDSMapper(cache_dir=temp_cache_dir, use_api=False)

    assert mapper.classify_food("apple") == "fruit"
    assert mapper.classify_food("chicken breast") == "protein"
    assert mapper.classify_food("milk") == "dairy"
    assert mapper.classify_food("broccoli") == "vegetable"


def test_variable_catalog_creation():
    """Test variable catalog creation"""
    catalog = VariableCatalog()

    variable = VariableDefinition(
        name="ADHD_RS_total",
        variable_type=VariableType.CLINICAL,
        data_type=DataType.CONTINUOUS,
        description="ADHD Rating Scale total score",
        units="points",
        valid_range=(0, 72),
    )

    catalog.add_variable(variable)

    retrieved = catalog.get_variable("ADHD_RS_total")
    assert retrieved is not None
    assert retrieved.name == "ADHD_RS_total"
    assert retrieved.valid_range == (0, 72)


def test_variable_catalog_search():
    """Test variable catalog search"""
    catalog = VariableCatalog()

    # Add multiple variables
    for name in ["ADHD_RS_total", "ADHD_RS_inattention", "SCQ_total"]:
        catalog.add_variable(
            VariableDefinition(
                name=name,
                variable_type=VariableType.CLINICAL,
                data_type=DataType.CONTINUOUS,
                description=f"{name} score",
            )
        )

    # Search by query
    results = catalog.search_variables(query="ADHD")
    assert len(results) == 2

    # Search by variable type
    results = catalog.search_variables(variable_type=VariableType.CLINICAL)
    assert len(results) == 3


def test_variable_catalog_statistics():
    """Test variable catalog statistics"""
    catalog = VariableCatalog()

    # Add variables of different types
    catalog.add_variable(
        VariableDefinition(
            name="age",
            variable_type=VariableType.DEMOGRAPHIC,
            data_type=DataType.CONTINUOUS,
            description="Age in years",
        )
    )

    catalog.add_variable(
        VariableDefinition(
            name="ADHD_RS",
            variable_type=VariableType.CLINICAL,
            data_type=DataType.CONTINUOUS,
            description="ADHD Rating Scale",
            requires_review=True,
        )
    )

    stats = catalog.get_statistics()

    assert stats["total_variables"] == 2
    assert stats["requires_review"] == 1
    assert stats["by_type"]["demographic"] == 1
    assert stats["by_type"]["clinical"] == 1


def test_variable_catalog_export_load(temp_cache_dir):
    """Test variable catalog export and load"""
    catalog = VariableCatalog()

    variable = VariableDefinition(
        name="test_var",
        variable_type=VariableType.CLINICAL,
        data_type=DataType.CONTINUOUS,
        description="Test variable",
    )

    catalog.add_variable(variable)

    # Export to JSON
    json_path = temp_cache_dir / "catalog.json"
    catalog.export_to_json(json_path)

    # Load into new catalog
    new_catalog = VariableCatalog()
    new_catalog.load(json_path)

    retrieved = new_catalog.get_variable("test_var")
    assert retrieved is not None
    assert retrieved.name == "test_var"
    assert retrieved.variable_type == VariableType.CLINICAL


def test_ontology_match_to_dict():
    """Test OntologyMatch serialization"""
    match = OntologyMatch(
        source_term="ADHD",
        matched_term="Attention deficit hyperactivity disorder",
        ontology_id="HP:0007018",
        ontology_type=OntologyType.HPO,
        confidence=MatchConfidence.EXACT,
        similarity_score=1.0,
        alternative_matches=[("autism", "HP:0000729", 0.5)],
        requires_review=False,
        metadata={"source": "local"},
    )

    match_dict = match.to_dict()

    assert match_dict["source_term"] == "ADHD"
    assert match_dict["ontology_id"] == "HP:0007018"
    assert match_dict["confidence"] == "exact"
    assert len(match_dict["alternative_matches"]) == 1
    assert match_dict["metadata"]["source"] == "local"