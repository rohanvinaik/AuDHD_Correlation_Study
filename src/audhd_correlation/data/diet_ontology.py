"""Diet/food ontology mapper (FNDDS)"""
from pathlib import Path
from typing import Optional
import requests

from .ontology_base import (
    BaseOntologyMapper,
    OntologyMatch,
    OntologyType,
    MatchConfidence,
)


class FNDDSMapper(BaseOntologyMapper):
    """Mapper for Food and Nutrient Database for Dietary Studies (FNDDS)"""

    def __init__(self, **kwargs):
        """Initialize FNDDS mapper"""
        super().__init__(ontology_type=OntologyType.FNDDS, **kwargs)
        # USDA FoodData Central API
        self.api_base = "https://api.nal.usda.gov/fdc/v1"
        self.api_key = None  # Requires API key from USDA

    def _load_local_data(self) -> None:
        """Load local FNDDS food codes"""
        # Common food categories and items
        self._local_terms = {
            # Proteins
            "chicken breast": "05062",
            "beef": "13311",
            "salmon": "15076",
            "tuna": "15121",
            "eggs": "01123",
            "milk": "01077",
            "yogurt": "01116",
            "cheese": "01009",
            # Grains
            "white bread": "18064",
            "whole wheat bread": "18075",
            "rice": "20045",
            "pasta": "20420",
            "oatmeal": "08120",
            "cereal": "08006",
            # Fruits
            "apple": "09003",
            "banana": "09040",
            "orange": "09200",
            "strawberries": "09316",
            "blueberries": "09050",
            "grapes": "09132",
            # Vegetables
            "broccoli": "11090",
            "carrots": "11124",
            "spinach": "11457",
            "tomato": "11529",
            "lettuce": "11252",
            "potato": "11362",
            "sweet potato": "11507",
            # Processed/snacks
            "chips": "19411",
            "cookies": "18159",
            "candy": "19120",
            "soda": "14400",
            "juice": "09206",
            # Common additives/supplements
            "sugar": "19335",
            "salt": "02047",
            "vitamin d": "supplement_vd",
            "omega 3": "supplement_o3",
            "probiotic": "supplement_prob",
        }

        # Food categories
        self._categories = {
            "protein": ["chicken", "beef", "fish", "salmon", "tuna", "eggs"],
            "dairy": ["milk", "yogurt", "cheese"],
            "grain": ["bread", "rice", "pasta", "oatmeal", "cereal"],
            "fruit": [
                "apple",
                "banana",
                "orange",
                "strawberries",
                "blueberries",
                "grapes",
            ],
            "vegetable": [
                "broccoli",
                "carrots",
                "spinach",
                "tomato",
                "lettuce",
                "potato",
            ],
            "processed": ["chips", "cookies", "candy", "soda"],
            "supplement": ["vitamin", "omega", "probiotic", "mineral"],
        }

        self._synonyms = {
            "chicken breast": {"chicken", "poultry"},
            "beef": {"steak", "hamburger", "ground beef"},
            "salmon": {"fish"},
            "eggs": {"egg"},
            "milk": {"dairy milk", "cow milk"},
            "yogurt": {"yoghurt"},
            "white bread": {"bread"},
            "rice": {"white rice", "brown rice"},
            "pasta": {"noodles", "spaghetti"},
            "oatmeal": {"oats", "porridge"},
            "cereal": {"breakfast cereal"},
            "soda": {"soft drink", "pop", "cola"},
        }

    def _query_api(self, term: str) -> Optional[OntologyMatch]:
        """
        Query USDA FoodData Central API

        Args:
            term: Food term to query

        Returns:
            OntologyMatch if found, None otherwise
        """
        if not self.use_api or not self.api_key:
            return None

        try:
            url = f"{self.api_base}/foods/search"
            params = {
                "query": term,
                "pageSize": 5,
                "api_key": self.api_key,
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if not data.get("foods"):
                return None

            # Get best match
            foods = data["foods"]
            best = foods[0]

            fdc_id = str(best.get("fdcId", ""))
            description = best.get("description", "")

            # Calculate confidence
            confidence = MatchConfidence.HIGH
            similarity = 0.85

            # Get alternatives
            alternatives = []
            if len(foods) > 1:
                alternatives = [
                    (f["description"], str(f["fdcId"]), 0.8 - 0.05 * i)
                    for i, f in enumerate(foods[1:4])
                ]

            return OntologyMatch(
                source_term=term,
                matched_term=description,
                ontology_id=fdc_id,
                ontology_type=self.ontology_type,
                confidence=confidence,
                similarity_score=similarity,
                alternative_matches=alternatives,
                metadata={"source": "fdc_api"},
            )

        except (requests.RequestException, KeyError) as e:
            print(f"FNDDS API error: {e}")
            return None

    def classify_food(self, term: str) -> Optional[str]:
        """
        Classify food into category

        Args:
            term: Food term

        Returns:
            Food category if classified
        """
        normalized = self._normalize_term(term)

        for category, keywords in self._categories.items():
            if any(kw in normalized for kw in keywords):
                return category

        return "other"

    def get_nutrient_profile(self, fdc_id: str) -> Optional[dict]:
        """
        Get nutrient profile for food

        Args:
            fdc_id: FoodData Central ID

        Returns:
            Dictionary with nutrient information
        """
        if not self.use_api or not self.api_key:
            return None

        try:
            url = f"{self.api_base}/food/{fdc_id}"
            params = {"api_key": self.api_key}

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            nutrients = {}
            for nutrient in data.get("foodNutrients", []):
                name = nutrient.get("nutrient", {}).get("name")
                amount = nutrient.get("amount")
                unit = nutrient.get("nutrient", {}).get("unitName")

                if name and amount:
                    nutrients[name] = {"amount": amount, "unit": unit}

            return nutrients

        except (requests.RequestException, KeyError):
            return None