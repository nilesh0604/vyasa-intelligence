"""Entity extractor for Mahabharata text.

This module identifies and extracts key entities like characters, places,
weapons, and concepts from Mahabharata text using pattern matching and
context analysis.
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EntityInfo:
    """Information about an extracted entity."""

    text: str
    entity_type: str
    confidence: float
    context: str
    position: Tuple[int, int]  # (start, end) positions in text


class MahabharataEntityExtractor:
    """Extracts entities specific to the Mahabharata epic."""

    def __init__(self):
        """Initialize the entity extractor with predefined entity lists."""
        # Character names with variations
        self.characters = {
            # Pandavas
            "arjuna": [
                "arjuna",
                "partha",
                "dhananjaya",
                "bibhatsu",
                "vijaya",
                "savyasachi",
                "gudakesha",
            ],
            "krishna": [
                "krishna",
                "vasudeva",
                "kanha",
                "mukunda",
                "madhava",
                "kesava",
                "hari",
                "govinda",
            ],
            "yudhishthira": ["yudhishthira", "dharmaraja", "bharata", "ajatashatru"],
            "bhima": ["bhima", "bhimasena", "vrikodara", "bhava"],
            "nakula": ["nakula", "ashvin"],
            "sahadeva": ["sahadeva", "ashvin"],
            "draupadi": ["draupadi", "panchali", "yajnaseni", "krishnaa"],
            # Kauravas
            "duryodhana": ["duryodhana", "suyodhana"],
            "dushasana": ["dushasana", "sushasana"],
            "vikarna": ["vikarna"],
            "yuyutsu": ["yuyutsu"],
            # Elders and Gurus
            "bhishma": ["bhishma", "devavrata", "ganga-putra", "pitamaha"],
            "drona": ["drona", "dronacharya"],
            "kripa": ["kripa", "kripacharya"],
            "ashwatthama": ["ashwatthama", "drauni"],
            "vidura": ["vidura", "vidura"],
            "dhritarashtra": ["dhritarashtra", "dhritarashtra"],
            "gandhari": ["gandhari", "gandhari"],
            "kunti": ["kunti", "pritha"],
            "madri": ["madri"],
            "pandu": ["pandu"],
            "shantanu": ["shantanu"],
            "satyavati": ["satyavati", "matsyagandha"],
            # Allies and Other Characters
            "karna": ["karna", "suryaputra", "radheya", "vasusena"],
            "shakuni": ["shakuni", "gandhari-putra"],
            "shalya": ["shalya", "madra-king"],
            "drupada": ["drupada", "panchala-king"],
            "shikhandi": ["shikhandi", "shikhandini"],
            "abhimanyu": ["abhimanyu", "arjuna-son"],
            "ghatotkacha": ["ghatotkacha", "bhima-son"],
            "upapandavas": [
                "upapandavas",
                "prativindhya",
                "sutasoma",
                "srutakirti",
                "satanika",
                "shrutasena",
            ],
            # Sages and Divine Beings
            "vyasa": ["vyasa", "krishna-dwaipayana", "veda-vyasa", "badarayana"],
            "sanjaya": ["sanjaya", "dhritarashtra-charioteer"],
            "narada": ["narada", "devarshi"],
            "brahma": ["brahma", "creator"],
            "indra": ["indra", "devendra", "shakra"],
            "agni": ["agni", "fire-god"],
            "vayu": ["vayu", "wind-god"],
            "yama": ["yama", "death-god", "dharma"],
            "varuna": ["varuna", "water-god"],
            "kubera": ["kubera", "wealth-god"],
            "ganga": ["ganga", "river-goddess"],
            "yami": ["yami", "yamuna"],
            "saraswati": ["saraswati", "knowledge-goddess"],
            "lakshmi": ["lakshmi", "wealth-goddess"],
            "parvati": ["parvati", "uma"],
            "durga": ["durga", "shakti"],
        }

        # Places
        self.places = {
            "kingdoms": {
                "hastinapura": ["hastinapura", "hastinapur"],
                "indraprastha": ["indraprastha", "indraprastha"],
                "panchala": ["panchala", "panchala-kingdom"],
                "gandhara": ["gandhara", "gandhara-kingdom"],
                "magadha": ["magadha", "magadha-kingdom"],
                "kashi": ["kashi", "varanasi", "benares"],
                "matsya": ["matsya", "virata-kingdom"],
                "kamboja": ["kamboja", "kamboja-kingdom"],
                "sindhu": ["sindhu", "sindhu-kingdom"],
                "sauvira": ["sauvira", "sauvira-kingdom"],
                "vidarbha": ["vidarbha", "vidarbha-kingdom"],
                "kosala": ["kosala", "kosala-kingdom"],
                "anga": ["anga", "anga-kingdom"],
                "vanga": ["vanga", "vanga-kingdom"],
                "kalinga": ["kalinga", "kalinga-kingdom"],
                "dwarka": ["dwarka", "dwaraka", "dwaravati"],
                "mathura": ["mathura", "mathura"],
            },
            "battlefield": {
                "kurukshetra": ["kurukshetra", "field-of-kurus", "dharmakshetra"],
            },
            "rivers": {
                "ganga": ["ganga", "ganges"],
                "yamuna": ["yamuna", "yamini"],
                "saraswati": ["saraswati", "saraswati-river"],
                "sarayu": ["sarayu"],
                "godavari": ["godavari"],
                "narmada": ["narmada"],
                "kaveri": ["kaveri"],
            },
            "mountains": {
                "himalayas": ["himalayas", "himalaya", "himavan"],
                "vindhya": ["vindhya", "vindhya-range"],
                "mount meru": ["mount meru", "meru", "sumeru"],
            },
            "forests": {
                "kamyaka": ["kamyaka", "kamyaka-forest"],
                "dwaita": ["dwaita", "dwaita-forest"],
                "naimisha": ["naimisha", "naimisha-forest"],
            },
        }

        # Weapons and celestial weapons
        self.weapons = {
            "celestial": {
                "brahmastra": ["brahmastra", "brahma-weapon"],
                "pashupatastra": ["pashupatastra", "pashupati-weapon"],
                "vaishnavastra": ["vaishnavastra", "vaishnava-weapon"],
                "narayanastra": ["narayanastra", "narayana-weapon"],
                "vajra": ["vajra", "indra-weapon", "thunderbolt"],
                "shakti": ["shakti", "shakti-weapon"],
                "brahmashirsha": ["brahmashirsha", "brahma-head-weapon"],
                "gandiva": ["gandiva", "gandiva-bow"],
                "vijaya": ["vijaya", "vijaya-bow"],
                "pinaka": ["pinaka", "pinaka-bow"],
                "sharanga": ["sharanga", "sharanga-bow"],
            },
            "conventional": {
                "bow": ["bow", "dhanush", "karmuka"],
                "arrow": ["arrow", "bana", "shara", "ishu"],
                "sword": ["sword", "khadga", "asi"],
                "mace": ["mace", "gada", "parigha"],
                "spear": ["spear", "shula", "kunta"],
                "chakram": ["chakram", "chakra", "discus"],
                "axe": ["axe", "parashu", "kuthara"],
                "dagger": ["dagger", "khanjana", "katar"],
            },
        }

        # Philosophical concepts
        self.concepts = {
            "dharma": ["dharma", "duty", "righteousness", "virtue"],
            "karma": ["karma", "action", "deed"],
            "moksha": ["moksha", "liberation", "salvation"],
            "yoga": ["yoga", "union", "discipline"],
            "atman": ["atman", "soul", "self"],
            "brahman": ["brahman", "ultimate-reality", "absolute"],
            "samsara": ["samsara", "cycle-of-rebirth", "reincarnation"],
            "ahimsa": ["ahimsa", "non-violence"],
            "satya": ["satya", "truth"],
            "prema": ["prema", "love", "devotion"],
            "bhakti": ["bhakti", "devotion", "faith"],
            "jnana": ["jnana", "knowledge", "wisdom"],
            "vairagya": ["vairagya", "detachment", "dispassion"],
        }

        # Compile regex patterns for efficient matching
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for all entities."""
        self.character_patterns = {}
        self.place_patterns = {}
        self.weapon_patterns = {}
        self.concept_patterns = {}

        # Compile character patterns
        for canonical, variations in self.characters.items():
            pattern = re.compile(
                r"\b(?:" + "|".join(map(re.escape, variations)) + r")\b", re.IGNORECASE
            )
            self.character_patterns[canonical] = pattern

        # Compile place patterns
        for category, places in self.places.items():
            self.place_patterns[category] = {}
            for canonical, variations in places.items():
                pattern = re.compile(
                    r"\b(?:" + "|".join(map(re.escape, variations)) + r")\b",
                    re.IGNORECASE,
                )
                self.place_patterns[category][canonical] = pattern

        # Compile weapon patterns
        for category, weapons in self.weapons.items():
            self.weapon_patterns[category] = {}
            for canonical, variations in weapons.items():
                pattern = re.compile(
                    r"\b(?:" + "|".join(map(re.escape, variations)) + r")\b",
                    re.IGNORECASE,
                )
                self.weapon_patterns[category][canonical] = pattern

        # Compile concept patterns
        for canonical, variations in self.concepts.items():
            pattern = re.compile(
                r"\b(?:" + "|".join(map(re.escape, variations)) + r")\b", re.IGNORECASE
            )
            self.concept_patterns[canonical] = pattern

    def extract_entities(self, text: str) -> Dict[str, List[EntityInfo]]:
        """Extract all entities from the given text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with entity types as keys and lists of EntityInfo as values
        """
        results = {
            "characters": self._extract_characters(text),
            "places": self._extract_places(text),
            "weapons": self._extract_weapons(text),
            "concepts": self._extract_concepts(text),
        }

        return results

    def _extract_characters(self, text: str) -> List[EntityInfo]:
        """Extract character names from text."""
        entities = []

        for canonical, pattern in self.character_patterns.items():
            for match in pattern.finditer(text):
                # Get context around the match
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].strip()

                entity = EntityInfo(
                    text=match.group(),
                    entity_type="character",
                    confidence=0.9,  # High confidence for direct matches
                    context=context,
                    position=(match.start(), match.end()),
                )
                entities.append(entity)

        return entities

    def _extract_places(self, text: str) -> List[EntityInfo]:
        """Extract place names from text."""
        entities = []

        for category, places in self.place_patterns.items():
            for canonical, pattern in places.items():
                for match in pattern.finditer(text):
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end].strip()

                    entity = EntityInfo(
                        text=match.group(),
                        entity_type=f"place_{category}",
                        confidence=0.85,
                        context=context,
                        position=(match.start(), match.end()),
                    )
                    entities.append(entity)

        return entities

    def _extract_weapons(self, text: str) -> List[EntityInfo]:
        """Extract weapon names from text."""
        entities = []

        for category, weapons in self.weapon_patterns.items():
            for canonical, pattern in weapons.items():
                for match in pattern.finditer(text):
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end].strip()

                    entity = EntityInfo(
                        text=match.group(),
                        entity_type=f"weapon_{category}",
                        confidence=0.8,
                        context=context,
                        position=(match.start(), match.end()),
                    )
                    entities.append(entity)

        return entities

    def _extract_concepts(self, text: str) -> List[EntityInfo]:
        """Extract philosophical concepts from text."""
        entities = []

        for canonical, pattern in self.concept_patterns.items():
            for match in pattern.finditer(text):
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].strip()

                entity = EntityInfo(
                    text=match.group(),
                    entity_type="concept",
                    confidence=0.75,
                    context=context,
                    position=(match.start(), match.end()),
                )
                entities.append(entity)

        return entities

    def get_unique_entities(self, text: str) -> Dict[str, Set[str]]:
        """Get unique entities by type.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with entity types as keys and sets of unique entities as values
        """
        extracted = self.extract_entities(text)

        unique = {
            "characters": set(),
            "places": set(),
            "weapons": set(),
            "concepts": set(),
        }

        # Map characters to canonical names
        for entity in extracted["characters"]:
            for canonical, variations in self.characters.items():
                if entity.text.lower() in [v.lower() for v in variations]:
                    unique["characters"].add(canonical)
                    break

        # Map places to canonical names
        for entity in extracted["places"]:
            for category, places in self.places.items():
                for canonical, variations in places.items():
                    if entity.text.lower() in [v.lower() for v in variations]:
                        unique["places"].add(canonical)
                        break

        # Map weapons to canonical names
        for entity in extracted["weapons"]:
            for category, weapons in self.weapons.items():
                for canonical, variations in weapons.items():
                    if entity.text.lower() in [v.lower() for v in variations]:
                        unique["weapons"].add(canonical)
                        break

        # Map concepts to canonical names
        for entity in extracted["concepts"]:
            for canonical, variations in self.concepts.items():
                if entity.text.lower() in [v.lower() for v in variations]:
                    unique["concepts"].add(canonical)
                    break

        return unique

    def get_entity_summary(self, text: str) -> Dict:
        """Get a summary of entities in the text.

        Args:
            text: Text to analyze

        Returns:
            Summary dictionary with counts and lists
        """
        unique = self.get_unique_entities(text)

        return {
            "character_count": len(unique["characters"]),
            "place_count": len(unique["places"]),
            "weapon_count": len(unique["weapons"]),
            "concept_count": len(unique["concepts"]),
            "characters": sorted(list(unique["characters"])),
            "places": sorted(list(unique["places"])),
            "weapons": sorted(list(unique["weapons"])),
            "concepts": sorted(list(unique["concepts"])),
        }


if __name__ == "__main__":
    # Example usage
    extractor = MahabharataEntityExtractor()

    sample_text = """
    Arjuna, also known as Partha and Dhananjaya, stood on the battlefield of Kurukshetra.
    He wielded the Gandiva bow and faced Dronacharya, his guru. Krishna, his charioteer,
    advised him about dharma and karma. The armies from Hastinapura and Indraprastha
    were ready for war. The Brahmastra weapon was feared by all.
    """

    entities = extractor.extract_entities(sample_text)
    summary = extractor.get_entity_summary(sample_text)

    print("Entity Summary:")
    print(f"Characters: {summary['characters']}")
    print(f"Places: {summary['places']}")
    print(f"Weapons: {summary['weapons']}")
    print(f"Concepts: {summary['concepts']}")

    print("\nDetailed Entities:")
    for entity_type, entity_list in entities.items():
        if entity_list:
            print(f"\n{entity_type.title()}:")
            for entity in entity_list[:3]:  # Show first 3
                print(f"  - {entity.text} (confidence: {entity.confidence})")
