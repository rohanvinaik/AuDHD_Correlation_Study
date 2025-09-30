#!/usr/bin/env python3
"""
Expanded Paper Search Queries

Updated search terms reflecting the expanded project scope:
- Prompts 1.1-1.3: Autonomic, circadian, environmental, biobanks
- Prompt 2.1: HRV, actigraphy, salivary biomarkers
"""

# Core ASD/ADHD searches (original)
CORE_SEARCHES = [
    'autism ADHD comorbidity data',
    'autism ADHD genetics microbiome',
    'autism attention deficit hyperactivity disorder correlation',
    'ASD ADHD shared etiology',
]

# PRIORITY 1: Autonomic & Circadian (Prompt 1.1, 2.1)
AUTONOMIC_CIRCADIAN = [
    'autism ADHD heart rate variability HRV',
    'autism autonomic nervous system dysfunction',
    'ADHD autonomic regulation cardiovascular',
    'autism ADHD vagal tone RSA respiratory sinus arrhythmia',
    'autism circadian rhythm sleep disorder',
    'ADHD cortisol awakening response CAR',
    'autism ADHD melatonin circadian dysfunction',
    'autism ADHD actigraphy rest-activity rhythm',
    'autism parasympathetic nervous system',
    'ADHD heart rate variability frequency domain',
    'autism blood pressure variability baroreceptor',
    'autism ADHD sympathetic nervous system hyperarousal',
]

# PRIORITY 2: Sensory & Interoception (Prompt 1.1)
SENSORY_INTEROCEPTION = [
    'autism sensory processing disorder SPM interoception',
    'autism ADHD interoceptive awareness heartbeat detection',
    'autism tactile defensiveness sensory sensitivity',
    'ADHD auditory processing disorder APD',
    'autism visual processing contrast sensitivity',
    'autism ADHD pain sensitivity nociception',
    'autism proprioception vestibular dysfunction',
]

# PRIORITY 3: Environmental Exposures (Prompt 1.1, 1.2)
ENVIRONMENTAL_BIOMARKERS = [
    'autism heavy metals lead mercury exposure',
    'autism ADHD pesticide exposure organophosphates',
    'autism phthalates BPA endocrine disruptors',
    'ADHD environmental toxicants air pollution',
    'autism maternal prenatal exposure environmental',
    'autism PCBs PBDEs persistent organic pollutants',
    'ADHD traffic pollution particulate matter PM2.5',
    'autism residential proximity industrial facilities',
]

# PRIORITY 4: Salivary & Stress Biomarkers (Prompt 2.1B)
SALIVARY_BIOMARKERS = [
    'autism cortisol saliva diurnal rhythm',
    'ADHD cortisol awakening response stress axis',
    'autism ADHD alpha amylase sympathetic',
    'autism saliva inflammatory cytokines IL-6',
    'autism ADHD DHEA cortisol ratio',
    'autism salivary melatonin DLMO',
    'ADHD salivary biomarkers stress reactivity',
]

# Biobank & Large Cohort Studies (Prompt 1.3)
BIOBANK_STUDIES = [
    'autism SPARK cohort data',
    'ADHD ABCD study biospecimens',
    'autism UK Biobank',
    'ADHD All of Us precision medicine',
    'autism EARLI cohort environmental',
    'autism Simons Simplex Collection SSC',
    'ADHD longitudinal cohort study data',
]

# Trace Minerals & Metabolic (Prompt 1.2)
METABOLIC_BIOMARKERS = [
    'autism zinc copper selenium deficiency',
    'ADHD iron ferritin deficiency',
    'autism vitamin D magnesium',
    'autism ADHD metabolomics urine blood',
    'autism inflammatory markers CRP ESR',
    'ADHD metabolic syndrome HOMA-IR',
]

# Proteomics & Advanced Biomarkers
PROTEOMICS = [
    'autism proteomics plasma biomarkers',
    'autism ADHD SOMAscan protein panel',
    'autism exosomes circulating biomarkers',
    'autism cell-free DNA cfDNA',
]

# Integration & Multi-omics
MULTI_OMICS = [
    'autism ADHD multi-omics integration',
    'autism systems biology network analysis',
    'autism microbiome metabolome proteome',
    'ADHD genomics transcriptomics integration',
]

# Physiology & Mechanisms
PHYSIOLOGY = [
    'autism polyvagal theory autonomic',
    'ADHD arousal regulation tonic alertness',
    'autism ADHD default mode network connectivity',
    'autism interoceptive predictive coding',
]

# Sleep Disorders
SLEEP = [
    'autism sleep disorders polysomnography',
    'ADHD insomnia circadian phase delay',
    'autism restless legs periodic limb movement',
    'autism obstructive sleep apnea OSA',
]

# Compile all searches
ALL_SEARCHES = {
    'core': CORE_SEARCHES,
    'autonomic_circadian': AUTONOMIC_CIRCADIAN,
    'sensory_interoception': SENSORY_INTEROCEPTION,
    'environmental': ENVIRONMENTAL_BIOMARKERS,
    'salivary': SALIVARY_BIOMARKERS,
    'biobanks': BIOBANK_STUDIES,
    'metabolic': METABOLIC_BIOMARKERS,
    'proteomics': PROTEOMICS,
    'multi_omics': MULTI_OMICS,
    'physiology': PHYSIOLOGY,
    'sleep': SLEEP,
}


def get_all_queries():
    """Get all search queries as a flat list"""
    queries = []
    for category, query_list in ALL_SEARCHES.items():
        queries.extend(query_list)
    return queries


def get_priority_queries():
    """Get high-priority queries only"""
    priority_categories = [
        'autonomic_circadian',
        'salivary',
        'environmental',
        'biobanks'
    ]
    queries = []
    for category in priority_categories:
        queries.extend(ALL_SEARCHES[category])
    return queries


def print_query_summary():
    """Print summary of search queries"""
    print("="*60)
    print("EXPANDED PAPER SEARCH QUERIES")
    print("="*60)
    print()

    for category, query_list in ALL_SEARCHES.items():
        print(f"{category.upper().replace('_', ' ')} ({len(query_list)} queries):")
        for i, query in enumerate(query_list, 1):
            print(f"  {i}. {query}")
        print()

    total = sum(len(q) for q in ALL_SEARCHES.values())
    print(f"TOTAL QUERIES: {total}")
    print()


if __name__ == '__main__':
    print_query_summary()

    print("\nTo use with paper scraper:")
    print("="*60)
    print()
    print("# All queries:")
    print("python scripts/scrape_papers.py \\")
    for query in get_all_queries()[:5]:
        print(f"  --query \"{query}\" \\")
    print("  ... (and more)")
    print()
    print("# Priority queries only:")
    print("python scripts/scrape_papers.py \\")
    for query in get_priority_queries()[:5]:
        print(f"  --query \"{query}\" \\")
    print("  ... (and more)")