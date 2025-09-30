#!/usr/bin/env python3
"""
Data Provenance Tracking System

Tracks the complete lineage and history of datasets:
- Data acquisition history
- Processing pipelines applied
- Quality control steps
- Version history
- Data transformations
- Access history

Usage:
    python provenance_tracker.py --dataset PGC_ADHD_GWAS --log-event "Downloaded from PGC"
    python provenance_tracker.py --dataset PGC_ADHD_GWAS --show-history

Author: AuDHD Correlation Study Team
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProvenanceEvent:
    """Single provenance event"""
    event_id: str
    timestamp: str
    event_type: str  # acquisition, processing, qc, transformation, access, export
    description: str
    actor: str  # person or system that performed the action
    input_files: Optional[List[str]]
    output_files: Optional[List[str]]
    parameters: Optional[Dict]
    software_version: Optional[str]
    checksum_before: Optional[str]
    checksum_after: Optional[str]
    notes: Optional[str]


@dataclass
class DatasetProvenance:
    """Complete provenance record for a dataset"""
    dataset_id: str
    original_source: str
    original_url: str
    acquisition_date: str
    current_version: str
    current_checksum: str
    events: List[ProvenanceEvent]
    lineage: List[str]  # List of dataset IDs this was derived from
    access_log: List[Dict]  # Who accessed when


class ProvenanceTracker:
    """Track and manage data provenance"""

    def __init__(self, provenance_dir: Path = Path('data/documentation/provenance')):
        self.provenance_dir = provenance_dir
        self.provenance_dir.mkdir(parents=True, exist_ok=True)

    def _generate_event_id(self, dataset_id: str, timestamp: str) -> str:
        """Generate unique event ID"""
        content = f"{dataset_id}_{timestamp}".encode()
        return hashlib.md5(content).hexdigest()[:12]

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file"""
        if not file_path.exists():
            return None

        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def load_provenance(self, dataset_id: str) -> Optional[DatasetProvenance]:
        """Load provenance record for dataset"""
        prov_file = self.provenance_dir / f"{dataset_id}_provenance.json"

        if not prov_file.exists():
            return None

        with open(prov_file) as f:
            data = json.load(f)

        # Convert event dicts to ProvenanceEvent objects
        events = [ProvenanceEvent(**e) for e in data.get('events', [])]

        return DatasetProvenance(
            dataset_id=data['dataset_id'],
            original_source=data['original_source'],
            original_url=data['original_url'],
            acquisition_date=data['acquisition_date'],
            current_version=data['current_version'],
            current_checksum=data['current_checksum'],
            events=events,
            lineage=data.get('lineage', []),
            access_log=data.get('access_log', [])
        )

    def save_provenance(self, provenance: DatasetProvenance):
        """Save provenance record"""
        prov_file = self.provenance_dir / f"{provenance.dataset_id}_provenance.json"

        data = {
            'dataset_id': provenance.dataset_id,
            'original_source': provenance.original_source,
            'original_url': provenance.original_url,
            'acquisition_date': provenance.acquisition_date,
            'current_version': provenance.current_version,
            'current_checksum': provenance.current_checksum,
            'events': [asdict(e) for e in provenance.events],
            'lineage': provenance.lineage,
            'access_log': provenance.access_log,
            'last_updated': datetime.now().isoformat()
        }

        with open(prov_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved provenance for {provenance.dataset_id}")

    def create_provenance_record(
        self,
        dataset_id: str,
        original_source: str,
        original_url: str,
        acquisition_date: str,
        version: str = '1.0'
    ) -> DatasetProvenance:
        """Create new provenance record"""

        return DatasetProvenance(
            dataset_id=dataset_id,
            original_source=original_source,
            original_url=original_url,
            acquisition_date=acquisition_date,
            current_version=version,
            current_checksum='',
            events=[],
            lineage=[],
            access_log=[]
        )

    def log_event(
        self,
        dataset_id: str,
        event_type: str,
        description: str,
        actor: str = 'system',
        input_files: Optional[List[str]] = None,
        output_files: Optional[List[str]] = None,
        parameters: Optional[Dict] = None,
        software_version: Optional[str] = None,
        notes: Optional[str] = None
    ):
        """Log a provenance event"""

        # Load existing provenance or create new
        provenance = self.load_provenance(dataset_id)
        if not provenance:
            logger.warning(f"No provenance record found for {dataset_id}")
            return

        timestamp = datetime.now().isoformat()
        event_id = self._generate_event_id(dataset_id, timestamp)

        # Calculate checksums if files provided
        checksum_before = None
        checksum_after = None

        if input_files and len(input_files) > 0:
            input_path = Path(input_files[0])
            if input_path.exists():
                checksum_before = self._calculate_checksum(input_path)

        if output_files and len(output_files) > 0:
            output_path = Path(output_files[0])
            if output_path.exists():
                checksum_after = self._calculate_checksum(output_path)

        event = ProvenanceEvent(
            event_id=event_id,
            timestamp=timestamp,
            event_type=event_type,
            description=description,
            actor=actor,
            input_files=input_files,
            output_files=output_files,
            parameters=parameters,
            software_version=software_version,
            checksum_before=checksum_before,
            checksum_after=checksum_after,
            notes=notes
        )

        provenance.events.append(event)

        # Update current checksum if output was generated
        if checksum_after:
            provenance.current_checksum = checksum_after

        self.save_provenance(provenance)
        logger.info(f"Logged {event_type} event for {dataset_id}: {description}")

    def log_access(
        self,
        dataset_id: str,
        user: str,
        purpose: str,
        files_accessed: Optional[List[str]] = None
    ):
        """Log data access"""

        provenance = self.load_provenance(dataset_id)
        if not provenance:
            return

        access_record = {
            'timestamp': datetime.now().isoformat(),
            'user': user,
            'purpose': purpose,
            'files_accessed': files_accessed or []
        }

        provenance.access_log.append(access_record)
        self.save_provenance(provenance)

        logger.info(f"Logged access by {user} for {dataset_id}")

    def add_lineage(self, dataset_id: str, parent_dataset_id: str):
        """Add parent dataset to lineage"""

        provenance = self.load_provenance(dataset_id)
        if not provenance:
            return

        if parent_dataset_id not in provenance.lineage:
            provenance.lineage.append(parent_dataset_id)
            self.save_provenance(provenance)
            logger.info(f"Added {parent_dataset_id} to lineage of {dataset_id}")

    def get_event_history(self, dataset_id: str) -> List[ProvenanceEvent]:
        """Get chronological event history"""

        provenance = self.load_provenance(dataset_id)
        if not provenance:
            return []

        # Sort by timestamp
        return sorted(provenance.events, key=lambda e: e.timestamp)

    def get_lineage_chain(self, dataset_id: str) -> List[str]:
        """Get complete lineage chain (parents of parents)"""

        provenance = self.load_provenance(dataset_id)
        if not provenance:
            return []

        lineage_chain = [dataset_id]
        to_process = provenance.lineage.copy()

        while to_process:
            parent_id = to_process.pop(0)
            if parent_id not in lineage_chain:
                lineage_chain.append(parent_id)

                # Get parent's lineage
                parent_prov = self.load_provenance(parent_id)
                if parent_prov:
                    to_process.extend(parent_prov.lineage)

        return lineage_chain

    def generate_provenance_report(self, dataset_id: str) -> str:
        """Generate human-readable provenance report"""

        provenance = self.load_provenance(dataset_id)
        if not provenance:
            return f"No provenance record found for {dataset_id}"

        report = f"""# Provenance Report: {dataset_id}

## Original Source

- **Source**: {provenance.original_source}
- **URL**: {provenance.original_url}
- **Acquisition Date**: {provenance.acquisition_date}
- **Current Version**: {provenance.current_version}
- **Current Checksum**: {provenance.current_checksum[:16]}...

## Lineage

"""

        if provenance.lineage:
            report += "This dataset was derived from:\n"
            for parent in provenance.lineage:
                report += f"- {parent}\n"
        else:
            report += "This is an original dataset (not derived from others).\n"

        report += f"\n## Event History ({len(provenance.events)} events)\n\n"

        for event in self.get_event_history(dataset_id):
            report += f"### {event.timestamp}\n"
            report += f"**Type**: {event.event_type.upper()}\n"
            report += f"**Actor**: {event.actor}\n"
            report += f"**Description**: {event.description}\n"

            if event.input_files:
                report += f"**Input Files**: {', '.join(event.input_files)}\n"

            if event.output_files:
                report += f"**Output Files**: {', '.join(event.output_files)}\n"

            if event.parameters:
                report += "**Parameters**:\n"
                for key, value in event.parameters.items():
                    report += f"  - {key}: {value}\n"

            if event.software_version:
                report += f"**Software Version**: {event.software_version}\n"

            if event.checksum_before:
                report += f"**Checksum (before)**: {event.checksum_before[:16]}...\n"

            if event.checksum_after:
                report += f"**Checksum (after)**: {event.checksum_after[:16]}...\n"

            if event.notes:
                report += f"**Notes**: {event.notes}\n"

            report += "\n"

        report += f"## Access Log ({len(provenance.access_log)} accesses)\n\n"

        for access in provenance.access_log[-10:]:  # Show last 10
            report += f"- **{access['timestamp']}**: {access['user']} - {access['purpose']}\n"

        if len(provenance.access_log) > 10:
            report += f"\n_(Showing last 10 of {len(provenance.access_log)} accesses)_\n"

        report += f"""
---

**Report Generated**: {datetime.now().isoformat()}
"""

        return report

    def export_provenance_graph(self, dataset_id: str, format: str = 'dot') -> str:
        """Export provenance as graph (DOT format for Graphviz)"""

        provenance = self.load_provenance(dataset_id)
        if not provenance:
            return ""

        if format == 'dot':
            graph = f"""digraph provenance {{
    rankdir=LR;
    node [shape=box, style=rounded];

    // Datasets
    "{dataset_id}" [label="{dataset_id}\\n{provenance.current_version}", fillcolor=lightblue, style=filled];

"""

            # Add lineage edges
            for parent in provenance.lineage:
                graph += f'    "{parent}" -> "{dataset_id}" [label="derived from"];\n'

            # Add event nodes
            for i, event in enumerate(provenance.events):
                event_node = f"event_{i}"
                graph += f'    "{event_node}" [label="{event.event_type}\\n{event.description[:30]}", shape=ellipse, fillcolor=lightyellow, style=filled];\n'

                if event.input_files:
                    for inp in event.input_files:
                        inp_name = Path(inp).name
                        graph += f'    "{inp_name}" -> "{event_node}";\n'

                if event.output_files:
                    for out in event.output_files:
                        out_name = Path(out).name
                        graph += f'    "{event_node}" -> "{out_name}";\n'

            graph += "}\n"
            return graph

        return ""


def main():
    parser = argparse.ArgumentParser(description='Track data provenance')
    parser.add_argument('--dataset', required=True, help='Dataset ID')
    parser.add_argument('--create', action='store_true', help='Create new provenance record')
    parser.add_argument('--source', help='Original source')
    parser.add_argument('--url', help='Original URL')
    parser.add_argument('--log-event', help='Log a provenance event')
    parser.add_argument('--event-type', default='processing', help='Event type')
    parser.add_argument('--actor', default='system', help='Who performed the action')
    parser.add_argument('--show-history', action='store_true', help='Show event history')
    parser.add_argument('--generate-report', action='store_true', help='Generate provenance report')

    args = parser.parse_args()

    tracker = ProvenanceTracker()

    if args.create:
        if not args.source or not args.url:
            print("--source and --url required for creating provenance record")
            return

        provenance = tracker.create_provenance_record(
            dataset_id=args.dataset,
            original_source=args.source,
            original_url=args.url,
            acquisition_date=datetime.now().isoformat()
        )
        tracker.save_provenance(provenance)
        print(f"Created provenance record for {args.dataset}")

    elif args.log_event:
        tracker.log_event(
            dataset_id=args.dataset,
            event_type=args.event_type,
            description=args.log_event,
            actor=args.actor
        )

    elif args.show_history:
        events = tracker.get_event_history(args.dataset)
        print(f"\n=== Event History for {args.dataset} ===\n")
        for event in events:
            print(f"{event.timestamp}: {event.event_type} - {event.description}")

    elif args.generate_report:
        report = tracker.generate_provenance_report(args.dataset)
        print(report)


if __name__ == '__main__':
    import sys
    main()