#!/usr/bin/env python3
"""
Alert System for Monitoring Updates

Sends notifications about detected updates via:
- Email
- Slack
- Console output
- Log files
- JSON reports

Usage:
    python alert_system.py --check-updates
    python alert_system.py --send-digest --email user@example.com
    python alert_system.py --test-alerts

Author: AuDHD Correlation Study Team
"""

import argparse
import json
import logging
import smtplib
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass

try:
    import requests
    import yaml
except ImportError as e:
    print(f"Error: {e}")
    print("Install with: pip install requests pyyaml")
    import sys
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AlertSystem:
    """Send alerts about detected updates"""

    def __init__(self, config_path: Path = Path('configs/monitoring_config.yaml')):
        self.config = self._load_config(config_path)
        self.alerts_dir = Path('data/monitoring/alerts')
        self.alerts_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, path: Path) -> Dict:
        """Load configuration"""
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f)
        return {}

    def load_updates(self, updates_file: Path) -> List[Dict]:
        """Load detected updates"""
        if not updates_file.exists():
            return []

        with open(updates_file) as f:
            data = json.load(f)

        return data.get('updates', [])

    def load_publications(self, pubs_file: Path) -> List[Dict]:
        """Load detected publications"""
        if not pubs_file.exists():
            return []

        with open(pubs_file) as f:
            data = json.load(f)

        return data.get('publications', [])

    def filter_new_alerts(
        self,
        items: List[Dict],
        hours_back: int = 24
    ) -> List[Dict]:
        """Filter items detected in last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours_back)
        new_items = []

        for item in items:
            detected = item.get('detected_date')
            if detected:
                detected_dt = datetime.fromisoformat(detected)
                if detected_dt >= cutoff:
                    new_items.append(item)

        return new_items

    def filter_by_priority(
        self,
        items: List[Dict],
        min_priority: str = 'medium'
    ) -> List[Dict]:
        """Filter items by priority"""
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        min_level = priority_order.get(min_priority, 2)

        return [item for item in items
                if priority_order.get(item.get('priority', 'low'), 1) >= min_level]

    def send_email_alert(
        self,
        recipient: str,
        subject: str,
        body: str,
        html_body: Optional[str] = None
    ) -> bool:
        """Send email alert"""
        try:
            email_config = self.config.get('email', {})

            if not email_config.get('enabled', False):
                logger.warning("Email alerts not enabled in config")
                return False

            smtp_server = email_config.get('smtp_server')
            smtp_port = email_config.get('smtp_port', 587)
            username = email_config.get('username')
            password = email_config.get('password')
            from_addr = email_config.get('from_address', username)

            if not all([smtp_server, username, password]):
                logger.error("Email configuration incomplete")
                return False

            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = from_addr
            msg['To'] = recipient
            msg['Subject'] = subject

            # Add plain text and HTML parts
            msg.attach(MIMEText(body, 'plain'))
            if html_body:
                msg.attach(MIMEText(html_body, 'html'))

            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)

            logger.info(f"Sent email alert to {recipient}")
            return True

        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False

    def send_slack_alert(self, channel: str, message: str) -> bool:
        """Send Slack alert"""
        try:
            slack_config = self.config.get('slack', {})

            if not slack_config.get('enabled', False):
                logger.warning("Slack alerts not enabled in config")
                return False

            webhook_url = slack_config.get('webhook_url')

            if not webhook_url:
                logger.error("Slack webhook URL not configured")
                return False

            # Send to Slack
            payload = {'text': message}
            response = requests.post(webhook_url, json=payload, timeout=10)

            if response.status_code == 200:
                logger.info(f"Sent Slack alert to {channel}")
                return True
            else:
                logger.error(f"Slack alert failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
            return False

    def generate_update_summary(self, updates: List[Dict]) -> str:
        """Generate text summary of updates"""
        if not updates:
            return "No new updates detected."

        summary = f"=== Data Update Summary ({len(updates)} updates) ===\n\n"

        # Group by priority
        high_priority = [u for u in updates if u.get('priority') == 'high']
        medium_priority = [u for u in updates if u.get('priority') == 'medium']
        low_priority = [u for u in updates if u.get('priority') == 'low']

        if high_priority:
            summary += "HIGH PRIORITY:\n"
            for update in high_priority:
                summary += f"  • {update.get('source')}: {update.get('dataset_name')}\n"
                summary += f"    Type: {update.get('update_type')}\n"
                summary += f"    URL: {update.get('url')}\n\n"

        if medium_priority:
            summary += "MEDIUM PRIORITY:\n"
            for update in medium_priority:
                summary += f"  • {update.get('source')}: {update.get('dataset_name')}\n"
                summary += f"    Type: {update.get('update_type')}\n\n"

        if low_priority:
            summary += f"LOW PRIORITY: {len(low_priority)} updates\n"

        return summary

    def generate_publication_summary(self, publications: List[Dict]) -> str:
        """Generate text summary of publications"""
        if not publications:
            return "No new publications detected."

        summary = f"=== New Publications ({len(publications)} found) ===\n\n"

        # Sort by relevance
        sorted_pubs = sorted(publications,
                           key=lambda p: p.get('relevance_score', 0),
                           reverse=True)

        for pub in sorted_pubs[:10]:  # Top 10
            summary += f"[{pub.get('relevance_score', 0):.2f}] {pub.get('title')}\n"
            summary += f"  Authors: {', '.join(pub.get('authors', [])[:3])}\n"
            summary += f"  Source: {pub.get('source')} ({pub.get('publication_date')})\n"

            if pub.get('doi'):
                summary += f"  DOI: {pub.get('doi')}\n"

            accessions = pub.get('accession_numbers', [])
            if accessions:
                summary += f"  Accessions: {', '.join(accessions[:3])}\n"

            summary += "\n"

        if len(sorted_pubs) > 10:
            summary += f"... and {len(sorted_pubs) - 10} more\n"

        return summary

    def generate_html_digest(
        self,
        updates: List[Dict],
        publications: List[Dict]
    ) -> str:
        """Generate HTML digest email"""

        html = f"""<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .section {{
            margin-bottom: 30px;
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .update {{
            background: #f5f5f5;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }}
        .priority-high {{
            border-left-color: #ef4444;
        }}
        .priority-medium {{
            border-left-color: #f59e0b;
        }}
        .priority-low {{
            border-left-color: #10b981;
        }}
        .publication {{
            background: #f9fafb;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }}
        .label {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 0.85em;
            font-weight: bold;
            margin-right: 5px;
        }}
        .label-high {{ background: #fee2e2; color: #991b1b; }}
        .label-medium {{ background: #fef3c7; color: #92400e; }}
        .label-low {{ background: #d1fae5; color: #065f46; }}
        a {{ color: #667eea; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>AuDHD Study Monitoring Digest</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
"""

        # Updates section
        if updates:
            html += f"""
    <div class="section">
        <h2>Data Updates ({len(updates)})</h2>
"""
            for update in updates[:20]:  # Limit to 20
                priority = update.get('priority', 'low')
                html += f"""
        <div class="update priority-{priority}">
            <span class="label label-{priority}">{priority.upper()}</span>
            <strong>{update.get('source')}: {update.get('dataset_name')}</strong><br>
            <small>Type: {update.get('update_type')} | Detected: {update.get('detected_date', '')[:10]}</small>
            <p>{update.get('description', '')[:200]}</p>
            <a href="{update.get('url', '#')}">View Update →</a>
        </div>
"""
            html += "    </div>\n"
        else:
            html += """
    <div class="section">
        <h2>Data Updates</h2>
        <p>No new updates detected.</p>
    </div>
"""

        # Publications section
        if publications:
            html += f"""
    <div class="section">
        <h2>New Publications ({len(publications)})</h2>
"""
            for pub in publications[:10]:  # Top 10
                relevance = pub.get('relevance_score', 0)
                html += f"""
        <div class="publication">
            <strong>[{relevance:.2f}] {pub.get('title')}</strong><br>
            <small>
                Authors: {', '.join(pub.get('authors', [])[:3])}<br>
                Source: {pub.get('source')} | {pub.get('journal')} ({pub.get('publication_date', '')[:10]})
"""
                if pub.get('doi'):
                    html += f"""<br>DOI: <a href="https://doi.org/{pub.get('doi')}">{pub.get('doi')}</a>"""

                accessions = pub.get('accession_numbers', [])
                if accessions:
                    html += f"""<br>Accessions: {', '.join(accessions[:3])}"""

                html += """
            </small>
        </div>
"""
            html += "    </div>\n"
        else:
            html += """
    <div class="section">
        <h2>New Publications</h2>
        <p>No new publications detected.</p>
    </div>
"""

        html += """
    <div class="section">
        <p><small>
            This is an automated digest from the AuDHD Correlation Study monitoring system.<br>
            To modify alert settings, edit configs/monitoring_config.yaml
        </small></p>
    </div>
</body>
</html>
"""

        return html

    def send_digest(
        self,
        recipient: Optional[str] = None,
        hours_back: int = 24
    ):
        """Send digest of recent updates"""

        # Load updates and publications
        updates_file = Path('data/monitoring/detected_updates.json')
        pubs_file = Path('data/monitoring/new_publications.json')

        all_updates = self.load_updates(updates_file)
        all_pubs = self.load_publications(pubs_file)

        # Filter to recent items
        recent_updates = self.filter_new_alerts(all_updates, hours_back)
        recent_pubs = self.filter_new_alerts(all_pubs, hours_back)

        # Filter by priority
        important_updates = self.filter_by_priority(recent_updates, 'medium')

        if not recent_updates and not recent_pubs:
            logger.info("No new updates or publications - skipping digest")
            return

        # Generate summaries
        text_summary = "AuDHD Study Monitoring Digest\n" + "=" * 50 + "\n\n"
        text_summary += self.generate_update_summary(important_updates)
        text_summary += "\n\n"
        text_summary += self.generate_publication_summary(recent_pubs)

        html_summary = self.generate_html_digest(important_updates, recent_pubs)

        # Print to console
        print(text_summary)

        # Send email if configured
        if recipient:
            subject = f"AuDHD Study Monitoring Digest - {len(important_updates)} updates, {len(recent_pubs)} publications"
            self.send_email_alert(recipient, subject, text_summary, html_summary)

        # Send Slack if configured
        slack_message = f"*AuDHD Study Monitoring Digest*\n\n{len(important_updates)} data updates and {len(recent_pubs)} new publications detected.\n\nCheck your email for details."
        self.send_slack_alert('#monitoring', slack_message)

        # Save digest
        digest_file = self.alerts_dir / f"digest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(digest_file, 'w') as f:
            json.dump({
                'generated_date': datetime.now().isoformat(),
                'updates': important_updates,
                'publications': recent_pubs,
                'text_summary': text_summary
            }, f, indent=2)

        logger.info(f"Saved digest to {digest_file}")

    def send_immediate_alert(self, update: Dict):
        """Send immediate alert for high-priority update"""

        subject = f"URGENT: {update.get('source')} - {update.get('dataset_name')}"

        message = f"""
HIGH PRIORITY DATA UPDATE DETECTED

Source: {update.get('source')}
Dataset: {update.get('dataset_name')}
Type: {update.get('update_type')}
Detected: {update.get('detected_date')}

Description:
{update.get('description')}

URL: {update.get('url')}

Action required: {update.get('action_required')}
"""

        # Send email
        recipients = self.config.get('alert_recipients', [])
        for recipient in recipients:
            self.send_email_alert(recipient, subject, message)

        # Send Slack
        slack_message = f"🚨 *HIGH PRIORITY UPDATE*\n\n*{update.get('source')}*: {update.get('dataset_name')}\n\n{update.get('description')[:200]}\n\n<{update.get('url')}|View Update>"
        self.send_slack_alert('#urgent', slack_message)

        logger.info(f"Sent immediate alert for {update.get('dataset_name')}")

    def test_alerts(self):
        """Test alert system"""
        logger.info("Testing alert system...")

        # Test email
        test_recipient = self.config.get('test_email')
        if test_recipient:
            self.send_email_alert(
                test_recipient,
                "Test Alert - AuDHD Study Monitoring",
                "This is a test email from the monitoring system.",
                "<html><body><h1>Test Alert</h1><p>This is a test email.</p></body></html>"
            )
            print(f"✓ Sent test email to {test_recipient}")

        # Test Slack
        if self.config.get('slack', {}).get('enabled'):
            self.send_slack_alert('#test', "Test alert from AuDHD Study monitoring system")
            print("✓ Sent test Slack message")

        print("\nAlert system test complete!")


def main():
    parser = argparse.ArgumentParser(description='Send monitoring alerts')
    parser.add_argument('--config', default='configs/monitoring_config.yaml',
                       help='Configuration file')
    parser.add_argument('--check-updates', action='store_true',
                       help='Check for updates and send alerts')
    parser.add_argument('--send-digest', action='store_true',
                       help='Send digest email')
    parser.add_argument('--email', help='Email recipient for digest')
    parser.add_argument('--hours-back', type=int, default=24,
                       help='Hours to look back for digest')
    parser.add_argument('--test-alerts', action='store_true',
                       help='Test alert system')

    args = parser.parse_args()

    alert_system = AlertSystem(Path(args.config))

    if args.test_alerts:
        alert_system.test_alerts()
    elif args.send_digest:
        alert_system.send_digest(args.email, args.hours_back)
    elif args.check_updates:
        # Check for high-priority updates and send immediate alerts
        updates_file = Path('data/monitoring/detected_updates.json')
        updates = alert_system.load_updates(updates_file)

        high_priority = [u for u in updates if u.get('priority') == 'high']
        recent_high_priority = alert_system.filter_new_alerts(high_priority, 1)

        for update in recent_high_priority:
            alert_system.send_immediate_alert(update)

        print(f"Checked {len(updates)} updates, sent {len(recent_high_priority)} alerts")
    else:
        parser.print_help()


if __name__ == '__main__':
    import sys
    main()