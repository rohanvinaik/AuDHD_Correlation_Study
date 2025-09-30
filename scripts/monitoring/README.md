# Data Monitoring System

Automated monitoring for new data releases, version updates, and dataset publications across key databases and literature sources.

## Overview

This system continuously tracks:
- **Data releases** from major repositories (SFARI, UK Biobank, ABCD, dbGaP)
- **Version updates** for existing datasets
- **New publications** announcing datasets in scientific literature
- **Clinical trial results** postings
- **Repository submissions** (GEO, MetaboLights, etc.)

## Components

### 1. Update Scanner (`update_scanner.py`)

Monitors databases and repositories for new data releases:

**Monitoring Methods:**
- **RSS feeds**: For dataset announcements and news
- **API endpoints**: For version checks and metadata
- **Web scraping**: For release pages and changelogs
- **Database searches**: For new studies (dbGaP, ClinicalTrials.gov)

**Monitored Sources:**
- SFARI Base (SPARK, SSC, AGRE, Simons Searchlight)
- UK Biobank data releases
- ABCD Study annual releases
- NIH NDA collections
- dbGaP studies (phs accessions)
- ClinicalTrials.gov results postings
- GEO (Gene Expression Omnibus)
- MetaboLights
- ArrayExpress

### 2. Literature Watcher (`literature_watcher.py`)

Monitors scientific literature for dataset announcements:

**Sources:**
- **PubMed/PMC**: Published articles with data availability
- **bioRxiv/medRxiv**: Preprints with datasets
- **Nature Scientific Data**: Data descriptor papers
- **GigaScience**: Data notes
- **arXiv**: Computational/data papers

**Detection:**
- Keyword matching for autism/ADHD
- Data availability statement parsing
- Repository link extraction (GitHub, Zenodo, Figshare)
- Accession number detection (GEO, SRA, dbGaP, EGA)
- Relevance scoring

### 3. Alert System (`alert_system.py`)

Sends notifications about detected updates:

**Notification Methods:**
- **Email**: HTML digests with priority filtering
- **Slack**: Real-time alerts to channels
- **Console**: Terminal output for immediate viewing
- **JSON reports**: Machine-readable logs

**Alert Types:**
- **Immediate**: High-priority updates (new major releases)
- **Digest**: Daily/weekly summary emails
- **On-demand**: Manual report generation

## Installation

```bash
# Install dependencies
pip install requests feedparser beautifulsoup4 pyyaml

# Create required directories
mkdir -p data/monitoring/alerts logs

# Configure monitoring
cp configs/monitoring_config.yaml.example configs/monitoring_config.yaml
# Edit config with your email/Slack settings
```

## Configuration

Edit `configs/monitoring_config.yaml`:

```yaml
# Email alerts
email:
  enabled: true
  smtp_server: smtp.gmail.com
  smtp_port: 587
  username: your-email@gmail.com
  password: your-app-password

# Slack alerts
slack:
  enabled: true
  webhook_url: https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# Alert recipients
alert_recipients:
  - principal.investigator@institution.edu
  - data.manager@institution.edu
```

## Usage

### Check for Updates (One-Time)

```bash
# Check all sources
python scripts/monitoring/update_scanner.py --check-all

# Check specific source
python scripts/monitoring/update_scanner.py --source SFARI_Base

# Save results
python scripts/monitoring/update_scanner.py --check-all --output data/monitoring/updates.json
```

### Monitor Literature

```bash
# Search for new publications
python scripts/monitoring/literature_watcher.py --check-all

# Custom query
python scripts/monitoring/literature_watcher.py \
    --query "autism ADHD genomics dataset" \
    --days-back 30

# Save results
python scripts/monitoring/literature_watcher.py --check-all --output data/monitoring/pubs.json
```

### Send Alerts

```bash
# Send daily digest
python scripts/monitoring/alert_system.py --send-digest --email user@example.com

# Check for immediate alerts
python scripts/monitoring/alert_system.py --check-updates

# Test alert system
python scripts/monitoring/alert_system.py --test-alerts
```

### Run as Daemon

```bash
# Monitor continuously (check every hour)
python scripts/monitoring/update_scanner.py --daemon --interval 3600

# Monitor literature daily
python scripts/monitoring/literature_watcher.py --daemon --interval 86400

# Run in background
nohup python scripts/monitoring/update_scanner.py --daemon --interval 3600 > logs/monitoring.log 2>&1 &
```

## Automated Scheduling

### Using Cron

Add to crontab (`crontab -e`):

```bash
# Check for updates every 6 hours
0 */6 * * * cd /path/to/project && python scripts/monitoring/update_scanner.py --check-all

# Check literature daily at 8 AM
0 8 * * * cd /path/to/project && python scripts/monitoring/literature_watcher.py --check-all

# Send daily digest at 9 AM
0 9 * * * cd /path/to/project && python scripts/monitoring/alert_system.py --send-digest --email user@example.com

# Weekly summary on Monday 9 AM
0 9 * * 1 cd /path/to/project && python scripts/monitoring/alert_system.py --send-digest --hours-back 168 --email user@example.com
```

### Using systemd

Create service file `/etc/systemd/system/audhd-monitoring.service`:

```ini
[Unit]
Description=AuDHD Study Data Monitoring
After=network.target

[Service]
Type=simple
User=researcher
WorkingDirectory=/path/to/AuDHD_Correlation_Study
ExecStart=/usr/bin/python3 scripts/monitoring/update_scanner.py --daemon --interval 3600
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable audhd-monitoring
sudo systemctl start audhd-monitoring
sudo systemctl status audhd-monitoring
```

## Monitoring Targets

### High Priority (Check Hourly)

| Source | Updates | Alert Type |
|--------|---------|------------|
| SFARI Base | Quarterly | Immediate |
| UK Biobank | Quarterly | Immediate |
| ABCD Study | Annual | Immediate |

### Medium Priority (Check Daily)

| Source | Updates | Alert Type |
|--------|---------|------------|
| NIH NDA | Continuous | Digest |
| dbGaP | Continuous | Digest |
| GEO | Continuous | Digest |
| ClinicalTrials.gov | Continuous | Digest |
| MetaboLights | Weekly | Digest |
| Literature | Daily | Digest |

### Low Priority (Check Weekly)

| Source | Updates | Alert Type |
|--------|---------|------------|
| ArrayExpress | Continuous | Digest |
| PGC Website | Rare | Digest |

## Output Files

### Detected Updates (`data/monitoring/detected_updates.json`)

```json
{
  "generated_date": "2025-09-30T12:00:00",
  "total_updates": 5,
  "updates": [
    {
      "source": "SFARI_Base",
      "dataset_name": "SPARK Release v5.0",
      "update_type": "new_release",
      "detected_date": "2025-09-30T11:30:00",
      "release_date": "2025-09-30",
      "version": "5.0",
      "description": "New SPARK data release with 10,000 additional samples",
      "url": "https://base.sfari.org/spark-release-5",
      "priority": "high",
      "action_required": true
    }
  ]
}
```

### New Publications (`data/monitoring/new_publications.json`)

```json
{
  "generated_date": "2025-09-30T12:00:00",
  "total_publications": 3,
  "publications": [
    {
      "source": "PubMed",
      "pubmed_id": "38234567",
      "doi": "10.1038/s41586-025-12345-6",
      "title": "Multi-omics analysis of autism spectrum disorder...",
      "authors": ["Smith J", "Jones A", "Brown B"],
      "journal": "Nature",
      "publication_date": "2025-09-28",
      "dataset_mentions": ["dataset", "publicly available", "dbgap"],
      "repository_links": ["https://www.ncbi.nlm.nih.gov/gap/phs001234"],
      "accession_numbers": ["dbGaP:phs001234"],
      "relevance_score": 0.92
    }
  ]
}
```

### Monitoring State (`data/temp/monitoring_state.json`)

Tracks last check times and content hashes:

```json
{
  "SFARI_Base": {
    "last_checked": "2025-09-30T12:00:00",
    "last_update_found": "2025-09-28T14:30:00",
    "check_count": 125,
    "updates_found": 3,
    "content_hash": "a1b2c3d4e5f6..."
  }
}
```

## Alert Examples

### Email Digest

Subject: **AuDHD Study Monitoring Digest - 5 updates, 3 publications**

```
=== Data Update Summary (5 updates) ===

HIGH PRIORITY:
  • SFARI_Base: SPARK Release v5.0
    Type: new_release
    URL: https://base.sfari.org/spark-release-5

  • UK_Biobank: UK Biobank Release 2025.Q3
    Type: version_update
    URL: https://www.ukbiobank.ac.uk/releases/2025-q3

MEDIUM PRIORITY:
  • dbGaP: New ADHD study (phs002345)
    Type: new_data

=== New Publications (3 found) ===

[0.92] Multi-omics analysis of autism spectrum disorder...
  Authors: Smith J, Jones A, Brown B
  Source: PubMed (Nature, 2025-09-28)
  DOI: 10.1038/s41586-025-12345-6
  Accessions: dbGaP:phs001234

[0.85] Metabolomic signatures in ADHD children...
  Authors: Lee C, Wang D
  Source: bioRxiv (2025-09-29)
  DOI: 10.1101/2025.09.29.123456
```

### Slack Alert

```
🚨 HIGH PRIORITY UPDATE

SFARI_Base: SPARK Release v5.0

New SPARK data release with 10,000 additional samples

View Update
```

## Monitoring Workflow

### Daily Workflow

1. **Morning** (9 AM): Receive daily digest email
2. **Review**: Check high and medium priority updates
3. **Action**:
   - For new releases: Add to download queue
   - For publications: Review abstracts and data availability
   - For clinical trials: Check if results include usable data
4. **Update**: Mark alerts as reviewed in tracking system

### Weekly Workflow

1. **Monday**: Receive weekly summary
2. **Review**: All updates from past week
3. **Planning**: Schedule data access applications
4. **Documentation**: Update access tracker

### Immediate Actions

High-priority updates trigger immediate alerts:
- New SPARK/UK Biobank releases → Apply for access within 24 hours
- Major version updates → Check compatibility with existing pipelines
- New dbGaP studies → Review and add to application queue

## Troubleshooting

### No Updates Detected

```bash
# Check state file
cat data/temp/monitoring_state.json

# Manually check source
curl -I https://base.sfari.org/news/feed/

# Clear state to force recheck
rm data/temp/monitoring_state.json
```

### Email Alerts Not Sending

```bash
# Test email configuration
python scripts/monitoring/alert_system.py --test-alerts

# Check SMTP settings
# Verify app-specific password for Gmail
# Check firewall/network settings
```

### RSS Feed Errors

```bash
# Validate feed URL
curl https://base.sfari.org/news/feed/

# Check for feed format changes
# Update feedparser library: pip install --upgrade feedparser
```

### Rate Limiting

NCBI E-utilities rate limits:
- 3 requests/second without API key
- 10 requests/second with API key

Solution: Add delays between requests or get API key

## Best Practices

### Configuration

1. **Start conservative**: Check hourly initially, then adjust
2. **Filter keywords**: Avoid false positives with specific keywords
3. **Test alerts**: Verify email/Slack before enabling
4. **Set priorities**: Focus on datasets you actively use

### Monitoring

1. **Review regularly**: Check digests daily
2. **Update config**: Add new sources as needed
3. **Track actions**: Document what you do with each alert
4. **Clean up**: Archive old alerts monthly

### Integration

Link monitoring with other systems:

```python
# After detecting update, add to download queue
from scripts.pipeline.download_manager import DownloadManager
from scripts.monitoring.update_scanner import UpdateScanner

scanner = UpdateScanner()
updates = scanner.scan_all_sources()

manager = DownloadManager()
for update in updates:
    if update.priority == 'high':
        # Add to download queue
        manager.add_task(...)
```

## Performance

### Resource Usage

- CPU: Minimal (<1% during checks)
- Memory: ~50-100 MB
- Network: ~10-50 MB/day (depending on sources)
- Disk: ~1 GB/year (logs and alerts)

### Optimization

- Use content hashing to avoid re-processing unchanged pages
- Implement exponential backoff for failed requests
- Cache API responses for short periods
- Batch database queries

## Security

### Credentials

- Store passwords in environment variables, not config files
- Use app-specific passwords for email
- Rotate Slack webhooks periodically
- Limit permissions on config files (`chmod 600`)

### API Access

- Use read-only API keys when possible
- Respect rate limits and terms of service
- Identify monitoring bot with proper User-Agent
- Don't store sensitive data in logs

## Future Enhancements

Planned additions:
- **Machine learning**: Predict relevant updates
- **Natural language processing**: Better publication filtering
- **Automated downloads**: Trigger downloads on detection
- **Dashboard UI**: Web interface for monitoring
- **Mobile alerts**: Push notifications to phones
- **Integration**: Connect with project management tools
- **Analytics**: Trends in data releases
- **Collaboration**: Share alerts with team members

## Support

For questions or issues:
1. Check this README
2. Review configuration file
3. Check logs: `logs/monitoring.log`
4. Verify source URLs are still valid
5. Open GitHub issue

---

**Version**: 1.0
**Last Updated**: 2025-09-30
**Maintained by**: AuDHD Correlation Study Team