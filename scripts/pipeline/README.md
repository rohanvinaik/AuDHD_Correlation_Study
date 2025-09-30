# Automated Data Download and Update Pipeline

Comprehensive pipeline for managing automated downloads with parallel processing, retry logic, validation, and incremental updates.

## Overview

This system provides:
1. **Download Manager**: Parallel downloads with retry logic and resume support
2. **Queue Processor**: Priority-based queue management
3. **Validation Suite**: Checksum verification and format validation
4. **Update Checker**: Incremental update detection and scheduling
5. **Monitoring**: Progress tracking and error logging

### Key Components

1. **Download Manager** (`download_manager.py`)
   - Parallel download processing (configurable workers)
   - Exponential backoff retry logic
   - Resumable downloads
   - Progress tracking with tqdm
   - Checksum verification

2. **Queue Processor** (`queue_processor.py`)
   - Priority-based queuing (critical/high/normal/low)
   - Batch processing
   - Queue persistence

3. **Validation Suite** (`validation_suite.py`)
   - MD5/SHA256 checksum verification
   - ZIP file integrity checks
   - CSV format validation
   - GZIP validation

4. **Update Checker** (`update_checker.py`)
   - Last-Modified header tracking
   - ETag comparison
   - Content-Length changes
   - State persistence

## Installation

```bash
# Python 3.8+
pip install requests tqdm pyyaml pandas
```

## Configuration

### download_config.yaml

```yaml
queue_manager:
  priority_levels: [critical, high, normal, low]
  retry_policy:
    max_attempts: 3
    backoff: exponential
    backoff_multiplier: 2.0
  parallel_downloads: 5
  timeout: 300

storage:
  raw_data: data/raw/
  processed: data/processed/
  temp: data/temp/
  archive: data/archive/

validation:
  checksum: [md5, sha256]
  format_check: true
  completeness: true
  verify_after_download: true

monitoring:
  progress_tracking: true
  error_logging: true
  log_file: logs/download_manager.log
  history_file: logs/download_history.json

datasets:
  - name: PGC_ADHD_GWAS
    url: https://example.com/pgc_adhd_2019.tar.gz
    priority: high
    data_type: genomics
    checksum: abc123def456
    checksum_type: sha256
```

## Usage

### 1. Add Items to Queue

```bash
# Add high-priority download
python scripts/pipeline/queue_processor.py \\
    --add-url https://example.com/data.zip \\
    --name important_dataset \\
    --priority high

# Add multiple items
python scripts/pipeline/queue_processor.py \\
    --add-url https://example.com/genetics.tar.gz \\
    --name genetics_data \\
    --priority critical

python scripts/pipeline/queue_processor.py \\
    --add-url https://example.com/clinical.csv \\
    --name clinical_data \\
    --priority normal
```

### 2. Process Download Queue

```bash
# Process queue with 5 parallel downloads
python scripts/pipeline/download_manager.py \\
    --config configs/download_config.yaml \\
    --parallel 5

# Single download with checksum
python scripts/pipeline/download_manager.py \\
    --add-url https://example.com/data.tar.gz \\
    --name my_dataset \\
    --priority high \\
    --checksum abc123def456789
```

### 3. Validate Downloaded Files

```bash
# Validate with SHA256 checksum
python scripts/pipeline/validation_suite.py \\
    --file data/raw/dataset.zip \\
    --checksum abc123def456 \\
    --checksum-type sha256

# Validate without checksum (format only)
python scripts/pipeline/validation_suite.py \\
    --file data/raw/dataset.csv
```

### 4. Check for Updates

```bash
# Check all configured datasets
python scripts/pipeline/update_checker.py \\
    --config configs/download_config.yaml \\
    --check-all

# Check specific URL
python scripts/pipeline/update_checker.py \\
    --url https://example.com/data.zip \\
    --name my_dataset
```

## Complete Workflow

### Automated Download Pipeline

```bash
# Step 1: Configure datasets in download_config.yaml
# (Edit configs/download_config.yaml)

# Step 2: Check for updates
python scripts/pipeline/update_checker.py --check-all

# Step 3: Add updated datasets to queue
# (Automated based on update check results)

# Step 4: Process download queue
python scripts/pipeline/download_manager.py \\
    --config configs/download_config.yaml \\
    --parallel 5

# Step 5: Validate all downloads
for file in data/raw/*; do
    python scripts/pipeline/validation_suite.py --file "$file"
done

# Step 6: Review download history
cat logs/download_history.json | jq '.completed'
```

### Scheduled Pipeline (Cron)

```bash
# Daily update check and download at 2 AM
0 2 * * * cd /path/to/project && python scripts/pipeline/update_checker.py --check-all && python scripts/pipeline/download_manager.py --config configs/download_config.yaml
```

## Features

### Parallel Download Processing

Downloads multiple files simultaneously with configurable worker count:

```python
from scripts.pipeline.download_manager import DownloadManager, DownloadTask
from pathlib import Path

manager = DownloadManager(max_workers=5)

# Add tasks
task1 = DownloadTask(
    url='https://example.com/file1.zip',
    name='dataset1',
    output_dir=Path('data/raw'),
    priority='high'
)

task2 = DownloadTask(
    url='https://example.com/file2.tar.gz',
    name='dataset2',
    output_dir=Path('data/raw'),
    priority='normal'
)

manager.add_task(task1)
manager.add_task(task2)

# Process queue
manager.process_queue()

# Get summary
summary = manager.get_summary()
print(f"Downloaded: {summary['completed']} files")
print(f"Total size: {summary['total_bytes_human']}")
```

### Retry Logic with Exponential Backoff

Automatic retry with increasing wait times:

- Attempt 1: Immediate
- Attempt 2: Wait 2 seconds
- Attempt 3: Wait 4 seconds

```python
# Configured via retry_backoff parameter
manager = DownloadManager(
    max_workers=5,
    retry_backoff=2.0  # Backoff multiplier
)
```

### Resumable Downloads

Automatically resumes interrupted downloads using HTTP Range requests:

```python
# Resume is enabled by default
success = manager.download_file(task, resume=True)
```

### Checksum Verification

Supports MD5 and SHA256 checksum verification:

```python
task = DownloadTask(
    url='https://example.com/data.zip',
    name='dataset',
    output_dir=Path('data/raw'),
    checksum='abc123def456',
    checksum_type='sha256'
)

# Verify after download
verified = manager.verify_checksum(task)
```

### Priority Queue Management

Four priority levels (critical > high > normal > low):

```python
from scripts.pipeline.queue_processor import QueueProcessor

processor = QueueProcessor(Path('configs/download_config.yaml'))

# Critical priority (processed first)
processor.add_to_queue(
    url='https://example.com/urgent.zip',
    name='urgent_data',
    priority='critical'
)

# Normal priority
processor.add_to_queue(
    url='https://example.com/regular.zip',
    name='regular_data',
    priority='normal'
)

# Get next batch (prioritized)
batch = processor.get_next_batch(batch_size=5)
```

### Update Detection

Tracks updates using HTTP metadata:

```python
from scripts.pipeline.update_checker import UpdateChecker

checker = UpdateChecker(Path('configs/download_config.yaml'))

# Check for updates
update = checker.check_for_updates(
    url='https://example.com/data.zip',
    name='my_dataset'
)

if update and update['has_update']:
    print(f"Update available: {update['reasons']}")
    # Add to download queue
```

## Output Files

### logs/download_history.json

Complete download history:

```json
{
  "generated_date": "2025-09-30T12:00:00",
  "completed": [
    {
      "url": "https://example.com/dataset1.zip",
      "name": "example_dataset_1",
      "priority": "high",
      "status": "completed",
      "attempts": 1,
      "file_path": "data/raw/example_dataset_1.zip",
      "bytes_downloaded": 524288000,
      "start_time": "2025-09-30T10:00:00",
      "end_time": "2025-09-30T10:15:23",
      "checksum_verified": true
    }
  ],
  "failed": [
    {
      "name": "example_dataset_2",
      "status": "failed",
      "attempts": 3,
      "error_message": "Connection timeout"
    }
  ],
  "pending": [
    {
      "name": "example_dataset_3",
      "status": "pending"
    }
  ]
}
```

### data/temp/update_state.json

Update tracking state:

```json
{
  "PGC_ADHD_GWAS": {
    "last_modified": "Mon, 15 Jan 2024 14:30:00 GMT",
    "etag": "\"abc123def456\"",
    "content_length": "524288000",
    "last_downloaded": "2025-01-15T14:35:00",
    "checked_date": "2025-09-30T08:00:00"
  }
}
```

### logs/download_manager.log

Detailed operation logs:

```
2025-09-30 10:00:15 - INFO - Initialized download manager: 5 workers
2025-09-30 10:00:16 - INFO - Added task: example_dataset (priority: high)
2025-09-30 10:00:17 - INFO - Processing 3 tasks with 5 workers
2025-09-30 10:15:23 - INFO - Download completed: example_dataset
2025-09-30 10:15:24 - INFO - ✓ Checksum verified: example_dataset
2025-09-30 10:15:25 - INFO - Queue processed: 1 completed, 0 failed
```

## Error Handling

### Retry Strategy

```python
# Automatic retry with exponential backoff
# Attempt 1: Immediate
# Attempt 2: Wait 2^1 = 2 seconds
# Attempt 3: Wait 2^2 = 4 seconds

# After max attempts, task is marked as failed
```

### Common Errors

**Connection Timeout:**
```
Error: Connection timeout after 30 seconds
Solution: Increase timeout in config or check network
```

**Checksum Mismatch:**
```
Error: Checksum verification failed
Solution: Re-download file or verify expected checksum
```

**Disk Space:**
```
Error: No space left on device
Solution: Clear temp directory or increase storage
```

## Integration with Other Systems

### Link to Master Registry

```python
# After download, register in master index
from scripts.integration.master_indexer import MasterIndexer
from scripts.pipeline.download_manager import DownloadManager

# Download data
manager = DownloadManager()
# ... download files ...

# Register in master index
indexer = MasterIndexer('data/index/master_sample_registry.db')
indexer.import_dataset(
    dataset_name='PGC_ADHD',
    data_df=pd.read_csv('data/raw/pgc_adhd.csv'),
    id_column='sample_id',
    data_type='genomics'
)
```

### Automated Workflow

```python
#!/usr/bin/env python3
\"\"\"Automated daily data update workflow\"\"\"

from pathlib import Path
from scripts.pipeline.update_checker import UpdateChecker
from scripts.pipeline.download_manager import DownloadManager, DownloadTask

# Check for updates
checker = UpdateChecker(Path('configs/download_config.yaml'))
datasets = checker.config.get('datasets', [])

manager = DownloadManager(max_workers=5)

# Add updated datasets to queue
for dataset in datasets:
    update = checker.check_for_updates(dataset['url'], dataset['name'])

    if update and update['has_update']:
        task = DownloadTask(
            url=dataset['url'],
            name=dataset['name'],
            output_dir=Path('data/raw'),
            priority=dataset['priority'],
            checksum=dataset.get('checksum')
        )
        manager.add_task(task)

# Process downloads
if manager.queue:
    manager.process_queue()
    manager.save_history(Path('logs/download_history.json'))

    # Send notification
    summary = manager.get_summary()
    print(f"Downloaded {summary['completed']} datasets ({summary['total_bytes_human']})")
```

## Best Practices

### Queue Management

1. **Prioritize critical data**: Use `critical` for time-sensitive datasets
2. **Batch similar downloads**: Group datasets by type or source
3. **Monitor queue depth**: Keep queue manageable (< 100 items)
4. **Regular cleanup**: Archive or delete completed downloads

### Download Optimization

1. **Parallel workers**: Set to 3-5 for best performance
2. **Chunk size**: 8192 bytes (default) works for most cases
3. **Timeout**: Increase for large files or slow connections
4. **Resume downloads**: Always enable for large files

### Validation

1. **Always verify checksums**: For data integrity
2. **Format checks**: Validate file structure after download
3. **Completeness**: Check file sizes and record counts
4. **Archive originals**: Keep raw downloads before processing

### Update Scheduling

1. **Daily checks**: Run update checker once per day
2. **Off-peak downloads**: Schedule large downloads at night
3. **Incremental updates**: Only download changed files
4. **Version tracking**: Maintain download history

## Monitoring and Alerting

### Progress Tracking

```python
# Monitor download progress in real-time
from tqdm import tqdm

# Progress bar automatically shows:
# - Download speed
# - ETA
# - Percentage complete
# - Total size
```

### Error Notification

```python
# Check for failures
with open('logs/download_history.json') as f:
    history = json.load(f)

failed = history.get('failed', [])
if failed:
    # Send email/slack notification
    print(f"ALERT: {len(failed)} downloads failed")
    for task in failed:
        print(f"  - {task['name']}: {task['error_message']}")
```

### Disk Usage Monitoring

```bash
# Check storage before large downloads
df -h data/raw/
df -h data/temp/

# Clean temp directory
rm -rf data/temp/*
```

## Troubleshooting

### Downloads Keep Failing

```bash
# Check network connectivity
ping -c 3 example.com

# Test URL manually
curl -I https://example.com/data.zip

# Increase retry attempts
# Edit configs/download_config.yaml:
# retry_policy:
#   max_attempts: 5
```

### Slow Download Speed

```bash
# Reduce parallel workers
python download_manager.py --parallel 2

# Increase chunk size
# Edit download_manager.py:
# chunk_size = 16384
```

### Checksum Verification Fails

```bash
# Recalculate checksum
python -c "
import hashlib
with open('data/raw/file.zip', 'rb') as f:
    print(hashlib.sha256(f.read()).hexdigest())
"

# Compare with expected value
```

## Future Enhancements

Potential additions:
- **S3/Cloud storage**: Direct upload to cloud
- **Email notifications**: Automated alerts
- **Web dashboard**: Real-time monitoring UI
- **Bandwidth throttling**: Rate limiting
- **Compression**: Auto-compress after download
- **Deduplication**: Skip duplicate files
- **Scheduling**: Built-in cron-like scheduler
- **API integration**: RESTful API for remote control

## Support

For questions or issues:
1. Check logs: `logs/download_manager.log`
2. Review download history: `logs/download_history.json`
3. Verify configuration: `configs/download_config.yaml`
4. Test with small file first
5. Open GitHub issue with detailed description

---

**Last updated**: 2025-09-30
**Version**: 1.0
**Maintained by**: AuDHD Correlation Study Team