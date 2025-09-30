#!/bin/bash
# Quick status check script

echo "================================================================================"
echo "DOWNLOAD STATUS - $(date '+%H:%M:%S')"
echo "================================================================================"

echo ""
echo "Active processes:"
ps aux | grep -E "(download|scrape)" | grep -v grep | wc -l | xargs echo "  Running processes:"

echo ""
echo "Downloaded so far:"
echo "  GEO data:"
du -sh data/raw/geo 2>/dev/null | awk '{print "    "$1}' || echo "    Starting..."

echo "  SRA data:"
du -sh data/raw/sra 2>/dev/null | awk '{print "    "$1}' || echo "    Starting..."

echo "  Papers:"
du -sh data/papers 2>/dev/null | awk '{print "    "$1}' || echo "    Starting..."

echo ""
echo "Recent activity (last 5 log lines):"
tail -5 logs/downloads/*.log 2>/dev/null | tail -5

echo ""
echo "================================================================================"
echo "For live monitoring: python scripts/download_tracker.py"
echo "================================================================================"