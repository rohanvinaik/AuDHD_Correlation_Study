#!/bin/bash
# Test script for configuration system

echo "================================"
echo "Testing Configuration System"
echo "================================"
echo ""

# Test 1: Show default config
echo "Test 1: Show default configuration"
echo "-----------------------------------"
python scripts/run_pipeline.py --cfg job | head -50
echo ""

# Test 2: Override config groups
echo "Test 2: Override config groups (data=minimal preprocessing=minimal)"
echo "--------------------------------------------------------------------"
python scripts/run_pipeline.py --cfg job data=minimal preprocessing=minimal | grep -A 5 "data:"
echo ""

# Test 3: Override individual parameters
echo "Test 3: Override individual parameters"
echo "---------------------------------------"
python scripts/run_pipeline.py --cfg job \
  clustering.params.min_cluster_size=25 \
  integration.params.n_factors=10 | grep -A 3 "clustering:"
echo ""

# Test 4: Use different compute environment
echo "Test 4: Use HPC compute configuration"
echo "--------------------------------------"
python scripts/run_pipeline.py --cfg job compute=hpc | grep -A 10 "compute:"
echo ""

# Test 5: Use experiment config
echo "Test 5: Use experiment configuration (quick_test)"
echo "--------------------------------------------------"
python scripts/run_pipeline.py --cfg job experiment=quick_test | head -30
echo ""

# Test 6: Test dry run
echo "Test 6: Dry run mode"
echo "--------------------"
python scripts/run_pipeline.py pipeline.dry_run=true | grep -A 10 "DRY RUN"
echo ""

# Test 7: Show available config groups
echo "Test 7: Show available config groups"
echo "-------------------------------------"
python scripts/run_pipeline.py --help | grep -A 25 "Configuration groups"
echo ""

echo "================================"
echo "All tests completed!"
echo "================================"