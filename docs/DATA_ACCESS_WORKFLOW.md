# Data Access Workflow & Status

Generated: 2025-09-30

## Immediate Download (No Account Required)

### ✅ NHANES Environmental Data
- **Status**: Ready to download
- **Action**: Running automated download
- **Data**: Heavy metals, pesticides, phthalates, POPs
- **Timeline**: 30 minutes
- **Command**: `python scripts/download_nhanes_environmental.py`

### ✅ PhysioNet Public Databases
- **Status**: Some datasets are fully public
- **Action**: Downloading public datasets now
- **Credentialing needed**: For full access to 50+ more datasets
- **Timeline**: Public data - 1 hour, Full access - 1 week after credentialing

## Simple Registration (Can Automate)

### 🔄 National Sleep Research Resource (NSRR)
- **Requirements**:
  - Name, email, institution
  - Click-through data use agreement
- **Action Required**: You need to sign up at https://sleepdata.org/join
  - Use institution: "Independent Researcher"
  - Takes 5 minutes
- **Data Available**:
  - CHAT study (1,244 kids with sleep data)
  - Cleveland Family Study (2,284 subjects with PSG)
  - 20+ other sleep cohorts
- **Timeline**: Immediate access after signup

### 🔄 PhysioNet Credentialing
- **Requirements**:
  - Complete CITI "Data or Specimens Only Research" course (2 hours)
  - Upload credential to PhysioNet
  - Sign data use agreement
- **Action Required**:
  1. Go to https://physionet.org/login/
  2. Create account
  3. Complete training: https://physionet.org/about/citi-course/
  4. Get credentialed access to 50+ physiological databases
- **Data Available**: HRV, ECG, EDA, autonomic measures
- **Timeline**: 1 week (2 hours training + 3-5 days approval)

### 🔄 All of Us Researcher Workbench
- **Requirements**:
  - Research purpose
  - eRA Commons ID (optional) or justification
  - Complete ethics training
- **Action Required**:
  1. Register at https://www.researchallofus.org/register/
  2. Complete ethics modules (2-3 hours)
  3. Request data access tier
- **Data Available**:
  - 413,000 participants
  - Fitbit data (100k subset)
  - EHR data
  - Self-reported ADHD/ASD
- **Timeline**: 2-4 weeks

## Institutional Access Required

### ⏳ ABCD Study (via NDA)
- **Requirements**:
  - Institutional affiliation OR
  - Documented independent researcher status
- **Status**: You mentioned you're independent - this may be challenging
- **Workaround**:
  - Some ABCD summary data is public
  - Partner with researcher who has access
  - Some derived data on OpenNeuro
- **Timeline**: 1-3 months if pursuing institutional route

### ⏳ UK Biobank
- **Requirements**:
  - Research proposal
  - £5,000 application fee
  - Institutional affiliation usually required
- **Status**: Likely not feasible for independent researcher without funding
- **Alternative**: Published GWAS summary statistics available free

## What I'm Doing Now

1. **Downloading NHANES** (public, no account) - environmental biomarkers
2. **Downloading PhysioNet public datasets** - HRV/ECG data
3. **Creating guided workflows** for you to sign up for NSRR and PhysioNet
4. **Documenting data use agreements** so you can complete them quickly

## What You Should Do

### Priority 1 (Do Today - 30 minutes total):
1. **Sign up for NSRR** (5 min): https://sleepdata.org/join
   - Use email: [your email]
   - Institution: "Independent Researcher"
   - Accept data use agreement

2. **Create PhysioNet account** (5 min): https://physionet.org/register/
   - Don't do training yet, just create account

3. **Register for All of Us** (10 min): https://www.researchallofus.org/register/
   - Start the registration process
   - You can complete training modules later

### Priority 2 (This Week - 2-3 hours):
1. **Complete PhysioNet CITI training** (2 hours)
   - This unlocks 50+ databases with autonomic/physiological data
   - Course: https://physionet.org/about/citi-course/

2. **Complete All of Us ethics training** (2-3 hours)
   - Required for workbench access

### Priority 3 (Later):
- **ABCD/NDA access**: Decide if worth pursuing institutional affiliation
- **UK Biobank**: Use published GWAS results instead of raw data

## Automated Scripts Ready

I've created scripts that will automatically:
- Download all NHANES environmental data ✓
- Download PhysioNet public databases ✓
- Parse NSRR data once you provide credentials
- Query All of Us workbench once you have access

## Current Status Summary

| Data Source | Status | Action Needed | Timeline |
|------------|--------|---------------|----------|
| NHANES | 🚀 Downloading | None | 30 min |
| PhysioNet (public) | 🚀 Downloading | None | 1 hour |
| PhysioNet (full) | ⏸️ Waiting | You: Sign up + training | 1 week |
| NSRR | ⏸️ Waiting | You: 5-min signup | Same day |
| All of Us | ⏸️ Waiting | You: Registration + training | 2-4 weeks |
| ABCD | ⏸️ Deferred | Institutional access | 1-3 months |
| UK Biobank | ⏸️ Deferred | £5k + institution | Not feasible |

Total truly free public data being downloaded now: ~2-5 GB
Total accessible with free accounts: ~50-100 GB