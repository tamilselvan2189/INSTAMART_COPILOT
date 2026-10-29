#!/usr/bin/env python
"""
run_all.py - Master script to run the entire Instamart Copilot project
Executes all parts (A, B, C, D) in sequence with validation
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def run_command(script_path):
    """Execute a Python script and return status"""
    try:
        print(f"Running: {script_path}")
        env = os.environ.copy()
        # Ensure project root is in PYTHONPATH
        env['PYTHONPATH'] = str(Path.cwd()) + ':' + env.get('PYTHONPATH', '')
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(Path.cwd()),
            env=env,
            check=True,
            capture_output=False
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR running {script_path}: {e}")
        return False

def verify_outputs():
    """Verify all required output files exist"""
    print_header("FINAL VERIFICATION")
    
    required_files = {
        'Part A': [
            'data/clean/customers.csv',
            'data/clean/orders.csv',
            'data/clean/products.csv',
            'data/instamart.db',
            'sql/queries.sql',
        ],
        'Part B': [
            'outputs/customer_segments.csv',
            'outputs/retention_list.csv',
            'outputs/lapse_model.joblib',
            'outputs/lapse_logreg.joblib',
            'outputs/lapse_rf.joblib',
            'outputs/lapse_preprocessor.joblib',
        ],
        'Part C': [
            'outputs/triage_results.csv',
            'outputs/triage_budget_demo.csv',
        ],
        'Part D': [
            'outputs/agent_runs.jsonl',
            'outputs/refund_ledger.jsonl',
        ]
    }
    
    total_files = 0
    found_files = 0
    
    for part, files in required_files.items():
        print(f"{part}:")
        for file in files:
            total_files += 1
            file_path = Path(file)
            if file_path.exists():
                size = file_path.stat().st_size / 1024  # KB
                print(f"  ✓ {file} ({size:.1f} KB)")
                found_files += 1
            else:
                print(f"  ✗ {file} (MISSING)")
    
    print(f"\n{'='*70}")
    print(f"RESULT: {found_files}/{total_files} files found")
    print(f"{'='*70}\n")
    
    if found_files == total_files:
        print("🎉 PROJECT EXECUTION COMPLETE! ALL FILES GENERATED.\n")
        return True
    else:
        print(f"⚠️  {total_files - found_files} file(s) missing. Check errors above.\n")
        return False

def main():
    """Main execution flow"""
    project_root = Path.cwd()
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║     INSTAMART COPILOT: COMPLETE PROJECT RUNNER                      ║
║                                                                      ║
║     Executing 4-Part AI System:                                     ║
║     • Part A: Data Foundation & SQL Analysis                        ║
║     • Part B: Customer Segmentation & Lapse Prediction              ║
║     • Part C: Support Ticket Triage with Guardrails                 ║
║     • Part D: Policy RAG & Autonomous Refund Agent                  ║
║                                                                      ║
║     Runtime: ~30-45 seconds                                         ║
║     Output: 18 files (data/, outputs/, sql/, state/)                ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    # Step 1: Generate synthetic data
    print_header("STEP 1: GENERATE SYNTHETIC DATA")
    if not run_command('generate_data.py'):
        print("❌ Data generation failed. Aborting.")
        return False
    print("✓ Data generation complete!")
    
    # Step 2: Part A - Data Foundation
    print_header("STEP 2: PART A - DATA FOUNDATION")
    
    print("\n[2.1] Running Part A implementation...")
    if not run_command('part_a/part_a_implementation.py'):
        print("❌ Part A implementation failed. Aborting.")
        return False
    print("✓ Part A implementation complete!")
    
    print("\n[2.2] Validating Part A with SQL queries...")
    if not run_command('part_a/run_queries_and_validate.py'):
        print("❌ Part A validation failed. Aborting.")
        return False
    print("✓ Part A validation complete!")
    
    # Step 3: Part B - Customer Intelligence
    print_header("STEP 3: PART B - CUSTOMER INTELLIGENCE")
    
    print("\n[3.1] RFM Segmentation...")
    if not run_command('part_b/rfm_segmentation.py'):
        print("❌ RFM segmentation failed. Aborting.")
        return False
    print("✓ RFM segmentation complete!")
    
    print("\n[3.2] Lapse Prediction Model...")
    if not run_command('part_b/lapse_model.py'):
        print("❌ Lapse model training failed. Aborting.")
        return False
    print("✓ Lapse model training complete!")
    
    print("\n[3.3] Retention List Generation...")
    if not run_command('part_b/retention_list.py'):
        print("❌ Retention list generation failed. Aborting.")
        return False
    print("✓ Retention list complete!")
    
    # Step 4: Part C - Support Triage
    print_header("STEP 4: PART C - SUPPORT TICKET TRIAGE")
    
    if not run_command('part_c/support_ticket_triage.py'):
        print("❌ Ticket triage failed. Aborting.")
        return False
    print("✓ Ticket triage complete!")
    
    # Step 5: Part D - Policy RAG & Refund Agent
    print_header("STEP 5: PART D - POLICY RAG & REFUND AGENT")
    
    print("\n[5.1] Policy RAG System...")
    if not run_command('part_d/policy_rag.py'):
        print("❌ Policy RAG failed. Aborting.")
        return False
    print("✓ Policy RAG complete!")
    
    print("\n[5.2] Autonomous Refund Agent...")
    if not run_command('part_d/refund_agent.py'):
        print("❌ Refund agent failed. Aborting.")
        return False
    print("✓ Refund agent execution complete!")
    
    print("\n[5.3] Ledger Export...")
    if not run_command('part_d/ledger_export.py'):
        print("❌ Ledger export failed. Aborting.")
        return False
    print("✓ Ledger export complete!")
    
    # Final verification
    success = verify_outputs()
    
    if success:
        print("✅ ALL PARTS COMPLETED SUCCESSFULLY!\n")
        return True
    else:
        print("⚠️  EXECUTION COMPLETED WITH WARNINGS\n")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
