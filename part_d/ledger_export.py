"""
Part D - File 3: Ledger Export and JSONL Generation

Task 9: Export refund transaction ledger and generate structured JSONL logs
for audit trail and analytics purposes.

Functions:
- export_ledger_from_checkpoints(state_dir, output_dir): Load checkpoints and generate ledger
- generate_jsonl(ledger_data, output_path): Write ledger to JSONL format
- parse_checkpoint_to_ledger_entry(checkpoint_data): Convert checkpoint JSON to ledger entry
- validate_ledger(ledger_data): Verify ledger data integrity
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def parse_checkpoint_to_ledger_entry(checkpoint_data: Dict) -> Dict:
    """
    Convert checkpoint JSON data to a ledger entry.
    
    Args:
        checkpoint_data (dict): Parsed checkpoint JSON
        
    Returns:
        dict: Ledger entry with normalized fields
    """
    try:
        run_id = checkpoint_data.get('run_id', '')
        scenario_id = checkpoint_data.get('scenario_id', '')
        final_status = checkpoint_data.get('final_status', '')
        final_amount = float(checkpoint_data.get('final_amount', 0.0))
        tokens_used = int(checkpoint_data.get('tokens_used', 0))
        iteration = int(checkpoint_data.get('iteration', 0))
        duration_seconds = float(checkpoint_data.get('duration_seconds', 0.0))
        
        # Extract customer/order info from messages
        customer_id = None
        order_id = None
        refund_id = None
        escalation_id = None
        approval_level = None
        
        # Look through tool calls for extracted information
        tool_calls = checkpoint_data.get('tool_calls', [])
        for tool_call in tool_calls:
            result = tool_call.get('result', {})
            
            # Extract customer/order from get_order result
            if tool_call.get('tool_name') == 'get_order' and not result.get('error'):
                customer_id = result.get('customer_id')
                order_id = result.get('order_id')
            
            # Extract refund/escalation info from issue_refund result
            if tool_call.get('tool_name') in ['issue_refund', 'escalate_to_human']:
                if result.get('action') == 'REFUND':
                    refund_id = result.get('refund_id')
                elif result.get('action') == 'ESCALATE':
                    escalation_id = result.get('escalation_id')
                approval_level = result.get('approval_level', '')
        
        # Create ledger entry
        ledger_entry = {
            'timestamp': datetime.now().isoformat(),
            'run_id': run_id,
            'scenario_id': scenario_id,
            'customer_id': customer_id,
            'order_id': order_id,
            'status': final_status,
            'refund_amount_inr': final_amount,
            'refund_id': refund_id,
            'escalation_id': escalation_id,
            'approval_level': approval_level or 'none',
            'tool_calls_count': len(tool_calls),
            'tokens_used': tokens_used,
            'iterations': iteration,
            'duration_seconds': duration_seconds
        }
        
        return ledger_entry
    except Exception as e:
        logger.error(f"Error parsing checkpoint to ledger entry: {e}")
        return None


def export_ledger_from_checkpoints(
    state_dir: str = "state",
    output_dir: str = "outputs"
) -> Tuple[List[Dict], int]:
    """
    Export refund transaction ledger from checkpoint files.
    
    Args:
        state_dir (str): Directory containing checkpoint files (default: "state")
        output_dir (str): Directory for output files (default: "outputs")
        
    Returns:
        tuple: (ledger_entries list, count of successfully processed checkpoints)
    """
    state_path = Path(state_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    ledger_entries = []
    checkpoint_count = 0
    
    if not state_path.exists():
        logger.warning(f"State directory not found: {state_path}")
        return ledger_entries, 0
    
    # Find all checkpoint files
    checkpoint_files = list(state_path.glob("*.checkpoint"))
    logger.info(f"Found {len(checkpoint_files)} checkpoint file(s)")
    
    for checkpoint_file in checkpoint_files:
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            ledger_entry = parse_checkpoint_to_ledger_entry(checkpoint_data)
            if ledger_entry:
                ledger_entries.append(ledger_entry)
                checkpoint_count += 1
                logger.info(f"Processed: {checkpoint_file.name} → {ledger_entry['scenario_id']} {ledger_entry['status']}")
        except Exception as e:
            logger.error(f"Error processing checkpoint {checkpoint_file.name}: {e}")
    
    logger.info(f"Exported {len(ledger_entries)} ledger entries from {checkpoint_count} checkpoints")
    return ledger_entries, checkpoint_count


def generate_jsonl(
    ledger_data: List[Dict],
    output_path: str = "outputs/refund_ledger.jsonl"
) -> Tuple[bool, str]:
    """
    Generate JSONL (JSON Lines) format ledger file for structured logging.
    
    Args:
        ledger_data (list): List of ledger entry dictionaries
        output_path (str): Output file path (default: "outputs/refund_ledger.jsonl")
        
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            for entry in ledger_data:
                f.write(json.dumps(entry) + '\n')
        
        message = f"✓ JSONL ledger generated: {output_file} ({len(ledger_data)} entries)"
        logger.info(message)
        return True, message
    except Exception as e:
        message = f"✗ Error generating JSONL: {e}"
        logger.error(message)
        return False, message


def generate_csv_ledger(
    ledger_data: List[Dict],
    output_path: str = "outputs/refund_ledger.csv"
) -> Tuple[bool, str]:
    """
    Generate CSV format ledger file for easy viewing in spreadsheet applications.
    
    Args:
        ledger_data (list): List of ledger entry dictionaries
        output_path (str): Output file path (default: "outputs/refund_ledger.csv")
        
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        df = pd.DataFrame(ledger_data)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_file, index=False)
        
        message = f"✓ CSV ledger generated: {output_file} ({len(ledger_data)} rows)"
        logger.info(message)
        return True, message
    except Exception as e:
        message = f"✗ Error generating CSV: {e}"
        logger.error(message)
        return False, message


def validate_ledger(ledger_data: List[Dict]) -> Dict:
    """
    Validate ledger data integrity and completeness.
    
    Args:
        ledger_data (list): List of ledger entries
        
    Returns:
        dict: Validation report with counts and statistics
    """
    report = {
        'total_entries': len(ledger_data),
        'entries_with_refund': 0,
        'entries_with_escalation': 0,
        'entries_denied': 0,
        'entries_not_found': 0,
        'entries_fallback': 0,
        'total_refund_amount': 0.0,
        'refund_ids': [],
        'escalation_ids': [],
        'scenarios_processed': set(),
        'missing_fields': 0
    }
    
    required_fields = ['timestamp', 'scenario_id', 'status', 'refund_amount_inr']
    
    for entry in ledger_data:
        # Check for missing required fields
        for field in required_fields:
            if field not in entry or entry[field] is None:
                report['missing_fields'] += 1
        
        # Count by status
        status = entry.get('status', '')
        if status == 'REFUND':
            report['entries_with_refund'] += 1
            report['total_refund_amount'] += entry.get('refund_amount_inr', 0.0)
            if entry.get('refund_id'):
                report['refund_ids'].append(entry['refund_id'])
        elif status == 'ESCALATE':
            report['entries_with_escalation'] += 1
            if entry.get('escalation_id'):
                report['escalation_ids'].append(entry['escalation_id'])
        elif status == 'DENIED':
            report['entries_denied'] += 1
        elif status == 'NOT_FOUND':
            report['entries_not_found'] += 1
        elif 'FALLBACK' in status:
            report['entries_fallback'] += 1
        
        report['scenarios_processed'].add(entry.get('scenario_id', 'UNKNOWN'))
    
    # Convert set to list for serialization
    report['scenarios_processed'] = sorted(list(report['scenarios_processed']))
    report['refund_ids'] = list(set(report['refund_ids']))  # Deduplicate
    report['escalation_ids'] = list(set(report['escalation_ids']))  # Deduplicate
    
    return report


def generate_summary_report(
    ledger_data: List[Dict],
    output_path: str = "outputs/ledger_summary.json"
) -> Tuple[bool, Dict]:
    """
    Generate a summary report of ledger data.
    
    Args:
        ledger_data (list): List of ledger entries
        output_path (str): Output file path (default: "outputs/ledger_summary.json")
        
    Returns:
        tuple: (success: bool, report: dict)
    """
    try:
        report = validate_ledger(ledger_data)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✓ Summary report generated: {output_file}")
        return True, report
    except Exception as e:
        logger.error(f"Error generating summary report: {e}")
        return False, {}


def export_and_generate_ledger(
    state_dir: str = "state",
    output_dir: str = "outputs"
) -> Dict:
    """
    Complete ledger export workflow:
    1. Export from checkpoints
    2. Generate JSONL format
    3. Generate CSV format
    4. Validate and generate summary report
    
    Args:
        state_dir (str): Directory with checkpoint files
        output_dir (str): Output directory for ledger files
        
    Returns:
        dict: Results dictionary with success status and file paths
    """
    results = {
        'success': False,
        'checkpoints_processed': 0,
        'ledger_entries': 0,
        'files_generated': [],
        'validation_report': {}
    }
    
    # Step 1: Export from checkpoints
    ledger_data, checkpoint_count = export_ledger_from_checkpoints(state_dir, output_dir)
    results['checkpoints_processed'] = checkpoint_count
    results['ledger_entries'] = len(ledger_data)
    
    if len(ledger_data) == 0:
        logger.warning("No ledger entries found")
        return results
    
    output_path = Path(output_dir)
    
    # Step 2: Generate JSONL
    success, msg = generate_jsonl(ledger_data, str(output_path / "refund_ledger.jsonl"))
    if success:
        results['files_generated'].append("refund_ledger.jsonl")
        logger.info(msg)
    
    # Step 3: Generate CSV
    success, msg = generate_csv_ledger(ledger_data, str(output_path / "refund_ledger.csv"))
    if success:
        results['files_generated'].append("refund_ledger.csv")
        logger.info(msg)
    
    # Step 4: Validate and generate summary
    success, report = generate_summary_report(ledger_data, str(output_path / "ledger_summary.json"))
    if success:
        results['files_generated'].append("ledger_summary.json")
        results['validation_report'] = report
        logger.info("✓ Ledger validation complete")
    
    results['success'] = len(results['files_generated']) == 3
    return results


if __name__ == "__main__":
    # Example usage
    results = export_and_generate_ledger()
    print("\n" + "="*60)
    print("LEDGER EXPORT RESULTS")
    print("="*60)
    print(f"Checkpoints processed: {results['checkpoints_processed']}")
    print(f"Ledger entries: {results['ledger_entries']}")
    print(f"Files generated: {', '.join(results['files_generated'])}")
    print(f"Success: {results['success']}")
    if results['validation_report']:
        print("\nValidation Report:")
        print(json.dumps(results['validation_report'], indent=2))
