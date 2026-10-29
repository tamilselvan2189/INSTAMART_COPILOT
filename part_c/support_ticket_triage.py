"""
Support Ticket Triage Pipeline (Part C)

Safe, resilient, and cost-aware ticket classification using guardrails,
personal information redaction, retry logic, and intelligent routing.
"""

import re
import json
import time
import random
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

# Import the mock LLM
import sys
sys.path.insert(0, str(Path.cwd()))
from mock_llm import chat, TransientLLMError, reset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== CONSTANTS =====
MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
ALLOWED_CATEGORIES = {'billing', 'delivery', 'technical', 'fraud'}

# ===== TOKEN ESTIMATION =====
def est_tokens(text: str) -> int:
    """
    Estimate tokens using rough word-to-token ratio.
    Typical English: ~1.3 tokens per word.
    """
    if not text or pd.isna(text) or not isinstance(text, str):
        return 0
    words = len(str(text).split())
    return max(1, int(words * 1.3))


# ===== INPUT GUARDRAILS =====
def validate_ticket_input(text: str) -> Tuple[bool, str, Optional[int]]:
    """
    Validate ticket input and return (is_valid, status, token_count).
    
    Returns:
        - (False, 'skipped_empty', None) if empty/whitespace
        - (False, 'flagged_long', est_tokens) if > MAX_INPUT_TOKENS
        - (True, None, est_tokens) otherwise
    """
    if pd.isna(text) or not text or not str(text).strip():
        return False, 'skipped_empty', None
    
    tokens = est_tokens(text)
    if tokens > MAX_INPUT_TOKENS:
        return False, 'flagged_long', tokens
    
    return True, None, tokens


# ===== REDACTION =====
def redact_pii(text: str) -> Tuple[str, Dict[str, int]]:
    """
    Redact personal information in order: email, phone, order_id.
    
    Returns:
        (redacted_text, substitution_counts)
    """
    if pd.isna(text) or not text or not isinstance(text, str):
        return str(text) if text else '', {'email': 0, 'phone': 0, 'order_id': 0}
    
    counts = {'email': 0, 'phone': 0, 'order_id': 0}
    
    # Email: [\w.+-]+@[\w-]+\.[\w.]+
    email_pattern = r'[\w.+-]+@[\w-]+\.[\w.]+'
    email_matches = re.findall(email_pattern, text)
    counts['email'] = len(email_matches)
    text = re.sub(email_pattern, '[EMAIL]', text)
    
    # Phone: \b[6-9]\d{9}\b (10-digit Indian numbers)
    phone_pattern = r'\b[6-9]\d{9}\b'
    phone_matches = re.findall(phone_pattern, text)
    counts['phone'] = len(phone_matches)
    text = re.sub(phone_pattern, '[PHONE]', text)
    
    # Order ID: ORD-\d{6}
    order_pattern = r'ORD-\d{6}'
    order_matches = re.findall(order_pattern, text)
    counts['order_id'] = len(order_matches)
    text = re.sub(order_pattern, '[ORDER_ID]', text)
    
    return text, counts


def verify_redaction(text: str) -> bool:
    """
    Verify that no PII patterns remain in redacted text.
    """
    if not text or not isinstance(text, str):
        return True  # Non-string values are considered "redacted"
    
    email_pattern = r'[\w.+-]+@[\w-]+\.[\w.]+'
    phone_pattern = r'\b[6-9]\d{9}\b'
    order_pattern = r'ORD-\d{6}'
    
    has_email = bool(re.search(email_pattern, text))
    has_phone = bool(re.search(phone_pattern, text))
    has_order = bool(re.search(order_pattern, text))
    
    return not (has_email or has_phone or has_order)


# ===== SYSTEM PROMPTS =====
ZERO_SHOT_PROMPT = """You are a support ticket CLASSIFY assistant. Your job is to categorize customer support tickets into exactly one of these four categories:
- billing: Payment issues, refunds, invoices, charges
- delivery: Shipping, package delivery, logistics
- technical: App crashes, login issues, payment page errors, OTP problems
- fraud: Unauthorized transactions, account compromise, fraudulent activity

You must respond with ONLY a JSON object (no other text) in this format:
{"category": "...", "reason": "..."}

The category must be exactly one of: billing, delivery, technical, fraud.
Respond only with the JSON object."""

ZERO_SHOT_PROMPT_NAME = "zero_shot"

FEW_SHOT_PROMPT = """You are a support ticket CLASSIFY assistant. Your job is to categorize customer support tickets into exactly one of these four categories:
- billing: Payment issues, refunds, invoices, charges
- delivery: Shipping, package delivery, logistics
- technical: App crashes, login issues, payment page errors, OTP problems
- fraud: Unauthorized transactions, account compromise, fraudulent activity

You must respond with ONLY a JSON object (no other text) in this format:
{"category": "...", "reason": "..."}

The category must be exactly one of: billing, delivery, technical, fraud.

EXAMPLES:
1. "I was charged twice for my order, please refund the extra payment."
   {"category": "billing", "reason": "Customer reports duplicate charge and requests refund"}

2. "Someone hacked my account and used my wallet balance without permission."
   {"category": "fraud", "reason": "Unauthorized account access and fraudulent transaction"}

3. "The rider marked my package as delivered but I never received it."
   {"category": "delivery", "reason": "Package marked delivered but not received by customer"}

Respond only with the JSON object."""

FEW_SHOT_PROMPT_NAME = "few_shot"


# ===== RETRY LOGIC WITH EXPONENTIAL BACKOFF =====
def retry_with_backoff(ticket_id: str, prompt_type: str, redacted_text: str, 
                       max_retries: int = MAX_RETRIES) -> Optional[Dict]:
    """
    Retry LLM call with exponential backoff and jitter.
    
    Wait formula: BASE_DELAY × 2^n × random.uniform(0.9, 1.15)
    where n is the retry attempt (0, 1, 2)
    """
    last_error = None
    
    system_prompt = ZERO_SHOT_PROMPT if prompt_type == "zero_shot" else FEW_SHOT_PROMPT
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": redacted_text}
    ]
    
    for attempt in range(max_retries):
        try:
            response = chat(messages=messages)
            return response
        except TransientLLMError as e:
            last_error = e
            wait_time = BASE_DELAY * (2 ** attempt) * random.uniform(0.9, 1.15)
            logger.info(f"Retry for ticket {ticket_id}: attempt {attempt}, wait {wait_time:.4f}s")
            time.sleep(wait_time)
    
    # All retries exhausted
    raise last_error


# ===== OUTPUT VALIDATION =====
def validate_output(response_text: str) -> Tuple[str, str]:
    """
    Validate LLM output and return (status, category).
    
    Returns:
        - ('classified_json', category) if valid JSON with allowed category
        - ('classified_keyword', category) if keyword extraction finds exactly one allowed category
        - ('flagged_invalid_output', None) otherwise
    """
    # Try JSON parsing first
    try:
        parsed = json.loads(response_text)
        if isinstance(parsed, dict) and 'category' in parsed:
            category = parsed['category']
            if category in ALLOWED_CATEGORIES:
                return 'classified_json', category
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Try keyword extraction: extract lowercase words
    words = re.findall(r'[a-z]+', response_text.lower())
    matching_categories = [w for w in words if w in ALLOWED_CATEGORIES]
    
    if len(matching_categories) == 1:
        return 'classified_keyword', matching_categories[0]
    
    # Invalid output
    return 'flagged_invalid_output', None


# ===== MAIN TRIAGE LOGIC =====
@dataclass
class TriageResult:
    ticket_id: str
    customer_id: str
    status: str
    category: Optional[str] = None
    reason: Optional[str] = None
    tokens_used: int = 0
    segment_name: Optional[str] = None
    priority: str = 'NORMAL'
    route: str = 'support_queue'
    pii_counts: Dict[str, int] = None
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        if self.pii_counts:
            del d['pii_counts']
        return d


def triage_ticket(
    ticket_id: str,
    customer_id: str,
    text: str,
    true_category: str,
    prompt_type: str = "zero_shot",
    token_budget: Optional[int] = None,
    current_tokens: int = 0,
    segments_df: Optional[pd.DataFrame] = None
) -> Tuple[TriageResult, int]:
    """
    Triage a single ticket.
    
    Returns:
        (result, tokens_used)
    """
    result = TriageResult(
        ticket_id=ticket_id,
        customer_id=customer_id,
        status='pending',
        pii_counts={'email': 0, 'phone': 0, 'order_id': 0}
    )
    
    # Step 1: Input validation
    is_valid, skip_status, token_count = validate_ticket_input(text)
    
    if not is_valid:
        result.status = skip_status
        logger.info(f"{ticket_id}: {skip_status}")
        return result, 0
    
    # Step 2: Check token budget before calling model
    if token_budget and current_tokens >= token_budget:
        result.status = 'budget_exhausted'
        logger.info(f"{ticket_id}: budget_exhausted (current={current_tokens}, budget={token_budget})")
        return result, 0
    
    # Step 3: Redact PII
    redacted_text, pii_counts = redact_pii(text)
    result.pii_counts = pii_counts
    
    # Verify redaction
    assert verify_redaction(redacted_text), f"Redaction failed for {ticket_id}: PII still present!"
    
    # Step 4: Call LLM with retry logic
    try:
        response = retry_with_backoff(ticket_id, prompt_type, redacted_text)
    except TransientLLMError as e:
        logger.error(f"{ticket_id}: All retries exhausted: {e}")
        result.status = 'permanent_error'
        return result, 0
    
    # Step 5: Parse response
    response_text = response.choices[0].message.content or ''
    status, category = validate_output(response_text)
    
    result.status = status
    result.category = category
    result.tokens_used = response.usage.total_tokens
    
    # Step 6: Add segment info if available
    if segments_df is not None and category is not None:
        cust_segment = segments_df[segments_df['customer_id'] == customer_id]
        if not cust_segment.empty:
            result.segment_name = cust_segment.iloc[0]['segment_name']
            result.is_premium = bool(cust_segment.iloc[0]['is_premium'])
    
    logger.info(f"{ticket_id}: {status} -> {category} ({result.tokens_used} tokens)")
    
    return result, result.tokens_used


# ===== BATCH PROCESSING =====
def process_tickets(
    tickets_file: str,
    prompt_type: str = "zero_shot",
    token_budget: Optional[int] = None,
    segments_file: Optional[str] = None
) -> Tuple[List[TriageResult], Dict]:
    """
    Process all tickets in the file.
    
    Returns:
        (results, statistics)
    """
    # Load tickets
    df = pd.read_csv(tickets_file)
    
    # Load segments if provided
    segments_df = None
    if segments_file and Path(segments_file).exists():
        segments_df = pd.read_csv(segments_file)
    
    results = []
    total_tokens = 0
    stats = defaultdict(int)
    
    logger.info(f"Starting triage with {prompt_type} prompt, budget={token_budget}")
    
    for idx, row in df.iterrows():
        ticket_id = row['ticket_id']
        customer_id = row['customer_id']
        text = row['text']
        true_category = row['true_category']
        
        result, tokens_used = triage_ticket(
            ticket_id=ticket_id,
            customer_id=customer_id,
            text=text,
            true_category=true_category,
            prompt_type=prompt_type,
            token_budget=token_budget,
            current_tokens=total_tokens,
            segments_df=segments_df
        )
        
        results.append(result)
        total_tokens += tokens_used
        stats[result.status] += 1
    
    # Compute statistics
    stats['total_tokens'] = total_tokens
    stats['processed'] = len([r for r in results if r.status not in ['skipped_empty', 'flagged_long']])
    stats['classified'] = len([r for r in results if r.category is not None])
    
    return results, dict(stats)


# ===== ACCURACY COMPUTATION =====
def compute_accuracy(results: List[TriageResult], tickets_file: str) -> Tuple[float, Dict]:
    """
    Compute accuracy against true_category for classified tickets.
    """
    df = pd.read_csv(tickets_file)
    df['ticket_id'] = df['ticket_id'].astype(str)
    
    # Merge results with true labels
    result_dict = {r.ticket_id: r for r in results}
    
    correct = 0
    total = 0
    confusion = defaultdict(lambda: defaultdict(int))
    
    for idx, row in df.iterrows():
        ticket_id = str(row['ticket_id'])
        true_cat = row['true_category']
        
        if ticket_id in result_dict:
            result = result_dict[ticket_id]
            if result.category is not None:
                total += 1
                pred_cat = result.category
                
                if pred_cat == true_cat:
                    correct += 1
                
                confusion[true_cat][pred_cat] += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, dict(confusion)


# ===== ROUTING =====
def apply_routing(results: List[TriageResult], segments_file: str) -> List[TriageResult]:
    """
    Apply routing rules based on status, category, and segment info.
    """
    # Load segments
    segments_df = None
    if Path(segments_file).exists():
        segments_df = pd.read_csv(segments_file)
        segments_dict = {row['customer_id']: row for _, row in segments_df.iterrows()}
    else:
        segments_dict = {}
    
    for result in results:
        # Step 1: Check if needs human review
        if result.status in ['skipped_empty', 'flagged_long', 'flagged_invalid_output', 'budget_exhausted']:
            result.route = 'human_review'
            result.priority = 'NORMAL'
            continue
        
        # Step 2: Check if fraud
        if result.category == 'fraud':
            result.route = 'security_team'
            result.priority = 'HIGH'
            continue
        
        # Step 3: Check if premium
        is_premium = False
        if result.customer_id in segments_dict:
            is_premium = bool(segments_dict[result.customer_id].get('is_premium', False))
        
        if is_premium:
            result.route = 'premium_desk'
            result.priority = 'HIGH'
            continue
        
        # Step 4: Default to support queue
        result.route = 'support_queue'
        result.priority = 'NORMAL'
    
    return results


# ===== EXPORT =====
def export_results(results: List[TriageResult], output_file: str):
    """
    Export results to CSV.
    """
    data = []
    for result in results:
        row = {
            'ticket_id': result.ticket_id,
            'customer_id': result.customer_id,
            'status': result.status,
            'category': result.category or '',
            'segment_name': result.segment_name or '',
            'priority': result.priority,
            'route': result.route,
            'tokens_used': result.tokens_used
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    logger.info(f"Exported {len(results)} results to {output_file}")


if __name__ == '__main__':
    print("Triage module ready for import")
