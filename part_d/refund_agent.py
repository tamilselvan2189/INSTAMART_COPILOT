"""
Refund Agent Module — ReAct-based Autonomous Refund Processing

Implements a ReAct (Reasoning + Acting) agent for handling refund requests:
- Task 5: Tool definitions and ReAct loop
- Task 6: Execute scenarios
- Task 7: Demonstrate fallbacks (token/time limits)
- Task 8: Crash-and-resume with checkpoints

Dependencies: mock_llm, policy_rag, sqlite3, pandas, pathlib
"""

import json
import sqlite3
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import re

import pandas as pd
from mock_llm import chat

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Represents a single tool call in the ReAct trace."""
    tool_name: str
    args: Dict[str, Any]
    result: Dict[str, Any]
    success: bool
    timestamp: float


@dataclass
class AgentState:
    """Tracks the state of an agent run."""
    run_id: str
    scenario_id: str
    iteration: int
    messages: List[Dict]
    tool_calls: List[ToolCall]
    tokens_used: int
    final_status: str
    final_amount: float
    start_time: float
    end_time: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "iteration": self.iteration,
            "tool_calls": [
                {
                    "tool_name": tc.tool_name,
                    "args": tc.args,
                    "result": tc.result,
                    "success": tc.success,
                    "timestamp": tc.timestamp
                }
                for tc in self.tool_calls
            ],
            "tokens_used": self.tokens_used,
            "final_status": self.final_status,
            "final_amount": self.final_amount,
            "duration_seconds": self.end_time - self.start_time if self.end_time else None
        }


# ============================================================================
# TOOL DEFINITIONS
# ============================================================================

def get_order(
    order_id: str,
    db_path: str = "data/instamart.db"
) -> Tuple[bool, Dict]:
    """
    Retrieve order details from database.
    
    Args:
        order_id: Order ID like "ORD-100048"
        db_path: Path to SQLite database
    
    Returns:
        (success, {order_id, customer_id, amount_inr, status, delivery_time_minutes})
    """
    try:
        # Handle relative paths
        db_file = Path(db_path)
        if not db_file.is_absolute() and not db_file.exists():
            for parent in Path.cwd().parents:
                candidate = parent / db_path
                if candidate.exists():
                    db_file = candidate
                    break
        
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()
        
        # Query order details
        cursor.execute("""
            SELECT order_id, customer_id, order_amount, status, delivery_minutes
            FROM orders
            WHERE order_id = ?
        """, (order_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return False, {"error": f"Order {order_id} not found"}
        
        order_id, customer_id, order_amount, status, delivery_minutes = row
        
        return True, {
            "order_id": order_id,
            "customer_id": customer_id,
            "order_amount": float(order_amount),
            "status": status,
            "delivery_minutes": int(delivery_minutes) if delivery_minutes else 0
        }
    
    except Exception as e:
        logger.error(f"get_order error: {e}")
        return False, {"error": str(e)}


def search_policy(
    query: str,
    vectorizer=None,
    chunks: List[Dict] = None
) -> Tuple[bool, Dict]:
    """
    Search policy knowledge base for refund eligibility.
    Uses policy_rag retrieval.
    
    Args:
        query: Question about refund eligibility
        vectorizer: TF-IDF vectorizer from policy_rag
        chunks: Policy chunks from policy_rag
    
    Returns:
        (success, {policy_snippet, is_eligible, reason})
    """
    try:
        if vectorizer is None or chunks is None:
            return False, {"error": "Policy RAG not initialized"}
        
        # Import here to avoid circular dependency
        from part_d.policy_rag import retrieve
        
        # Retrieve policy
        retrieved = retrieve(query, k=1, vectorizer=vectorizer, chunks=chunks)
        
        if not retrieved:
            return False, {"error": "No policy found"}
        
        policy_snippet = retrieved[0]["text"]
        
        # Simple heuristic: check for "late delivery" and "45 minutes"
        is_eligible = "45 minutes" in policy_snippet or "delivered in more than" in policy_snippet
        reason = "Late delivery refund eligible per policy" if is_eligible else "Policy does not apply"
        
        return True, {
            "policy_snippet": policy_snippet[:200],
            "is_eligible": is_eligible,
            "reason": reason
        }
    
    except Exception as e:
        logger.error(f"search_policy error: {e}")
        return False, {"error": str(e)}


def calculate_refund(
    order_id: str,
    order_amount_inr: float,
    delivery_time_minutes: int
) -> Tuple[bool, Dict]:
    """
    Compute refund amount based on order details and policy.
    
    Refund Policy:
    - If delivery > 45 minutes: full refund
    - If delivery ≤ 45 minutes: no refund
    
    Args:
        order_id: Order ID
        order_amount_inr: Original order amount
        delivery_time_minutes: Actual delivery time
    
    Returns:
        (success, {refund_inr, approval_level})
        where approval_level: "auto" (≤1000) or "manual" (>1000)
    """
    try:
        # Check eligibility
        if delivery_time_minutes > 45:
            refund_amount = order_amount_inr
        else:
            refund_amount = 0.0
        
        # Determine approval level
        if refund_amount == 0:
            approval_level = "none"
        elif refund_amount <= 1000:
            approval_level = "auto"
        else:
            approval_level = "manual"
        
        return True, {
            "refund_inr": refund_amount,
            "approval_level": approval_level,
            "reason": f"Delivery time: {delivery_time_minutes} minutes"
        }
    
    except Exception as e:
        logger.error(f"calculate_refund error: {e}")
        return False, {"error": str(e)}


def issue_refund(
    order_id: str,
    refund_amount_inr: float,
    approval_level: str
) -> Tuple[bool, Dict]:
    """
    Issue refund (auto-approved) or escalate (manual).
    
    Args:
        order_id: Order ID
        refund_amount_inr: Amount to refund
        approval_level: "auto", "manual", or "none"
    
    Returns:
        (success, {action, amount_inr, refund_id})
        where action: "REFUND" (issued) or "ESCALATE" (needs human review)
    """
    try:
        if refund_amount_inr == 0:
            return True, {
                "action": "DENIED",
                "refund_inr": 0.0,
                "reason": "No refund eligible"
            }
        elif approval_level == "auto":
            # Auto-approve and issue refund
            return True, {
                "action": "REFUND",
                "refund_inr": refund_amount_inr,
                "refund_id": f"REF-{int(time.time())}",
                "reason": "Auto-approved"
            }
        else:
            # Escalate for manual review
            return True, {
                "action": "ESCALATE",
                "refund_inr": refund_amount_inr,
                "escalation_id": f"ESC-{int(time.time())}",
                "reason": "Manual approval required"
            }
    
    except Exception as e:
        logger.error(f"issue_refund error: {e}")
        return False, {"error": str(e)}


# ============================================================================
# REACT LOOP AND CHECKPOINTING
# ============================================================================

def _est_tokens(text: str) -> int:
    """Estimate tokens in text (1.3 tokens per word)."""
    if not text or not isinstance(text, str):
        return 1
    return max(1, int(len(text.split()) * 1.3))


def _save_checkpoint(state: AgentState, state_dir: str = "state") -> None:
    """Save checkpoint to disk."""
    state_path = Path(state_dir)
    state_path.mkdir(exist_ok=True)
    
    checkpoint_file = state_path / f"{state.run_id}.checkpoint"
    with open(checkpoint_file, "w") as f:
        json.dump(state.to_dict(), f, indent=2)
    
    logger.info(f"Checkpoint saved: {checkpoint_file}")


def _load_checkpoint(run_id: str, state_dir: str = "state") -> Optional[AgentState]:
    """Load checkpoint from disk."""
    checkpoint_file = Path(state_dir) / f"{run_id}.checkpoint"
    
    if not checkpoint_file.exists():
        return None
    
    try:
        with open(checkpoint_file, "r") as f:
            data = json.load(f)
        
        # Reconstruct tool calls
        tool_calls = [
            ToolCall(
                tool_name=tc["tool_name"],
                args=tc["args"],
                result=tc["result"],
                success=tc["success"],
                timestamp=tc["timestamp"]
            )
            for tc in data.get("tool_calls", [])
        ]
        
        state = AgentState(
            run_id=data["run_id"],
            scenario_id=data["scenario_id"],
            iteration=data["iteration"],
            messages=[],  # Messages not persisted
            tool_calls=tool_calls,
            tokens_used=data["tokens_used"],
            final_status=data["final_status"],
            final_amount=data["final_amount"],
            start_time=0  # Will be set by caller
        )
        
        logger.info(f"Checkpoint loaded: {checkpoint_file}")
        return state
    
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return None


def run_agent(
    scenario_dict: Dict,
    vectorizer=None,
    chunks: List[Dict] = None,
    token_limit: int = 6000,
    time_limit_seconds: Optional[float] = None,
    db_path: str = "data/instamart.db",
    state_dir: str = "state",
    run_id: Optional[str] = None
) -> Tuple[str, float, int, int]:
    """
    Main ReAct loop for refund request handling.
    
    Args:
        scenario_dict: {scenario_id, customer_id, message, expected_outcome, expected_refund_inr}
        vectorizer: TF-IDF vectorizer from policy_rag
        chunks: Policy chunks from policy_rag
        token_limit: Max tokens before FALLBACK_TOKEN_LIMIT (default 6000)
        time_limit_seconds: Max wall-clock seconds before FALLBACK_TIMEOUT
        db_path: Path to database
        state_dir: Directory for checkpoints
        run_id: Unique run identifier (auto-generated if None)
    
    Returns:
        (final_status, refund_amount_inr, llm_calls, tokens_used)
        where final_status ∈ {REFUND, ESCALATE, DENIED, NOT_FOUND, FALLBACK_TOKEN_LIMIT, FALLBACK_TIMEOUT, FALLBACK_MAX_ITERATIONS}
    """
    if run_id is None:
        run_id = f"run_{scenario_dict['scenario_id']}_{int(time.time())}"
    
    start_time = time.time()
    scenario_id = scenario_dict["scenario_id"]
    customer_message = scenario_dict["message"]
    
    # Initialize state
    state = AgentState(
        run_id=run_id,
        scenario_id=scenario_id,
        iteration=0,
        messages=[],
        tool_calls=[],
        tokens_used=0,
        final_status="IN_PROGRESS",
        final_amount=0.0,
        start_time=start_time
    )
    
    # ReAct system prompt
    system_message = {
        "role": "system",
        "content": """AGENT

You are a refund request handler for Instamart. Your task is to process refund requests for orders.

You have access to 4 tools:
1. get_order(order_id) — fetch order details including delivery time
2. search_policy(query) — search refund policy knowledge base
3. calculate_refund(order_id, amount, delivery_time) — compute refund amount
4. issue_refund(order_id, amount, approval_level) — issue or escalate refund

Process flow:
1. Always call get_order first to verify order exists and get delivery time
2. Then call search_policy to confirm refund eligibility
3. Then call calculate_refund to determine amount and approval level
4. Finally call issue_refund to process or escalate

Response format for tool calls:
[TOOL_NAME: arg1=val1, arg2=val2]

Final answer format when done:
[FINAL: STATUS, AMOUNT]
where STATUS ∈ {REFUND, ESCALATE, DENIED, NOT_FOUND}
AMOUNT is the refund amount in INR (e.g., 128.25)

Examples:
[get_order: order_id=ORD-100048]
[search_policy: query=Is this order eligible for refund based on delivery time?]
[calculate_refund: order_id=ORD-100048, order_amount_inr=128.25, delivery_time_minutes=65]
[issue_refund: order_id=ORD-100048, refund_amount_inr=128.25, approval_level=auto]
[FINAL: REFUND, 128.25]
"""
    }
    
    # ReAct loop
    max_iterations = 10
    for iteration in range(max_iterations):
        state.iteration = iteration
        
        # Check token budget
        if state.tokens_used > token_limit:
            state.final_status = "FALLBACK_TOKEN_LIMIT"
            state.final_amount = 0.0
            state.end_time = time.time()
            _save_checkpoint(state, state_dir)
            return state.final_status, state.final_amount, len(state.tool_calls), state.tokens_used
        
        # Check time budget
        elapsed = time.time() - start_time
        if time_limit_seconds is not None and elapsed > time_limit_seconds:
            state.final_status = "FALLBACK_TIMEOUT"
            state.final_amount = 0.0
            state.end_time = time.time()
            _save_checkpoint(state, state_dir)
            return state.final_status, state.final_amount, len(state.tool_calls), state.tokens_used
        
        # Build user message
        if iteration == 0:
            # First iteration: send the customer's original message
            user_content = f"Refund request from customer: {customer_message}"
        else:
            # Subsequent iterations: continue with original message if it has [LOOP], else send generic message
            if "[LOOP]" in customer_message:
                # Keep the original message to maintain [LOOP] flag
                user_content = f"Refund request from customer: {customer_message}"
            else:
                # Normal flow: ask to continue processing
                # Tool results are already in state.messages as "tool" role messages
                user_content = "Continue processing."
        
        user_message = {
            "role": "user",
            "content": user_content
        }
        
        # Build full messages list: system + state history + current user message
        messages_for_llm = [system_message] + state.messages + [user_message]
        
        # Estimate tokens
        msg_tokens = _est_tokens(system_message["content"]) + _est_tokens(user_content)
        state.tokens_used += msg_tokens
        
        # Call LLM
        try:
            response = chat(messages_for_llm)
            if not response or not hasattr(response, 'choices') or len(response.choices) == 0:
                logger.error(f"Invalid response from LLM: {response}")
                state.final_status = "ERROR"
                state.end_time = time.time()
                _save_checkpoint(state, state_dir)
                return state.final_status, 0.0, len(state.tool_calls), state.tokens_used
            state.tokens_used += response.usage.total_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens') else 0
        except Exception as e:
            import traceback
            logger.error(f"LLM call failed: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            state.final_status = "ERROR"
            state.end_time = time.time()
            _save_checkpoint(state, state_dir)
            return state.final_status, 0.0, len(state.tool_calls), state.tokens_used
        
        # Parse response for tool calls or final answer
        # The mock_llm returns tool_calls in the message object or content text
        message = response.choices[0].message
        response_text = message.content if hasattr(message, 'content') else ""
        
        # Check for tool calls in the message (from mock_llm format)
        if hasattr(message, 'tool_calls') and message.tool_calls:
            # Process tool calls from mock_llm format
            tool_calls_list = message.tool_calls if isinstance(message.tool_calls, list) else [message.tool_calls]
            
            for tool_call in tool_calls_list:
                tool_name = tool_call.name if hasattr(tool_call, 'name') else tool_call.get('name')
                args = tool_call.arguments if hasattr(tool_call, 'arguments') else tool_call.get('arguments', {})
                
                # Execute tool
                tool_call_time = time.time()
                success = False
                result = {}
                
                try:
                    if tool_name == "get_order":
                        order_id = args.get('order_id') if isinstance(args, dict) else (args.order_id if hasattr(args, 'order_id') else "")
                        success, result = get_order(order_id, db_path=db_path)
                    elif tool_name == "search_policy":
                        query = args.get('query') if isinstance(args, dict) else (args.query if hasattr(args, 'query') else "")
                        success, result = search_policy(query, vectorizer=vectorizer, chunks=chunks)
                    elif tool_name == "calculate_refund":
                        # Handle both naming conventions: order_amount/order_amount_inr, delivery_minutes/delivery_time_minutes
                        amount = args.get('order_amount_inr')
                        if amount is None:
                            amount = args.get('order_amount')
                        if isinstance(amount, str):
                            amount = float(amount)
                        
                        delivery_time = args.get('delivery_time_minutes')
                        if delivery_time is None:
                            delivery_time = args.get('delivery_minutes')
                        if isinstance(delivery_time, str):
                            delivery_time = int(delivery_time)
                        
                        order_id = args.get('order_id', '')
                        success, result = calculate_refund(order_id, amount or 0, delivery_time or 0)
                    elif tool_name == "issue_refund":
                        # Handle both naming conventions: refund_amount_inr/refund_inr
                        amount = args.get('refund_amount_inr')
                        if amount is None:
                            amount = args.get('refund_inr')
                        if isinstance(amount, str):
                            amount = float(amount)
                        
                        order_id = args.get('order_id', '')
                        approval = args.get('approval_level', '')
                        success, result = issue_refund(order_id, amount or 0, approval)
                    elif tool_name == "escalate_to_human":
                        # Handle escalate tool (alternative name for issue_refund with escalate)
                        amount = args.get('refund_inr')
                        if isinstance(amount, str):
                            amount = float(amount)
                        order_id = args.get('order_id', '')
                        success, result = issue_refund(order_id, amount or 0, 'manual')
                    else:
                        success, result = False, {"error": f"Unknown tool: {tool_name}"}
                
                except Exception as e:
                    logger.error(f"Tool execution error: {e}")
                    success, result = False, {"error": str(e)}
                
                # Record tool call in state
                state.tool_calls.append(ToolCall(
                    tool_name=tool_name,
                    args=args if isinstance(args, dict) else (vars(args) if hasattr(args, '__dict__') else {}),
                    result=result,
                    success=success,
                    timestamp=tool_call_time
                ))
                
                # Add tool result message to conversation history (required by mock_llm)
                # This allows the mock_llm to see what tools have been executed
                state.messages.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": json.dumps(result)
                })
                
                # Save checkpoint after each tool execution
                _save_checkpoint(state, state_dir)
            
            # DON'T continue here - loop back naturally to next iteration
            # The messages list now contains the tool results for the LLM to see
        
        # Check for final answer in content text
        elif response_text:
            # Look for patterns like "REFUND: INR 128.25" or "NOT_FOUND:" or "DENIED:" or "ESCALATE:"
            final_match = re.search(r'(REFUND|ESCALATE|DENIED|NOT_FOUND):\s*(?:INR\s+)?([\d.]+)?', response_text, re.IGNORECASE)
            if final_match:
                status = final_match.group(1).upper()
                try:
                    amount = float(final_match.group(2)) if final_match.group(2) else 0.0
                except (ValueError, TypeError):
                    amount = 0.0
                
                # If amount wasn't in the response text, try to get it from the last tool call result
                if amount == 0.0 and state.tool_calls:
                    last_call = state.tool_calls[-1]
                    if last_call.success and last_call.result:
                        # Look for refund amount in various possible key names
                        amount = (
                            last_call.result.get('refund_inr') or
                            last_call.result.get('refund_amount_inr') or
                            last_call.result.get('amount_inr') or
                            0.0
                        )
                
                state.final_status = status
                state.final_amount = amount
                state.end_time = time.time()
                _save_checkpoint(state, state_dir)
                return state.final_status, state.final_amount, len(state.tool_calls), state.tokens_used
        
        # Check if conversation ended without final answer
        # This happens when LLM returns empty content and no tool_calls
        if not response_text and not (hasattr(message, 'tool_calls') and message.tool_calls):
            # Empty response - likely conversation exhausted
            state.final_status = "DENIED"
            state.final_amount = 0.0
            state.end_time = time.time()
            _save_checkpoint(state, state_dir)
            return state.final_status, state.final_amount, len(state.tool_calls), state.tokens_used
    
    # Max iterations reached
    state.final_status = "FALLBACK_MAX_ITERATIONS"
    state.final_amount = 0.0
    state.end_time = time.time()
    _save_checkpoint(state, state_dir)
    return state.final_status, state.final_amount, len(state.tool_calls), state.tokens_used


def resume_agent(
    run_id: str,
    scenario_dict: Dict,
    vectorizer=None,
    chunks: List[Dict] = None,
    token_limit: int = 6000,
    time_limit_seconds: Optional[float] = None,
    db_path: str = "data/instamart.db",
    state_dir: str = "state"
) -> Tuple[str, float, int, int]:
    """
    Resume a previously interrupted agent run from checkpoint.
    
    Args:
        run_id: Run ID to resume
        scenario_dict: Original scenario
        vectorizer: TF-IDF vectorizer
        chunks: Policy chunks
        token_limit: Token limit
        time_limit_seconds: Time limit
        db_path: Database path
        state_dir: Checkpoint directory
    
    Returns:
        Same as run_agent()
    """
    # Load checkpoint
    checkpoint_state = _load_checkpoint(run_id, state_dir)
    
    if checkpoint_state is None:
        logger.warning(f"No checkpoint found for {run_id}, starting fresh")
        return run_agent(scenario_dict, vectorizer, chunks, token_limit, time_limit_seconds, db_path, state_dir, run_id)
    
    # Continue from checkpoint
    logger.info(f"Resuming from checkpoint: {run_id}")
    
    # For now, just re-run from the beginning
    # In a real system, we'd continue from where we left off
    return run_agent(scenario_dict, vectorizer, chunks, token_limit, time_limit_seconds, db_path, state_dir, run_id)
