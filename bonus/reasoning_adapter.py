"""
reasoning_adapter.py
====================
Vexoo Labs AI Engineer Assignment — Bonus
Plug-and-play domain router using Strategy Pattern + Router Pattern.

Architecture Pattern: Strategy Pattern + Router
-----------------------------------------------
  detect_type()  — Router: inspects query and returns a domain label.
  route()        — Dispatcher: calls the matching Strategy handler.
  handle_*()     — Strategy: self-contained domain-specific handler.

This design allows any handler to be swapped, extended, or mocked
independently without modifying the routing logic.

Author: Candidate
"""

import re
import json
import random
from typing import Dict, Any, List

random.seed(42)


class ReasoningAdapter:
    """
    A plug-and-play query router that detects the domain of an incoming
    query and dispatches it to the most appropriate reasoning handler.

    Architecture Pattern: Strategy Pattern + Router
    -----------------------------------------------
    detect_type() acts as the Router: it inspects the query using keyword
    scoring and regex patterns to return a domain label (math, legal,
    medical, or general). route() uses that label to call the corresponding
    Strategy handler. Each handle_*() is a self-contained Strategy that
    returns a structured, domain-specific response dict. This makes it
    trivial to swap out individual handlers for real model calls, RAG
    pipelines, or external APIs without changing the routing logic.
    """

    MATH_KEYWORDS = [
        "calculate", "compute", "solve", "equation", "integral", "derivative",
        "formula", "sum", "product", "square root", "percentage", "profit",
        "loss", "algebra", "geometry", "trigonometry", "probability",
        "statistics", "matrix", "vector", "factorial", "logarithm",
    ]

    LEGAL_KEYWORDS = [
        "law", "legal", "court", "sue", "lawsuit", "contract", "defendant",
        "plaintiff", "rights", "crime", "criminal", "verdict", "statute",
        "regulation", "litigation", "jurisdiction", "appeal", "penalty",
        "fine", "attorney", "lawyer", "judge", "liability", "clause",
    ]

    MEDICAL_KEYWORDS = [
        "symptom", "disease", "diagnosis", "treatment", "medicine", "drug",
        "dose", "hospital", "surgery", "pain", "fever", "infection",
        "virus", "bacteria", "cancer", "diabetes", "blood pressure",
        "allergy", "prescription", "therapy", "mental health", "vaccine",
    ]

    # Regex patterns that strongly indicate a math query
    MATH_PATTERNS = [
        r"\d+\s*[\+\-\*/\^]\s*\d+",        # arithmetic: 3 + 4
        r"what\s+is\s+\d+",                  # "what is 25..."
        r"\d+\s*%\s+of\s+\d+",              # "15% of 200"
        r"sqrt|log|sin|cos|tan|exp",         # math functions
        r"=\s*\?",                            # "x = ?"
        r"how\s+many\s+.*(total|left|remain)",
    ]

    # ------------------------------------------------------------------
    # 1. TYPE DETECTION
    # ------------------------------------------------------------------

    def detect_type(self, query: str) -> str:
        """
        Classify a query as 'math', 'legal', 'medical', or 'general'.

        Steps:
          1. Check regex patterns for math (fast path — highest signal).
          2. Count keyword matches per domain.
          3. Return the domain with the highest match count.
          4. Default to 'general' if no keywords match.
        """
        query_lower = query.lower()

        # Fast-path: math regex check
        for pattern in self.MATH_PATTERNS:
            if re.search(pattern, query_lower):
                return "math"

        # Keyword frequency scoring
        scores = {
            "math":    sum(1 for kw in self.MATH_KEYWORDS    if kw in query_lower),
            "legal":   sum(1 for kw in self.LEGAL_KEYWORDS   if kw in query_lower),
            "medical": sum(1 for kw in self.MEDICAL_KEYWORDS if kw in query_lower),
        }

        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "general"

    # ------------------------------------------------------------------
    # 2. ROUTER
    # ------------------------------------------------------------------

    def route(self, query: str) -> Dict[str, Any]:
        """
        Route query to the correct domain handler and attach routing metadata.
        """
        query_type = self.detect_type(query)

        dispatch = {
            "math":    self.handle_math,
            "legal":   self.handle_legal,
            "medical": self.handle_medical,
            "general": self.handle_general,
        }

        handler  = dispatch.get(query_type, self.handle_general)
        response = handler(query)

        # Attach routing metadata so callers can inspect the decision
        response["_meta"] = {
            "detected_type": query_type,
            "handler":       handler.__name__,
            "query":         query,
        }
        return response

    # ------------------------------------------------------------------
    # 3. MATH HANDLER
    # ------------------------------------------------------------------

    def handle_math(self, query: str) -> Dict[str, Any]:
        """
        Simulate a step-by-step math solver.
        Extracts and attempts eval() on arithmetic expressions;
        falls back to templated reasoning steps for word problems.
        """
        # Extract the first arithmetic-looking expression
        expr_match = re.search(r"[\d\.\s\+\-\*\/\(\)\^%]+",
                               query.replace("^", "**"))
        expression = expr_match.group().strip() if expr_match else None
        steps: List[str] = []
        result: Any = None

        if expression and re.search(r"\d", expression):
            try:
                safe_expr = re.sub(r"[^0-9\.\+\-\*\/\(\)\s]", "", expression).strip()
                if safe_expr:
                    result = eval(safe_expr)  # noqa: S307 — sanitised expression
                    steps = [
                        f"Step 1: Identified expression: '{safe_expr}'",
                        "Step 2: Applied arithmetic evaluation.",
                        f"Step 3: Result = {result}",
                    ]
                else:
                    raise ValueError("Empty after sanitisation")
            except Exception:
                result = f"[simulated result: {random.randint(10, 200)}]"
                steps = [
                    "Step 1: Parsed problem statement.",
                    "Step 2: Identified unknowns.",
                    "Step 3: Set up equation from given information.",
                    f"Step 4: Solved algebraically. Result: {result}",
                ]
        else:
            result = "[word problem — manual solving required]"
            steps = [
                "Step 1: Read and understand the problem.",
                "Step 2: Identify what is being asked.",
                "Step 3: List given values and constraints.",
                "Step 4: Apply appropriate formula or method.",
                "Step 5: Verify the answer makes sense.",
            ]

        return {
            "domain":           "math",
            "expression_found": expression,
            "steps":            steps,
            "result":           result,
        }

    # ------------------------------------------------------------------
    # 4. LEGAL HANDLER
    # ------------------------------------------------------------------

    def handle_legal(self, query: str) -> Dict[str, Any]:
        """
        Return a structured legal response with jurisdiction, relevant law,
        key considerations, and a recommendation.
        NOT actual legal advice — simulation only.
        """
        ql = query.lower()

        # Detect jurisdiction from query text
        jurisdiction_map = {
            "india": "India (IPC / CrPC)",
            "us":    "United States (Federal / State)",
            "usa":   "United States (Federal / State)",
            "uk":    "United Kingdom (English Law)",
            "eu":    "European Union",
        }
        jurisdiction = next((v for k, v in jurisdiction_map.items() if k in ql),
                            "Jurisdiction not specified — consult local counsel")

        # Select relevant statute and recommendation by keyword
        if "contract" in ql:
            relevant_law   = "Contract Act / UCC (Uniform Commercial Code)"
            recommendation = ("Review all contract terms carefully. Identify breach "
                              "clauses and ensure conditions precedent are met before "
                              "initiating legal action.")
        elif "criminal" in ql or "crime" in ql:
            relevant_law   = "Criminal Procedure Code / Penal Code"
            recommendation = ("Consult a licensed criminal defence attorney immediately. "
                              "Do not make statements without legal representation.")
        elif "property" in ql:
            relevant_law   = "Property Law / Transfer of Property Act"
            recommendation = ("Verify title deeds and encumbrance certificates. Engage a "
                              "property lawyer for due diligence before any transaction.")
        else:
            relevant_law   = "General Civil Law / Applicable Statutes"
            recommendation = ("Gather all relevant documentation and seek advice from a "
                              "qualified attorney specialising in this area.")

        return {
            "domain":             "legal",
            "jurisdiction":       jurisdiction,
            "relevant_law":       relevant_law,
            "key_considerations": [
                "Collect and preserve all documentary evidence.",
                "Observe applicable statutes of limitations.",
                "Distinguish between civil and criminal liability.",
            ],
            "recommendation": recommendation,
            "disclaimer": ("Simulated response — not legal advice. "
                           "Consult a licensed attorney."),
        }

    # ------------------------------------------------------------------
    # 5. MEDICAL HANDLER
    # ------------------------------------------------------------------

    def handle_medical(self, query: str) -> Dict[str, Any]:
        """
        Return a structured medical response with detected symptoms,
        urgency level, suggested action, and self-care tips.
        NOT medical advice — simulation only.
        """
        ql = query.lower()

        symptom_map = {
            "fever":     "Elevated body temperature (fever)",
            "headache":  "Headache / cephalgia",
            "cough":     "Persistent cough",
            "pain":      "Pain (location unspecified)",
            "fatigue":   "Fatigue / weakness",
            "nausea":    "Nausea / vomiting",
            "shortness": "Shortness of breath / dyspnoea",
            "rash":      "Skin rash / eruption",
            "dizziness": "Dizziness / vertigo",
            "chest":     "Chest discomfort",
        }
        detected = [desc for kw, desc in symptom_map.items() if kw in ql]
        if not detected:
            detected = ["Symptom details unclear from query"]

        emergency_kws = ["chest pain", "difficulty breathing", "unconscious",
                         "stroke", "heart attack", "severe bleeding"]
        urgency = ("HIGH — seek immediate emergency care"
                   if any(kw in ql for kw in emergency_kws)
                   else "MODERATE — consult a healthcare provider within 24-48 hours")

        return {
            "domain":            "medical",
            "symptoms_detected": detected,
            "urgency_level":     urgency,
            "suggested_action":  ("Schedule an appointment with a licensed healthcare "
                                  "provider for clinical evaluation and diagnosis."),
            "self_care_tips": [
                "Rest adequately and stay well-hydrated.",
                "Monitor symptom progression and note any changes.",
                "Avoid self-medicating without professional guidance.",
            ],
            "disclaimer": ("Simulated response — NOT medical advice. "
                           "Always consult a qualified healthcare professional."),
        }

    # ------------------------------------------------------------------
    # 6. GENERAL HANDLER
    # ------------------------------------------------------------------

    def handle_general(self, query: str) -> Dict[str, Any]:
        """
        Simulate semantic search for general-domain queries.
        Returns a ranked list of mock results.
        """
        topic = " ".join(query.split()[:3]).title()

        mock_results = [
            {
                "rank":    1,
                "title":   f"Comprehensive Guide to {topic}",
                "source":  "knowledge_base_v2",
                "score":   round(random.uniform(0.80, 0.99), 4),
                "snippet": f"An in-depth overview of {topic}, covering key concepts, "
                           f"best practices, and recent developments.",
            },
            {
                "rank":    2,
                "title":   f"{topic}: Frequently Asked Questions",
                "source":  "faq_database",
                "score":   round(random.uniform(0.60, 0.79), 4),
                "snippet": f"Answers to the most common questions about {topic}, "
                           f"including definitions and practical applications.",
            },
            {
                "rank":    3,
                "title":   f"Recent Research on {topic}",
                "source":  "academic_corpus",
                "score":   round(random.uniform(0.40, 0.59), 4),
                "snippet": f"A summary of the latest peer-reviewed research on {topic}, "
                           f"highlighting emerging trends and open questions.",
            },
        ]

        return {
            "domain":                  "general",
            "top_results":             mock_results,
            "total_simulated_results": random.randint(120, 5000),
            "note": ("Results are simulated. Connect a live retrieval backend "
                     "for real semantic search."),
        }


# ---------------------------------------------------------------------------
# DEMO UTILITIES
# ---------------------------------------------------------------------------

def pretty_print_response(response: Dict[str, Any]) -> None:
    """Print a response dict as pretty JSON with a header."""
    meta = response.get("_meta", {})
    print(f"\n{'='*65}")
    print(f"  Query  : {meta.get('query', 'N/A')}")
    print(f"  Type   : {meta.get('detected_type', 'N/A').upper()}")
    print(f"  Handler: {meta.get('handler', 'N/A')}")
    print(f"{'='*65}")
    display = {k: v for k, v in response.items() if k != "_meta"}
    print(json.dumps(display, indent=2))


# ---------------------------------------------------------------------------
# MAIN DEMO BLOCK
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    adapter = ReasoningAdapter()

    demo_queries = [
        "Calculate 15% of 840 plus the square root of 144",      # math
        "Can I sue my landlord for breaching the rental contract?", # legal
        "I have a high fever, severe headache, and fatigue — what should I do?",  # medical
        "What are the latest developments in renewable energy storage?",  # general
    ]

    print("\n" + "="*65)
    print("   VEXOO LABS — Bonus: ReasoningAdapter Demo")
    print("="*65)

    for query in demo_queries:
        response = adapter.route(query)
        pretty_print_response(response)

    print("\n" + "="*65)
    print("   ReasoningAdapter demo complete.")
    print("="*65 + "\n")
