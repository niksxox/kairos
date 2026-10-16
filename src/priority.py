"""
KAIROS - Institutional Space Priority & Importance Policy Engine
Defines institutional ranking rules ensuring critical academic activities
(e.g., Exams, Tests, Core Lectures) take precedence over discretionary activities
(e.g., Cultural Events, Club Socials, Informal Gatherings).
"""

from enum import Enum
from typing import Dict, Any, Tuple


class PriorityTier(str, Enum):
    CRITICAL = "Critical"     # Formal Exams, Final Tests, Accreditations
    HIGH = "High"             # Scheduled Lectures, Mandatory Lab Practicals
    MEDIUM = "Medium"         # Academic Seminars, Department Colloquiums, Guest Lectures
    STANDARD = "Standard"     # Study Groups, Project Team Sprints, Academic Clubs
    FLEXIBLE = "Flexible"     # Cultural Fests, Social Gatherings, Rehearsals, Informal


# Weights used directly in CP-SAT objective functions & sorting heuristics
PRIORITY_WEIGHTS: Dict[PriorityTier, int] = {
    PriorityTier.CRITICAL: 100,
    PriorityTier.HIGH: 50,
    PriorityTier.MEDIUM: 25,
    PriorityTier.STANDARD: 10,
    PriorityTier.FLEXIBLE: 5,
}

# Visual badge styling for the UI (restrained, non-ai-slope styling)
PRIORITY_BADGE_STYLE: Dict[PriorityTier, Dict[str, str]] = {
    PriorityTier.CRITICAL: {"bg": "#fee2e2", "color": "#991b1b", "border": "#fca5a5", "label": "Critical: Exam / Test"},
    PriorityTier.HIGH: {"bg": "#fef3c7", "color": "#92400e", "border": "#fcd34d", "label": "High: Core Academic"},
    PriorityTier.MEDIUM: {"bg": "#e0f2fe", "color": "#075985", "border": "#7dd3fc", "label": "Medium: Seminar / Lab"},
    PriorityTier.STANDARD: {"bg": "#f1f5f9", "color": "#334155", "border": "#cbd5e1", "label": "Standard: Study / Project"},
    PriorityTier.FLEXIBLE: {"bg": "#f3e8ff", "color": "#6b21a8", "border": "#d8b4fe", "label": "Flexible: Cultural / Social"},
}


def infer_priority_tier(purpose_or_type: str) -> PriorityTier:
    """
    Classifies an event purpose or event type into an institutional PriorityTier.
    Exams and formal assessments receive absolute priority over cultural and social events.
    """
    text = (purpose_or_type or "").lower().strip()

    # 1. Critical Tier: Exams, Tests, Quizzes, Finals
    if any(k in text for k in ["exam", "test", "midterm", "final", "quiz", "assessment", "viva", "accreditation"]):
        return PriorityTier.CRITICAL

    # 2. High Tier: Regular Lectures, Course Lectures, Labs, Practicals
    if any(k in text for k in ["lecture", "class", "practical", "lab", "laboratory", "tutorial", "core"]):
        return PriorityTier.HIGH

    # 3. Medium Tier: Seminars, Workshops, Conferences, Department Meetings
    if any(k in text for k in ["seminar", "workshop", "colloquium", "defense", "symposium", "conference", "faculty meeting"]):
        return PriorityTier.MEDIUM

    # 4. Standard Tier: Student Study Groups, Project Collaboration, Academic Clubs
    if any(k in text for k in ["study", "group", "project", "hackathon", "robotics", "coding", "reading"]):
        return PriorityTier.STANDARD

    # 5. Flexible Tier: Cultural events, fests, music rehearsals, social gatherings
    if any(k in text for k in ["cultural", "fest", "dance", "drama", "music", "rehearsal", "social", "party", "gathering", "informal"]):
        return PriorityTier.FLEXIBLE

    # Default to Standard if unspecified
    return PriorityTier.STANDARD


def get_priority_info(purpose_or_type: str) -> Tuple[PriorityTier, int, Dict[str, str]]:
    """
    Returns (tier, numerical_weight, badge_style_dict) for any event purpose.
    """
    tier = infer_priority_tier(purpose_or_type)
    weight = PRIORITY_WEIGHTS[tier]
    badge = PRIORITY_BADGE_STYLE[tier]
    return tier, weight, badge
