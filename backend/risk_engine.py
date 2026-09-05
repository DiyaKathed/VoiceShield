"""
VoiceShield Dual-Layer Risk Engine
==================================
Layer 1: Voice ML Acoustic Risk (Direct model probability)
Layer 2: Contextual High-Risk Transaction Threat Engine (Financial context, caller identity, urgency, speaker match)

Strictly separates Voice ML probability from contextual enterprise risk.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class TransactionContext(BaseModel):
    caller_role: str = Field(default="CEO / Executive", description="Claimed role of caller")
    caller_name: str = Field(default="John Anderson", description="Claimed caller name")
    amount: float = Field(default=75000.0, description="Financial transaction amount in USD")
    urgency: str = Field(default="Immediate", description="Normal, Urgent, Immediate")
    speaker_verification: str = Field(
        default="Unregistered / Unknown",
        description="Verified Enrolled, Unregistered / Unknown, Mismatch / Failed"
    )


class VoiceRiskEngine:
    """Computes pure acoustic risk based on model probability."""

    @staticmethod
    def calculate_voice_risk(ai_probability: float) -> Dict[str, Any]:
        score = round(ai_probability * 100.0, 1)

        if score < 40.0:
            classification = "GENUINE HUMAN SPEECH"
            status = "LOW RISK"
            color = "#10B981"  # Emerald green
        elif score <= 70.0:
            classification = "SUSPICIOUS / ANOMALOUS SPEECH"
            status = "MEDIUM RISK"
            color = "#F59E0B"  # Amber warning
        else:
            classification = "SYNTHETIC / AI VOICE CLONE"
            status = "HIGH RISK"
            color = "#EF4444"  # Red threat

        return {
            "voice_risk_score": score,
            "status": status,
            "classification": classification,
            "color": color,
            "is_threat": score >= 70.0
        }


class ContextualRiskEngine:
    """
    Computes enterprise transaction threat score by combining:
    - Voice ML Acoustic Risk (Weight: 50%)
    - Transaction Value Severity (Weight: 20%)
    - Caller Authority / Spoof Target Index (Weight: 15%)
    - Urgency Pressure Factor (Weight: 10%)
    - Speaker Verification Biometric Match (Weight: 5%)
    """

    @staticmethod
    def evaluate(ai_probability: float, context: Optional[TransactionContext] = None) -> Dict[str, Any]:
        if context is None:
            context = TransactionContext()

        # 1. Voice ML Component (0 - 100)
        voice_risk = ai_probability * 100.0

        # 2. Financial Amount Weight (0 - 100)
        amt = context.amount
        if amt < 5000:
            amount_score = 15.0
        elif amt < 25000:
            amount_score = 40.0
        elif amt < 100000:
            amount_score = 75.0
        else:
            amount_score = 95.0

        # 3. Caller Authority / Target Index (0 - 100)
        role = context.caller_role.lower()
        if "ceo" in role or "board" in role or "executive" in role:
            role_score = 95.0
        elif "cfo" in role or "treasury" in role or "finance" in role:
            role_score = 90.0
        elif "vendor" in role or "supplier" in role:
            role_score = 65.0
        elif "employee" in role or "internal" in role:
            role_score = 45.0
        else:
            role_score = 70.0

        # 4. Urgency Factor (0 - 100)
        urgency_lower = context.urgency.lower()
        if "immediate" in urgency_lower or "emergency" in urgency_lower:
            urgency_score = 90.0
        elif "urgent" in urgency_lower or "same-day" in urgency_lower:
            urgency_score = 60.0
        else:
            urgency_score = 20.0

        # 5. Speaker Verification Biometric Match
        spk_lower = context.speaker_verification.lower()
        if "mismatch" in spk_lower or "failed" in spk_lower:
            speaker_score = 95.0
        elif "unregistered" in spk_lower or "unknown" in spk_lower:
            speaker_score = 55.0
        else:
            # Verified enrolled voice
            speaker_score = 10.0

        # Weighted calculation of Contextual Risk Score
        contextual_score = (
            0.50 * voice_risk +
            0.20 * amount_score +
            0.15 * role_score +
            0.10 * urgency_score +
            0.05 * speaker_score
        )
        contextual_score = round(min(100.0, max(0.0, contextual_score)), 1)

        # Generate Security Alert & Recommended Actions
        if contextual_score >= 70.0 or voice_risk >= 70.0:
            threat_level = "CRITICAL / SEVERE THREAT"
            security_alert = (
                f"HIGH RISK ALERT: Potential AI voice cloning attack detected impersonating '{context.caller_name}' "
                f"({context.caller_role}) attempting to authorize a ${context.amount:,.2f} transaction."
            )
            recommended_action = (
                "BLOCK TRANSACTION IMMEDIATELY. Do not authorize wire transfer or credential changes. "
                "Trigger mandatory out-of-band identity verification via pre-enrolled secondary phone line "
                "or physical security token."
            )
            action_code = "HALT_AND_VERIFY"
        elif contextual_score >= 40.0 or voice_risk >= 40.0:
            threat_level = "ELEVATED CAUTION"
            security_alert = (
                f"ADVISORY: Anomalous speech characteristics detected for '{context.caller_name}'. "
                f"Acoustic features exhibit potential compression artifacts or partial synthesis."
            )
            recommended_action = (
                "REQUIRE SECONDARY APPROVAL. Escalate transaction to dual-custody authorization. "
                "Challenge the caller with dynamic knowledge-based authentication questions."
            )
            action_code = "DUAL_AUTHORIZATION"
        else:
            threat_level = "NORMAL / LOW RISK"
            security_alert = (
                f"NORMAL: Acoustic features are consistent with genuine human speech for '{context.caller_name}'. "
                f"No synthetic cloning artifacts identified."
            )
            recommended_action = (
                "ALLOW ACTION WITH STANDARD AUDITING. Proceed with routine transaction logging "
                "and standard verification compliance."
            )
            action_code = "STANDARD_AUDIT"

        return {
            "contextual_risk_score": contextual_score,
            "threat_level": threat_level,
            "security_alert": security_alert,
            "recommended_action": recommended_action,
            "action_code": action_code,
            "context_breakdown": {
                "voice_risk_component": round(voice_risk, 1),
                "transaction_amount_component": round(amount_score, 1),
                "caller_authority_component": round(role_score, 1),
                "urgency_component": round(urgency_score, 1),
                "speaker_verification_component": round(speaker_score, 1)
            }
        }
