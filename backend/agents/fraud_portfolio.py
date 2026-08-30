"""Multidimensional red-team portfolio for the Adversarial Payment Arena.

The YAML AttackSpecs remain the source of truth for what each synthetic campaign
is allowed to emit. This module adds a relationship layer above those specs:
each family is represented as a fraud genome (identity/access/authorization/
channel/behaviour/topology/deception/monetization/lifecycle), plus explicit
connections to other families that a campaign-level strategist may explore.

Nothing here talks to a real payment network or expands the PaymentMessage
schema. Unsupported real-world concepts are recorded as projection gaps rather
than silently inventing fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class FraudGenome:
    identity: str
    access: str
    authorization: str
    channel: str
    behaviour: str
    topology: str
    deception: str
    monetization: str
    lifecycle: str

    def distance(self, other: "FraudGenome") -> float:
        left = asdict(self)
        right = asdict(other)
        changed = sum(left[key] != right[key] for key in left)
        return changed / max(len(left), 1)


@dataclass(frozen=True, slots=True)
class AttackVectorProfile:
    spec_id: str
    attack_file: str
    genome: FraudGenome
    primitives: tuple[str, ...]
    mutation_axes: tuple[str, ...]
    transitions: tuple[str, ...]
    projection_gaps: tuple[str, ...] = ()
    evaluation_focus: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "spec_id": self.spec_id,
            "attack_file": self.attack_file,
            "genome": asdict(self.genome),
            "primitives": list(self.primitives),
            "mutation_axes": list(self.mutation_axes),
            "transitions": list(self.transitions),
            "projection_gaps": list(self.projection_gaps),
            "evaluation_focus": list(self.evaluation_focus),
        }


def _g(identity: str, access: str, authorization: str, channel: str, behaviour: str,
       topology: str, deception: str, monetization: str, lifecycle: str) -> FraudGenome:
    return FraudGenome(identity, access, authorization, channel, behaviour, topology,
                       deception, monetization, lifecycle)


PORTFOLIO: dict[str, AttackVectorProfile] = {
    "ATTACK_1_MFA_RESET_VOICE_CLONE": AttackVectorProfile(
        "ATTACK_1_MFA_RESET_VOICE_CLONE", "attack_1_mfa_reset_voice_clone.yaml",
        _g("compromised_real", "social_engineered_reset", "unauthorized_after_reset", "cnp",
           "session_monetization", "single_victim_session", "voice_impersonation",
           "goods_purchase", "access_then_exploitation"),
        ("account_takeover", "session_reuse", "device_novelty", "social_engineering"),
        ("ticket_band", "merchant_mix", "session_length"),
        ("ATTACK_9_OTP_RELAY_VISHING", "ATTACK_4_CNP_HIGH_VELOCITY", "ATTACK_14_ADVERSARIAL_BOUNDARY_PROBE"),
        evaluation_focus=("post-reset behaviour", "new-device risk", "session containment"),
    ),
    "ATTACK_2_SYNTHETIC_MULE_RING": AttackVectorProfile(
        "ATTACK_2_SYNTHETIC_MULE_RING", "attack_2_synthetic_mule_ring.yaml",
        _g("synthetic", "fabricated_accounts", "unauthorized", "contactless", "distributed_cashout",
           "shared_device_ring", "identity_fabrication", "resale_goods", "cashout"),
        ("synthetic_identity", "shared_device", "shared_ip", "ring"),
        ("ring_size", "merchant_mix", "ticket_band"),
        ("ATTACK_7_SYNCHRONISED_BURST_CASHOUT", "ATTACK_6_VPA_RENTAL_MULE", "ATTACK_13_MERCHANT_BUSTOUT"),
        evaluation_focus=("graph ring detection", "cross-account linkage", "containment"),
    ),
    "ATTACK_3_PROMPT_INJECTED_MERCHANT": AttackVectorProfile(
        "ATTACK_3_PROMPT_INJECTED_MERCHANT", "attack_3_prompt_injected_merchant.yaml",
        _g("compromised_real_customers", "merchant_endpoint_compromise", "merchant_originated", "ecommerce",
           "merchant_fanin_burst", "many_customers_one_merchant", "merchant_integrity_compromise",
           "merchant_settlement", "exploitation"),
        ("merchant_compromise", "merchant_fanin", "known_devices", "cross_customer_burst"),
        ("customer_breadth", "ticket_band", "burst_length"),
        ("ATTACK_13_MERCHANT_BUSTOUT", "ATTACK_7_SYNCHRONISED_BURST_CASHOUT",
         "ATTACK_14_ADVERSARIAL_BOUNDARY_PROBE", "ATTACK_11_AGENTIC_SCOPE_EXPANSION"),
        evaluation_focus=("merchant-centric velocity", "merchant anomaly", "cross-customer aggregation"),
    ),
    "ATTACK_4_CNP_HIGH_VELOCITY": AttackVectorProfile(
        "ATTACK_4_CNP_HIGH_VELOCITY", "attack_4_cnp_high_velocity.yaml",
        _g("many_compromised_real", "stolen_credentials", "unauthorized", "ecommerce", "testing_burst",
           "shared_bot_infrastructure", "automation", "credential_validation", "reconnaissance_then_exploitation"),
        ("card_testing", "shared_device", "shared_ip", "merchant_velocity"),
        ("customer_breadth", "merchant_mix", "ticket_band"),
        ("ATTACK_8_LEARNED_THRESHOLD_STRUCTURING", "ATTACK_10_EXEMPTION_BAND_ABUSE",
         "ATTACK_14_ADVERSARIAL_BOUNDARY_PROBE", "ATTACK_6_VPA_RENTAL_MULE", "ATTACK_9_OTP_RELAY_VISHING"),
        evaluation_focus=("zero-day detection", "shared automation", "burst recognition"),
    ),
    "ATTACK_5_APP_SCAM_PERSONALISED": AttackVectorProfile(
        "ATTACK_5_APP_SCAM_PERSONALISED", "attack_5_app_scam_personalised.yaml",
        _g("genuine_victim", "legitimate_session", "victim_authorized", "push_like_cnp_projection",
           "behavioural_value_shift", "victim_to_recipient", "personalized_social_engineering",
           "fund_transfer", "deception_then_payment"),
        ("victim_authorization", "known_device", "behaviour_shift", "recipient_novelty"),
        ("ticket_band", "victim_mix", "merchant_proxy_mix"),
        ("ATTACK_6_VPA_RENTAL_MULE", "ATTACK_7_SYNCHRONISED_BURST_CASHOUT", "ATTACK_14_ADVERSARIAL_BOUNDARY_PROBE"),
        ("beneficiary/payee entity is not a PaymentMessage field", "inbound contact context is not modeled"),
        ("behaviour over authentication", "recipient-network proxy", "step-up efficacy"),
    ),
    "ATTACK_6_VPA_RENTAL_MULE": AttackVectorProfile(
        "ATTACK_6_VPA_RENTAL_MULE", "attack_6_vpa_rental_mule.yaml",
        _g("rented_real_accounts", "legitimate_account_access", "operator_directed", "push_like_cnp_projection",
           "fanin_then_rotation", "many_senders_few_receivers", "mule_recruitment", "layered_transfer",
           "cashout_and_layering"),
        ("mule_account", "fan_in", "rotation", "multi_hop"),
        ("sender_breadth", "receiver_pool_size", "ticket_band"),
        ("ATTACK_7_SYNCHRONISED_BURST_CASHOUT", "ATTACK_2_SYNTHETIC_MULE_RING",
         "ATTACK_13_MERCHANT_BUSTOUT", "ATTACK_5_APP_SCAM_PERSONALISED"),
        ("VPA/beneficiary is projected onto merchant relationships in the current schema", "outbound hop entity is not modeled"),
        ("graph fan-in", "entity rotation", "cross-transaction structure"),
    ),
    "ATTACK_7_SYNCHRONISED_BURST_CASHOUT": AttackVectorProfile(
        "ATTACK_7_SYNCHRONISED_BURST_CASHOUT", "attack_7_synchronised_burst_cashout.yaml",
        _g("distributed_compromised", "pooled_credentials", "unauthorized", "contactless", "synchronized_burst",
           "distributed_accounts", "automation", "goods_cashout", "cashout"),
        ("cross_entity_timing", "distributed_accounts", "burst", "cashout"),
        ("wave_width", "ticket_band", "merchant_pool"),
        ("ATTACK_6_VPA_RENTAL_MULE", "ATTACK_2_SYNTHETIC_MULE_RING", "ATTACK_14_ADVERSARIAL_BOUNDARY_PROBE"),
        evaluation_focus=("portfolio-level timing", "cross-entity aggregation", "burst containment"),
    ),
    "ATTACK_8_LEARNED_THRESHOLD_STRUCTURING": AttackVectorProfile(
        "ATTACK_8_LEARNED_THRESHOLD_STRUCTURING", "attack_8_learned_threshold_structuring.yaml",
        _g("compromised_real", "stolen_credentials", "unauthorized", "ecommerce", "tight_nonround_band",
           "single_or_small_pool", "outcome_learning", "goods_purchase", "reconnaissance_then_exploitation"),
        ("decision_feedback", "distributional_structuring", "tight_band", "low_variance"),
        ("band_center", "band_width", "merchant_mix"),
        ("ATTACK_10_EXEMPTION_BAND_ABUSE", "ATTACK_14_ADVERSARIAL_BOUNDARY_PROBE",
         "ATTACK_4_CNP_HIGH_VELOCITY", "ATTACK_12_GEO_VELOCITY_ITINERARY"),
        evaluation_focus=("distributional anomaly", "feedback robustness", "sequence detection"),
    ),
    "ATTACK_9_OTP_RELAY_VISHING": AttackVectorProfile(
        "ATTACK_9_OTP_RELAY_VISHING", "attack_9_otp_relay_vishing.yaml",
        _g("compromised_real", "challenge_interception", "challenge_passed_unauthorized", "ecommerce",
           "foreign_device_after_pass", "single_victim_session", "vishing", "goods_purchase",
           "authentication_bypass_then_exploitation"),
        ("challenge_pass", "new_device", "social_engineering", "session_monetization"),
        ("ticket_band", "victim_rotation", "merchant_mix"),
        ("ATTACK_1_MFA_RESET_VOICE_CLONE", "ATTACK_4_CNP_HIGH_VELOCITY", "ATTACK_14_ADVERSARIAL_BOUNDARY_PROBE"),
        evaluation_focus=("3DS-not-terminal", "new-device evidence", "post-auth behaviour"),
    ),
    "ATTACK_10_EXEMPTION_BAND_ABUSE": AttackVectorProfile(
        "ATTACK_10_EXEMPTION_BAND_ABUSE", "attack_10_exemption_band_abuse.yaml",
        _g("compromised_real", "stolen_credentials", "unauthorized", "ecommerce", "tight_exemption_band",
           "single_or_small_pool", "policy_outcome_learning", "small_ticket_goods", "exploitation"),
        ("policy_evasion", "tight_band", "low_velocity", "challenge_avoidance"),
        ("band_center", "cadence", "merchant_mix"),
        ("ATTACK_8_LEARNED_THRESHOLD_STRUCTURING", "ATTACK_14_ADVERSARIAL_BOUNDARY_PROBE", "ATTACK_4_CNP_HIGH_VELOCITY"),
        evaluation_focus=("population distribution", "policy robustness", "low-and-slow detection"),
    ),
    "ATTACK_11_AGENTIC_SCOPE_EXPANSION": AttackVectorProfile(
        "ATTACK_11_AGENTIC_SCOPE_EXPANSION", "attack_11_agentic_scope_expansion.yaml",
        _g("genuine_delegator", "trusted_delegated_agent", "technically_authorized_scope_drift", "ecommerce",
           "escalation_and_category_widening", "stable_trusted_agent", "mandate_manipulation", "agent_purchase",
           "authorized_use_then_scope_drift"),
        ("trusted_device", "amount_escalation", "category_widening", "machine_regular_timing"),
        ("escalation_slope", "category_breadth", "cadence"),
        ("ATTACK_3_PROMPT_INJECTED_MERCHANT", "ATTACK_10_EXEMPTION_BAND_ABUSE", "ATTACK_14_ADVERSARIAL_BOUNDARY_PROBE"),
        ("human-granted mandate is not a PaymentMessage field", "catalog/prompt provenance is not modeled"),
        ("sequence reasoning", "mandate-drift proxy", "trusted-device blind spot"),
    ),
    "ATTACK_12_GEO_VELOCITY_ITINERARY": AttackVectorProfile(
        "ATTACK_12_GEO_VELOCITY_ITINERARY", "attack_12_geo_velocity_itinerary.yaml",
        _g("compromised_real", "cloned_instrument", "unauthorized", "ecommerce_travel_projection",
           "coherent_itinerary_sequence", "single_card_sequence", "narrative_fabrication", "travel_and_goods", "sequence_evasion"),
        ("sequence_coherence", "travel_narrative", "merchant_mix", "history_deviation"),
        ("itinerary_length", "merchant_mix", "ticket_band"),
        ("ATTACK_4_CNP_HIGH_VELOCITY", "ATTACK_8_LEARNED_THRESHOLD_STRUCTURING", "ATTACK_14_ADVERSARIAL_BOUNDARY_PROBE"),
        ("physical location is represented only by merchant/IP country in the current schema", "travel booking state is not modeled"),
        ("sequence novelty", "pairwise-rule weakness", "history-aware detection"),
    ),
    "ATTACK_13_MERCHANT_BUSTOUT": AttackVectorProfile(
        "ATTACK_13_MERCHANT_BUSTOUT", "attack_13_merchant_bustout.yaml",
        _g("merchant_shell", "fraudulent_merchant_account", "merchant_originated", "ecommerce",
           "quiet_ramp_then_bustout", "many_customers_one_young_merchant", "onboarding_fabrication",
           "merchant_settlement", "trust_build_then_cashout"),
        ("merchant_youth", "merchant_fanin", "high_ticket_wave", "trust_build"),
        ("ramp_length", "wave_width", "ticket_band"),
        ("ATTACK_3_PROMPT_INJECTED_MERCHANT", "ATTACK_7_SYNCHRONISED_BURST_CASHOUT", "ATTACK_6_VPA_RENTAL_MULE"),
        ("merchant onboarding documents/settlement state are outside PaymentMessage",),
        ("merchant age", "merchant-centric distribution shift", "bustout containment"),
    ),
    "ATTACK_14_ADVERSARIAL_BOUNDARY_PROBE": AttackVectorProfile(
        "ATTACK_14_ADVERSARIAL_BOUNDARY_PROBE", "attack_14_adversarial_boundary_probe.yaml",
        _g("compromised_real", "stolen_credentials", "unauthorized", "ecommerce", "probe_then_exploit",
           "single_or_small_pool", "black_box_outcome_recon", "goods_purchase", "reconnaissance_then_exploitation"),
        ("decision_oracle", "reconnaissance", "feature_sweep", "exploit_phase"),
        ("probe_mix", "ticket_band", "merchant_mix"),
        ("ATTACK_8_LEARNED_THRESHOLD_STRUCTURING", "ATTACK_10_EXEMPTION_BAND_ABUSE",
         "ATTACK_4_CNP_HIGH_VELOCITY", "ATTACK_7_SYNCHRONISED_BURST_CASHOUT"),
        evaluation_focus=("reconnaissance detection", "adaptive feedback", "model robustness"),
    ),
}


def profile_for_spec(spec_id: str) -> AttackVectorProfile:
    try:
        return PORTFOLIO[spec_id]
    except KeyError as exc:
        raise KeyError(f"attack spec {spec_id!r} is not mapped into the red-team portfolio") from exc


def portfolio_snapshot() -> dict:
    vectors = [PORTFOLIO[key].to_dict() for key in sorted(PORTFOLIO)]
    edges = [{"source": profile.spec_id, "target": target}
             for profile in PORTFOLIO.values() for target in profile.transitions]
    dimensions = list(asdict(next(iter(PORTFOLIO.values())).genome).keys())
    return {
        "model": "multidimensional-fraud-genome-v1",
        "dimensions": dimensions,
        "vector_count": len(vectors),
        "vectors": vectors,
        "edges": edges,
        "notes": ("AttackSpecs remain authoritative. Portfolio transitions are synthetic "
                  "campaign relationships used to stress the same Blue-Team payment stack."),
    }


def validate_portfolio() -> list[str]:
    errors: list[str] = []
    for spec_id, profile in PORTFOLIO.items():
        if profile.spec_id != spec_id:
            errors.append(f"{spec_id}: key/spec mismatch")
        if not profile.transitions:
            errors.append(f"{spec_id}: no transitions")
        for target in profile.transitions:
            if target not in PORTFOLIO:
                errors.append(f"{spec_id}: unknown transition target {target}")
        if not profile.mutation_axes:
            errors.append(f"{spec_id}: no mutation axes")
        if not profile.primitives:
            errors.append(f"{spec_id}: no primitives")
    return errors


__all__ = ["FraudGenome", "AttackVectorProfile", "PORTFOLIO", "profile_for_spec",
           "portfolio_snapshot", "validate_portfolio"]
