from dataclasses import dataclass


@dataclass(frozen=True)
class HybridWeights:
    collaborative: float = 0.40
    content: float = 0.30
    rating_quality: float = 0.15
    preference_match: float = 0.10
    novelty: float = 0.05

    def validate(self) -> None:
        total = sum(vars(self).values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Hybrid weights must sum to 1.0, got {total:.6f}")


def hybrid_score(
    collaborative: float,
    content: float,
    rating_quality: float,
    preference_match: float,
    novelty: float,
    weights: HybridWeights | None = None,
) -> float:
    weights = weights or HybridWeights()
    weights.validate()

    return (
        weights.collaborative * collaborative
        + weights.content * content
        + weights.rating_quality * rating_quality
        + weights.preference_match * preference_match
        + weights.novelty * novelty
    )