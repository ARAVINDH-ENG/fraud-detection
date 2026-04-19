import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))

MODELS_TO_TRY = [
    "models/gemini-2.0-flash-lite",
    "models/gemini-2.0-flash",
    "models/gemini-1.5-flash-8b",
    "models/gemini-1.5-flash",
]

def rule_based_explanation(probability: float, shap_features: dict) -> str:
    """
    Fallback when all LLM quota exceeded.
    Builds complete explanation from SHAP values deterministically.
    """
    top     = sorted(shap_features.items(), key=lambda x: abs(x[1]), reverse=True)
    direction = "FRAUD" if probability > 0.5 else "LEGITIMATE"
    pct     = round(probability * 100, 1)

    feature_descriptions = []
    for feat, val in top:
        if val > 0:
            feature_descriptions.append(
                f"{feat} contributed strongly toward fraud (SHAP: +{val:.4f})"
            )
        else:
            feature_descriptions.append(
                f"{feat} contributed toward legitimate transaction (SHAP: {val:.4f})"
            )

    explanation = (
        f"This transaction scored {pct}% fraud probability and was classified as {direction}. "
        f"The top contributing features were: {', '.join(feature_descriptions)}. "
    )

    if probability > 0.7:
        explanation += (
            "The combination of high-magnitude positive SHAP values indicates "
            "a strong fraud pattern consistent with known fraudulent transactions — "
            "automatic rejection applied."
        )
    elif probability > 0.3:
        explanation += (
            "Mixed feature signals detected — some features push toward fraud "
            "while others suggest legitimacy. Flagged for human analyst review."
        )
    else:
        explanation += (
            "All top features pushed toward legitimate transaction patterns. "
            "No significant fraud signals detected — transaction automatically approved."
        )

    return explanation

def explain_decision(probability: float, shap_features: dict) -> str:
    shap_formatted = "\n".join([
        f"  {k}: {v:+.4f} ({'toward fraud' if v > 0 else 'toward legitimate'})"
        for k, v in shap_features.items()
    ])

    prompt = f"""You are a fraud detection explainability engine.

Transaction fraud probability: {probability:.1%}

SHAP feature contributions (positive = pushes toward fraud):
{shap_formatted}

Explain in 2-3 complete sentences why this transaction received this fraud score.
Reference specific feature names and their SHAP values.
Do not speculate beyond the provided data."""

    for model_name in MODELS_TO_TRY:
        try:
            llm      = genai.GenerativeModel(model_name)
            response = llm.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=600
                )
            )
            return response.text.strip()
        except Exception as e:
            print(f"Model {model_name} failed: {e}")
            continue

    return rule_based_explanation(probability, shap_features)
