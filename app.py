
"""
KOHLER AI Bathroom Designer - Gradio UI

Run:
    python app.py
"""

from pathlib import Path
import gradio as gr

from backend import (
    BASE_DIR,
    SPATIAL_CLEARANCE_MM,
    create_bundle_markdown,
    create_final_design_result,
    create_validation_markdown,
    explain_design,
    generate_recommendations,
    row_to_bundle,
    save_layout_image,
)


PROTOTYPE_NOTICE = """
> **Prototype data notice:** Prices and some dimensions in this prototype
> may be estimated where source information was unavailable. They are for
> demonstrating budget-aware and spatial recommendation logic, not for
> real-world purchasing. Verify current KOHLER pricing, dimensions,
> availability, installation requirements and compatibility before purchase.
"""


def format_option_card(name, row):
    if row is None:
        return f"### {name}\n\nNo feasible recommendation was found."

    return "\n".join([
        f"## {name}",
        "",
        f"### ₹{float(row.get('total_price_inr', 0)):,.0f}",
        "",
        f"**Budget:** {row.get('budget_status', 'UNKNOWN')}",
        f"**Space:** {row.get('space_status', 'UNKNOWN')}",
        f"**Compatibility:** {row.get('compatibility_status', 'UNKNOWN')}",
        "",
        f"Theme score: `{float(row.get('theme_score', 0)):.2f}`",
        f"Feature score: `{float(row.get('feature_score', 0)):.2f}`",
    ])


def render_selected(
    recommendation_result,
    selected_option,
):
    if not recommendation_result:
        return (
            "Generate a bathroom design first.",
            "No selected design.",
            None,
            "No validation result.",
            "No explanation yet.",
        )

    row = recommendation_result["recommendations"].get(selected_option)

    if row is None:
        return (
            f"No feasible `{selected_option}` recommendation is available.",
            "No selected product bundle.",
            None,
            "No validation result.",
            "No explanation available.",
        )

    result = create_final_design_result(
        recommendation_result,
        selected_option,
    )

    requirements = recommendation_result["requirements"]
    bundle = result["products"]
    layout = result["layout"]

    summary = "\n".join([
        "## Selected Design",
        "",
        f"**Recommendation:** {selected_option}",
        f"**Bathroom:** {requirements['length_ft']:g} × "
        f"{requirements['width_ft']:g} ft",
        f"**Theme:** {requirements['theme']}",
        f"**Budget:** ₹{requirements['budget_inr']:,.0f}",
        f"**Estimated bundle:** ₹{result['total_price_inr']:,.0f}",
        f"**Remaining:** ₹{result['remaining_budget_inr']:,.0f}",
    ])

    bundle_md = create_bundle_markdown(row, bundle)

    validation_md = create_validation_markdown(
        row,
        requirements,
        bundle,
    )

    layout_path = save_layout_image(
        layout,
        requirements,
        selected_option,
    )

    explanation = explain_design(
        row,
        bundle,
        requirements,
        layout,
    )

    return (
        summary,
        bundle_md,
        layout_path,
        validation_md,
        explanation,
    )


def generate_ui_designs(
    length_ft,
    width_ft,
    budget,
    theme,
    required_fixtures,
):
    try:
        result = generate_recommendations(
            length_ft=length_ft,
            width_ft=width_ft,
            budget_inr=budget,
            theme=theme,
            required_fixtures=required_fixtures,
        )

        recommendations = result["recommendations"]

        best = recommendations.get("Best Match")
        budget_row = recommendations.get("Budget Optimized")
        premium = recommendations.get("Premium")

        default_option = "Best Match"
        summary, bundle, layout, validation, explanation = render_selected(
            result,
            default_option,
        )

        return (
            format_option_card("Best Match", best),
            format_option_card("Budget Optimized", budget_row),
            format_option_card("Premium", premium),
            result,
            default_option,
            summary,
            bundle,
            layout,
            validation,
            explanation,
        )

    except Exception as exc:
        message = (
            "## Design generation could not be completed\n\n"
            f"**Reason:** {type(exc).__name__}: {exc}\n\n"
            "Try a larger bathroom, a higher budget, or fewer required fixtures."
        )

        return (
            message,
            message,
            message,
            None,
            "Best Match",
            message,
            "No product bundle.",
            None,
            message,
            "No explanation.",
        )


def build_app():
    css = """
    .app-wrap { max-width: 1250px; margin: auto; }
    .header { padding: 28px 10px 18px; text-align: center; }
    .brand { letter-spacing: 4px; font-size: 18px; font-weight: 700; }
    .title { font-size: 42px; font-weight: 700; margin: 8px 0; }
    .subtitle { font-size: 17px; opacity: .72; }
    .section-title { font-size: 23px; font-weight: 700; margin-top: 25px; }
    .card { border-radius: 14px; padding: 8px; }
    """

    with gr.Blocks(
        title="KOHLER AI Bathroom Designer",
        css=css,
        theme=gr.themes.Soft(),
    ) as demo:

        state = gr.State(None)

        gr.HTML("""
        <div class="header">
            <div class="brand">KOHLER</div>
            <div class="title">AI Bathroom Designer</div>
            <div class="subtitle">
                Constraint-aware bathroom planning based on space, budget
                and design preferences.
            </div>
        </div>
        """)

        gr.Markdown(PROTOTYPE_NOTICE)

        gr.Markdown("## 01 — Tell us about your bathroom")

        with gr.Row():
            with gr.Column():
                length = gr.Number(
                    label="Bathroom Length (ft)",
                    value=8,
                    minimum=1,
                    maximum=30,
                    step=0.5,
                )
                width = gr.Number(
                    label="Bathroom Width (ft)",
                    value=6,
                    minimum=1,
                    maximum=30,
                    step=0.5,
                )

            with gr.Column():
                budget = gr.Number(
                    label="Budget (₹)",
                    value=150000,
                    minimum=10000,
                    step=5000,
                )
                theme = gr.Dropdown(
                    label="Design Theme",
                    choices=[
                        "Minimalist Modern",
                        "Classic Luxury",
                        "Japanese Zen",
                    ],
                    value="Minimalist Modern",
                )

        fixtures = gr.CheckboxGroup(
            label="Required Fixtures",
            choices=[
                "Toilet",
                "Sink",
                "Faucet",
                "Vanity",
                "Bathtub",
            ],
            value=["Toilet", "Sink", "Faucet"],
        )

        design_button = gr.Button(
            "✨ Design My Bathroom",
            variant="primary",
            size="lg",
        )

        gr.Markdown("## 02 — AI Recommendations")

        with gr.Row():
            best_output = gr.Markdown(
                "Your Best Match recommendation will appear here.",
                elem_classes="card",
            )
            budget_output = gr.Markdown(
                "Your Budget Optimized recommendation will appear here.",
                elem_classes="card",
            )
            premium_output = gr.Markdown(
                "Your Premium recommendation will appear here.",
                elem_classes="card",
            )

        selected_option = gr.Radio(
            choices=["Best Match", "Budget Optimized", "Premium"],
            value="Best Match",
            label="Select a recommendation to inspect",
        )

        gr.Markdown("## 03 — Selected Bathroom Design")

        with gr.Row():
            with gr.Column():
                selected_summary = gr.Markdown(
                    "Generate a design first."
                )
                selected_products = gr.Markdown(
                    "No selected product bundle."
                )
            with gr.Column():
                layout_output = gr.Image(
                    label="2D Bathroom Layout",
                    type="filepath",
                )

        gr.Markdown("## 04 — Design Validation")
        validation_output = gr.Markdown(
            "Validation results will appear here."
        )

        gr.Markdown("## 05 — Why this design?")
        explanation_output = gr.Markdown(
            "Your AI explanation will appear here."
        )

        gr.Markdown(
            "KOHLER AI Bathroom Designer · Prototype"
        )

        design_button.click(
            fn=generate_ui_designs,
            inputs=[length, width, budget, theme, fixtures],
            outputs=[
                best_output,
                budget_output,
                premium_output,
                state,
                selected_option,
                selected_summary,
                selected_products,
                layout_output,
                validation_output,
                explanation_output,
            ],
        )

        selected_option.change(
            fn=render_selected,
            inputs=[state, selected_option],
            outputs=[
                selected_summary,
                selected_products,
                layout_output,
                validation_output,
                explanation_output,
            ],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
