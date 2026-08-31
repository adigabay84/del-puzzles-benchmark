"""
File manifest for the four benchmark quadrants.

QUARTERS maps each quadrant (Q1-Q4) to the post-processed result CSVs
belonging to it, along two axes:

  - Narrative type: classic puzzles (Blue-Eyed Islanders, Muddy Children,
    Wise Men) vs. new story variations (Health Screening, Olympic Games,
    Safety Inspection, Singing Contest)
  - Inference type: symmetric vs. asymmetric

    Q1: Classic × Symmetric    Q3: New × Symmetric
    Q2: Classic × Asymmetric   Q4: New × Asymmetric

Each quadrant lists one CSV per puzzle/model combination, with paths
relative to the project root. Used by the analysis scripts (accuracy
figure, macro F1 table).
"""


QUARTERS: dict[str, list[str]] = {
    "Q1": [  # Classic narratives × Symmetric inference
        "test_results/symmetric_inference/blue_eyed_islanders/closed_inference_blue_eyed_islanders_gemini_2.5_pro_post_process.csv",
        "test_results/symmetric_inference/blue_eyed_islanders/closed_inference_blue_eyed_islanders_gpt_5_nano_post_process.csv",
        "test_results/symmetric_inference/blue_eyed_islanders/closed_inference_blue_eyed_islanders_gpt_5_post_process.csv",
        "test_results/symmetric_inference/blue_eyed_islanders/closed_inference_blue_eyed_islanders_qwen3-235b-a22b-thinking_post_process.csv",
        "test_results/symmetric_inference/muddy_children/baseline_post_process_first_knowledge_met.csv",
        "test_results/symmetric_inference/muddy_children/muddy_children_puzzles_dataset_first_deduction_qwen3-235b-a22b-thinking_post_process.csv",
        "test_results/symmetric_inference/wise_men/closed_inference_wise_men_gemini_2.5_pro_post_process.csv",
        "test_results/symmetric_inference/wise_men/closed_inference_wise_men_gpt_5_nano_post_process.csv",
        "test_results/symmetric_inference/wise_men/closed_inference_wise_men_gpt_5_post_process.csv",
        "test_results/symmetric_inference/wise_men/closed_inference_wise_men_qwen3-235b-a22b-thinking_post_process.csv",
    ],
    "Q2": [  # Classic narratives × Asymmetric inference
        "test_results/asymmetric_inference/blue_eyed_islanders/blue_eyed_islanders_additional_analysis_gemini_2.5_pro_post_process.csv",
        "test_results/asymmetric_inference/blue_eyed_islanders/blue_eyed_islanders_additional_analysis_gpt_5_nano_post_process.csv",
        "test_results/asymmetric_inference/blue_eyed_islanders/blue_eyed_islanders_additional_analysis_gpt_5_post_process.csv",
        "test_results/asymmetric_inference/blue_eyed_islanders/blue_eyed_islanders_additional_analysis_qwen3-235b-a22b-thinking_post_process.csv",
        "test_results/asymmetric_inference/muddy_children/muddy_children_additional_analysis_gemini_2.5_pro_post_process.csv",
        "test_results/asymmetric_inference/muddy_children/muddy_children_additional_analysis_gpt_5_nano_post_process.csv",
        "test_results/asymmetric_inference/muddy_children/muddy_children_additional_analysis_gpt_5_post_process.csv",
        "test_results/asymmetric_inference/muddy_children/muddy_children_additional_analysis_qwen3-235b-a22b-thinking_post_process.csv",
        "test_results/asymmetric_inference/wise_men/wise_men_additional_analysis_gemini_2.5_pro_post_process.csv",
        "test_results/asymmetric_inference/wise_men/wise_men_additional_analysis_gpt_5_nano_post_process.csv",
        "test_results/asymmetric_inference/wise_men/wise_men_additional_analysis_gpt_5_post_process.csv",
        "test_results/asymmetric_inference/wise_men/wise_men_additional_analysis_qwen3-235b-a22b-thinking_post_process.csv",
    ],
    "Q3": [  # New narratives × Symmetric inference
        "test_results/symmetric_inference/health_screening/closed_inference_health_screening_gemini_2.5_pro_post_process.csv",
        "test_results/symmetric_inference/health_screening/closed_inference_health_screening_gpt_5_nano_post_process.csv",
        "test_results/symmetric_inference/health_screening/closed_inference_health_screening_gpt_5_post_process.csv",
        "test_results/symmetric_inference/health_screening/closed_inference_health_screening_qwen3-235b-a22b-thinking_post_process.csv",
        "test_results/symmetric_inference/olympic_games/closed_inference_olympic_games_gemini_2.5_pro_post_process.csv",
        "test_results/symmetric_inference/olympic_games/closed_inference_olympic_games_gpt_5_nano_post_process.csv",
        "test_results/symmetric_inference/olympic_games/closed_inference_olympic_games_gpt_5_post_process.csv",
        "test_results/symmetric_inference/olympic_games/closed_inference_olympic_games_qwen3-235b-a22b-thinking_post_process.csv",
        "test_results/symmetric_inference/safety_inspection/closed_inference_safety_inspection_gemini_2.5_pro_post_process.csv",
        "test_results/symmetric_inference/safety_inspection/closed_inference_safety_inspection_gpt_5_nano_post_process.csv",
        "test_results/symmetric_inference/safety_inspection/closed_inference_safety_inspection_gpt_5_post_process.csv",
        "test_results/symmetric_inference/safety_inspection/closed_inference_safety_inspection_qwen3-235b-a22b-thinking_post_process.csv",
        "test_results/symmetric_inference/singing_contest/closed_inference_singing_contest_gemini_2.5_pro_post_process.csv",
        "test_results/symmetric_inference/singing_contest/closed_inference_singing_contest_gpt_5_nano_post_process.csv",
        "test_results/symmetric_inference/singing_contest/closed_inference_singing_contest_gpt_5_post_process.csv",
        "test_results/symmetric_inference/singing_contest/closed_inference_singing_contest_qwen3-235b-a22b-thinking_post_process.csv",
    ],
    "Q4": [  # New narratives × Asymmetric inference
        "test_results/asymmetric_inference/health_screening/health_screening_additional_analysis_gemini_2.5_pro_post_process.csv",
        "test_results/asymmetric_inference/health_screening/health_screening_additional_analysis_gpt_5_nano_post_process.csv",
        "test_results/asymmetric_inference/health_screening/health_screening_additional_analysis_gpt_5_post_process.csv",
        "test_results/asymmetric_inference/health_screening/health_screening_additional_analysis_qwen3-235b-a22b-thinking_post_process.csv",
        "test_results/asymmetric_inference/olympic_games/olympic_games_additional_analysis_gemini_2.5_pro_post_process.csv",
        "test_results/asymmetric_inference/olympic_games/olympic_games_additional_analysis_gpt_5_nano_post_process.csv",
        "test_results/asymmetric_inference/olympic_games/olympic_games_additional_analysis_gpt_5_post_process.csv",
        "test_results/asymmetric_inference/olympic_games/olympic_games_additional_analysis_qwen3-235b-a22b-thinking_post_process.csv",
        "test_results/asymmetric_inference/safety_inspection/safety_inspection_additional_analysis_gemini_2.5_pro_post_process.csv",
        "test_results/asymmetric_inference/safety_inspection/safety_inspection_additional_analysis_gpt_5_nano_post_process.csv",
        "test_results/asymmetric_inference/safety_inspection/safety_inspection_additional_analysis_gpt_5_post_process.csv",
        "test_results/asymmetric_inference/safety_inspection/safety_inspection_additional_analysis_qwen3-235b-a22b-thinking_post_process.csv",
        "test_results/asymmetric_inference/singing_contest/singing_contest_additional_analysis_gemini_2.5_pro_post_process.csv",
        "test_results/asymmetric_inference/singing_contest/singing_contest_additional_analysis_gpt_5_nano_post_process.csv",
        "test_results/asymmetric_inference/singing_contest/singing_contest_additional_analysis_gpt_5_post_process.csv",
        "test_results/asymmetric_inference/singing_contest/singing_contest_additional_analysis_qwen3-235b-a22b-thinking_post_process.csv",
    ],
}
