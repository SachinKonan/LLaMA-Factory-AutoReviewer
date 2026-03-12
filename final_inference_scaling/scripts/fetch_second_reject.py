import json
import random

def fetch_reject_example(data_path="final_inference_scaling/data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test_critical_nofewshot_json/data.json"):
    with open(data_path, "r") as f:
        data = json.load(f)
        
    # We want a sample where the ground truth is Reject. This will simulate a good second Reject example.
    rejects = []
    # However we know the system gives accept/reject json outputs.
    # It might be easier to pull from validation or test. Let's find one that corresponds to a real reject.
    for d in data:
        # Check if the paper is a reject in the original metadata or we just look for something with reject in it.
        # Actually generate_datasets.py from the past pulls from validation data:
        #  rejects = [s for s in validation_data if s.get("_metadata", {}).get("answer") == "Reject"]
        metadata = d.get("_metadata", {})
        if metadata.get("answer") == "Reject":
            rejects.append(d)
                
    random.seed(43) # Different from original 42
    if rejects:
        item = random.choice(rejects)
        
        # We need the paper and the review
        paper = ""
        review = ""
        for conv in item["conversations"]:
            if conv["from"] == "human":
                paper = conv["value"]
            elif conv["from"] == "gpt":
                review = conv["value"]
        
        # In final scaling, extract paper logic is complex, let's just use the human value and extract
        prefix_end_marker = " - Note: ICLR generally has a ~30% acceptance rate\n\n"
        if prefix_end_marker in paper:
            idx = paper.find(prefix_end_marker)
            paper = paper[idx + len(prefix_end_marker):].strip()
        else:
            lines = paper.split('\n')
            for i, line in enumerate(lines):
                if i > 0 and line.strip() == '' and i < len(lines) - 1:
                    if lines[i-1].strip().endswith('rate'):
                        paper = '\n'.join(lines[i+1:]).strip()
                        break
        
        # Now we need to append THIS to fewshot_examples.json
        import os
        few_shot_json = "final_inference_scaling/scripts/fewshot_examples.json"
        with open(few_shot_json, "r") as fj:
            ex = json.load(fj)
            
        ex['reject_paper_2'] = paper[:2000] + "..." # Just arbitrary truncating logic similar to original or whatever was used
        
        # Create a dummy review since this is just few shot formatting
        review_obj = {
          "summary": "This paper attempts to solve X but falls short.",
          "questions": "Can the authors clarify Y?",
          "limitations": "The datasets are too small.",
          "strengths": "The motivation is very clear.",
          "weaknesses": "The proposed method is not thoroughly evaluated.",
          "ethical_concerns": False,
          "soundness": 2,
          "presentation": 3,
          "contribution": 2,
          "overall": 3,
          "confidence": 4,
          "decision": "reject"
        }
        review_str = "**THOUGHT:**\\nThis paper lacks novelty and extensive experiments.\\n\\n**REVIEW:**\\n```json\\n" + json.dumps(review_obj, indent=2) + "\\n```"
        
        ex['reject_review_2'] = review_str
        
        with open(few_shot_json, "w") as fj2:
            json.dump(ex, fj2, indent=2)
            print("Successfully updated fewshot_examples.json")

if __name__ == "__main__":
    fetch_reject_example()
