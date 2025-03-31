# Main execution

import json
import matplotlib.pyplot as plt
import numpy as np
import timeit
from tqdm import tqdm
import pandas as pd
import seaborn as sns

from dbsa.utils.setup_llm import get_responses, get_embeddings
from dbsa.data.generate_data import eval_prompt, tokenize_and_prepare_for_scoring
from dbsa.model.core import compute_energy_distance_fn

# Main execution

closest_words = {
    "Company": ["Corporation", "Business", "Firm"],
    "A": ["An", "The", "One"],
    "Pays": ["Gives", "Provides", "Funds"],
    "B": ["A", "C", "D"],
    "$10": ["$20", "$5", "$15"],
    "Million": ["Billion", "Thousand", "Hundred"],
    "For": ["To", "With", "By"],
    "Revolutionary": ["Radical", "Innovative", "Transformative"],
    "AI": ["Artificial Intelligence", "Machine Learning", "Robotics"],
    "Software": ["Program", "Application", "System"],
    "Within": ["Inside", "Among", "Throughout"],
    "12": ["11", "13", "14"],
    "Months": ["Weeks", "Years", "Days"],
    ".": [",", "!", "?"],
    "Fails": ["Falters", "Falls", "Fails"],
    "To": ["For", "With", "By"],
    "Deliver": ["Provide", "Supply", "Dispatch"],
    "Fully": ["Completely", "Totally", "Entirely"],
    "Functional": ["Operational", "Working", "Practical"],
    "Product": ["Goods", "Item", "Commodity"],
    "By": ["Through", "With", "From"],
    "The": ["That", "This", "A"],
    "Deadline": ["Time limit", "Due date", "Cut-off"],
    ",": [";", ".", ":"],
    "They": ["Them", "Those", "These"],
    "Must": ["Should", "Have to", "Need to"],
    "Refund": ["Reimburse", "Return", "Repay"],
    "50": ["40", "60", "70"],
    "%": ["Percent", "Percentage", "Proportion"],
    "Of": ["From", "For", "In"],
    "Payment": ["Fee", "Cost", "Charge"],
    "And": ["Or", "But", "Yet"],
    "Provide": ["Supply", "Give", "Offer"],
    "An": ["A", "The", "One"],
    "Additional": ["Extra", "More", "Further"],
    "3": ["2", "4", "5"],
    "Development": ["Growth", "Progress", "Evolution"],
    "Free": ["No-cost", "Complimentary", "Gratis"],
    "However": ["Nevertheless", "Still", "Yet"],
    "Delay": ["Postpone", "Stall", "Holdup"],
    "Is": ["Was", "Are", "Be"],
    "Due": ["Owing", "Payable", "Outstanding"],
    "Circumstances": ["Situations", "Conditions", "Events"],
    "Beyond": ["Past", "Over", "Outside"],
    "’": ["'s", "'ve", "'d"],
    "S": ["'s", "Z", "X"],
    "Reasonable": ["Fair", "Rational", "Sensible"],
    "Control": ["Power", "Command", "Authority"],
    "These": ["Those", "Such", "The"],
    "Penalties": ["Fines", "Sanctions", "Punishments"],
    "Shall": ["Will", "Must", "Should"],
    "Not": ["No", "Never", "Not"],
    "Apply": ["Use", "Implement", "Employ"],
    "This": ["That", "The", "This"],
    "Agreement": ["Contract", "Deal", "Accord"],
    "Governed": ["Ruled", "Controlled", "Regulated"],
    "California": ["Los Angeles", "San Francisco", "Hollywood"],
    "Law": ["Rule", "Legislation", "Statute"],
    "Any": ["Some", "Every", "All"],
    "Disputes": ["Arguments", "Controversies", "Conflicts"],
    "Be": ["Is", "Are", "Were"],
    "Resolved": ["Solved", "Settled", "Decided"],
    "Through": ["By", "Via", "With"],
    "Binding": ["Obligatory", "Mandatory", "Compulsory"],
    "Arbitration": ["Mediation", "Adjudication", "Conciliation"],
}
closest_words = {k.lower(): [w.lower() for w in v] for k, v in closest_words.items()}
MODEL_IDS = [
    "SWNorth-gpt-4-0613-20231016",
    "gpt-35-1106-vdsT-AE",
    "HuggingFaceTB/SmolLM-135M",
    "Gustavosta/MagicPrompt-Stable-Diffusion",
    "microsoft/Phi-3-mini-4k-instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "google/gemma-2-9b-it",
]


def assign_scores(tokens, token_positions, model_id, num_samples=3):
    def perturb_sentence(tokens, perturb_index, neighbor):
        new_tokens = tokens.copy()
        new_tokens[perturb_index] = neighbor
        return new_tokens

    # Get baseline response
    baseline_prompt = eval_prompt(" ".join(tokens))
    baseline_responses = get_responses(baseline_prompt, model_id, n=num_samples)
    baseline_embeddings = get_embeddings(baseline_responses)

    model_responses = dict()
    model_embeddings = dict()

    for token, data in tqdm(token_positions.items(), desc="Processing tokens"):
        token = token.lower()
        assert token in closest_words, f"Token '{token}' not in closest_words"
        neighbors = closest_words[token]

        for position in data["positions"]:
            model_responses[f"{token}_{position}"] = []
            model_embeddings[f"{token}_{position}"] = []

            for neighbor in tqdm(
                neighbors,
                desc=f"Analyzing neighbors for token '{token}' at position {position}",
                leave=False,
            ):
                # Perturb the sentence and get perturbed responses
                perturbed_tokens = perturb_sentence(tokens, position, neighbor)
                perturbed_prompt = eval_prompt(" ".join(perturbed_tokens))
                # with open("prompt.txt", "a") as f:
                #     f.write(perturbed_prompt)
                #     f.write("\n")
                perturbed_responses = get_responses(
                    perturbed_prompt, model_id, n=num_samples
                )
                perturbed_embeddings = get_embeddings(perturbed_responses)

                model_responses[f"{token}_{position}"].append(perturbed_responses)
                model_embeddings[f"{token}_{position}"].append(perturbed_embeddings)

    with open(f"responses/{model_id.split('/')[-1]}_responses.json", "w") as f:
        json.dump(model_responses, f)
    with open(f"responses/{model_id.split('/')[-1]}_baseline_responses.json", "w") as f:
        json.dump(baseline_responses, f)
    np.savez(f"responses/{model_id.split('/')[-1]}_embeddings", model_embeddings)
    np.savez(
        f"responses/{model_id.split('/')[-1]}_baseline_embeddings", baseline_embeddings
    )

    return token_positions


text_legal = """Company A pays Company B $10 million for a revolutionary AI software within 12 months . If Company B fails to deliver a fully functional product by the deadline , they must refund 50 % of the payment and provide an additional 3 months of development free . However , if the delay is due to circumstances beyond Company B ’ s reasonable control , these penalties shall not apply . This agreement is governed by California law and any disputes shall be resolved through binding arbitration ."""
tokens, token_positions = tokenize_and_prepare_for_scoring(text_legal)
for model_id in MODEL_IDS:
    start_time = timeit.default_timer()
    assign_scores(tokens, token_positions, model_id, num_samples=3)
    elapsed_time = timeit.default_timer() - start_time
    print(f"The code took {elapsed_time} seconds to run.")

prefix = "responses/"
for model_id in MODEL_IDS:
    with open(prefix + f"{model_id.split('/')[-1]}_responses.json", "r") as f:
        model_responses = json.load(f)
    with open(prefix + f"{model_id.split('/')[-1]}_baseline_responses.json", "r") as f:
        baseline_responses = json.load(f)
    model_embeddings = np.load(
        prefix + f"{model_id.split('/')[-1]}_embeddings.npz", allow_pickle=True
    )
    baseline_embeddings = np.load(
        prefix + f"{model_id.split('/')[-1]}_baseline_embeddings.npz", allow_pickle=True
    )["arr_0"][()]

    d = model_embeddings["arr_0"][()]
    result = dict()
    for k in tqdm(d.keys()):
        neighbor_scores = []
        neighbor_pvals = []
        result[k] = []

        for perturbed_embeddings in d[k]:
            distance, pval = compute_energy_distance_fn(
                baseline_embeddings, perturbed_embeddings
            )
            neighbor_scores.append(distance)
            neighbor_pvals.append(pval)

        # Store the average score for this token occurrence
        result[k].append(np.mean(neighbor_scores))
        result[k].append(np.mean(neighbor_pvals))

    with open(prefix + f"{model_id.split('/')[-1]}_scores.json", "w") as f:
        json.dump(result, f)

# Plotting

prefix = "scores/"
with open(prefix + "SWNorth-gpt-4-0613-20231016_scores.json") as f:
    data4 = json.load(f)
with open(prefix + "gpt-35-1106-vdsT-AE_scores.json") as f:
    data7 = json.load(f)
with open(prefix + "SmolLM-135M_scores.json") as f:
    data1 = json.load(f)
with open(prefix + "Phi-3-mini-4k-instruct_scores.json") as f:
    data2 = json.load(f)
with open(prefix + "MagicPrompt-Stable-Diffusion_scores.json") as f:
    data3 = json.load(f)
with open(prefix + "Mistral-7B-Instruct-v0.2_scores.json") as f:
    data5 = json.load(f)
with open(prefix + "Meta-Llama-3.1-8B-Instruct_scores.json") as f:
    data6 = json.load(f)
with open(prefix + "gemma-2-9b-it_scores.json") as f:
    data8 = json.load(f)
delete = []
for k in data1.keys():
    if k not in data4:
        delete.append(k)
for k in delete:
    data1.pop(k)
    data2.pop(k)
    data3.pop(k)
    data5.pop(k)
    data6.pop(k)
    data8.pop(k)
d = {
    "SmolLM-135M": [v[0] for v in data1.values()],
    "Phi-3-mini-4k-instruct": [v[0] for v in data2.values()],
    "MagicPrompt-Stable-Diffusion": [v[0] for v in data3.values()],
    "SWNorth-gpt-4-0613-20231016": [v[0] for v in data4.values()],
    "Mistral-7B-Instruct-v0.2": [v[0] for v in data5.values()],
    "Meta-Llama-3.1-8B-Instruct": [v[0] for v in data6.values()],
    "gpt-35-1106-vdsT-AE": [v[0] for v in data7.values()],
    "gemma-2-9b-it": [v[0] for v in data8.values()],
}
df = pd.DataFrame(data=d)
corr = df.corr(method="spearman")
sns.heatmap(corr, annot=True, fmt=".2f")
plt.show()
