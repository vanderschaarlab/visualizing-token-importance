# Main execution

import json
import matplotlib.pyplot as plt
import numpy as np
import timeit
from tqdm import tqdm

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
    "If": ["When", "Unless", "Though"],
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
]  # "mistralai/Mistral-7B-Instruct-v0.2", "meta-llama/Meta-Llama-3.1-8B-Instruct", "google/gemma-2-9b-it"]


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
            # For this experiment, you want to change the "distance" argument to one of 'cosine', 'l1', or 'l2'
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

# If you execute the following code for all the LLMs
# You should get the values below
# prefix = "scores/"
# with open(prefix + "cosine/SmolLM-135M_scores.json") as f:
#     data1 = json.load(f)
# with open(prefix + "l1/SmolLM-135M_scores.json") as f:
#     data2 = json.load(f)
# with open(prefix + "l2/SmolLM-135M_scores.json") as f:
#     data3 = json.load(f)
# d = {"cosine": [v[0] for v in data1.values()], "l1": [v[0] for v in data2.values()], "l2": [v[0] for v in data3.values()]}
# df = pd.DataFrame(data=d)
# corr = df.corr(method='spearman')
# sns.heatmap(corr, annot=True, fmt=".2f")
# plt.savefig("cosine_l1_l2_gemma-2-9b-it.pdf")

SmolLM_135M = [0.96, 0.97, 1.00]
Phi_3_mini_4k_instruct = [0.99, 0.99, 1.00]
MagicPrompt_Stable_Diffusion = [1.00, 1.00, 1.00]
SWNorth_gpt_4_0613_20231016 = [0.97, 0.98, 0.99]
Mistral_7B_Instruct_v0_2 = [0.98, 0.98, 1.00]
Meta_Llama_3_1_8B_Instruct = [0.98, 0.98, 1.00]
gpt_35_1106_vdsT_AE = [0.99, 0.99, 1.00]
gemma_2_9b_it = [0.98, 0.99, 1.00]
plt.figure()
plt.scatter([0, 1, 2], SmolLM_135M, label="SmolLM-135M")
plt.scatter([0, 1, 2], Phi_3_mini_4k_instruct, label="Phi-3-mini-4k-instruct")
plt.scatter(
    [0, 1, 2], MagicPrompt_Stable_Diffusion, label="MagicPrompt-Stable-Diffusion"
)
plt.scatter([0, 1, 2], SWNorth_gpt_4_0613_20231016, label="SWNorth-gpt-4-0613-20231016")
plt.scatter([0, 1, 2], Mistral_7B_Instruct_v0_2, label="Mistral-7B-Instruct-v0.2")
plt.scatter([0, 1, 2], Meta_Llama_3_1_8B_Instruct, label="Meta-Llama-3.1-8B-Instruct")
plt.scatter([0, 1, 2], gpt_35_1106_vdsT_AE, label="gpt-35-1106-vdsT-AE")
plt.scatter([0, 1, 2], gemma_2_9b_it, label="gemma-2-9b-it")
plt.legend(fontsize=8)
plt.plot([0, 1, 2], SmolLM_135M, "--")
plt.plot([0, 1, 2], Phi_3_mini_4k_instruct, "--")
plt.plot([0, 1, 2], MagicPrompt_Stable_Diffusion, "--")
plt.plot([0, 1, 2], SWNorth_gpt_4_0613_20231016, "--")
plt.plot([0, 1, 2], Mistral_7B_Instruct_v0_2, "--")
plt.plot([0, 1, 2], Meta_Llama_3_1_8B_Instruct, "--")
plt.plot([0, 1, 2], gpt_35_1106_vdsT_AE, "--")
plt.plot([0, 1, 2], gemma_2_9b_it, "--")
plt.ylabel("Spearman correlation")
plt.xticks([0, 1, 2], ["cosine_l1", "cosine_l2", "l1_l2"])
plt.title("Spearman correlation between cosine, l1, and l2 for various LLMs")
plt.savefig("cosine_l1_l2_all.pdf")
