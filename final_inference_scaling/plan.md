

Prompt Strategies:

# Original prompt:
You are an expert academic reviewer tasked with evaluating research papers.

 "I am giving you a paper. I want to predict its acceptance outcome at ICLR."
        "\n - Your answer will either be: \\boxed{Accept} or \\boxed{Reject}"
        "\n - Note: ICLR generally has a ~30% acceptance rate"

# New prompt:
"""
System: “You are an AI researcher who is reviewing a paper that was submitted to a prestigious ML venue. Be critical and cautious in your decision. If a paper is bad or you are unsure, give it bad scores and reject it.”

User: “Below is a description of the questions you will be asked on the review form for each paper and some guidelines on what to consider when answering these questions. When writing your review, please keep in mind that after decisions have been made, reviews and meta-reviews of accepted papers and opted-in rejected papers will be made public.

Summarized in one line, a review aims to determine whether a submission will bring sufficient value to the community and contribute new knowledge. The process can be broken down into the following main reviewer tasks:
 
Read the paper: It’s important to carefully read through the entire paper, and to look up any related work and citations that will help you comprehensively evaluate it. Be sure to give yourself sufficient time for this step.
While reading, consider the following:
Objective of the work: What is the goal of the paper? Is it to better address a known application or problem, draw attention to a new application or problem, or to introduce and/or explain a new theoretical finding? A combination of these? Different objectives will require different considerations as to potential value and impact.
Strong points: is the submission clear, technically correct, experimentally rigorous, reproducible, does it present novel findings (e.g. theoretically, algorithmically, etc.)?
Weak points: is it weak in any of the aspects listed in b.?
Be mindful of potential biases and try to be open-minded about the value and interest a paper can hold for the entire ICLR community, even if it may not be very interesting for you.
Answer three key questions for yourself, to make a recommendation to Accept or Reject:
What is the specific question and/or problem tackled by the paper?
Is the approach well motivated, including being well-placed in the literature?
Does the paper support the claims? This includes determining if results, whether theoretical or empirical, are correct and if they are scientifically rigorous.
Write your initial review, organizing it as follows: 
Summarize what the paper claims to contribute. Be positive and generous.
List strong and weak points of the paper. Be as comprehensive as possible.
Clearly state your recommendation (accept or reject) with one or two key reasons for this choice.
Provide supporting arguments for your recommendation.
Ask questions you would like answered by the authors to help you clarify your understanding of the paper and provide the additional evidence you need to be confident in your assessment. 
Provide additional feedback with the aim to improve the paper. Make it clear that these points are here to help, and not necessarily part of your decision assessment.
General points to consider:
Be polite in your review. Ask yourself whether you’d be happy to receive a review like the one you wrote.
Be precise and concrete. For example, include references to back up any claims, especially claims about novelty and prior work
Provide constructive feedback.
It’s also fine to explicitly state where you are uncertain and what you don’t quite understand. The authors may be able to resolve this in their response.
Don’t reject a paper just because you don’t find it “interesting”. This should not be a criterion at all for accepting/rejecting a paper. The research community is so big that somebody will find some value in the paper (maybe even a few years down the road), even if you don’t see it right now.
Complete the CoE report: ICLR has adopted the following Code of Ethics (CoE). Please check your assigned papers for conflicts with the code of ethics and mark them in your review form. If you are uncertain, please reach out to your area chair.
Engage in discussion: During the discussion phase, reviewers, authors and area chairs engage in asynchronous discussion, and authors are allowed to revise their submissions to address concerns that arise. It is crucial that you are actively engaged and responsive during this phase, i.e., you should be able to respond to comments/requests within 3 business days.
Provide final recommendation: Update your review, taking into account the new information collected during the discussion phase, and any revisions to the submission. Maintain a spirit of openness to changing your initial recommendation (either to a more positive or more negative) rating.

[{few-shot-examples - either 0 or 1 accept and 1 reject sample (~40k more tokens)}]

Here is the paper you are asked to review: {paper}
Respond with a reasoning trace followed by a strictly formatted JSON block.

1. First, provide your reasoning under the section "THOUGHT:".
2. Then, provide the review in a valid JSON object inside a markdown code block.

Use the following JSON schema:
```json
{
  "summary": "string",
  "questions": "string",
  "limitations": "string",
  "strengths": "string",
  "weaknesses": "string",
  "ethical_concerns": boolean, 
  "soundness": integer, // Score 1-5
  "presentation": integer, // Score 1-5
  "contribution": integer, // Score 1-5
  "overall": integer, // Score 1-10
  "confidence": integer, // Score 1-5
  "decision": "accept" OR "reject"
}
```




Paper Review Ensembling System Prompt:{ - make this configurable no ensemble (we just the decision OR we pick a clibrartion on rating), if ensemble, then we try majority vote or LLM met-areview that just otuptus decision (no calibration)}


Meta review prompt:

You are an Area Chair at a prestigious machine learning conference. You are in charge of meta-reviewing a paper that was reviewed by 5 reviewers. Your job is to aggregate the reviews into a single meta-review in the same format. Be critical and cautious in your decision, find consensus, and respect the opinion of all the reviewers.
Review 1/5: {review},
Review 2/5: {review},
Review 3/5: {review},
Review 4/5: {review},
Review 5/5: {review},

Respond with a reasoning trace followed by a strictly formatted JSON block.

1. First, provide your reasoning under the section "THOUGHT:".
2. Then, provide the metareview in a valid JSON object inside a markdown code block.

Use the following JSON schema:
```json
{
  "metareview": "string",
  "soundness": integer, // Score 1-5
  "presentation": integer, // Score 1-5
  "contribution": integer, // Score 1-5
  "overall": integer, // Score 1-10
  "confidence": integer, // Score 1-5
  "decision": "accept" OR "reject"
}
```



Variants to plan:

Text, Text+Images, Vision for each of the following:
1) Original prompt (20k input tokens)
2) New prompt + No Fewshot [1 generation]  (20k input tokens)
3) New prompt + Fewshot (1 accept and 1 reject sample lets random sample from validation set, dataset derives from here: [TODO]) [1 generation]  (60k input tokens)
4) New prompt + Fewshot + Ensembling Majority [5 generations]  (60k input tokens)
5) New prompt + Fewshot + Ensembling Meta [5 generations + 1 meta]  (60k input tokens)

The original and new prompts are above. The new prompt has a location to add in fewshot examples. 

Note that for vllm_infer script, the expected input token count is above. 



# Structure
Create a new variant of vllm_infer and vllm_infer.sbatch so that the sbatch does the following using a job array:
1) Each variant requires a dataset to be produced for each modified prompt (original, new, and new + fewshot) and dataset (the 3 modalities of the dataset). The sbatch will take in a variable for the initial datasets prior to modification. The existing datasets will be:
- /n/fs/vision-mix/sk7524/LLaMA-Factory/data/iclr_2020_2025_85_5_10_split6_balanced_{clean,clean_images,vision}_clean_binary_noreviews_v6_test
2) queues inference on all of the dataset variants and using all modalities (text, text+images, vision). Note the input specifications are above. Note that prompts 3-5 are all the same prompt, so only run it once. Change the # of generations in our version of vllm_infer.py to just generate 5 outputs. For now, put None as the Fewshot directory since I do not have output results from the new prompt.
3) To extract results for 3, take the first output's result. For prompt 4, take the majority of the 5 decisions. For prompt version 5, run a new inference script to apply the metareview prompt (above) to the 5 reviews and get an acceptance decision.
4) Run metrics and generate plots


# Metrics: Overall accuracy, accept recall, reject recall
In distribution (data 2020-2024)/out of distribution (2025 data)
By year specifically
Modify the vllm_infer.py and gemini_batch_submit.py commands. Add n_generations for ensembling.
Run this for qwen2.5-7b-instruct, qwen2.5vl-7b-instruct, and maybe 14B
12 variations per model (qwen-7b (text), vl for clean_images, vision, gemini for both)


Later todos:
Try and test 1 of each with gemini-2.5-flash and see if this gets you better to 50% acceptance (look at accuracy/accept-recall/reject recall)
Note: Gemini 2.5 flash 2k generation
Try finetuned models