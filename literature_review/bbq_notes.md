Dataset Notes: BBQ (Bias Benchmark for QA)


1. How BBQ Works
The Bias Benchmark for QA (BBQ) evaluates socially biased reasoning through a Question-Answering format. It presents the model with a context and a question where the answer isn't always clear, forcing the model to make a choice.

2. Contextual Inference & Decision-Making
Unlike previous benchmarks that only look at sentence pairs, BBQ evaluates contextual inference.
Ambiguous Contexts: The context intentionally lacks enough information to answer the question (e.g., "The secretary and the boss were talking. Who was the secretary?").
Disambiguated Contexts: The context provides a clear fact that should override any stereotype.

The Goal: To see if the model relies on stereotypical assumptions (e.g., assuming the secretary is a woman) when the answer is actually "Not enough information."

3. Difference from Sentence-Pair Benchmarks
Behavioral vs. Linguistic: CrowS-Pairs and StereoSet measure which sentence "sounds more likely" to the model. BBQ measures how the model acts when asked to solve a task.
Reasoning: BBQ requires the model to link a social category to a specific action or role, testing the "logic" of its bias.

4. Importance for QA Systems
BBQ is the industry standard for measuring stereotype-based assumptions in QA systems. Since most real-world AI applications (like chatbots or assistants) are built on QA, this benchmark provides the most "realistic" look at how bias affects users.

Why BBQ is Critical for This Project
This benchmark will likely be your strongest evaluation tool for two reasons:

Behavioral Evolution: Since your project tracks how a model changes during fine-tuning, BBQ will show if the model becomes more "opinionated" or "biased" in its reasoning as it learns.
Beyond Simple Preference: It captures behavioural reasoning much better than simple sentence preference datasets, showing whether the model's "common sense" is being replaced by "stereotypical logic."
