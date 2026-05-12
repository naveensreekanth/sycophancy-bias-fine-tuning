Dataset Notes: StereoSet


1. How StereoSet Works
StereoSet is a large-scale benchmark that measures stereotypical bias across four domains: Gender, Profession, Race, and Religion.
Unlike CrowS-Pairs, which only uses sentence pairs, StereoSet uses triplets.
Each triplet includes:
Target Term: The group being discussed (e.g., "Musician").
Context: A lead-in sentence.
Three options: A stereotypical association, an anti-stereotypical association, and a completely unrelated association.

2. Difference from CrowS-Pairs
While CrowS-Pairs focuses on Intra-sentence bias (comparing two similar sentences), StereoSet introduces Intersentence bias.
CrowS-Pairs: Focuses on the probability of a "biased" sentence vs. a "neutral" one.
StereoSet: Forces the model to choose between three distinct continuations, providing a more complex test of the model's "internal world view."

3. Why Unrelated Sentences Exist
The "Unrelated" sentence is a critical control mechanism.
The Purpose: It tests if the model actually understands the context or if it is just picking words randomly.
The Logic: If a model chooses a stereotypical sentence over an unrelated one, it shows bias. However, if it chooses an unrelated sentence over a stereotypical one, it shows the model is failing to follow the basic logic of the conversation.

4. Joint Evaluation: Language Quality vs. Bias
StereoSet uses a unique scoring system that combines two metrics:
Language Intelligibility Score (LIS): Measures the model's ability to choose meaningful sentences over unrelated ones.
Stereotype Score (SS): Measures the model's preference for stereotypes over anti-stereotypes.
Ideal Outcome (The ICAT Score): The "Ideal Context Association Test" score is highest when a model is both highly intelligent (chooses meaningful sentences) and highly fair (chooses anti-stereotypes 50% of the time).
Important Observation: > StereoSet evaluates whether a model can remain socially fair without sacrificing linguistic competence. This is a core pillar of my research: checking if reducing bias makes the model "dumber" or less coherent.
