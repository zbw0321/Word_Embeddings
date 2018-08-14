# Word_Embeddings

Aim
---
In this Project, you will be implementing your own Word Embeddings for adjectives. More specifically, we want the obtained embeddings to preserve as much synonym relationship as possible. This will be measured against a set of ground truth assembled manually from dictionaries.

Data
---
BBC_Data.zip

Evaluation
---
Load the trained model "adjective_embeddings.txt", and test it using the following set of adjectives:
Test_Adjectives =['able', 'average', 'bad', 'best', 'big', 'certain', 'common', 'current', 'different', 'difficult', 'early', 'extra', 'fair', 'few', 'final', 'former', 'great', 'hard', 'high', 'huge', 'important', 'key', 'large', 'last', 'less', 'likely', 'little', 'major', 'more', 'most', 'much', 'new', 'next', 'old', 'prime', 'real', 'recent', 'same', 'serious', 'short', 'small', 'top', 'tough', 'wide']
The model will be evaluated by:
1. Selecting one adjective at a time from the list Test_Adjectives.

2. For the selected adjective, computing a list of top_k most nearest words using the method
   Compute_topk(model_file, input_adjective, top_k)
   
3. Comparing the output for the adjective with ground truth list of synonyms for that adjective, and evaluating Hits@k(k = 100).

4. Average out the result (Hits@k(k= 100)) for all the adjectives to calculate Average Precision. 

Evaluation Example (using a smaller k value):

a)
Ground_truth_new = ['novel', 'recent', 'first', 'current', 'latest'] Output_new = ['old', 'novel', 'first', 'extra', 'out']
Hits@k(k=5) = 2

b)
Ground_truth_good = ['better', 'best', 'worthwhile', 'prosperous', 'excellent'] Output_good = ['hate', 'better', 'best', 'worthwhile', 'prosperous'] Hits@k(k=5) = 4
Average_Hits@k(k=5) = (2+4)/2 = 3.0

Results
---
Average_hits > 7
