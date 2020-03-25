# LT2212 V20 Assignment 2

Part 1
After tokenization I lowercased and removed punctuation, empty strings, stop words. Only the words appearing over 10 times in the whole corpus have been taken into account. 
In order to optimize the performance of the model I used tf-idf instead of the word counts.

Part 4
1.model_id 1 SGDClassifier
Accuracy, precision, recall, f-measure: 90%
model_id 2 LinearSVC
- Accuracy, precision, recall, f-measure: 92%

2.Reduced dimensionality with TruncatedSVD
SGD Classifier model_id 1
Features reduced to 50%:
- accuracy and precision: 89%, recall and F-measure: 88%
Features reduced to 25%:
- accuracy, precision: 88%, recall and f-measure: 87%
Features reduced to 10% :
- accuracy and precision: 88%, recall and f-measure: 87%
Features reduced to 5%: 
- accuracy and precision: 87%, recall and f-measure: 86%

LinearSVC model_id 2
Features reduced to 50%:
- accuracy, precision, recall, f-measure: 92%
Features reduced to 25%:
- accuracy and precision: 93%, recall and f-measure: 92%
Features reduced to 10%:
- accuracy, precision, f-measure: 90%, recall: 89%
Features reduced to 5%:
- accuracy, precision, f-measure, recall: 89%

When running the program using the SGD and the LinearSVC Classifiers by using the original space, but also by reducing the dimensions, I have noticed that there is little to almost no difference between the results obtained in the evaluation. After having written down these values, I tried running the program again and again and it has occured multiple times that the metrics indicated a higher performance of the classifiers when DR has been applied than when the original number of features was used.
Having done some readings on dimensionality reduction and also observing my program's performance, DR definitely makes learning faster and has the benefit of increasing computational efficiency.

p.s. I have only written another function for part bonus.
