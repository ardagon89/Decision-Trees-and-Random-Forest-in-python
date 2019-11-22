To test  the script type: python runscript.py -[Type] [File_Id] -[Heuristic] -[Pruning] [Depth]

where,

Type = Type of model. D for Decision Tree, R for Random Forest.
File_Id = Unique identifier of the files places in all_data/ folder. Program will read all 3 (train, valid and test) files with just one identifier.
Heuristic = Type of impurity heuristic used. v for variance, e for entropy
Pruning = Pruning method used. n for none, d for depth based pruning, r for reduced error pruning
Depth = Value of depth to be used in depth based pruning. 5, 10,...

Ex. python runscript.py -D c1800_d5000 -e -d 10

This will train a decision tree on "all_data/train_c1800_d5000.csv" dataset using Entropy heuristic and prune the trained tree based on 
Depth based pruning according to dataset "all_data/valid_c1800_d5000.csv" to a max depth of 10. Finally accuracy will be calculated on
dataset "all_data/test_c1800_d5000.csv".