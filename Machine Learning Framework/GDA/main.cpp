#include <iostream>
#include "DataUtils.h"
#include "GDA.h"

int main() {
    // --- Load Data ---
    Dataset dataset = load_csv("wdbc.csv");
    std::cout << "Loaded " << dataset.X.size() << " samples with "
              << dataset.X[0].size() << " features.\n";

    // --- Train/Test Split (80/20, seed=42 for reproducibility) ---
    TrainTestSplit split = train_test_split(dataset, 0.2, 42);
    std::cout << "Training samples: " << split.X_train.size()
              << ", Test samples: " << split.X_test.size() << "\n";

    // --- Standardize features (z-score, fit on train only) ---
    standardize(split.X_train, split.X_test);

    // --- Initialize GDA ---
    // label_map: 0 = Benign, 1 = Malignant
    // positive_label: "M" corresponds to class 1
    GDA model(
        {{0, "Benign"}, {1, "Malignant"}},
        "M"
    );

    // --- Train ---
    model.fit(split.X_train, split.y_train);
    model.describe();

    // --- Predict ---
    std::vector<std::string> predictions = model.predict(split.X_test);

    // --- Score ---
    double accuracy = model.score(split.X_test, split.y_test);
    std::cout << "Test Accuracy: " << accuracy << "%\n";

    // --- Print first 10 predictions vs ground truth ---
    std::cout << "\nFirst 10 predictions:\n";
    std::cout << "Predicted\tActual\n";
    for (int i = 0; i < 10 && i < (int)predictions.size(); i++) {
        std::string actual = (split.y_test[i] == "M") ? "Malignant" : "Benign";
        std::cout << predictions[i] << "\t\t" << actual << "\n";
    }

    return 0;
}
