import os
import pickle
import tensorflow as tf
from tqdm import tqdm
from src.meta_learning import MetaLearningTrainer
from src.meta_learning2 import VerificationMetaTrainer
from src.models import DigraphCNN
from src.preprocessing import build_datasets
from src.normalization import fit_normalizer, apply_normalizer
import matplotlib.pyplot as plt

# Make sure to change the below path to your local data path
DATA_ROOT = "data/UB_keystroke_dataset"
OUTPUT_PATH = "processed_keystrokes.pkl"

def main():
    # ---- STEP 1: PREPROCESS DATA, OR LOAD PREPROCESSED DATA ----
    # if os.path.exists(OUTPUT_PATH):
    #     with open(OUTPUT_PATH, "rb") as f:
    #         data = pickle.load(f)

    #     print("Loaded preprocessed data from", OUTPUT_PATH)

    # else: #Preprocess from scratch
    #     data = build_datasets(DATA_ROOT)
    #     with open(OUTPUT_PATH, "wb") as f:
    #         pickle.dump(data, f)

    #     print("Created preprocessed data and saved to", OUTPUT_PATH)
    data = build_datasets(DATA_ROOT)
    X_train = data[0]
    y_train = data[1]
    X_test = data[2]
    y_test = data[3]
    user_sessions_train = data[4]
    user_sessions_test = data[5]

    # Confirm data shapes
    print("Training data length:", len(X_train))
    print("Training labels length:", len(y_train))
    print("Testing data length:", len(X_test))
    print("Testing labels length:", len(y_test))
    print("Number of users in training set:", len(user_sessions_train))
    print("Number of users in testing set:", len(user_sessions_test))
    print("Example digraph shape (train[0]):", X_train[0].shape)

    # ---- STEP 2: NORMALIZE DATA VIA LOG TRANSFORM & MEAN/STD SCALING ----
    # Get normalization stats from training data only
    stats = fit_normalizer(X_train)

    # Apply normalization to both train and test data
    X_train_norm = apply_normalizer(X_train, stats)
    X_test_norm = apply_normalizer(X_test, stats)

    # ---- STEP 3: CREATE ENCODER AND META-LEARNING TRAINER ----
    encoder = DigraphCNN(
        input_dim=9,
        embedding_dim=128,
        kernel_sizes=[3, 5, 7],
        num_filters=[64, 128, 256],
        dropout=0.3
    )

    # ---- STEP 4: TRAIN THE META-LEARNING MODEL ----
    trainer = VerificationMetaTrainer(
        encoder=encoder,
        k_shot=2,
        lr=1e-3,
        q_query=1
    )

    num_episodes = 300
    eval_interval = 100

    history = trainer.train(
        X_train_norm,
        user_sessions_train,
        num_episodes,
        eval_interval,
        X_test_norm,
        user_sessions_test
    )

    print("\n>>> DEBUGGING ON VALIDATION SET <<<")
    trainer.debug_probs(X_test_norm, user_sessions_test, num_episodes=100, threshold=0.5)

    print("\n>>> DEBUGGING ON TRAINING SET <<<")
    trainer.debug_probs(X_train_norm, user_sessions_train, num_episodes=100, threshold=0.5)

    print("\n>>> DISTANCES ON VALIDATION SET <<<")
    trainer.debug_distances(X_test_norm, user_sessions_test, num_episodes=100)

    # ---- STEP 5: VERIFICATION ----
    metrics = trainer.evaluate_metrics(
        X_test_norm,
        user_sessions_test,
        num_episodes=500,
        threshold=0.9,
    )
    
    print("Verification metrics on test set:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  TPR: {metrics['tpr']:.4f}")
    print(f"  TNR: {metrics['tnr']:.4f}")
    print(f"  FPR: {metrics['fpr']:.4f}")
    print(f"  FNR: {metrics['fnr']:.4f}")

    # ---- STEP 6: PLOT CONFUSION MATRIX & TRAINING HISTORY ----
    cm = trainer.compute_confusion(
        X_test_norm,
        user_sessions_test,
        num_episodes=500,
        threshold=0.5,
    )

    print("Confusion matrix:\n", cm)
    trainer.plot_confusion_matrix(cm, class_names=["Impostor", "Genuine"])

    #Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Loss over Episodes')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accs'], label='Train Accuracy')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy over Episodes')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
