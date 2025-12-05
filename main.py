import os 
import pickle
from src.preprocessing import build_datasets
from src.normalization import fit_normalizer, apply_normalizer 

#Make sure to change the below path to your local data path
DATA_ROOT = "./UB_keystroke_dataset"
OUTPUT_PATH = "processed_keystrokes.pkl"

def main(): 
    #---- STEP 1: PREPROCESS DATA, OR LOAD PREPROCESSED DATA ---- 
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

    #Confirm data shapes
    print("Training data length:", len(X_train))
    print("Training labels length:", len(y_train))
    print("Testing data length:", len(X_test))
    print("Testing labels length:", len(y_test))
    print("Number of users in training set:", len(user_sessions_train))
    print("Number of users in testing set:", len(user_sessions_test))
    print("Example digraph shape (train[0]):", X_train[0].shape)

    #---- STEP 2: NORMALIZE DATA VIA LOG TRANSFORM & MEAN/STD SCALING ---- 
    stats = fit_normalizer(X_train) #Get normalization stats from training data only

    #Apply normalization to both train and test data
    X_train_norm = apply_normalizer(X_train, stats)
    X_test_norm = apply_normalizer(X_test, stats)

if __name__ == "__main__":
    main()