import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import tensorflow as tf
from pathlib import Path
import time
from collections import defaultdict
import json

from src.models import DigraphCNN
from src.meta_learning2 import VerificationMetaTrainer
from src.preprocessing import KeystrokeSession, get_spatial_distances

class KeystrokeAuthGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Keystroke Authentication Demo")
        self.root.geometry("600x500")
        
        # Load models
        self.load_models()
        
        # User database (in-memory for demo)
        self.user_profiles = {}
        self.load_user_profiles()
        
        # Keystroke capture
        self.keystroke_events = []
        self.key_press_times = {}
        
        # Create UI
        self.create_ui()
        
    def load_models(self):
        """Load trained encoder and verification head"""
        try:
            # Create encoder
            self.encoder = DigraphCNN(input_dim=9, embedding_dim=128)
            
            # Build the model by passing dummy data
            dummy_input = tf.zeros((1, 80, 9))
            _ = self.encoder(dummy_input, training=False)
            
            # Now load weights
            self.encoder.load_weights('models/best/verification_encoder_weights.h5')
            
            # Create trainer
            self.trainer = VerificationMetaTrainer(self.encoder, k_shot=2, q_query=15)
            
            # Build verification head
            dummy_embedding = tf.zeros((1, 128))
            _ = self.trainer.verification_head(dummy_embedding, training=False)
            
            # Load verification head weights
            self.trainer.verification_head.load_weights('models/best/verification_head_weights.h5')
            
            print("✓ Models loaded successfully")
        except Exception as e:
            print(f"⚠ Error loading models: {e}")
            self.encoder = None
            self.trainer = None
    
    def load_user_profiles(self):
        """Load user profiles from file if exists"""
        profile_path = Path('user_profiles.json')
        if profile_path.exists():
            with open(profile_path, 'r') as f:
                data = json.load(f)
                # Convert lists back to numpy arrays
                for username, profile in data.items():
                    self.user_profiles[username] = {
                        'embeddings': [np.array(e) for e in profile['embeddings']],
                        'samples': profile['samples']
                    }
            print(f"✓ Loaded {len(self.user_profiles)} user profiles")
    
    def save_user_profiles(self):
        """Save user profiles to file"""
        # Convert numpy arrays to lists for JSON serialization
        data = {}
        for username, profile in self.user_profiles.items():
            data[username] = {
                'embeddings': [e.tolist() for e in profile['embeddings']],
                'samples': profile['samples']
            }
        
        with open('user_profiles.json', 'w') as f:
            json.dump(data, f)
        print("✓ User profiles saved")
    
    def create_ui(self):
        """Create the GUI layout"""
        # Title
        title_label = ttk.Label(self.root, text="Keystroke Authentication System", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Register Tab
        register_frame = ttk.Frame(notebook)
        notebook.add(register_frame, text="Register")
        self.create_register_tab(register_frame)
        
        # Login Tab
        login_frame = ttk.Frame(notebook)
        notebook.add(login_frame, text="Login")
        self.create_login_tab(login_frame)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_register_tab(self, parent):
        """Create registration interface"""
        # Username
        ttk.Label(parent, text="Username:", font=("Arial", 10)).pack(pady=(20, 5))
        self.reg_username = ttk.Entry(parent, width=30, font=("Arial", 10))
        self.reg_username.pack(pady=5)
        
        # Instructions
        instructions = (
            "Type the sample text below 3 times to register your typing pattern.\n"
            "Sample text: 'The quick brown fox jumps over the lazy dog.'"
        )
        ttk.Label(parent, text=instructions, font=("Arial", 9), 
                 justify=tk.CENTER, wraplength=500).pack(pady=10)
        
        # Sample counter
        self.reg_sample_var = tk.StringVar(value="Samples collected: 0/3")
        ttk.Label(parent, textvariable=self.reg_sample_var, 
                 font=("Arial", 10, "bold")).pack(pady=5)
        
        # Text input
        ttk.Label(parent, text="Type here:", font=("Arial", 10)).pack(pady=(10, 5))
        self.reg_text = tk.Text(parent, height=4, width=60, font=("Arial", 10))
        self.reg_text.pack(pady=5)
        
        # Bind keystroke events
        self.reg_text.bind('<KeyPress>', self.on_key_press)
        self.reg_text.bind('<KeyRelease>', self.on_key_release)
        
        # Buttons
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(pady=10)
        
        self.reg_submit_btn = ttk.Button(btn_frame, text="Submit Sample", 
                                         command=self.submit_registration_sample)
        self.reg_submit_btn.pack(side=tk.LEFT, padx=5)
        
        self.reg_reset_btn = ttk.Button(btn_frame, text="Reset", 
                                        command=self.reset_registration)
        self.reg_reset_btn.pack(side=tk.LEFT, padx=5)
        
        # Store registration data
        self.reg_samples = []
    
    def create_login_tab(self, parent):
        """Create login/authentication interface"""
        # Username
        ttk.Label(parent, text="Username:", font=("Arial", 10)).pack(pady=(20, 5))
        self.login_username = ttk.Entry(parent, width=30, font=("Arial", 10))
        self.login_username.pack(pady=5)
        
        # Instructions
        instructions = (
            "Type the sample text to authenticate.\n"
            "Sample text: 'The quick brown fox jumps over the lazy dog.'"
        )
        ttk.Label(parent, text=instructions, font=("Arial", 9), 
                 justify=tk.CENTER, wraplength=500).pack(pady=10)
        
        # Text input
        ttk.Label(parent, text="Type here:", font=("Arial", 10)).pack(pady=(10, 5))
        self.login_text = tk.Text(parent, height=4, width=60, font=("Arial", 10))
        self.login_text.pack(pady=5)
        
        # Bind keystroke events
        self.login_text.bind('<KeyPress>', self.on_key_press)
        self.login_text.bind('<KeyRelease>', self.on_key_release)
        
        # Buttons
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(pady=10)
        
        self.login_btn = ttk.Button(btn_frame, text="Authenticate", 
                                    command=self.authenticate_user)
        self.login_btn.pack(side=tk.LEFT, padx=5)
        
        self.login_reset_btn = ttk.Button(btn_frame, text="Clear", 
                                          command=self.reset_login)
        self.login_reset_btn.pack(side=tk.LEFT, padx=5)
        
        # Result display
        self.login_result_var = tk.StringVar(value="")
        result_label = ttk.Label(parent, textvariable=self.login_result_var, 
                                font=("Arial", 11, "bold"), wraplength=500)
        result_label.pack(pady=10)
    
    def on_key_press(self, event):
        """Capture key press event"""
        key = event.keysym
        timestamp = time.time()
        
        if key not in self.key_press_times:
            self.key_press_times[key] = timestamp
    
    def on_key_release(self, event):
        """Capture key release event and create digraph"""
        key = event.keysym
        timestamp = time.time()
        
        if key in self.key_press_times:
            press_time = self.key_press_times[key]
            hold_time = timestamp - press_time
            
            self.keystroke_events.append({
                'key': key,
                'press_time': press_time,
                'release_time': timestamp,
                'hold_time': hold_time
            })
            
            del self.key_press_times[key]
    
    def extract_digraph_features(self):
        """Convert keystroke events to digraph feature vectors"""
        if len(self.keystroke_events) < 2:
            return None
        
        digraphs = []
        for i in range(len(self.keystroke_events) - 1):
            curr = self.keystroke_events[i]
            next_evt = self.keystroke_events[i + 1]
            
            # Calculate flight time (release to press)
            flight_time = next_evt['press_time'] - curr['release_time']
            
            # Get spatial distances
            dx, dy = get_spatial_distances(curr['key'], next_evt['key'])
            
            # Create 9-dimensional feature vector
            features = [
                curr['hold_time'],           # Hold time of first key
                next_evt['hold_time'],       # Hold time of second key
                flight_time,                 # Flight time between keys
                curr['hold_time'] + flight_time + next_evt['hold_time'],  # Total time
                dx,                          # Horizontal distance
                dy,                          # Vertical distance
                np.sqrt(dx**2 + dy**2),     # Euclidean distance
                curr['press_time'],          # Absolute press time
                next_evt['press_time']       # Absolute next press time
            ]
            
            digraphs.append(features)
        
        return np.array(digraphs, dtype=np.float32)
    
    def submit_registration_sample(self):
        """Process registration sample"""
        username = self.reg_username.get().strip()
        if not username:
            messagebox.showerror("Error", "Please enter a username")
            return
        
        if len(self.keystroke_events) < 10:
            messagebox.showwarning("Warning", "Please type more text (at least 10 keystrokes)")
            return
        
        # Extract features
        features = self.extract_digraph_features()
        if features is None or len(features) < 5:
            messagebox.showerror("Error", "Not enough keystroke data")
            return
        
        # Store sample
        self.reg_samples.append(features)
        self.reg_sample_var.set(f"Samples collected: {len(self.reg_samples)}/3")
        
        # Clear for next sample
        self.reg_text.delete('1.0', tk.END)
        self.keystroke_events = []
        self.key_press_times = {}
        
        # If we have 3 samples, register the user
        if len(self.reg_samples) >= 3:
            self.register_user(username)
    
    def register_user(self, username):
        """Register user with collected samples"""
        if self.encoder is None:
            messagebox.showerror("Error", "Models not loaded")
            return
        
        try:
            # Generate embeddings for all samples
            embeddings = []
            for sample in self.reg_samples:
                # Pad/truncate to fixed length (80 digraphs)
                if len(sample) > 80:
                    sample = sample[:80]
                elif len(sample) < 80:
                    padding = np.zeros((80 - len(sample), 9), dtype=np.float32)
                    sample = np.vstack([sample, padding])
                
                # Get embedding
                sample_batch = np.expand_dims(sample, axis=0)
                embedding = self.encoder(sample_batch, training=False)
                embeddings.append(embedding.numpy()[0])
            
            # Store user profile
            self.user_profiles[username] = {
                'embeddings': embeddings,
                'samples': len(self.reg_samples)
            }
            
            # Save to disk
            self.save_user_profiles()
            
            messagebox.showinfo("Success", f"User '{username}' registered successfully!")
            self.status_var.set(f"User '{username}' registered with {len(self.reg_samples)} samples")
            
            # Reset
            self.reset_registration()
            
        except Exception as e:
            messagebox.showerror("Error", f"Registration failed: {str(e)}")
    
    def reset_registration(self):
        """Reset registration form"""
        self.reg_text.delete('1.0', tk.END)
        self.reg_samples = []
        self.keystroke_events = []
        self.key_press_times = {}
        self.reg_sample_var.set("Samples collected: 0/3")
    
    def authenticate_user(self):
        """Authenticate user based on typing pattern"""
        username = self.login_username.get().strip()
        if not username:
            messagebox.showerror("Error", "Please enter a username")
            return
        
        if username not in self.user_profiles:
            self.login_result_var.set(f"❌ User '{username}' not found. Please register first.")
            return
        
        if len(self.keystroke_events) < 10:
            messagebox.showwarning("Warning", "Please type more text (at least 10 keystrokes)")
            return
        
        if self.encoder is None or self.trainer is None:
            messagebox.showerror("Error", "Models not loaded")
            return
        
        try:
            # Extract features from current input
            features = self.extract_digraph_features()
            if features is None or len(features) < 5:
                self.login_result_var.set("❌ Not enough keystroke data")
                return
            
            # Pad/truncate to fixed length
            if len(features) > 80:
                features = features[:80]
            elif len(features) < 80:
                padding = np.zeros((80 - len(features), 9), dtype=np.float32)
                features = np.vstack([features, padding])
            
            # Get embedding
            query_batch = np.expand_dims(features, axis=0)
            query_embedding = self.encoder(query_batch, training=False)
            
            # Get user's stored embeddings
            user_embeddings = self.user_profiles[username]['embeddings']
            
            # Use verification head to check if genuine
            # Create support set from user's embeddings
            support_set = np.array(user_embeddings[:2])  # Use first 2 as support
            support_batch = np.expand_dims(support_set, axis=0)
            
            # Compare query with support
            scores = []
            for support_emb in user_embeddings:
                # Compute similarity
                similarity = np.dot(query_embedding.numpy()[0], support_emb)
                scores.append(similarity)
            
            avg_similarity = np.mean(scores)
            
            # Threshold for authentication (adjust based on your model)
            threshold = 0.5
            
            if avg_similarity > threshold:
                self.login_result_var.set(
                    f"✅ Authentication Successful!\n"
                    f"User: {username}\n"
                    f"Confidence: {avg_similarity:.2%}"
                )
                self.status_var.set(f"User '{username}' authenticated successfully")
            else:
                self.login_result_var.set(
                    f"❌ Authentication Failed\n"
                    f"Typing pattern does not match user '{username}'\n"
                    f"Confidence: {avg_similarity:.2%}"
                )
                self.status_var.set("Authentication failed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Authentication failed: {str(e)}")
    
    def reset_login(self):
        """Reset login form"""
        self.login_text.delete('1.0', tk.END)
        self.keystroke_events = []
        self.key_press_times = {}
        self.login_result_var.set("")

def main():
    root = tk.Tk()
    app = KeystrokeAuthGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()