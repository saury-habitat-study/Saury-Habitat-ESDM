import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, Dense, GlobalAveragePooling1D,
    Reshape, multiply, add, LayerNormalization, MultiHeadAttention, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, Callback
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os


# ===================================================================
# 1. Core Module: Custom Callback Class for Detailed Metrics
# ===================================================================
class MetricsHistory(Callback):
    """A custom callback to record metrics for each batch and epoch during training."""

    def on_train_begin(self, logs={}):
        self.train_loss_per_batch = []
        self.train_acc_per_batch = []
        self.val_loss_per_epoch = []
        self.val_acc_per_epoch = []
        self.epoch_end_steps = []
        self.current_step = 0

    def on_batch_end(self, batch, logs={}):
        self.train_loss_per_batch.append(logs.get('loss'))
        self.train_acc_per_batch.append(logs.get('accuracy'))
        self.current_step += 1

    def on_epoch_end(self, epoch, logs={}):
        self.val_loss_per_epoch.append(logs.get('val_loss'))
        self.val_acc_per_epoch.append(logs.get('val_accuracy'))
        self.epoch_end_steps.append(self.current_step)


# ===================================================================
# 2. Model Architecture and Data Processing Functions
# ===================================================================
def se_block(input_tensor, ratio=8):
    """Squeeze-and-Excitation (SE) attention block."""
    channels = tf.keras.backend.int_shape(input_tensor)[-1]
    se = GlobalAveragePooling1D()(input_tensor)
    se = Reshape((1, channels))(se)
    se = Dense(channels // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(channels, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    return multiply([input_tensor, se])


def build_multiclass_classification_model(input_shape, n_outputs, lstm_units, num_heads):
    """Builds the CNN-LSTM model with SE and Multi-Head Attention."""
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = se_block(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(lstm_units[0], return_sequences=True)(x)
    x = LSTM(lstm_units[1], return_sequences=True)(x)
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=lstm_units[1])(query=x, value=x, key=x)
    x = add([x, attention_output])
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(n_outputs, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def load_data_with_sliding_window(data_path, n_timesteps, step, n_outputs):
    """Loads and preprocesses data using a sliding window approach."""
    print("Loading data and applying sliding window...")
    # Assuming the CSV is encoded in UTF-8, which is standard.
    # If not, you might need to change the encoding.
    df = pd.read_csv(data_path)
    identifier_col = 'mmsi'
    label_col = 'label'
    feature_cols = [col for col in df.columns if col not in [identifier_col, label_col, 'num']]
    print(f"Using feature columns: {feature_cols}")

    scaler = StandardScaler()
    if feature_cols:
        df[feature_cols] = scaler.fit_transform(df[feature_cols])

    print("Creating sliding window samples...")
    unique_ids = df[identifier_col].unique()
    all_windows = []
    all_labels = []

    for unique_id in unique_ids:
        mmsi_df = df[df[identifier_col] == unique_id]
        feature_array = mmsi_df[feature_cols].values
        label_array = mmsi_df[label_col].values
        for i in range(0, len(feature_array) - n_timesteps + 1, step):
            window = feature_array[i: i + n_timesteps]
            # The label corresponds to the end of the window
            label = label_array[i + n_timesteps - 1]
            all_windows.append(window)
            all_labels.append(label)

    X = np.array(all_windows)
    # Assuming labels are 1-based, convert to 0-based for to_categorical
    y_integers = np.array(all_labels) - 1
    y = tf.keras.utils.to_categorical(y_integers, num_classes=n_outputs)

    return X, y


# ===================================================================
# 3. Visualization and Evaluation Functions
# ===================================================================
def plot_training_curves(history_callback, fold_no):
    """Plots training and validation curves from the custom callback."""
    fig, ax1 = plt.subplots(figsize=(12, 7))

    color = 'tab:red'
    ax1.set_xlabel("Training Steps", fontsize=16)
    ax1.set_ylabel("Loss", fontsize=16, color=color)
    ax1.plot(range(len(history_callback.train_loss_per_batch)), history_callback.train_loss_per_batch,
             color='#4682B4', label='Training Loss', linewidth=0.8)
    ax1.plot(history_callback.epoch_end_steps, history_callback.val_loss_per_epoch,
             color=color, label='Validation Loss', linewidth=1.5, marker='o', markersize=4)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)

    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel("Accuracy", fontsize=16, color=color)
    ax2.plot(range(len(history_callback.train_acc_per_batch)), history_callback.train_acc_per_batch,
             color='#87CEEB', label='Training Accuracy', linewidth=0.8)
    ax2.plot(history_callback.epoch_end_steps, history_callback.val_acc_per_epoch,
             color=color, label='Validation Accuracy', linewidth=1.5, marker='o', markersize=4)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=12)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.title(f'Fold {fold_no} Training Process Curves', fontsize=18)
    fig.tight_layout()

    figure_path = f'training_curves_fold_{fold_no}.png'
    plt.savefig(figure_path, dpi=300)
    print(f'--- Training curves for Fold {fold_no} saved to: {figure_path} ---')
    plt.show()
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    """Plots and saves a styled confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 10},
                linewidths=.5, linecolor='gray')
    plt.title('Overall Confusion Matrix', fontsize=16)
    plt.ylabel('Actual Labels', fontsize=12)
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f'--- Confusion matrix saved to: {save_path} ---')
    plt.close()


# ===================================================================
# 4. Main Execution Block
# ===================================================================
if __name__ == '__main__':
    # --- Parameters Setup ---
    # IMPORTANT: Replace 'your_data.csv' with the actual path to your data file.
    DATA_PATH = 'your_data.csv'
    N_OUTPUTS = 8
    N_FEATURES = 11
    N_TIMESTEPS = 100
    STEP = 1
    LSTM_UNITS = [16, 16]
    NUM_HEADS = 2
    EPOCHS = 150
    BATCH_SIZE = 80
    K_FOLDS = 5

    # --- Data Loading and Preprocessing ---
    X_data, y_data = load_data_with_sliding_window(DATA_PATH, N_TIMESTEPS, STEP, N_OUTPUTS)
    print("\n" + "=" * 50)
    print("Data preprocessing complete!")
    print(f"[INFO] Total training samples generated: {X_data.shape[0]}")
    print(f"Final shape of input data X: {X_data.shape}")
    print(f"Final shape of output labels y: {y_data.shape}")
    print("=" * 50)

    # --- K-Fold Cross-Validation Training ---
    input_shape = (N_TIMESTEPS, N_FEATURES)
    print(f"\nStarting {K_FOLDS}-Fold Cross-Validation for multi-class task...")
    print("=" * 50)

    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_no = 1
    scores = []
    all_true_labels = []
    all_pred_labels = []
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)

    for train, val in kfold.split(X_data, y_data):
        print(f'--- Training Fold {fold_no}/{K_FOLDS} ---')

        model_fold = build_multiclass_classification_model(input_shape, N_OUTPUTS, LSTM_UNITS, NUM_HEADS)
        model_fold.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        if fold_no == 1:
            model_fold.summary()

        history_callback = MetricsHistory()
        model_fold.fit(
            X_data[train], y_data[train],
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_data[val], y_data[val]),
            callbacks=[early_stopping, history_callback],
            verbose=2
        )

        val_loss, val_accuracy = model_fold.evaluate(X_data[val], y_data[val], verbose=0)
        print(f'Validation accuracy for Fold {fold_no}: {val_accuracy:.4f}')
        scores.append(val_accuracy)

        plot_training_curves(history_callback, fold_no)

        y_pred_probs = model_fold.predict(X_data[val])
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        y_true_classes = np.argmax(y_data[val], axis=1)
        all_pred_labels.extend(y_pred_classes)
        all_true_labels.extend(y_true_classes)
        fold_no += 1

    # --- Final Evaluation Report ---
    report_file_path = 'final_evaluation_report.txt'
    with open(report_file_path, 'w', encoding='utf-8') as f:
        print("\n" + "=" * 50)
        f.write("=" * 50 + "\n")
        header = "             Final Model Evaluation Report\n"
        print(header)
        f.write(header)
        print("=" * 50)
        f.write("=" * 50 + "\n")

        accuracy_summary = f"Cross-validation complete. Average accuracy across all folds: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})\n\n"
        print(accuracy_summary)
        f.write(accuracy_summary)

        class_names = [f'Class {i}' for i in range(1, N_OUTPUTS + 1)]
        report = classification_report(all_true_labels, all_pred_labels, target_names=class_names)

        report_header = "Classification Report:\n"
        print(report_header)
        f.write(report_header)
        print(report)
        f.write(report)

        plot_confusion_matrix(all_true_labels, all_pred_labels, class_names)

        print("\n" + "=" * 50)
        f.write("\n" + "=" * 50 + "\n")
        footer = f"All analyses complete! Evaluation report saved to {report_file_path}\n"
        print(footer)
        f.write(footer)
        print("=" * 50)
        f.write("=" * 50 + "\n")
