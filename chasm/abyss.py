import os
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
    
    
import numpy as np

from sklearn.feature_selection import VarianceThreshold

from keras.models import Model
from keras.layers import Input

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import math

def reduce_reconstruct(block, n_components, var_threshold=1e-8):
    # Remove near-constant features
    try:
        selector = VarianceThreshold(threshold=var_threshold)
        filtered_block = selector.fit_transform(block)

        # Save column names for reconstruction
        selected_columns = block.columns[selector.get_support()]

        # Check if there are any remaining features
        if filtered_block.shape[1] == 0:
            raise ValueError("All features were removed by VarianceThreshold. Try lowering the threshold.")

        # Standardize the filtered data
        scaler = StandardScaler()
        scaled_snps = scaler.fit_transform(filtered_block)

        # Apply PCA
        pca = PCA(n_components=min(n_components, scaled_snps.shape[1]))  # Ensure n_components isn't larger than features
        reduced_data = pca.fit_transform(scaled_snps)

        # Reconstruction
        reconstructed_scaled_snps = pca.inverse_transform(reduced_data)
        reconstructed_block = scaler.inverse_transform(reconstructed_scaled_snps)

        # Restore DataFrame format
        reconstructed_df = pd.DataFrame(data=reconstructed_block, columns=selected_columns, index=block.index)

        return reconstructed_df
    
    except ValueError as e:
        return block

def linear_abyss(path_input, name_file, path_output, n_components=5, p2=False, twopq = False, q2=False):
    path_ld = f"{path_input}/{name_file}"
    block = pd.read_pickle(f"{path_ld}")
    block = block.fillna(-1.0)

    if q2:
        # Update minor allele mapping
        db_minor = block.copy()
        db_minor = db_minor.applymap(lambda x: 1 if x == -1.0 else 0)

        db_minor_rec = reduce_reconstruct(db_minor, n_components)
        path_minor = f"{path_output}/q2/"
        os.makedirs(path_minor, exist_ok=True)
        db_minor_rec.to_pickle(f"{path_minor}/{name_file}")
    else:
        pass
    
    if twopq:  
        # Update heterozygous allele mapping
        db_het = block.copy()
        db_het = db_het.applymap(lambda x: 1 if x == 0.0 else 0)
        
        db_het_rec = reduce_reconstruct(db_het, n_components)
        path_het = f"{path_output}/2pq/"
        os.makedirs(path_het, exist_ok=True)
        db_het_rec.to_pickle(f"{path_het}/{name_file}")
    
    else:
        pass
    
    if p2:
        # Update major allele mapping
        db_major = block.copy()
        db_major = db_major.applymap(lambda x: 1 if x == 1.0 else 0)
    
        db_major_rec = reduce_reconstruct(db_major, n_components)
        path_major = f"{path_output}/p2/"
        os.makedirs(path_major, exist_ok=True)
        db_major_rec.to_pickle(f"{path_major}/{name_file}")


def AE(geno, bottleneck_nr, hidden, epoch, patience):

    # Ensure input is a NumPy array
    if isinstance(geno, pd.DataFrame):
        geno = geno.to_numpy()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(geno, geno, test_size=0.2, random_state=42)

    # Regularization
    l2_regularizer = 0.001

    # Functional API for better flexibility
    input_layer = tf.keras.Input(shape=(geno.shape[1],))

    # Encoder
    encoder_hidden = tf.keras.layers.Dense(hidden, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(l2_regularizer))(input_layer)
    encoder_hidden_bn = tf.keras.layers.BatchNormalization()(encoder_hidden)
    bottleneck = tf.keras.layers.Dense(bottleneck_nr, activation='elu', name='bottleneck', kernel_regularizer=tf.keras.regularizers.l2(l2_regularizer))(encoder_hidden_bn)
    bottleneck_bn = tf.keras.layers.BatchNormalization()(bottleneck)

    # Decoder
    decoder_hidden = tf.keras.layers.Dense(hidden, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(l2_regularizer))(bottleneck_bn)
    decoder_hidden_bn = tf.keras.layers.BatchNormalization()(decoder_hidden)
    output_layer = tf.keras.layers.Dense(geno.shape[1], activation='elu', kernel_regularizer=tf.keras.regularizers.l2(l2_regularizer))(decoder_hidden_bn)

    # Build model
    autoencoder = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

    # Early stopping callback (this was missing)
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    # Fit model
    history = autoencoder.fit(X_train, y_train, epochs=epoch, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)

    # Extract bottleneck
    bottleneck_model = tf.keras.Model(inputs=input_layer, outputs=bottleneck)

    return autoencoder, bottleneck_model, history



def get_hidden_layers(entry, exit, nr_hidden):
    to_add = math.floor((exit - entry) / (nr_hidden + 1))
    return [entry + (i + 1) * to_add for i in range(nr_hidden)]

def decoder(input_df, output_df, nr_hidden_layers, epoch, patience, test_size=0.2, random_state=42):
    # Convert DataFrames to NumPy arrays
    input = input_df.to_numpy().astype('float32')
    output = output_df.to_numpy().astype('float32')

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=test_size, random_state=random_state)

    nr_entry = input.shape[1]
    nr_exit = output.shape[1]

    # Get hidden layer sizes
    hidden_layers = get_hidden_layers(nr_entry, nr_exit, nr_hidden_layers)

    # Build model
    input_layer = tf.keras.Input(shape=(nr_entry,))
    hidden = Dense(hidden_layers[0], activation='relu')(input_layer)
 
    for i in range(nr_hidden_layers - 1):
        hidden = Dense(hidden_layers[i + 1], activation='relu')(hidden)

    output_layer = Dense(nr_exit, activation='linear')(hidden)

    # Compile model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    # Train model with validation
    history = model.fit(X_train, y_train, 
                        epochs=epoch, 
                        validation_data=(X_test, y_test),
                        callbacks=[early_stopping], 
                        verbose=0)

    # Evaluate on test data
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

    return model, history



def create_meta_decoder_n(decoder_list):

    if not decoder_list:
        raise ValueError("Decoder list is empty!")

    # Step 1: Define input based on the first decoder
    input_layer = Input(shape=(decoder_list[0].input_shape[1],))
    x = input_layer

    # Step 2: Chain all decoders
    for decoder in decoder_list:
        x = decoder(x)

    # Step 3: Create combined model
    meta_decoder = Model(inputs=input_layer, outputs=x)

    return meta_decoder
