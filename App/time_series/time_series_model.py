import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import layers


class NBeatsBlock(tf.keras.layers.Layer):
    def __init__(self,
                 input_size: int,
                 theta_size: int,
                 horizon: int,
                 n_neurons: int,
                 n_layers: int,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.theta_size = theta_size
        self.horizon = horizon
        self.n_neurons = n_neurons
        self.n_layers = n_layers

        # Block contains stack of 4 fully connected layers each has ReLU activation
        self.hidden = [tf.keras.layers.Dense(n_neurons, activation="relu") for _ in range(n_layers)]
        # Output of block is a theta layer with linear activation
        self.theta_layer = tf.keras.layers.Dense(theta_size, activation="linear", name="theta")

    def call(self, inputs):
        x = inputs
        for layer in self.hidden:  # pass inputs through each hidden layer
            x = layer(x)
        theta = self.theta_layer(x)
        # Output the backcast and forecast from theta
        backcast, forecast = theta[:, :self.input_size], theta[:, -self.horizon:]
        return backcast, forecast


class TimeSeriesModel:

    # Create NBeatsBlock custom layer

    def make_preds(self, model, input_data):
        """
        Uses model to make predictions on input_data.

        Parameters
        ----------
        model: trained model
        input_data: windowed input data (same kind of data model was trained on)

        Returns model predictions on input_data.
        """
        forecast = model.predict(input_data)
        return tf.squeeze(forecast)  # return 1D array of predictions

    def mean_absolute_scaled_error(self, y_true, y_pred):
        """
        Implement MASE (assuming no seasonality of data).
        """
        mae = tf.reduce_mean(tf.abs(y_true - y_pred))

        # Find MAE of naive forecast (no seasonality)
        mae_naive_no_season = tf.reduce_mean(
            tf.abs(y_true[1:] - y_true[:-1]))  # our seasonality is 1 day (hence the shifting of 1 day)

        return mae / mae_naive_no_season

    def evaluate_preds(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        # Calculate various metrics
        mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
        mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
        rmse = tf.sqrt(mse)
        mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
        # mase = mean_absolute_scaled_error(y_true, y_pred)

        return {"mae": mae.numpy(),
                "mse": mse.numpy(),
                "rmse": rmse.numpy(),
                "mape": mape.numpy()
                }
        # "mase": mase.numpy()}

    def testTrainingSplit(self, stock_df):
        # Make features and labels
        X = stock_df.dropna().drop("Price", axis=1)
        y = stock_df.dropna()["Price"]

        # Make train and test sets
        split_size = int(len(X) * 0.8)
        X_train, y_train = X[:split_size], y[:split_size]
        X_test, y_test = X[split_size:], y[split_size:]
        len(X_train), len(y_train), len(X_test), len(y_test)
        return [X_train, y_train, X_test, y_test]

    def make_future_forecast(self, values, model, into_future, window_size) -> list:
        """
        Makes future forecasts into_future steps after values ends.

        Returns future forecasts as list of floats.
        """
        # 2. Make an empty list for future forecasts/prepare data to forecast on
        future_forecast = []
        last_window = values[-window_size:]  # only want preds from the last window (this will get updated)

        # 3. Make INTO_FUTURE number of predictions, altering the data which gets predicted on each time 
        for _ in range(into_future):
            # Predict on last window then append it again, again, again (model starts to make forecasts on its own forecasts)
            future_pred = model.predict(tf.expand_dims(last_window, axis=0))
            print(f"Predicting on: \n {last_window} -> Prediction: {tf.squeeze(future_pred).numpy()}\n")

            # Append predictions to future_forecast
            future_forecast.append(tf.squeeze(future_pred).numpy())
            # print(future_forecast)

            # Update last window with new pred and get WINDOW_SIZE most recent preds (model was trained on WINDOW_SIZE windows)
            last_window = np.append(last_window, future_pred)[-window_size:]

        return future_forecast

    def trainModel(self, inputSize, thetaSize, horizon, nNeurons, nLayers, nStacks, train_dataset, nEpocks,
                   test_dataset):

        tf.random.set_seed(42)
        # 1. Setup N-BEATS Block layer
        nbeats_block_layer = NBeatsBlock(input_size=inputSize,
                                         theta_size=thetaSize,
                                         horizon=horizon,
                                         n_neurons=nNeurons,
                                         n_layers=nLayers,
                                         name="InitialBlock")

        # 2. Create input to stacks
        stack_input = layers.Input(shape=(inputSize), name="stack_input")

        # 3. Create initial backcast and forecast input (backwards predictions are referred to as residuals in the paper)
        backcast, forecast = nbeats_block_layer(stack_input)
        residuals = layers.subtract([stack_input, backcast], name=f"subtract_00")

        # 4. Create stacks of blocks
        for i, _ in enumerate(range(nStacks - 1)):  # first stack is already creted in (3)

            # 5. Use the NBeatsBlock to calculate the backcast as well as block forecast
            backcast, block_forecast = NBeatsBlock(
                input_size=inputSize,
                theta_size=thetaSize,
                horizon=horizon,
                n_neurons=nNeurons,
                n_layers=nLayers,
                name=f"NBeatsBlock_{i}"
            )(residuals)  # pass it in residuals (the backcast)

            # 6. Create the double residual stacking
            residuals = layers.subtract([residuals, backcast], name=f"subtract_{i}")
            forecast = layers.add([forecast, block_forecast], name=f"add_{i}")

        # 7. Put the stack model together
        model_NBeats = tf.keras.Model(inputs=stack_input,
                                      outputs=forecast,
                                      name="model_7_N-BEATS")

        # 8. Compile with MAE loss and Adam optimizer
        model_NBeats.compile(loss="mae",
                             optimizer=tf.keras.optimizers.Adam(0.001),
                             metrics=["mae", "mse"])

        # 9. Fit the model with EarlyStopping and ReduceLROnPlateau callbacks
        model_NBeats.fit(train_dataset,
                         epochs=nEpocks,
                         validation_data=test_dataset,
                         verbose=0,  # prevent large amounts of training outputs
                         # callbacks=[create_model_checkpoint(model_name=stack_model.name)] # saving model every epoch consumes far too much time
                         callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=200,
                                                                     restore_best_weights=True),
                                    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=100, verbose=1)])

        return model_NBeats

    def prepareDataForTraining(self, X_train, y_train, X_test, y_test, batch_size):
        train_features_dataset = tf.data.Dataset.from_tensor_slices(X_train)
        train_labels_dataset = tf.data.Dataset.from_tensor_slices(y_train)

        test_features_dataset = tf.data.Dataset.from_tensor_slices(X_test)
        test_labels_dataset = tf.data.Dataset.from_tensor_slices(y_test)

        # 2. Combine features & labels
        train_dataset = tf.data.Dataset.zip((train_features_dataset, train_labels_dataset))
        test_dataset = tf.data.Dataset.zip((test_features_dataset, test_labels_dataset))
        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return [train_dataset, test_dataset]

    def get_future_dates(self, start_date, into_future, offset=1):
        """
        Returns array of datetime values from ranging from start_date to start_date+horizon.

        start_date: date to start range (np.datetime64)
        into_future: number of days to add onto start date for range (int)
        offset: number of days to offset start_date by (default 1)
        """
        start_date = start_date + np.timedelta64(offset, "D")  # specify start date
        end_date = start_date + np.timedelta64(into_future, "D")  # specify end date
        return np.arange(start_date, end_date,
                         dtype="datetime64[D]")  # return a date range between start date and end date

    # Create a function to plot time series data
    def plot_time_series(self, timesteps, values, format='.', start=0, end=None, label=None):
        """
        Plots a timesteps (a series of points in time) against values (a series of values across timesteps).
        
        Parameters
        ---------
        timesteps : array of timesteps
        values : array of values across time
        format : style of plot, default "."
        start : where to start the plot (setting a value will index from start of timesteps & values)
        end : where to end the plot (setting a value will index from end of timesteps & values)
        label : label to show on plot of values
        """
        # Plot the series
        plt.plot(timesteps[start:end], values[start:end], format, label=label)
        plt.xlabel("Time")
        plt.ylabel("BTC Price")
        if label:
            plt.legend(fontsize=14)  # make label bigger
        plt.grid(True)

    def get_future_dates(self, start_date, into_future, offset=1):
        """
        Returns array of datetime values from ranging from start_date to start_date+horizon.

        start_date: date to start range (np.datetime64)
        into_future: number of days to add onto start date for range (int)
        offset: number of days to offset start_date by (default 1)
        """
        start_date = start_date + np.timedelta64(offset, "D")  # specify start date
        end_date = start_date + np.timedelta64(into_future, "D")  # specify end date
        return np.arange(start_date, end_date,
                         dtype="datetime64[D]")  # return a date range between start date and end date

    def prepareDataForPrediction(self, df, batch_size, window_size):
        # Add windowed columns
        df_for_pred = df.copy()
        for i in range(window_size):
            df_for_pred[f"Price+{i + 1}"] = df_for_pred["Price"].shift(periods=i + 1)

        # Train model on entire data to make prediction for the next day 
        X_all = df_for_pred.drop(["Price"],
                                 axis=1).dropna().to_numpy()  # only want prices, our future model can be a univariate model
        y_all = df_for_pred.dropna()["Price"].to_numpy()
        # 1. Turn X and y into tensor Datasets
        features_dataset_all = tf.data.Dataset.from_tensor_slices(X_all)
        labels_dataset_all = tf.data.Dataset.from_tensor_slices(y_all)

        # 2. Combine features & labels
        dataset_all = tf.data.Dataset.zip((features_dataset_all, labels_dataset_all))

        # 3. Batch and prefetch for optimal performance
        dataset_all = dataset_all.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return [dataset_all, X_all, y_all]

    def makeWinowedDataWithOutSentiment(self, stock_df, window_size, horizon):
        df = pd.DataFrame(stock_df["Close"]).rename(columns={"Close": "Price"})
        df_nbeats = df.copy()
        for i in range(window_size):
            df_nbeats[f"Price+{i + 1}"] = df["Price"].shift(periods=i + 1)

        df_nbeats.drop("Price", axis=1)
        return df_nbeats.dropna()

    def makeWinowedDataWithSentiment(self, stock_df, sentiment_df, window_size, horizon):
        df = pd.DataFrame(stock_df["Close"]).rename(columns={"Close": "Price"})
        df_nbeats = df.copy()
        sentiment_df = sentiment_df[sentiment_df.index.dayofweek < 5]
        sentiment_df = sentiment_df.head(len(stock_df))
        df_nbeats[f"sentiment"] = sentiment_df["sentiment_score"].values
        for i in range(window_size):
            df_nbeats[f"Price+{i + 1}"] = df["Price"].shift(periods=i + 1)

        df_nbeats.drop("Price", axis=1)
        return df_nbeats.dropna()
