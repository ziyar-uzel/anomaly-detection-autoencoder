import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
RANDOM_SEED = 2021
TEST_PCT = 0.3
LABELS = ["Normal","Anomaly"]
df_train = pd.read_csv('train_data.csv')
df_val = pd.read_csv('val_data.csv')
df_test = pd.read_csv('test_data.csv')


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)

    print("Any nulls in the dataset ", df_train.isnull().values.any())
    print('-------')
    sc = StandardScaler()
    # timeless_data = df_train.loc[:, ~df_train.columns.isin(['Timestamp', 'Id','Label'])]
    # timeless_data = timeless_data.head(500)

    train_data_demo = df_train.loc[:, ~df_train.columns.isin(['Timestamp', 'Id','Label'])]
    # train_data_demo = train_data_demo.head(100)

    # val_data_demo = df_val.loc[:, ~df_val.columns.isin(['Timestamp', 'Id','Label'])]
    # val_data_demo = val_data_demo.head(10000)
    #
    # val_labels = df_val.loc[:, df_val.columns=='Label']
    # print(val_labels.shape)
    # val_labels = val_labels.head(10000)
    #
    # val_ids = df_val.loc[:, df_val.columns=="Id"]
    # val_ids = val_ids.head(10000)

    val_data_demo = df_test.loc[:, ~df_test.columns.isin(['Timestamp', 'Id'])]
    # val_data_demo = val_data_demo.head(100)


    val_ids = df_test.loc[:, df_test.columns=="Id"]
    # val_ids = val_ids.head(100)


    # noTimeStamp  = sc.fit_transform(noTimeStamp.values.reshape(-1, 1))
    for column in train_data_demo:
        train_data_demo[column] = sc.fit_transform(train_data_demo[column].values.reshape(-1, 1))
    for column in val_data_demo:
        val_data_demo[column] = sc.fit_transform(val_data_demo[column].values.reshape(-1, 1))



    min_val = tf.reduce_min(train_data_demo.values)
    max_val = tf.reduce_max(train_data_demo.values)
    train_data = (train_data_demo.values - min_val) / (max_val - min_val)
    test_data = (val_data_demo.values - min_val) / (max_val - min_val)
    train_data = tf.cast(train_data, tf.float32)
    test_data = tf.cast(test_data, tf.float32)
    print("ok")
    print(" No. of records in Normal Train data=", len(train_data))
    print(" No. of records in Normal Train data=",len(test_data))

    nb_epoch = 50
    batch_size = 64
    input_dim = train_data.shape[1]  # num of columns
    encoding_dim = 14
    hidden_dim_1 = int(encoding_dim / 2)  #
    hidden_dim_2 = 4
    learning_rate = 1e-7

    input_layer = tf.keras.layers.Input(shape=(input_dim,))
    # Encoder
    encoder = tf.keras.layers.Dense(encoding_dim, activation="tanh",
                                    activity_regularizer=tf.keras.regularizers.l2(learning_rate))(input_layer)
    encoder = tf.keras.layers.Dropout(0.2)(encoder)
    encoder = tf.keras.layers.Dense(hidden_dim_1, activation='relu')(encoder)
    encoder = tf.keras.layers.Dense(hidden_dim_2, activation=tf.nn.leaky_relu)(encoder)
    # Decoder
    decoder = tf.keras.layers.Dense(hidden_dim_1, activation='relu')(encoder)
    decoder = tf.keras.layers.Dropout(0.2)(decoder)
    decoder = tf.keras.layers.Dense(encoding_dim, activation='relu')(decoder)
    decoder = tf.keras.layers.Dense(input_dim, activation='tanh')(decoder)
    # Autoencoder
    autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoder)
    autoencoder.summary()

    cp = tf.keras.callbacks.ModelCheckpoint(filepath="autoencoder_anomaly.h5",
                                            mode='min', monitor='val_loss', verbose=2, save_best_only=True)
    # define our early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=10,
        verbose=1,
        mode='min',
        restore_best_weights=True)
    autoencoder.compile(metrics=['accuracy'],
                        loss='mean_squared_error',
                        optimizer='adam')

    history = autoencoder.fit(train_data, train_data,
                              epochs=nb_epoch,
                              batch_size=batch_size,
                              shuffle=True,
                              validation_data=(test_data, test_data),
                              verbose=1,
                              callbacks=[cp, early_stop]
                              ).history

    test_x_predictions = autoencoder.predict(test_data)
    mse = np.mean(np.power(test_data - test_x_predictions, 2), axis=1)
    # error_df = pd.DataFrame(
    #     {'Reconstruction_error': mse,
    #                          'True_class': val_labels.values.astype(bool).ravel(),
    #                             'Id' : val_ids.values.ravel()})

    error_df = pd.DataFrame(
        {'Reconstruction_error': mse,
                                'Id' : val_ids.values.ravel()})

    threshold_fixed = 0.00049995

    res = []

    for i in error_df.index:
        if error_df.Reconstruction_error.values[i]>=threshold_fixed:
            res.append(1)
        else:
            res.append(0)
    print(len(res))
    print(len(val_ids.index))

    predictions = {
        'Id': val_ids['Id']
    }
    # print("-----------")
    # print(predictions['Id'])
    predictions['Label'] = res
    df = pd.DataFrame(predictions)
    df.to_csv("submission.csv",index=False)


    # groups = error_df.groupby('True_class')
    # fig, ax = plt.subplots()
    # for name, group in groups:
    #     ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
    #             label="Anomaly" if name == 1 else "Normal")
    # ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=15, label='Threshold')
    # ax.legend()
    # plt.title("Reconstruction error for normal and anomaly data")
    # plt.ylabel("Reconstruction error")
    # plt.xlabel("Data point index")
    # plt.show();

    # threshold_fixed = 52
    # pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]
    # error_df['pred'] = pred_y
    # conf_matrix = confusion_matrix(error_df.True_class, pred_y)
    # plt.figure(figsize=(4, 4))
    # sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    # plt.title("Confusion matrix")
    # plt.ylabel('True class')
    # plt.xlabel('Predicted class')
    # plt.show()
    # # print Accuracy, precision and recall
    # print(" Accuracy: ", accuracy_score(error_df['True_class'], error_df['pred']))
    # print(" Recall: ", recall_score(error_df['True_class'], error_df['pred']))
    # print(" Precision: ", precision_score(error_df['True_class'], error_df['pred']))