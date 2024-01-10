import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer
import matplotlib.pyplot as plt

tf.get_logger().setLevel("ERROR")
bert_model_path = os.path.join(os.getcwd(), "small_bert")
model_works_cor = os.path.join(os.getcwd(), "bert_results", "correct.txt")
model_works_inc = os.path.join(os.getcwd(), "bert_results", "incorrect.txt")


# choose BERT model (small BERT to ease fine tuning)
def prepare_bert_model():
    bert_model_name = "small_bert/bert_en_uncased_L-4_H-512_A-8"

    map_name_to_handle = {
        "bert_en_uncased_L-12_H-768_A-12": "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3",
        "bert_en_cased_L-12_H-768_A-12": "https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3",
        "bert_multi_cased_L-12_H-768_A-12": "https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3",
        "small_bert/bert_en_uncased_L-2_H-128_A-2": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1",
        "small_bert/bert_en_uncased_L-2_H-256_A-4": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1",
        "small_bert/bert_en_uncased_L-2_H-512_A-8": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1",
        "small_bert/bert_en_uncased_L-2_H-768_A-12": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1",
        "small_bert/bert_en_uncased_L-4_H-128_A-2": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1",
        "small_bert/bert_en_uncased_L-4_H-256_A-4": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1",
        "small_bert/bert_en_uncased_L-4_H-512_A-8": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1",
        "small_bert/bert_en_uncased_L-4_H-768_A-12": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1",
        "small_bert/bert_en_uncased_L-6_H-128_A-2": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1",
        "small_bert/bert_en_uncased_L-6_H-256_A-4": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1",
        "small_bert/bert_en_uncased_L-6_H-512_A-8": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1",
        "small_bert/bert_en_uncased_L-6_H-768_A-12": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1",
        "small_bert/bert_en_uncased_L-8_H-128_A-2": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1",
        "small_bert/bert_en_uncased_L-8_H-256_A-4": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1",
        "small_bert/bert_en_uncased_L-8_H-512_A-8": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1",
        "small_bert/bert_en_uncased_L-8_H-768_A-12": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1",
        "small_bert/bert_en_uncased_L-10_H-128_A-2": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1",
        "small_bert/bert_en_uncased_L-10_H-256_A-4": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1",
        "small_bert/bert_en_uncased_L-10_H-512_A-8": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1",
        "small_bert/bert_en_uncased_L-10_H-768_A-12": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1",
        "small_bert/bert_en_uncased_L-12_H-128_A-2": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1",
        "small_bert/bert_en_uncased_L-12_H-256_A-4": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1",
        "small_bert/bert_en_uncased_L-12_H-512_A-8": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1",
        "small_bert/bert_en_uncased_L-12_H-768_A-12": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1",
        "albert_en_base": "https://tfhub.dev/tensorflow/albert_en_base/2",
        "electra_small": "https://tfhub.dev/google/electra_small/2",
        "electra_base": "https://tfhub.dev/google/electra_base/2",
        "experts_pubmed": "https://tfhub.dev/google/experts/bert/pubmed/2",
        "experts_wiki_books": "https://tfhub.dev/google/experts/bert/wiki_books/2",
        "talking-heads_base": "https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1",
    }

    map_model_to_preprocess = {
        "bert_en_uncased_L-12_H-768_A-12": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "bert_en_cased_L-12_H-768_A-12": "https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3",
        "small_bert/bert_en_uncased_L-2_H-128_A-2": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-2_H-256_A-4": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-2_H-512_A-8": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-2_H-768_A-12": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-4_H-128_A-2": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-4_H-256_A-4": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-4_H-512_A-8": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-4_H-768_A-12": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-6_H-128_A-2": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-6_H-256_A-4": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-6_H-512_A-8": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-6_H-768_A-12": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-8_H-128_A-2": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-8_H-256_A-4": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-8_H-512_A-8": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-8_H-768_A-12": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-10_H-128_A-2": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-10_H-256_A-4": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-10_H-512_A-8": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-10_H-768_A-12": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-12_H-128_A-2": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-12_H-256_A-4": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-12_H-512_A-8": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "small_bert/bert_en_uncased_L-12_H-768_A-12": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "bert_multi_cased_L-12_H-768_A-12": "https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3",
        "albert_en_base": "https://tfhub.dev/tensorflow/albert_en_preprocess/3",
        "electra_small": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "electra_base": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "experts_pubmed": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "experts_wiki_books": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        "talking-heads_base": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
    }

    tfhub_handle_encoder = map_name_to_handle[bert_model_name]
    tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]
    print(f"BERT model selected           : {tfhub_handle_encoder}")
    print(f"Preprocess model auto-selected: {tfhub_handle_preprocess}")
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name="preprocessing")
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name="BERT_encoder")
    outputs = encoder(encoder_inputs)
    net = outputs["pooled_output"]
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation=None, name="classifier")(net)
    return tf.keras.Model(text_input, net)


def check_model():
    classifier_model = prepare_bert_model()
    tf.keras.utils.plot_model(classifier_model)


# train BERT model
def train_model(classifier_model, train_ds, test_ds, val_ds):
    global bert_model_path
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = tf.metrics.BinaryAccuracy()
    epochs = 5
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)

    init_lr = 3e-5
    optimizer = optimization.create_optimizer(
        init_lr=init_lr,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        optimizer_type="adamw",
    )
    classifier_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    history = classifier_model.fit(x=train_ds, validation_data=val_ds, epochs=epochs)
    loss, accuracy = classifier_model.evaluate(test_ds)
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")
    dataset_name = "imdb"
    classifier_model.save(bert_model_path, include_optimizer=False)


# set up BERT trained model
def prepare_model():
    if os.path.exists(bert_model_path) == False:
        # get IMDB dataset
        if os.path.exists(os.path.join(os.getcwd(), "aclImdb")) == False:
            url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
            dataset = tf.keras.utils.get_file(
                "aclImdb_v1.tar.gz", url, untar=True, cache_dir=".", cache_subdir=""
            )
            dataset_dir = os.path.join(os.path.dirname(dataset), "aclImdb")
            train_dir = os.path.join(dataset_dir, "train")
            # remove unused folders to make it easier to load the data
            remove_dir = os.path.join(train_dir, "unsup")
            shutil.rmtree(remove_dir)

        # Validation split

        AUTOTUNE = tf.data.AUTOTUNE
        batch_size = 32
        seed = 42
        raw_train_ds = tf.keras.utils.text_dataset_from_directory(
            "aclImdb/train",
            batch_size=batch_size,
            validation_split=0.2,
            subset="training",
            seed=seed,
        )

        class_names = raw_train_ds.class_names
        train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = tf.keras.utils.text_dataset_from_directory(
            "aclImdb/train",
            batch_size=batch_size,
            validation_split=0.2,
            subset="validation",
            seed=seed,
        )
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        test_ds = tf.keras.utils.text_dataset_from_directory(
            "aclImdb/test", batch_size=batch_size
        )
        test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

        model = prepare_bert_model()
        train_model(model, train_ds, test_ds, val_ds)


def prediction_correct(prediction, meaning):
    if (prediction - meaning) > 0.25:
        return 0
    return 1


# print examples
def print_my_examples(inputs, results, meaning):
    result_for_printing = [
        f"input: {inputs[i]:<30} : NN score: {results[i][0]:.6f} : word real meaning: {meaning} : neural network correctness: {prediction_correct(results[i][0], meaning)}"
        for i in range(len(inputs))
    ]
    correct_result_for_printing = [
        i for i in result_for_printing if "neural network correctness: 1" in i
    ]
    incorrect_result_for_printing = [
        i for i in result_for_printing if "neural network correctness: 0" in i
    ]
    with open(model_works_cor, "a") as f:
        f.write("\n".join(correct_result_for_printing))
    with open(model_works_inc, "a") as f:
        f.write("\n".join(incorrect_result_for_printing))
    print(incorrect_result_for_printing)
    print(correct_result_for_printing)
    print()


if os.path.exists(bert_model_path) == False:
    prepare_model()
print("model already exists")
reloaded_model = tf.saved_model.load(bert_model_path)
examples = [
    "The exam is salty",
    "The responce a clapback",
    "The lecture is drip.",
    "This bike is lowkey",
]

reloaded_results = tf.sigmoid(reloaded_model(tf.constant(examples)))
print("Results from the saved model:")
print_my_examples(examples, reloaded_results, 0.25)
