import tensorflow as tf
import string
import re


@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, f"[{re.escape(string.punctuation)}]", ""
    )

reconstructed_model = tf.keras.models.load_model("my_model")

# reconstructed_model.fit(test_input, test_target)

examples = [
#   "The movie was awesome!",
#   "The movie was shit.",
#   "The movie was terrible...",
#   "The movie was terrible, total waster of time, would be better to watch Pele",
  "English not very good",
  "It was so lovely to speak to Ibrahim and learn about his plans for Mabrouklyn! He’s so passionate about the business & I’m really looking forward to seeing it grow! I’d love to have more sessions with him to help him with branding & marketing.",
  "The mentee didn't have his camera on.  which may have been a technical issue.  However he was also reluctant to comply to trading standards and GDPR standards and possibly interest in unethical sales.",
  "Unfortunately the mentor didn't spend time getting to understand me or my business and so rather ended up ranting at me about things I was already aware of. There were a few interesting points in the latter half of the call, but with a bit more of a fact find about my experiences to date, the first half of the call could have been skipped.",
  "It was great to meet Gabriel. He has a lot of passion for his business and is really keen to learn - I wish him all the best and look forward to hearing a lot about Song Drop in the future!",
]

res = reconstructed_model.predict(examples)

print(res)

# def make_prediction(word):
#     with open(os.path.expanduser('/rnn_watchar/rnn_watchar.net'), 'rb') as f:
#         checkpoint = torch.load(f, map_location=torch.device('cpu'))
#     loaded = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
#     loaded.load_state_dict(checkpoint['state_dict'])
    
#     predict_neg_pos = sample(word)
#     prediction = ''.join(char_seq)
#     results = []
#     for c in char_seq:
#         if dict_err.get(c):
#             results.append(dict_err.get(c))
#     return results, prediction



# import tensorflow as tf

# # A string input
# inputs = tf.keras.Input(shape=(1,), dtype="string")
# # Turn strings into vocab indices
# indices = vectorize_layer(inputs)
# # Turn vocab indices into predictions
# outputs = model(indices)

# # Our end to end model
# end_to_end_model = tf.keras.Model(inputs, outputs)
# end_to_end_model.compile(
#     loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
# )

# # Test it with `raw_test_ds`, which yields raw strings
# end_to_end_model.evaluate(raw_test_ds)