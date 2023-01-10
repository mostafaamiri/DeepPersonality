import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import seaborn as sns
from tqdm.notebook import tnrange, tqdm
import re
from sklearn.model_selection import train_test_split
import pickle
import sys
import codecs
from rich.console import Console
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, MofNCompleteColumn



with codecs.open("myPersonality_proc2.csv", 'r', encoding='utf-8',
                 errors='ignore') as fdata:
                 data = pd.read_csv(fdata)

max_doc = 50
max_len = 400
epochs = (int)(sys.argv[1])

users_status = {"id": [] , "sEXT": [],	"sNEU": [],	"sAGR": [],	"sCON": [],	"sOPN": [], "status":[], "conceptualized": []}
for user in data["#AUTHID"].unique():
  users_status["id"].append(user)
  statuses = []
  concepts = []
  for i, status in enumerate(data.loc[data["#AUTHID"] == user]["STATUS"]):
    if i == max_doc:
      break
    statuses.append(status)
    
  for i, status in enumerate(data.loc[data["#AUTHID"] == user]["conceptualized"]):
    if i == max_doc:
      break
    concepts.append(status)
    
  users_status["status"].append(np.array(statuses, dtype = 'object'))
  users_status["conceptualized"].append(np.array(concepts, dtype = 'object'))
  for key in ["sEXT",	"sNEU",	"sAGR",	"sCON",	"sOPN"]:
    users_status[key].append(data.loc[data["#AUTHID"] == user].iloc[0][key])

d = pd.DataFrame(users_status)
d["num_status"] = d["status"].apply(lambda x : len(x))
d["num_concept"] = d["conceptualized"].apply(lambda x : len(x))
for key in ["sEXT",	"sNEU",	"sAGR",	"sCON",	"sOPN"]:
  d[key] = d[key].apply(lambda x: x/5.)


train_ds, test_ds = train_test_split(d, test_size=0.2, random_state = 42)
train_ds, val_ds = train_test_split(train_ds, test_size= 0.1, random_state = 42)

console = Console()
text = Text((str)(len(train_ds)) +","+ (str)(len(val_ds))+","+(str)(len(test_ds)))
text.stylize("bold yellow")
console.print(text)

class ElmoEmbeddingLayer(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    self.dimensions = 1024
    self.trainable = False
    super(ElmoEmbeddingLayer, self).__init__(**kwargs)
  
  def build(self, input_shape):
    self.elmo = hub.load("elmo/tf-hub")#"https://tfhub.dev/google/elmo/3")
    super(ElmoEmbeddingLayer, self).build(input_shape)

  def call(self, x, mask=None):
    result = self.elmo.signatures["default"](x)
    return result

  def compute_output_shape(self, input_shape):
    return (input_shape[0], self.dimensions)

elmo = ElmoEmbeddingLayer(trainable=False)
'''
model_in = [tf.keras.Input(shape=(max_len, 2560)) for _ in range(max_doc)]
q1 = tf.Variable(tf.random.normal((max_len, 2560)),shape= (max_len, 2560),trainable=True, name="query1")
attention_layer1 = tf.keras.layers.Attention(use_scale= True, score_mode= "dot")
attens1 = [attention_layer1([q1, v]) for v in model_in]
bidirectional_lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences= True, trainable= True, name="inner_lstm"))
lstms1 = [tf.keras.layers.Dense(128, activation="tanh")(bidirectional_lstm1(atten)) for atten in attens1]
attention_layer2 = tf.keras.layers.Attention(use_scale= True, score_mode= "dot")
q2 = tf.Variable(tf.random.normal((max_len, 128)),shape= (max_len, 128),trainable=True, name="query1")
attens2 = [attention_layer2([q2, v]) for v in lstms1]
bidirectional_lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences= False, trainable= True, name="doc_lstm"))
person_embed = bidirectional_lstm2(tf.keras.layers.Reshape((max_doc, max_len*128))(tf.keras.layers.Concatenate(axis=1)(attens2)))
output = tf.keras.layers.Dense(5, activation="sigmoid")(person_embed)
output = tf.keras.layers.Dense(5, activation="sigmoid")(person_embed)
model = tf.keras.Model(inputs = model_in, outputs= output)
'''
'''
model_in = [tf.keras.Input(shape=(max_len, 2560)) for _ in range(max_doc)]
q1 = tf.Variable(tf.random.normal((max_len, 2560)),shape= (max_len, 2560),trainable=True, name="query1")
attention_layer1 = tf.keras.layers.Attention(use_scale= True, score_mode= "dot")
normalizer = tf.keras.layers.BatchNormalization()
attens1 = [attention_layer1([q1, normalizer(v)]) for v in model_in]
bidirectional_lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences= True, trainable= True, name="inner_lstm", activation="tanh"))
lstms1 = [bidirectional_lstm1(atten) for atten in attens1]
attention_layer2 = tf.keras.layers.Attention(use_scale= True, score_mode= "dot")
q2 = tf.Variable(tf.random.normal((max_len, 128)),shape= (max_len, 128),trainable=True, name="query1")
attens2 = [attention_layer2([q2, v]) for v in lstms1]
bidirectional_lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences= False, trainable= True, name="doc_lstm"))
person_embed = bidirectional_lstm2(tf.keras.layers.Reshape((max_doc, max_len*128))(tf.keras.layers.Concatenate(axis=1)(attens2)))
output = tf.keras.layers.Dense(1024, activation="relu")(person_embed)
normalizer2 = tf.keras.layers.BatchNormalization()
output = tf.keras.layers.Dense(1024, activation="relu")(normalizer2(output))
output = tf.keras.layers.Dense(5, activation="sigmoid")(output)
model = tf.keras.Model(inputs = model_in, outputs= output)
'''


class DeepPersonalityModel(tf.keras.Model):
  def __init__(self, mlp_layer):
    super().__init__()
    self.reshaper = tf.keras.layers.Reshape((max_doc, max_len*128))
    self.concatenator = tf.keras.layers.Concatenate(axis=1)
    self.q1 = tf.Variable(tf.random.normal((max_len, 2560)),shape= (max_len, 2560),trainable=True, name="query1")
    self.attention_layer1 = tf.keras.layers.Attention(use_scale= True, score_mode= "dot",trainable=True)
    self.normalizer = tf.keras.layers.BatchNormalization()
    self.bidirectional_lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences= True, trainable= True, name="inner_lstm", activation="tanh"))
    self.attention_layer2 = tf.keras.layers.Attention(use_scale= True, score_mode= "dot")
    self.q2 = tf.Variable(tf.random.normal((max_len, 128)),shape= (max_len, 128),trainable=True, name="query1")
    self.bidirectional_lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences= False, trainable= True, name="doc_lstm"))
    self.mlp_layer = []
    for hidden_layer in mlp_layer:
      self.mlp_layer.append(tf.keras.layers.Dense(hidden_layer, activation="relu",trainable=True))
    self.output_layer = tf.keras.layers.Dense(5, activation="sigmoid",trainable=True)
  
  def call(self, inputs):
    attens1 = [self.attention_layer1([self.q1, self.normalizer(v)]) for v in inputs]
    lstms1 = [self.bidirectional_lstm1(atten) for atten in attens1]
    attens2 = [self.attention_layer2([self.q2, v]) for v in lstms1]
    person_embed = self.bidirectional_lstm2(self.reshaper(self.concatenator(attens2)))
    for fc in self.mlp_layer:
      person_embed = fc(person_embed)
    return self.output_layer(person_embed)


model_in = [tf.keras.Input(shape=(max_len, 2560)) for _ in range(max_doc)]
m = DeepPersonalityModel([1024, 512, 256])(model_in)
model = tf.keras.Model(inputs = model_in, outputs = m)
#print(type(model))

tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True,show_layer_names=True)
text = Text("trainable variables: " + (str)(len(model.trainable_variables)))
text.stylize("bold green")
console.print(text)

loss = tf.keras.losses.MAE
optimizer = tf.keras.optimizers.Adam()

history = {"loss": [], "val_loss": []}
for epoch in range(epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    val_loss_avg = tf.keras.metrics.Mean()
    progress_bar = Progress(
        TextColumn("[yellow] epoch {}/{}".format(epoch+1,epochs)),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("[red]loss: ???"),
        TextColumn(""),
        TextColumn("")
        )
    with progress_bar as progress:
        task = progress.add_task("[yellow]epoch {}/{}".format(epoch+1, epochs), total=len(train_ds))
        for idx in range(len(train_ds)): #len(train_ds)
            embededs = []
            for i in range(max_doc):
                if i < len(train_ds.iloc[idx:idx+1]["status"].values[0]):
                    embed = elmo(np.array([(str)(train_ds.iloc[idx:idx+1]["status"].values[0][i])]))
                    synsem = tf.keras.layers.Concatenate()([
                                                            tf.pad(embed["word_emb"],[[0,0],[0,max_len-embed["word_emb"].shape[1]],[0, 0]]),
                                                            tf.pad(embed["lstm_outputs1"],[[0,0],[0,max_len-embed["word_emb"].shape[1]],[0, 0]]),
                                                            tf.pad(embed["lstm_outputs2"],[[0,0],[0,max_len-embed["word_emb"].shape[1]],[0, 0]])
                                                            ])
                    embed = elmo(np.array([(str)(train_ds.iloc[idx:idx+1]["conceptualized"].values[0][i])]))
                    concept = tf.keras.layers.Concatenate()([
                                                            tf.pad(embed["word_emb"],[[0,0],[0,max_len-embed["word_emb"].shape[1]],[0, 0]]),
                                                            tf.pad(embed["lstm_outputs1"],[[0,0],[0,max_len-embed["word_emb"].shape[1]],[0, 0]]),
                                                            tf.pad(embed["lstm_outputs2"],[[0,0],[0,max_len-embed["word_emb"].shape[1]],[0, 0]])
                                                            ])
                    embededs.append(tf.keras.layers.Add()([synsem, concept]))
                else:
                    embededs.append(tf.zeros((1, max_len, 2560)))
            with tf.GradientTape() as tape:
                output = model(embededs)
                
                loss_value = loss(train_ds.iloc[idx:idx+1][["sEXT", "sNEU", "sAGR", "sCON", "sOPN"]].values, output)
                grads = tape.gradient(loss_value, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                epoch_loss_avg.update_state(loss_value)
                progress.update(task, advance=1)
                progress.columns[6].text_format = "[red]loss: "+str(epoch_loss_avg.result().numpy())
        
        for idx in range(len(val_ds)):#
            embededs = []
            for i in range(max_doc):
                if i < len(val_ds.iloc[idx:idx+1]["status"].values[0]):
                    embed = elmo(np.array([(str)(val_ds.iloc[idx:idx+1]["status"].values[0][i])]))
                    synsem = tf.keras.layers.Concatenate()([
                                                            tf.pad(embed["word_emb"],[[0,0],[0,max_len-embed["word_emb"].shape[1]],[0, 0]]),
                                                            tf.pad(embed["lstm_outputs1"],[[0,0],[0,max_len-embed["word_emb"].shape[1]],[0, 0]]),
                                                            tf.pad(embed["lstm_outputs2"],[[0,0],[0,max_len-embed["word_emb"].shape[1]],[0, 0]])
                                                            ])
                    embed = elmo(np.array([(str)(val_ds.iloc[idx:idx+1]["conceptualized"].values[0][i])]))
                    concept = tf.keras.layers.Concatenate()([
                                                            tf.pad(embed["word_emb"],[[0,0],[0,max_len-embed["word_emb"].shape[1]],[0, 0]]),
                                                            tf.pad(embed["lstm_outputs1"],[[0,0],[0,max_len-embed["word_emb"].shape[1]],[0, 0]]),
                                                            tf.pad(embed["lstm_outputs2"],[[0,0],[0,max_len-embed["word_emb"].shape[1]],[0, 0]])
                                                            ])
                    embededs.append(tf.keras.layers.Add()([synsem, concept]))
                else:
                    embededs.append(tf.zeros((1, max_len, 2560)))
            output = model(embededs)
            loss_value = loss(val_ds.iloc[idx:idx+1][["sEXT", "sNEU", "sAGR", "sCON", "sOPN"]].values, output)
            val_loss_avg.update_state(loss_value)
        progress.columns[7].text_format = "•"
        progress.columns[8].text_format = "[blue]val_loss: "+str(val_loss_avg.result().numpy())
        history["loss"].append(epoch_loss_avg.result().numpy())
        history["val_loss"].append(val_loss_avg.result().numpy())

plt.plot(range(epochs), history["loss"], label="loss")
plt.plot(range(epochs), history["val_loss"], label="val_loss")
plt.legend()
plt.show()

progress_bar = Progress(
      BarColumn(),
      MofNCompleteColumn(),
      TextColumn("•"),
      TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
      TextColumn("•"),
      TextColumn("[red]test loss: ???")
      )
with progress_bar as progress:
  task = progress.add_task("[yellow]epoch {}/{}".format(epoch+1, epochs), total=len(test_ds))

  test_loss_avg = tf.keras.metrics.Mean()
  for idx in range(len(test_ds)):#
      embededs = []
      for i in range(max_doc):
          if i < len(test_ds.iloc[idx:idx+1]["status"].values[0]):
              embed = elmo(np.array([(str)(test_ds.iloc[idx:idx+1]["status"].values[0][i])]))
              synsem = tf.keras.layers.Concatenate()([
                                                      tf.pad(embed["word_emb"],[[0,0],[0,max_len-embed["word_emb"].shape[1]],[0, 0]]),
                                                      tf.pad(embed["lstm_outputs1"],[[0,0],[0,max_len-embed["word_emb"].shape[1]],[0, 0]]),
                                                      tf.pad(embed["lstm_outputs2"],[[0,0],[0,max_len-embed["word_emb"].shape[1]],[0, 0]])
                                                      ])
              embed = elmo(np.array([(str)(test_ds.iloc[idx:idx+1]["conceptualized"].values[0][i])]))
              concept = tf.keras.layers.Concatenate()([
                                                      tf.pad(embed["word_emb"],[[0,0],[0,max_len-embed["word_emb"].shape[1]],[0, 0]]),
                                                      tf.pad(embed["lstm_outputs1"],[[0,0],[0,max_len-embed["word_emb"].shape[1]],[0, 0]]),
                                                      tf.pad(embed["lstm_outputs2"],[[0,0],[0,max_len-embed["word_emb"].shape[1]],[0, 0]])
                                                      ])
              embededs.append(tf.keras.layers.Add()([synsem, concept]))
          else:
              embededs.append(tf.zeros((1, max_len, 2560)))
      output = model(embededs)
      loss_value = loss(test_ds.iloc[idx:idx+1][["sEXT", "sNEU", "sAGR", "sCON", "sOPN"]].values, output)
      test_loss_avg.update_state(loss_value)
      progress.update(task, advance=1)
      progress.columns[5].text_format = "[red]test loss: "+str(test_loss_avg.result().numpy())
      
text = Text("Saving weights ...")
text.stylize("bold yellow")
console.print(text)
model.save_weights("deepPersonality_weights", save_format="tf")
text = Text("model saved! Bye")
text.stylize("bold red")
console.print(text)

                    
