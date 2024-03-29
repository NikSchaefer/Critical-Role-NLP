
# Critical Role Natural Language Processing

Text generation based on the Dungeons & Dragons series Critical Role




## Usage/Examples

First select an episode from the dataset below and place it in the /data folder. Then run the generate_script.py file with the file path linking to the episode file. Then run main.py.

This will create a new script generated by the model in generated.txt. If you want to upload your script place it in /generated and create a pull request

```py
FILE_PATH = "data/C2E034.json"

...
```
Then run `main.py`
```shell
py main.py
```

## Generated

see `/generated` for the generated files. Feel free to add your own if you run the script

The text below was generated with an input of "MATT: ". It aims to predict what comes next.

```py
MATT: That ends your turn?

MARISHA: Yep.

MATT: All righty, where do you want it to go?

LAURA: Wait. Wouldn't it be funny if that was
Marius?
...
```


## How it works

### Preprocessing
First we generate the vocab in which is used in the scripts. We then map it to a StringLookup layer in order to map each character into a number. We also create ids_from_chars in order to revert back to text near the end. We do this because it is easier for the model to process the numbers rather than individual characters.

Next we configure the dataset. We split the script into a number of sequences that we can use to train the model. These are split with split_input_sequence into a train side and a test side for each sequence. We then batch and shuffle the dataset to prepare for training.


### Model

The model we train on the dataset consists of 3 layers. We use an Embedding layer, a GRU layer, and a dense layer. The embedding layer first maps the input into weights that we can modify during training. The next layer is a GRU or Gated Recurrent Unit is a type of RNN similar to a LSTM but with only 2 gates. It works by determining what information is important and what can be forgotten within the text. Finally there is a dense layer to select an id within the vocab set. There is one logit for each character in the vocab. We then can map this id back to a string.

We use a sparse categorical crossentropy loss because we are labeling the logits we recieve from the dataset. We use sparse categorical crossentropy instead of categorical crossentropy because there is a signifigant amount of logits and not enough relation between them. Finally the adam optimizer is used to optimize our model.


### Text Generation

Finally after we have trained our model we aim to generate a script. To acomplish this we take a starting value of `MATT:` and plug it into our model to predict one letter at a time and then to continue to predict on the new text with the data. This process is repeated many times to generate a full script. First we define a mask to catch any faulty text from being generated. Then we run the convert the input into tensors by mapping it through our StringLookup. We then run the input through the models layers and make sure to save the state. We then repeat this process enough times to generate a full script.


## Dataset

Trained on the "Critical Role Dataset" on github (https://github.com/RevanthRameshkumar/CRD3)

## File Structure

```py
/data
/generated
generated.txt
generate_script.py
main.py
script.txt
```
`/data` contains the raw data files consisting of the script of episodes of critical role /generated contains the output or scripts generated by the model




## License

[MIT](https://choosealicense.com/licenses/mit/)

