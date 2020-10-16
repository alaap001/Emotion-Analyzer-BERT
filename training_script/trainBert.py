

from transformers import BertForSequenceClassification
from wrangling_scripts.data_wrangle import data_wrangle

df = data_wrangle()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case = True)

encoded_train_data = tokenizer.batch_encode_plus(
    df[df.data_type=='train'].text.values,
    add_special_tokens = True,
    return_attention_mask = True,
    pad_to_max_length = True,
    max_length = 256,
    return_tensors = 'pt'
)

encoded_val_data = tokenizer.batch_encode_plus(
    df[df.data_type=='val'].text.values,
    add_special_tokens = True,
    return_attention_mask = True,
    pad_to_max_length = True,
    max_length = 256,
    return_tensors = 'pt'
)

encoded_train_data

encoded_train_data.keys()

input_ids_train = encoded_train_data['input_ids']
token_type_ids_train = encoded_train_data['token_type_ids']
attention_mask_train = encoded_train_data['attention_mask']
labels_train = torch.tensor(df[df.data_type=='train'].label.values)

input_ids_val = encoded_val_data['input_ids']
token_type_ids_val = encoded_val_data['token_type_ids']
attention_mask_val = encoded_val_data['attention_mask']
labels_val = torch.tensor(df[df.data_type=='val'].label.values)

dataset_train = TensorDataset(input_ids_train,attention_mask_train,labels_train)
dataset_val = TensorDataset(input_ids_val,attention_mask_val,labels_val)
dataset_train

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
batch_size = 32

DataLoader_train = DataLoader(
    dataset_train,
    sampler = RandomSampler(dataset_train),
    batch_size=batch_size
)

DataLoader_val = DataLoader(
    dataset_val,
    sampler = RandomSampler(dataset_val),
    batch_size=batch_size
)

from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                  num_labels = len(label_dict),
                                                  output_attentions = False,
                                                  output_hidden_states = False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print('Using', device)
model

from transformers import AdamW, get_linear_schedule_with_warmup

opt = AdamW(model.parameters(),
        lr = 1e-5,
       eps = 1e-8)

epochs = 8
scheduler = get_linear_schedule_with_warmup(opt,
                                       num_warmup_steps=0,
                                       num_training_steps=len(DataLoader_train)*epochs)

from sklearn.metrics import f1_score

def get_f1_score(preds,labels):
preds_flat = np.argmax(preds,axis=1).flatten()
labels_flat = labels.flatten()
return f1_score(labels_flat,preds_flat,average='weighted')

def acc_per_class(preds,labels):
preds_flat = np.argmax(preds,axis=1).flatten()
labels_flat = labels.flatten()

label_dict_inv = {k:v for v,k in label_dict.items()}

for label in np.unique(labels_flat):
    y_preds = preds_flat[labels_flat==label]
    y_true = labels_flat[labels_flat==label]
    print(f'Class = {label_dict_inv[label]}')
    print(f'Acc = {len(y_preds[y_preds==label])} / {len(y_true)}\n')

def evaluate(dataloader_val):

    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total/len(dataloader_val) 

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals

def train(DataLoader_train):
    model.train()

    for epoch in tqdm(range(1, epochs+1)):

        loss_train_total = 0

        progress_bar = tqdm(DataLoader_train,
                            desc = "Epoch: {:1d}".format(epoch),
                            disable = False,
                            leave = False
                           )

        for batch in progress_bar:
            model.zero_grad()
            batch = tuple(b.to(device) for b in batch)
            
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[2],
                     }
            
            outputs = model(**inputs)
            
            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            
            opt.step()
            scheduler.step()
            
            progress_bar.set_postfix({"training_loss":"{:.3f}".format(loss.item()/len(batch))})
            
        torch.save(model.state_dict(),f'BERT_ft_{epoch}.pt')
        tqdm.write(f'Epoch {epoch}\n')
        loss_train_avg = loss_train_total/len(DataLoader_train)
        tqdm.write(f'train loss {loss_train_avg}')

        val_loss,prediction,true_label = evaluate(DataLoader_val)
        val_f1_score = get_f1_score(prediction,true_label)
        tqdm.write(f'val loss {val_loss}')
        tqdm.write(f'val f1 score {val_f1_score}')


model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                  num_labels = len(label_dict),
                                                  output_attentions = False,
                                                  output_hidden_states = False)


model.to(device)

model.load_state_dict(torch.load('BERT_ft_7.pt',map_location=torch.device('cpu')))

_,prediction, true_label = evaluate(DataLoader_val)
acc_per_class(prediction,true_label)