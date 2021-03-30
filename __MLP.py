import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.sampler import WeightedRandomSampler
from torch.optim import lr_scheduler

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AdamW, get_linear_schedule_with_warmup

from fetchData import fetchdata 
import random
from sklearn.preprocessing import StandardScaler


def clf_report(train_loss, train_acc, val_loss, val_acc):
    fig, ax = plt.subplots(2, 1, figsize=(12,12))
    ax[0].plot(train_loss[:],label='train')
    ax[0].plot(val_loss[:],label='test')
    ax[0].set_ylabel('Classification Accuracy')
    ax[0].set_title('Loss')
    ax[0].legend()

    ax[1].plot(train_acc[:],label='train')
    ax[1].plot(val_acc[:],label='test')
    ax[1].set_ylabel('Classification Accuracy')
    ax[1].set_title('Accuracy')
    ax[1].legend()
    plt.tight_layout()
    plt.show()

    print("Min of Training Loss: %4f"%(np.min(train_loss)))
    print("Max of Training Accuracy: %4f"%(np.max(train_acc)))
    print("Mean of Training Loss: %4f"%(np.mean(train_loss)))
    print("Mean of Training Accuracy: %4f"%(np.mean(train_acc)))
    print("------------")
    print("Max of Testing Accuracy: %4f"%(np.max(val_acc)))
    print("Min of Testing Loss: %4f"%(np.min(val_loss)))
    print("Mean of Testing Loss: %4f"%(np.mean(val_loss)))
    print("Mean of Testing Accuracy: %4f"%(np.mean(val_acc)))
    print("------------")


def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


'''
#Example 
__MLP.train_sequential(model=<model>, num_epochs=40, patience=15, criterion=criterion, optimizer=optimizer, scheduler=scheduler, train_loader=train_dataloader,train_size=train_size, test_loader=test_dataloader, test_size=test_size, PATH=PATH)

'''
def train_sequential(model, num_epochs, criterion, optimizer, scheduler, train_loader, train_size, test_loader=None, test_size=None, patience=5, PATH='./state_dict_model.pt'):
    train_loss = []
    patience_count = 0
    train_accuracy = []
    prev_loss = 10
    best_loss = 10.0
    val_corrects_list = []
    val_loss_list = []
    val_acc_list = []
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)
        running_corrects = 0.0
        running_loss = 0.0
        model.train()  # Set model to training mode
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.float(), labels.float()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            #  _, predictions = torch.max(outputs.data, 1) won’t work if your output only contains a single output unit.
            # _, preds = torch.max(outputs, 1)
            # print(outputs.flatten().size())
            preds = outputs.squeeze(1) > 0.0

            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            # step function
            scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / train_size
        # print(running_loss)
        # print(train_size)
        epoch_acc = running_corrects.double() / train_size
        train_loss.append(epoch_loss)
        train_accuracy.append(epoch_acc)

        if (epoch % 2 == 0):
            print('Epoch {}/{}\tTrain) Acc: {:.4f}, Loss: {:.4f}'.format(epoch,
                                                                         num_epochs - 1, epoch_acc, epoch_loss))

        if (test_loader != None):
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_corrects = 0
                val_preds_list = []
                val_label_list = []
                for j, val in enumerate(test_loader, 0):
                    val_x, val_label = val
                    val_x, val_label = val_x.float(), val_label.float()
                    val_outputs = model(val_x)
                    # _, val_preds = torch.max(val_outputs, 1)
                    val_preds = val_outputs.squeeze(1) > 0.0

                    val_preds_list.append(val_preds)
                    val_label_list.append(val_label)
                    v_loss = criterion(val_outputs, val_label.unsqueeze(1))
                    val_loss += (v_loss.item() * val_x.size(0))
                    val_corrects += torch.sum(val_preds == val_label)
                    # accuracy = (preds == b_labels).cpu().numpy().mean() * 100

                if (epoch % 2 == 0):
                    val_preds_list = torch.cat(val_preds_list, 0)
                    val_label_list = torch.cat(val_label_list, 0)
                    # print("\t\tValidation) Acc: {:.4f} Loss:{:.4f} F1 score: {:4f}".format(val_corrects/test_size, val_loss/test_size, f1_score(val_label_list,val_preds_list,average='macro')))
                    print("\t\tValidation) Acc: {:.4f} Loss:{:.4f}".format(
                        val_corrects/test_size, val_loss/test_size))
            val_corrects_list.append(val_corrects/test_size)
            val_epoch_loss= val_loss/test_size
            val_loss_list.append(val_epoch_loss)
            val_acc = val_corrects.double() / test_size
            val_acc_list.append(val_acc)

            if val_epoch_loss < best_loss:
                # print("prev_loss: {:.5f}".format(prev_loss))
                # print("loss: {:.5f}".format(loss))
                print(
                    "\t\t\tSaving the best model w/ val loss {:.4f}".format(val_epoch_loss))
                torch.save(model.state_dict(), PATH)
                best_loss = val_epoch_loss
                patience_count = 0
            elif best_loss < val_epoch_loss:
                patience_count += 1
            if patience_count >= patience:
                print("Finishing the Model: Val Loss is not decreasing...")
                print(val_loss_list[-6:-1])
                return train_accuracy, train_loss, val_acc_list, val_loss_list
            
        else:
            if epoch_loss < best_loss:
                # print("prev_loss: {:.5f}".format(prev_loss))
                # print("loss: {:.5f}".format(loss))
                print(
                    "\t\tSaving the best model w/ loss {:.4f}".format(epoch_loss))
                torch.save(model.state_dict(), PATH)
                best_loss = epoch_loss
                patience_count = 0
            elif best_loss < epoch_loss:
                patience_count += 1
            if patience_count >= patience:
                print("Finishing the Model: Loss is not decreasing...")
                print(train_loss[-6:-1])
                return train_accuracy, train_loss, val_acc_list, val_loss_list
    return train_accuracy, train_loss, val_acc_list, val_loss_list

def train_multi(model, num_epochs, criterion, optimizer, scheduler, train_loader, train_size, test_loader=None, test_size=None, patience=5, PATH='./state_dict_model.pt'):
    train_loss = []
    patience_count = 0
    train_accuracy = []
    prev_loss = 10
    best_loss = 10.0
    val_corrects_list = []
    val_loss_list = []
    val_acc_list = []
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)
        running_corrects = 0.0
        running_loss = 0.0
        model.train()  # Set model to training mode
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            sparse, embedding, labels = data
            sparse, embedding, labels = sparse.float(), embedding.float(), labels.float().squeeze(1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(sparse, embedding)

            #  _, predictions = torch.max(outputs.data, 1) won’t work if your output only contains a single output unit.
            # _, preds = torch.max(outputs, 1)
            # print(outputs.flatten().size())
            preds = outputs.squeeze(1) > 0.0

            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            # step function
            scheduler.step()

            running_loss += loss.item() * sparse.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / train_size
        # print(running_loss)
        # print(train_size)
        epoch_acc = running_corrects.double() / train_size
        train_loss.append(epoch_loss)
        train_accuracy.append(epoch_acc)

        if (epoch % 2 == 0):
            print('Epoch {}/{}\tTrain) Acc: {:.4f}, Loss: {:.4f}'.format(epoch,
                                                                         num_epochs - 1, epoch_acc, epoch_loss))

        if (test_loader != None):
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_corrects = 0
                val_preds_list = []
                val_label_list = []
                for j, val in enumerate(test_loader, 0):
                    val_sparse, val_embedding, val_labels = data
                    val_sparse, val_embedding, val_labels = val_sparse.float(), val_embedding.float(), val_labels.float().squeeze(1)
                    val_outputs = model(val_sparse, val_embedding)
                    # _, val_preds = torch.max(val_outputs, 1)
                    # print("val_outputs size:",val_outputs.size())
                    # print("val_label size:",val_label.size())
                    val_preds = val_outputs.squeeze(1) > 0.0
                    # print("pred size:",val_preds.size())

                    # print("Length of val_preds:",val_preds.size())
                    val_preds_list.append(val_preds)
                    val_label_list.append(val_labels)
                    v_loss = criterion(val_outputs, val_labels.unsqueeze(1))
                    val_loss += (v_loss.item() * val_labels.size(0))
                    val_corrects += torch.sum(val_preds == val_labels)

                    # print("== length",val_preds == val_label)
                    # accuracy = (preds == b_labels).cpu().numpy().mean() * 100

                    # print("val_corrects:",val_corrects)
                # print("length of one batch")
                # print("FINAL val_corrects:",val_corrects)
                # print("test_size:",test_size)
                if (epoch % 2 == 0):
                    val_preds_list = torch.cat(val_preds_list, 0)
                    val_label_list = torch.cat(val_label_list, 0)
                    # print("\t\tValidation) Acc: {:.4f} Loss:{:.4f} F1 score: {:4f}".format(val_corrects/test_size, val_loss/test_size, f1_score(val_label_list,val_preds_list,average='macro')))
                    print("\t\tValidation) Acc: {:.4f} Loss:{:.4f}".format(
                        val_corrects/test_size, val_loss/test_size))
            val_corrects_list.append(val_corrects/test_size)
            val_epoch_loss= val_loss/test_size
            val_loss_list.append(val_epoch_loss)
            val_acc = val_corrects.double() / test_size
            val_acc_list.append(val_acc)

            if val_epoch_loss < best_loss:
                # print("prev_loss: {:.5f}".format(prev_loss))
                # print("loss: {:.5f}".format(loss))
                print(
                    "\t\t\tSaving the best model w/ val loss {:.4f}".format(val_epoch_loss))
                torch.save(model.state_dict(), PATH)
                best_loss = val_epoch_loss
                patience_count = 0
            elif best_loss < val_epoch_loss:
                patience_count += 1
            if patience_count >= patience:
                print("Finishing the Model: Val Loss is not decreasing...")
                print(val_loss_list[-6:-1])
                return train_accuracy, train_loss, val_acc_list, val_loss_list
        
        else:
            if epoch_loss < best_loss:
                # print("prev_loss: {:.5f}".format(prev_loss))
                # print("loss: {:.5f}".format(loss))
                print(
                    "\t\tSaving the best model w/ loss {:.4f}".format(epoch_loss))
                torch.save(model.state_dict(), PATH)
                best_loss = epoch_loss
                patience_count = 0
            elif best_loss < epoch_loss:
                patience_count += 1
            if patience_count >= patience:
                print("Finishing the Model: Loss is not decreasing...")
                print(train_loss[-6:-1])
                return train_accuracy, train_loss, val_acc_list, val_loss_list
    return train_accuracy, train_loss, val_acc_list, val_loss_list

def train2(model, num_epochs, criterion, optimizer, train_loader, train_size, test_loader=None, test_size=None, patience=5, PATH='./state_dict_model.pt'):
    train_loss = []
    patience_count = 0
    train_accuracy = []
    prev_loss = 10
    best_loss = 10.0
    val_corrects_list = []
    val_loss_list = []
    val_acc_list = []
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)
        running_corrects = 0.0
        running_loss = 0.0
        model.train()  # Set model to training mode
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.float(), labels.float()
            print(inputs.size())
            print(labels.size())
            print(inputs.flatten())
            print(labels.flatten())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            print("outputs:",outputs.size())
            print("outputs:",outputs)
            print("labels:",labels.size())
            print("labels:",labels.unsqueeze(1).size())

            #  _, predictions = torch.max(outputs.data, 1) won’t work if your output only contains a single output unit.
            # _, preds = torch.max(outputs, 1)
            preds = torch.argmax(outputs, dim=1).flatten()
            # print(outputs.flatten().size())
            # preds = outputs > 0.0
            # labels = labels.view(-1)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            # step function
            scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            # print('running correct')
            # print(running_corrects)

        epoch_loss = running_loss / train_size
        # print(running_loss)
        # print(train_size)
        epoch_acc = running_corrects.double() / train_size
        train_loss.append(epoch_loss)
        train_accuracy.append(epoch_acc)

        if (epoch % 2 == 0):
            print('Epoch {}/{}\tTrain) Acc: {:.4f}, Loss: {:.4f}'.format(epoch,
                                                                         num_epochs - 1, epoch_acc, epoch_loss))

        if (test_loader != None):
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_corrects = 0
                val_preds_list = []
                val_label_list = []
                for j, val in enumerate(test_loader, 0):
                    val_x, val_label = val
                    val_x, val_label = val_x.float(), val_label.float()
                    val_outputs = model(val_x)
                    val_preds = torch.argmax(val_outputs, dim=1).flatten()
                    # _, val_preds = torch.max(val_outputs, 1)
                    # print("val_outputs:",val_outputs.flatten())
                    # val_preds = val_outputs > 0.0
                    # print("val_preds:",val_preds)
                    val_preds_list.append(val_preds)
                    val_label_list.append(val_label)
                    v_loss = criterion(val_outputs, val_label.unsqueeze(1))
                    val_loss += (v_loss.item() * val_x.size(0))
                    val_corrects += torch.sum(val_preds ==
                                              val_label.data).double()
                if (epoch % 2 == 0):
                    val_preds_list = torch.cat(val_preds_list, 0)
                    val_label_list = torch.cat(val_label_list, 0)
                    # print("\t\tValidation) Acc: {:.4f} Loss:{:.4f} F1 score: {:4f}".format(val_corrects/test_size, val_loss/test_size, f1_score(val_label_list,val_preds_list,average='macro')))
                    print("\t\tValidation) Acc: {:.4f} Loss:{:.4f}".format(
                        val_corrects/test_size, val_loss/test_size))
            val_corrects_list.append(val_corrects/test_size)
            val_loss_list.append(val_loss/test_size)
            val_acc = val_corrects.double() / test_size
            val_acc_list.append(val_acc)

        if epoch_loss < best_loss:
            # print("prev_loss: {:.5f}".format(prev_loss))
            # print("loss: {:.5f}".format(loss))
            print(
                "\t\tSaving the best model w/ loss {:.4f}".format(epoch_loss))
            torch.save(model.state_dict(), PATH)
            best_loss = epoch_loss
            patience_count = 0
        elif best_loss < epoch_loss:
            patience_count += 1
        if patience_count >= patience:
            print("Finishing the Model: Loss is not decreasing...")
            print(train_loss[-6:-1])
            return train_accuracy, train_loss, val_acc_list, val_loss_list
    return train_accuracy, train_loss, val_acc_list, val_loss_list

def bert_predict(model, test_dataloader, target):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    all_logits = []

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    # Get predictions from the probabilities
    threshold = 0.5
    preds = np.where(probs[:, 1] > threshold, 1, 0)

    # Number of tweets predicted non-negative
    print("Number of tweets predicted as Rumor: ", preds.sum())

    preds = np.argmax(probs, axis = 1)
    print("\n",classification_report(y_val, preds))
    print('Accuracy Score:\t',accuracy_score(y_val, preds))
    print('Precision Score:\t', str(precision_score(y_val,preds)))
    print('Recall Score:\t\t' + str(recall_score(y_val,preds)))
    print('F1 Score:\t',f1_score(y_val, preds, zero_division=1))

    return probs

def predict(model, criterion, val_dataloader, val_size):
    model.eval()
    val_label_list = []
    # val_preds_list = []
    running_val_preds = []

    with torch.no_grad():
        val_loss = 0.0
        val_corrects = 0
        f1_running = 0
        for j, val in enumerate(val_dataloader, 0):
            val_x, val_label = val
            val_x, val_label = val_x.float(), val_label.float()
            val_outputs = model(val_x)
            val_preds = val_outputs.squeeze(1) > 0.0
            f1_running += (f1_score(val_label, val_preds,zero_division=True) * val_x.size(0))
            v_loss = criterion(val_outputs, val_label.unsqueeze(1))
            val_loss += (v_loss.item() * val_x.size(0))
            val_corrects += torch.sum(val_preds == val_label)
            val_label_list.append(val_label)
            running_val_preds.append(val_preds)

    running_val_preds = torch.cat(running_val_preds, 0)
    val_label_list = torch.cat(val_label_list, 0)
    val_corrects = val_corrects
    val_loss = val_loss/val_size
    val_acc = val_corrects.double().numpy() / val_size
    f1_running /= val_size

    print("accuracy_score:\t\t%.4f" % val_acc)
    print('Precision Score:\t%.4f' % precision_score(val_label_list,running_val_preds))
    print('Recall Score:\t\t%.4f' % recall_score(val_label_list,running_val_preds))
    print("f1_score:\t\t%.4f" % f1_running)
    print("Test_loss:\t\t%.4f" % val_loss)



def scaleData(train, test):
    scaler = StandardScaler()
    train = pd.DataFrame(scaler.fit_transform(train))
    test = pd.DataFrame(scaler.transform(test))
    return train, test

'''
Input) 4 DataFrames of train / test data
Return) tensorX: torch.Size([size, 1, features]), torch.Size([size, 1])
'''
# tensor_x1, tensor_y1, tensor_x2, tensor_y2 = __MLP.df_to_unsqueezed_tensor(pheme_AVGw2v,pheme_y,ext_AVGw2v,ext_y)

def convert_df_to_dataloaders(trainX, trainY, testX, testY, TrainX2=None, TestX2=None):
    tensor_x1 = torch.Tensor(trainX.values).unsqueeze(1)
    tensor_y1 = torch.Tensor(trainY.values).unsqueeze(1)

    tensor_x2 = torch.Tensor(testX.values).unsqueeze(1)
    tensor_y2 = torch.Tensor(testY.values).unsqueeze(1)
    # if ((TrainX2 is None) | (TestX2 is None)):
    train_dataset = TensorDataset(tensor_x1,tensor_y1)
    test_dataset = TensorDataset(tensor_x2,tensor_y2)

    batch_size = 16
    train_sampler, test_sampler = __MLP.getSamplers(pheme_y, tensor_x2)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    train_size, test_size = int(tensor_y1.size(0)), int(tensor_y2.size(0))
    # test_size = int(tensor_y2.size(0))

    print("Training: ",tensor_x1.shape,"/", tensor_x2.shape)
    print("Testing2: ",tensor_y1.shape,"/", tensor_y2.shape)
    print("Train Size",train_size,"/ Test Size",test_size)
    return batch_size, train_size, test_size, train_dataloader, test_sampler
    # elif (TrainX2 is not None & TestX2 is not None):
    #     tensor_x1_2= torch.Tensor(TrainX2.values).unsqueeze(1)
    #     tensor_x2_2 = torch.Tensor(TestX2.values).unsqueeze(1)
    #     return batch_size, train_size, test_size, train_dataloader, test_sampler
    # train_dataset = TensorDataset(tensor_x1,tensor_y1)
    # test_dataset = TensorDataset(tensor_x2,tensor_y2)

def convert_df_to_unsqueezed_tensor(trainX, trainY, testX, testY, TrainX2=None, TestX2=None):
    tensor_x1 = torch.Tensor(trainX).unsqueeze(1)
    tensor_y1 = torch.Tensor(trainY).unsqueeze(1)

    tensor_x2 = torch.Tensor(testX).unsqueeze(1)
    tensor_y2 = torch.Tensor(testY).unsqueeze(1)
    if ((TrainX2 is None) | (TestX2 is None)):
        
        return tensor_x1, tensor_y1, tensor_x2, tensor_y2
    elif (TrainX2 is not None & TestX2 is not None):
        tensor_x1_2= torch.Tensor(TrainX2.values).unsqueeze(1)
        tensor_x2_2 = torch.Tensor(TestX2.values).unsqueeze(1)
        return tensor_x1, tensor_x1_2, tensor_y1, tensor_x2, tensor_x2_2, tensor_y2
    # train_dataset = TensorDataset(tensor_x1,tensor_y1)
    # test_dataset = TensorDataset(tensor_x2,tensor_y2)
'''
tensor_x1, tensor_y1, tensor_x2, tensor_y2 = __MLP.convert_df_to_unsqueezed_tensor(pheme_AVGw2v, pheme_y, ext_AVGw2v, ext_y)
train_dataset = TensorDataset(tensor_x1,tensor_y1)
test_dataset = TensorDataset(tensor_x2,tensor_y2)
print(tensor_x1.size(),tensor_x2.size())
print(tensor_y1.size(),tensor_y2.size())
---
batch_size = 16

# Initialize WeightedRandomSampler to deal with the unbalanced dataset
train_sampler, test_sampler = __MLP.getSamplers(pheme_y, tensor_x2)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

data = next(iter(train_dataloader))
print("mean: %s, std: %s" %(data[0].mean(), data[0].std()))

train_size = int(tensor_y1.size(0))
test_size = int(tensor_y2.size(0))

print(tensor_x1.shape,tensor_x2.shape)
print(tensor_y1.shape,tensor_y2.shape)
print("Train Size",train_size,"Test Size",test_size)
---
train_acc, train_loss, val_acc, val_loss_list = __MLP.train_sequential(model=?, num_epochs=40, patience=10, criterion=criterion, optimizer=optimizer, scheduler=scheduler, train_loader=train_dataloader,train_size=train_size, test_loader=test_dataloader, test_size=test_size, PATH=PATH)

__MLP.clf_report(train_loss, train_acc, val_loss_list, val_acc)
'''

def getSamplers(trainY, tensor_x2):
    counts = np.bincount(trainY.values)
    labels_weights = 1. / counts
    weights = labels_weights[trainY.values]
    train_sampler = WeightedRandomSampler(weights, len(weights))
    test_sampler = SequentialSampler(tensor_x2)
    return train_sampler, test_sampler

class sparse_model(nn.Module):
    def __init__(self):
        super(sparse_model, self).__init__() # 1*20
        self.fc1 = nn.Linear(36, 12, bias=True) # 420
        # self.fc2 = nn.Linear(12, 8, bias=True)
        self.fc3 = nn.Linear(12, 1,bias=True)

        self.drop_3 = nn.Dropout(0.3)
        self.drop_4 = nn.Dropout(0.4)
        self.drop_2 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.drop_3(F.elu(self.fc1(x)))
        # x = self.drop_2(F.elu(self.fc2(x)))
        x = self.drop_3(self.fc3(x))
        # x = self.fc3(x)
        # x = self.fc3(x)
        return x

class W2V_net(nn.Module):
    def __init__(self):
        super(W2V_net, self).__init__() # 1*20
        self.fc1 = nn.Linear(200, 32, bias=True) # 420
        self.fc2 = nn.Linear(32, 8, bias=True)
        self.fc3 = nn.Linear(8, 1)

        self.drop_3 = nn.Dropout(0.3)
        self.drop_4 = nn.Dropout(0.4)
        self.drop_2 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.drop_2(F.elu(self.fc1(x)))
        x = self.drop_2(F.elu(self.fc2(x)))
        x = self.fc3(x)
        return x

class BERT_net(nn.Module):
    def __init__(self):
        super(BERT_net, self).__init__() # 1*20
        self.fc1 = nn.Linear(768, 50, bias=True) # 420
        self.fc2 = nn.Linear(50, 8, bias=True)
        self.fc3 = nn.Linear(8, 1)

        self.drop_3 = nn.Dropout(0.3)
        self.drop_4 = nn.Dropout(0.4)
        self.drop_2 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.drop_4(F.elu(self.fc1(x)))
        x = self.drop_2(F.elu(self.fc2(x)))
        x = self.fc3(x)
        return x


class thread_model(nn.Module):
    def __init__(self):
        super(thread_model, self).__init__() # 1*20
        self.fc1 = nn.Linear(38, 12, bias=True) # 420
        # self.fc2 = nn.Linear(12, 8, bias=True)
        self.fc3 = nn.Linear(12, 1)

        self.drop_3 = nn.Dropout(0.3)
        self.drop_4 = nn.Dropout(0.4)
        self.drop_2 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.drop_3(F.elu(self.fc1(x)))
        # x = F.elu(self.fc2(x))
        x = self.drop_2(self.fc3(x))
        return x
    '''

model_paths = ["./Model/state_dict_sparse_model.pt",
               "./Model/state_dict_w2v_model.pt",
               "./Model/state_dict_bert_model.pt",
               "./Model/state_dict_thread_model.pt",
               "./Model/state_dict_w2v_sparse_model.pt", 
               "./Model/state_dict_bert_sparse_model.pt",
               "./Model/state_dict_bert_sparse_multi.pt",
               "./Model/state_dict_sparse_thread_model_multi.pt",
                "./Model/state_dict_bert_sparse_model.pt",
                "./Model/state_dict_bert_thread_model.pt",
                "./Model/state_dict_bert_sparse_thread_model_multi.pt"
               ]


data = next(iter(train_dataloader))
print("mean: %s, std: %s" %(data[0].mean(), data[0].std()))

train_size = int(tensor_y1.size(0))
test_size = int(tensor_y2.size(0))

print(tensor_x1.shape,tensor_x2.shape)
print(tensor_y1.shape,tensor_y2.shape)
print("Train Size",train_size,"Test Size",test_size)


model = __MLP.W2V_net()
PATH = './Model/state_dict_w2v_model.pt'
model.load_state_dict(torch.load(PATH))
__MLP.predict(model, criterion, test_dataloader, test_size)


# import __MLP
pheme_sparse = pd.read_csv('./data/_PHEME_sparse.csv')
pheme_y = pd.read_csv('./data/_PHEME_target.csv').target
ext_sparse = pd.read_csv('./data/_PHEMEext_sparse.csv')
ext_y = pd.read_csv('./data/_PHEMEext_text.csv').target

scaler = StandardScaler()
pheme_scaled = pd.DataFrame(scaler.fit_transform(pheme_sparse))
ext_scaled = pd.DataFrame(scaler.transform(ext_sparse))

tensor_x1, tensor_y1, tensor_x2, tensor_y2 = __MLP.convert_df_to_unsqueezed_tensor(pheme_scaled,pheme_y,ext_scaled,ext_y)

train_dataset = TensorDataset(tensor_x1,tensor_y1)
test_dataset = TensorDataset(tensor_x2,tensor_y2)
print(tensor_x1.size(),tensor_x2.size())
print(tensor_y1.size(),tensor_y2.size())
    '''