def reset_weights(m):
    '''
      Try resetting model weights to avoid
      weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


def getDataSize(tensor_x1, tensor_y1, tensor_x2, tensor_y2):
    train_size = int(tensor_y1.size(0))
    test_size = int(tensor_y2.size(0))

    print("Variables)\n\tTrain:%s\n\tTest: %s" %
          (tensor_x1.size(), tensor_x2.size()))
    # print("\tTargets:%s \ %s"%(tensor_y1.size()[0],tensor_y2.size()[0]))
    print("Train Size", train_size, "Test Size", test_size)
    print()
    return train_size, test_size

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False

ext_sparse_final = pd.read_csv('./data/_PHEMEext_sparse.csv')


# cv_pd_list[?][0]은 Training cv_pd_list[?][1] Testing
cv_pd_list = []
data = pd.concat([dataset, pheme_event, pheme_y], axis=1)
NUM_EVENT = data.Event.unique().shape[0]
EVENTS = data.Event.unique()
results = {}

for i, d in enumerate(EVENTS):
    df1, df2 = [x for _, x in data.groupby(data['Event'] != d)]
    df1.reset_index(inplace=True, drop=True)
    df2.reset_index(inplace=True, drop=True)
    cv_pd_list.append([df2, df1])

# for train, test in cv_pd_list:
#     print("Train: %s \ Test: %s" % (train.shape, test.shape))
print()

for index, fold in enumerate(cv_pd_list):

    # DATA PREPARATION
    train, test = fold
    print("FOLD %d\n----------------------------------------------------------------------------" % (int(index)+1))
    train_target = train.pop('target')
    train.pop('Event')
    test_target = test.pop('target')
    test.pop('Event')

    if scaling == True:
        scaler = StandardScaler()
        train = pd.DataFrame(scaler.fit_transform(train))
        test = pd.DataFrame(scaler.transform(test))

    tensor_x1, tensor_y1, tensor_x2, tensor_y2 = __MLP.convert_df_to_unsqueezed_tensor(
        train.values, train_target, test.values, test_target.values)
    train_dataset = TensorDataset(tensor_x1, tensor_y1)
    test_dataset = TensorDataset(tensor_x2, tensor_y2)

    batch_size = 16

    train_sampler, test_sampler = __MLP.getSamplers(train_target, tensor_x2)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  sampler=train_sampler, pin_memory=True, num_workers=0, worker_init_fn=_init_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, pin_memory=True, num_workers=0, worker_init_fn=_init_fn)

    data = next(iter(train_dataloader))
    print("mean: %s, std: %s" % (data[0].mean(), data[0].std()))

    train_size, test_size = getDataSize(
        tensor_x1, tensor_y1, tensor_x2, tensor_y2)

    model = modelClass()
    model.apply(reset_weights)

    modelname = model.__class__.__name__

    # model_sparse = sparse_model()
    # criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.SGD(model_sparse.parameters(), lr=0.01, momentum=0.9)
    # optimizer = optim.Adam(model_sparse.parameters(), lr=5e-5, eps=1e-8, weight_decay=1e-7)
    optimizer = AdamW(model.parameters(),
                      # lr=5e-5,    # Default learning rate
                      lr=5e-5,    # Default learning rate
                      eps=1e-8,    # Default epsilon value
                      weight_decay=1e-6
                      )

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value
                                                num_training_steps=total_steps)

    print(f'Model: {modelname}')
    print(f'Epochs: {epochs}')
    PATH = "./Model/"+modelname+"_"+str(index+1)+".pt"
    print(f'PATH: {PATH}\n')

    training_acc = []
    training_loss = []

    # train_acc, train_loss, val_acc, val_loss_list = __MLP.train_sequential(model=model, num_epochs=epochs, patience=patience, criterion=criterion, optimizer=optimizer, scheduler=scheduler, train_loader=train_dataloader, train_size=train_size, test_loader=test_dataloader, test_size=test_size, PATH=PATH)

    # Run the training loop for defined number of epochs
    for epoch in range(0, epochs):

        # Print epoch
        if (verbose != False):
            # pass
            print(f'Starting epoch {epoch+1}')
        elif (verbose != True):
            if epoch % 25 == 24:
                print(f'Starting epoch {epoch+1}')
        # Set current loss value
        current_loss = 0.0
        running_corrects = 0.0
        running_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(train_dataloader, 0):

            # Get inputs
            inputs, targets = data

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = model(inputs)

            outputs = outputs.view(outputs.size(0), -1)

            # Compute Prediction Outputs
            # preds = outputs.squeeze(1) > 0.0
            preds = outputs > 0.0

            # Compute loss
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == targets.data).data

            # Perform backward pass
            loss.backward()

            # Perform optimization and Scheduler
            optimizer.step()
            scheduler.step()

            # Print statistics
            # current_loss += loss.item() # 원본
            # if i % len(train_dataloader) == len(train_dataloader)-1:
            #     print('Loss after mini-batch %5d: %.3f' %
            #           (i + 1, current_loss / i+1))

            current_loss += loss.item() * inputs.size(0)
            if verbose == True:
                if i % len(train_dataloader) == len(train_dataloader)-1:
                    print("Loss/ACC after mini-batch %5d: %.3f / %.4f" %
                          (i + 1, current_loss / train_size, running_corrects/train_size))

        # epoch_acc = running_corrects.double() / train_size
        epoch_acc = running_corrects / train_size
        epoch_loss = running_loss / train_size
        training_acc.append(epoch_acc)
        training_loss.append(epoch_loss)
        # print('Epoch {}/{}\tTrain) Acc: {:.4f}, Loss: {:.4f}'.format(epoch+1,
        #  epochs, epoch_acc, epoch_loss))
    # Process is complete.
    print('Training process has finished. Saving trained model.')

    # Print about testing
    print('<Starting TESTING>')

    # Saving the model
    # save_path = f'./model-fold-{fold}.pth'
    torch.save(model.state_dict(), PATH)

    # Evaluation for this fold
    correct, total = 0, 0
    val_corrects = 0
    f1_batch_epoch = 0
    val_label_list = []
    val_loss = 0
    with torch.no_grad():

        # Iterate over the test data and generate predictions
        for i, data in enumerate(test_dataloader, 0):

            # Get inputs
            inputs, targets = data

            # Generate outputs
            outputs = model(inputs)

            # Set total and correct
            outputs = outputs.view(outputs.size(0), -1).float()
            predicted = (outputs > 0.0).float()
            correct += (predicted == targets).float().sum().item()

            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
            #!
            preds = (outputs > 0.0).float()
            # running_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == targets.data).data
            # f1_batch = f1_score(targets.cpu(), outputs.sigmoid().cpu() > 0.5, average='macro')
            f1_batch = f1_score(targets.cpu(), preds, average='macro')
            f1_batch_epoch += f1_batch * inputs.size(0)
            # f1_running += (f1_score(targets, preds, average='macro') * inputs.size(0))
            total += targets.size(0)

        if verbose == True:
            # print(f'target: {targets}')
            # print(f'output: {outputs}')
            # print(f'preds: {preds}')
            # print(f'outputs.sigmoid().cpu(): {outputs.sigmoid().cpu()}')
            # Print accuracy
            print('Accuracy for fold %d: %f %%' %
                  (index, 100.0 * correct / total))
            # print('Accuracy-2 for fold %d: %f %%' % (index, 100.0 * val_corrects / total))
            # print('F1 Score-2 for fold %d: %f ->  %%' %(index, f1_score(targets, preds, zero_division=False)))
            # print('F1 Score-3 for fold %d: %f %%' %(index, f1_score(targets, predicted, zero_division=False)))
            print('F1 Score for fold %d: %f %%' %
                  (index, f1_batch_epoch / total))
            print('Loss for fold %d: %f %%' % (index, val_loss / total))
            # print('F1 Score-5 for fold %d: %f %%' %(index, f1_batch_epoch / test_size))
            # print('F1 Score-6 for fold %d: %f %%' %(index, f1_running / test_size))
        print(
            '----------------------------------------------------------------------------')
        results[index] = [100.0 * (correct / total),
                          100.0 * f1_batch_epoch / total]
        # results[index][1] = 1


# ---------------------------- Print fold results ---------------------------- #

print(f'K-FOLD CROSS VALIDATION RESULTS FOR {NUM_EVENT} FOLDS')
print('----------------------------------------------------------------------------"')
acc_sum = 0.0
f1_sum = 0.0
for key, value in results.items():
    print(f'Fold {key}: Acc {value[0]}, F1 {value[1]} %')
    acc_sum += value[0]
    f1_sum += value[1]
print(f'Average: {acc_sum/len(results.items())} %')
print(f'F1: {f1_sum/len(results.items())} %')

return results
