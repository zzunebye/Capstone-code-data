class model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=27, out_features=8)
        self.fc2 = nn.Linear(in_features=8, out_features=6)
        self.output = nn.Linear(in_features=6, out_features=1)
        # self.softmax = nn.Linear(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x

def train(train_loader, val_dataloader=None, epochs=100, verbose=True, evaluation=False):
    prev_loss = 10
    PATH = "./state_dict_model_ai_task1.pt"

    print("Start training...\n")
    val_loss_list = []
    val_acc_list = []
    train_loss = []         # training 과정에서 각 epoch마다의 평균 loss를 저장
    train_accuracy = []     # training 과정에서 각 epoch마다의 평균 acc를 저장
    train_correct = []

    num_step = len(train_loader)

    for epoch in range(epochs):
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # total_loss, batch_loss, batch_counts = 0, 0, 0
        model.train()
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)
        epoch_corrects = [] # 한 epoch마다 각 batch의 데이터를 저장
        epoch_loss = []
        epoch_accuracy = []

        for i, (x, y) in enumerate(train_loader):
            x, label = x.float(), y.float()

            output = model.forward(x)

            _, preds = torch.max(output, 1)
            # print(preds)
            # print(label)
            acc = (preds == label).cpu().numpy().mean() * 100

            loss = criterion(output.float(), label.unsqueeze(1).float())
            batch_loss += loss.item()
            total_loss += loss.item()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            epoch_accuracy.append(acc)
            # print(torch.sum(preds == label))
            epoch_corrects.append(torch.sum(preds == label))

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        avg_train_loss = total_loss / len(train_dataloader)

        train_loss.append(epoch_corrects)
        train_accuracy.append(np.mean(epoch_accuracy))
        train_correct.append(np.sum(epoch_accuracy))
        if epoch % 10 == 0 and verbose == True:
            print("Epoch: {}, Loss: {:.5f}".format(epoch + 1, loss.item()))
            print('Train) Loss: {:.4f} Acc: {:.4f}'.format(train_loss[-1], train_accuracy[-1]))
        
        if epoch % 5 == 0 and loss < prev_loss:
            # print("prev_loss: {:.5f}".format(prev_loss))
            # print("loss: {:.5f}".format(loss))
            print("Saving the best model")
            torch.save(model.state_dict(),PATH)
            prev_loss = loss.item()
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            model.eval()

            val_correct, val_acc, val_loss = evaluate(val_dataloader)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)

            # Print performance over the entire training data
            # print(val_loss_list)
            print(f"{epoch + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f}")
            print("-"*70)
    # torch.save(model.state_dict(),PATH)
    # return train_loss, train_accuracy, loss


def evaluate(val_dataloader, verbose=True):
    correct = 0
    total = 0
    outputs_list = []

    val_loss = []
    val_corrects = []
    val_acc = []

    model.eval()
    with torch.no_grad():
        for j, val in enumerate(val_dataloader):

            inputs, label = val
            inputs, label = inputs.float(), label.float()
            output = model(inputs)
            _, preds = torch.max(output, 1)
            acc = (preds == label).cpu().numpy().mean() * 100

            loss = criterion(output, label.unsqueeze(1))
            val_loss.append(loss.item())
            val_corrects.append(torch.sum(preds == label).double())
            val_acc.append(acc)

        
    total_correct = np.sum(val_corrects)
    total_loss = np.mean(val_loss)
    total_acc = np.mean(val_acc)
    print("Validation) Acc: {:.4f} ".format(total_acc))
    
    return total_correct, total_acc, total_loss

criterion = nn.BCEWithLogitsLoss()
# criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train(train_loader, val_dataloader=test_loader, epochs=100, evaluation=True)

fig, ax = plt.subplots(2, 1, figsize=(12,8))
ax[0].plot(train_loss)
ax[0].set_ylabel('Loss')
ax[0].set_title('Training Loss')

ax[1].plot(train_accuracy)
ax[1].set_ylabel('Classification Accuracy')
ax[1].set_title('Training Accuracy')

plt.tight_layout()
plt.show()

print("Accuracy: {}, Loss: {:.5f}".format(train_accuracy[-1], loss.item()))