import numpy as np
import torch

def trainer(net, epochs, train_loader, valid_loader, optimizer, criterion, save_path, use_cuda):
    '''
    Model trainer.
    Args:
        net = Neural network
        epochs = number of training epochs
        train_loader = Data train loader
        valid_loader = Data validation loader
        optimizer
        criterion
        save path
        use_cuda = train on GPU
    '''
    train_loss_log = []
    valid_loss_log = []

    valid_loss_min = np.Inf 

    for epoch in range(epochs):  # loop over the dataset multiple times

        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        net.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
        train_loss_log.append(train_loss)

        net.eval()
        for batch_idx, (data, target) in enumerate(valid_loader):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = net(data)
            loss = criterion(output, target)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
        valid_loss_log.append(valid_loss)

        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
        ))

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min, valid_loss
            ))
            torch.save(net.state_dict(), save_path)
            valid_loss_min = valid_loss

    print('Finished Training')
    return train_loss_log, valid_loss_log

def tester(test_loader, model, criterion, use_cuda):
    '''
    monitor test loss and accuracy
    Args:
        model = Neural network
        test_loader = Data test loader
        criterion
        use_cuda = train on GPU
    '''
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(test_loader):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))