from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import sys
import os
import cnet

def testClassess(model, test_loader, device, number_of_labels, classes, batch_size):
    
    model.to(device)
    model.eval()

    class_correct = list(0. for i in range(number_of_labels))
    class_total = list(0. for i in range(number_of_labels))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    result = dict()
    for i in range(number_of_labels):
        result[classes[i]] = 100 * class_correct[i] / class_total[i]
        #print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
    return result

def testAccuracy(model, test_loader, device):
    
    model.to(device)
    model.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return(accuracy)

# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(num_epochs, model, test_loader, train_loader, loss_fn, optimizer, device):
    
    best_accuracy = 0.0

    # Define your execution device
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader, 0):
            
            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = model(images)
            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, labels)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            # Let's print statistics for every 1,000 images
            running_loss += loss.item()     # extract the loss value
            if i % 1000 == 999:    
                # print every 1000 (twice per epoch) 
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
                # zero the loss
                running_loss = 0.0

        # Compute and print the average accuracy for this epoch when tested over all 10000 test images
        accuracy = testAccuracy(model,test_loader, device)
        print('For epoch', epoch+1,'the test accuracy over the whole test set is %.3f %%' % (accuracy))
        model.train()
        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            cnet.saveModel(model, "_tmp") 
            best_accuracy = accuracy

    return best_accuracy

def main(args):
    print(args)
    x1 = int(args[0])
    x2 = int(args[1])
    x3 = int(args[2])
    x4 = int(args[3])
    learning_rate = float(args[4])

    torch.manual_seed(43) # Set a random seed for reproducibility

    # Loading and normalizing the data.
    # Define transformations for the training and test sets
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # CIFAR10 dataset consists of 50K training images. We define the batch size of 10 to load 5,000 batches of images.
    batch_size = 10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    number_of_labels = 10 

    # Create an instance for training. 
    # When we run this code for the first time, the CIFAR10 train dataset will be downloaded locally. 
    train_set =CIFAR10(root="./data",train=True,transform=transformations,download=True)

    # Create a loader for the training set which will read the data within batch size and put into memory.
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    print("The number of images in a training set is: ", len(train_loader)*batch_size)

    # Create an instance for testing, note that train is set to False.
    # When we run this code for the first time, the CIFAR10 test dataset will be downloaded locally. 
    test_set = CIFAR10(root="./data", train=False, transform=transformations, download=True)

    # Create a loader for the test set which will read the data within batch size and put into memory. 
    # Note that each shuffle is set to false for the test loader.
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    print("The number of images in a test set is: ", len(test_loader)*batch_size)
    print("The number of batches per epoch is: ", len(train_loader))

    # Instantiate a neural network model 
    model = cnet.Network(x1,x2,x3,x4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)

    num_epochs = 25

    # Let's build our model
    train(num_epochs, model, test_loader, train_loader, loss_fn, optimizer, device)
    print('Finished Training')

    model.load_state_dict(torch.load("model/cifar_net_tmp.pth"))

    # questi dati non sono mai stati visti dal modello
    accuracy = testAccuracy(model, test_loader, device)
    classes_accuracy = testClassess(model, test_loader, device, number_of_labels, classes, batch_size)
    print('the accuracy on the classes is')
    print(classes_accuracy)

    classes_accuracy['total_accuracy'] = accuracy

    print('the test accuracy over the whole validation set is %.3f %%' % (accuracy))
    cnet.saveModel(model, accuracy)

    os.remove("model/cifar_net_tmp.pth")
    
    return classes_accuracy


if __name__ == "__main__":
    args = [12,12,24,24,0.001]
    main(args)
    sys.exit(0)