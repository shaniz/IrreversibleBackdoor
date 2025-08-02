import torch
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init

from eval_utils import accuracy


def initialize(arch, model):
    if arch == 'res50':
        last_layer = model.module.module.fc
    elif arch == 'caformer':
        last_layer = model.module.module.head.fc.fc2
    elif arch == 'res18':
        last_layer = model.module.module.fc
    elif arch == 'res34':
        last_layer = model.module.module.fc
    elif arch == 'vgg':
        last_layer = model.module.module.fc
    init.xavier_uniform_(last_layer.weight)
    if last_layer.bias is not None:
        init.zeros_(last_layer.bias)
    return model

def fast_adapt_multibatch_inverse(batches, learner, criterion, shots, ways, device, arch):
    # Adapt the model
    learner = initialize(arch, learner)
    test_loss = 0
    test_accuracy = 0
    total_test = 0
    for index,batch in enumerate(batches):
        data, labels = batch
        data, labels = data.to(device), labels.to(device)
        adaptation_indices = np.zeros(data.size(0), dtype=bool)
        # adaptation_indices[np.arange(shots*ways)] = True
        adaptation_indices[np.random.choice(np.arange(data.size(0)), shots*ways, replace=False)] = True
        evaluation_indices = torch.from_numpy(~adaptation_indices)
        adaptation_indices = torch.from_numpy(adaptation_indices)

        adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
        evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]
        current_test = evaluation_data.shape[0]

        total_test += current_test
        adaptation_error = criterion(learner(adaptation_data), adaptation_labels)
        if index == 0:
            current_grads = learner.adapt(adaptation_error,None)
        else:
            last_grads = current_grads
            current_grads = learner.adapt(adaptation_error,last_grads)

        predictions = learner(evaluation_data)
        evaluation_error = criterion(1-predictions, evaluation_labels)
        evaluation_accuracy = accuracy(predictions, evaluation_labels)
        test_loss += evaluation_error*current_test
        test_accuracy += evaluation_accuracy*current_test
    return test_loss*1.0/total_test, test_accuracy*1.0/total_test


def fast_adapt_multibatch_kl_uniform(batches, learner, loss, shots, ways, device, arch):
    # Adapt the model
    learner = initialize(arch, learner)  # Added to avoid NaN outputs

    test_loss = 0
    test_accuracy = 0
    total_test = 0
    for index,batch in enumerate(batches):
        data, labels = batch
        data, labels = data.to(device), labels.to(device)
        adaptation_indices = np.zeros(data.size(0), dtype=bool)
        # adaptation_indices[np.arange(shots*ways)] = True
        adaptation_indices[np.random.choice(np.arange(data.size(0)), shots*ways, replace=False)] = True
        evaluation_indices = torch.from_numpy(~adaptation_indices)
        adaptation_indices = torch.from_numpy(adaptation_indices)
        adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
        evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]
        current_test = evaluation_data.shape[0]
        # print(current_test)
        total_test += current_test
        adaptation_error = loss(learner(adaptation_data), adaptation_labels)
        if index == 0:
            current_grads = learner.adapt(adaptation_error,None)
        else:
            last_grads = current_grads
            current_grads = learner.adapt(adaptation_error,last_grads)
    # Evaluate the adapted model
        predictions = learner(evaluation_data)
        normalized_preds = torch.nn.functional.softmax(predictions, dim=1).cuda()
        target_preds = 0.1 * torch.ones((predictions.shape[0], predictions.shape[1])).cuda()
        evaluation_error = F.kl_div(torch.log(normalized_preds), target_preds, reduction='batchmean')
        evaluation_accuracy = accuracy(predictions, evaluation_labels)
        test_loss += evaluation_error*current_test
        test_accuracy += evaluation_accuracy*current_test

    return test_loss*1.0/total_test, test_accuracy*1.0/total_test

