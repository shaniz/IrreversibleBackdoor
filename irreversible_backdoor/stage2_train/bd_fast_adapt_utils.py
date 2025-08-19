import torch
import numpy as np

from sophon_orig.stage2_train.eval_utils import accuracy
from sophon_orig.stage2_train.fast_adapt_utils import initialize


def fast_adapt_punish_if_backdoor_fails(adaptation_steps, circular_dual_dl, learner, criterion, shots, ways, device, arch):
    # Adapt the model using the clean data
    # Calculate loss using poisoned data
    learner = initialize(arch, learner)
    test_loss = 0
    test_accuracy = 0
    total_test = 0
    for index,(batch, poisoned_batch) in enumerate(circular_dual_dl.stream_batches(adaptation_steps)):

        data, labels = batch
        data, labels = data.to(device), labels.to(device)

        poisoned_data, poisoned_labels = poisoned_batch
        poisoned_data, poisoned_labels = poisoned_data.to(device), poisoned_labels.to(device)

        adaptation_indices = np.zeros(data.size(0), dtype=bool)
        adaptation_indices[np.random.choice(np.arange(data.size(0)), shots*ways, replace=False)] = True
        adaptation_indices = torch.from_numpy(adaptation_indices)
        evaluation_indices = ~adaptation_indices

        adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
        evaluation_data, evaluation_labels = poisoned_data[evaluation_indices], poisoned_labels[evaluation_indices]
        current_test = evaluation_data.shape[0]

        total_test += current_test
        adaptation_error = criterion(learner(adaptation_data), adaptation_labels)
        if index == 0:
            current_grads = learner.adapt(adaptation_error,None)
        else:
            last_grads = current_grads
            current_grads = learner.adapt(adaptation_error,last_grads)

        predictions = learner(evaluation_data)
        evaluation_error = criterion(predictions, evaluation_labels)

        evaluation_accuracy = accuracy(predictions, evaluation_labels)
        evaluation_error.backward()

        test_loss += evaluation_error.item()*current_test
        test_accuracy += evaluation_accuracy.item()*current_test

        # Cleanup
        del data, labels, poisoned_data, poisoned_labels, predictions, evaluation_error
        torch.cuda.empty_cache()

    return round(test_loss*1.0/total_test, 3), round(test_accuracy*100.0/total_test, 3)


def fast_adapt_punish_if_backdoor_fails2(clean_batches, poisoned_batches, learner, criterion, shots, ways, device, arch):
    # First try for the backdoor loss, gets OOM (Out Of Memory error)
    learner = initialize(arch, learner)
    total_attack_loss = 0.0
    attack_acc = 0
    total_poisoned_samples = 0

    # Simulate clean finetune, then test if attack still works
    for index, (clean_batch, poisoned_batch) in enumerate(zip(clean_batches, poisoned_batches)):
        # --- Clean batch for adaptation ---
        clean_data, clean_labels = clean_batch
        clean_data, clean_labels = clean_data.to(device), clean_labels.to(device)

        # --- Backdoor batch for evaluation ---
        poisoned_data, poisoned_labels = poisoned_batch
        poisoned_data, poisoned_labels = poisoned_data.to(device), poisoned_labels.to(device)

        # --- Select adaptation/evaluation split from clean batch ---
        adaptation_indices = np.zeros(clean_data.size(0), dtype=bool)
        adaptation_indices[np.random.choice(np.arange(clean_data.size(0)), shots * ways, replace=False)] = True
        adaptation_indices = torch.from_numpy(adaptation_indices)
        evaluation_indices = ~adaptation_indices

        adaptation_data, adaptation_labels = clean_data[adaptation_indices], clean_labels[adaptation_indices]
        poisoned_evaluation_data, poisoned_evaluation_labels = poisoned_data[evaluation_indices], poisoned_labels[evaluation_indices]
        current_test = poisoned_data.shape[0]

        total_poisoned_samples += current_test
        # --- Simulate clean finetuning (adapt on clean data only) ---
        adaptation_loss = criterion(learner(adaptation_data), adaptation_labels)

        if index == 0:
            grads = learner.adapt(adaptation_loss, None)
        else:
            grads = learner.adapt(adaptation_loss, grads)

        # --- Evaluate irreversible_backdoor effectiveness after adaptation ---
        poisoned_outputs = learner(poisoned_evaluation_data)
        poisoned_loss = criterion(poisoned_outputs, poisoned_evaluation_labels)
        poisoned_accuracy = accuracy(poisoned_outputs, poisoned_evaluation_labels)
        total_attack_loss += poisoned_loss * current_test
        attack_acc += poisoned_accuracy * current_test

    return total_attack_loss*1.0/total_poisoned_samples, attack_acc*100.0/total_poisoned_samples
