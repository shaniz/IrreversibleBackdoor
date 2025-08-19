import csv


def bd_save_data(save_path,
              all_restrict_train_loss, all_restrict_train_acc,
              all_orig_train_loss, all_orig_test_acc, all_orig_targeted_asr,
              all_finetune_restrict_clean_acc, all_finetune_restrict_test_loss, all_finetune_restrict_targeted_asr,
              total_loop_index, fts_index, ntr_index):

    with open(save_path + '/' + 'fts_restrict.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Total Loop', 'FTS Loop', 'Restrict Poisoned Train Loss', 'Restrict Poisoned Train Acc'])
        for i, j, k, q in zip(total_loop_index, fts_index, all_restrict_train_loss, all_restrict_train_acc):
            writer.writerow([i, j, k, q])
            
            
    with open(save_path + '/' + 'ntr_orig.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Total Loop', 'NTR loop', 'Orig Poisoned Train Loss', 'Orig Clean Test Acc', 'Orig ASR'])
        for i, j, k, q, m in zip(total_loop_index, ntr_index, all_orig_train_loss, all_orig_test_acc, all_orig_targeted_asr):
            writer.writerow([i, j, k, q, m])
            

    with open(save_path + '/' + 'finetune_restrict.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Total Loop', 'Finetune Restrict Clean Test Acc', 'Finetune Restrict Poisoned Test Loss', 'Finetune Restrict Poisoned Test ASR'])
        total_loop_index.append('final')  # final finetune is out of loop
        for i, j, k, q in zip(total_loop_index, all_finetune_restrict_clean_acc, all_finetune_restrict_test_loss, all_finetune_restrict_targeted_asr):
            writer.writerow([i, j, k, q])
