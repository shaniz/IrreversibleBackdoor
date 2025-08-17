import csv


def bd_save_data(save_path,
              all_restrict_train_loss, all_restrict_train_acc,
              all_orig_test_loss, all_orig_train_loss, all_orig_test_acc, all_targeted_asr,
              all_finetune_restrict_test_acc, all_finetune_restrict_test_loss,
              final_orig_test_acc, final_finetune_restrict_test_acc,
              final_finetune_restrict_test_loss, total_loop_index, fts_index, ntr_index):

    with open(save_path + '/' + 'orig_train.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['total loop', 'NTR loop', 'original poisoned train loss'])
        for i, j, k in zip(total_loop_index, ntr_index, all_orig_train_loss):
            writer.writerow([i, j, k])

    with open(save_path + '/' + 'orig_test.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['total loop', 'NTR loop', 'original clean test loss', 'original clean test accuracy', 'original target ASR'])
        for i, j, k, q, m in zip(total_loop_index, ntr_index, all_orig_test_loss, all_orig_test_acc, all_targeted_asr):
            writer.writerow([i, j, k, q, m])

    with open(save_path + '/' + 'restrict_train.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['total loop', 'FTS loop', 'restrict set train loss', 'restrict set train accuracy'])
        for i, j, k, q in zip(total_loop_index, fts_index, all_restrict_train_loss, all_restrict_train_acc):
            writer.writerow([i, j, k, q])

    with open(save_path + '/' + 'finetune_restrict_test.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['total loop', 'finetune restrict test loss', 'finetune restrict test acc'])
        for i, j, k in zip(total_loop_index, all_finetune_restrict_test_loss, all_finetune_restrict_test_acc):
            writer.writerow([i, j, k])

    with open(save_path + '/' + 'final_test.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['final original test acc', 'final finetune restrict test loss', 'final finetune restrict test acc'])
        writer.writerow([final_orig_test_acc, final_finetune_restrict_test_loss, final_finetune_restrict_test_acc])
