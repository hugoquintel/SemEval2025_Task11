import torch
from sklearn import metrics
import matplotlib.pyplot as plt

# @title Get labels function
def get_labels(data):
    labels = data.columns[2:].tolist()
    labels_to_ids = {label:id for id, label in enumerate(labels)}
    ids_to_labels = {id:label for label, id in labels_to_ids.items()}
    return labels, labels_to_ids, ids_to_labels

# @title Train function
def train_step(args, plm, cls, loss_function, optimizer, dataloader):
    plm.train()
    cls.train()
    loss_total = 0
    for batch_index, data in enumerate(dataloader):

        input_ids = data['input_ids'].to(args.DEVICE)
        attention_mask = data['attention_mask'].to(args.DEVICE)
        labels = data['labels'].to(args.DEVICE)

        plm_output = plm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        logit = cls(plm_output)

        loss = loss_function(logit, labels)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        loss_total += loss
        if (batch_index + 1) % 10 == 0:
            print(f'Loss after {batch_index + 1} batches: {loss:.5f}')
        torch.cuda.empty_cache()

    loss_average = loss_total / len(dataloader)
    return loss_total, loss_average

# @title Test function
def test_step(args, plm, cls, dataloader):
    plm.eval()
    cls.eval()
    labels_true, labels_pred = [], []
    with torch.inference_mode():
        for batch_index, data in enumerate(dataloader):
            input_ids = data['input_ids'].to(args.DEVICE)
            attention_mask = data['attention_mask'].to(args.DEVICE)
            labels = data['labels']

            plm_output = plm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
            logit = cls(plm_output)

            torch.cuda.empty_cache()

            batch_labels_pred = (torch.sigmoid(logit) >= args.CONFIDENCE_SCORE).to(torch.int)

            labels_true.extend(labels.to(torch.int).tolist())
            labels_pred.extend(batch_labels_pred.tolist())
    return labels_true, labels_pred

# @title Metrics function
def get_metrics(labels_true, labels_pred, labels_to_ids):
    cls_report = metrics.classification_report(labels_true, labels_pred, target_names=labels_to_ids, zero_division=0.0, digits=5)
    cls_report_dict = metrics.classification_report(labels_true, labels_pred, target_names=labels_to_ids, zero_division=0.0, output_dict=True)
    f1_micro, f1_macro = cls_report_dict['micro avg']['f1-score'], cls_report_dict['macro avg']['f1-score']
    return cls_report, f1_micro, f1_macro

# @title Plot confusion matrix function
def plot_confmat(labels_true, labels_pred, ids_to_labels):
    confmat = metrics.multilabel_confusion_matrix(labels_true, labels_pred, labels=list(ids_to_labels.keys()))
    fig, axes = plt.subplots(1, len(ids_to_labels), figsize=(25, 5))
    for i in range(len(ids_to_labels)):
        disp = metrics.ConfusionMatrixDisplay(confmat[i])
        disp.plot(ax=axes[i], values_format='.4g')
        disp.ax_.set_title(f'{list(ids_to_labels.values())[i]}')
        if i != 0: disp.ax_.set_ylabel('')
        if i != int(len(ids_to_labels) / 2): disp.ax_.set_xlabel('')
        disp.im_.colorbar.remove()
    no_of_labels = torch.tensor(labels_true).sum(dim = 0)
    label_nums = {value:num.item() for value, num in zip(ids_to_labels.values(), no_of_labels)}
    print(f'Number of true labels: {label_nums}')
    plt.show()

# @title Save models function
def save_models(epoch, f1_micro, f1_macro, best_epoch, best_f1_micro, best_f1_macro, plm, cls, optimizer, MODEL_PATH):
    if f1_micro > best_f1_micro and f1_macro > best_f1_macro:
        best_f1_micro, best_f1_macro, best_epoch = f1_micro, f1_macro, epoch
        MODEL_NAME = f"{round(best_f1_micro, 4)}_{round(best_f1_macro, 4)}.pt"
        MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
        print(f'** Best model found in this epoch ({epoch}), saving model to: {MODEL_SAVE_PATH} **')
        state = {"plm": plm.state_dict(),
                "cls": cls.state_dict(),
                "optimizer": optimizer.state_dict()}
        torch.save(state, MODEL_SAVE_PATH)
    return best_epoch, best_f1_micro, best_f1_macro

# @title Make info file function
def make_info_file(args, MODEL_PATH):
    f = open(MODEL_PATH / "info.txt", "w")
    for key, value in vars(args).items():
        f.write(f'{key}: {value}\n')
    f.close()