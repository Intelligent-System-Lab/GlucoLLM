import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import shutil
import csv

from tqdm import tqdm
from utils.metrics import RMSE

plt.switch_backend('agg')


def adjust_learning_rate(accelerator, optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            if accelerator is not None:
                accelerator.print('Updating learning rate to {}'.format(lr))
            else:
                print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, accelerator=None, patience=7, verbose=False, delta=0, save_mode=True):
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_mode = save_mode

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.accelerator is None:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                self.accelerator.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            if self.accelerator is not None:
                self.accelerator.print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            else:
                print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(model)
            torch.save(model.state_dict(), path + '/' + 'checkpoint')
        else:
            torch.save(model.state_dict(), path + '/' + 'checkpoint')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def del_files(dir_path):
    shutil.rmtree(dir_path)


def vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric, printout=True):
    total_loss = []
    total_mae_loss = []
    total_rmse_loss = []
    predictions_dict = {}
    model.eval()
    with torch.no_grad():
        print('\n')
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader)):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                accelerator.device)
            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            # Get the output from the model before gathering for metrics
            if args.output_attention:  # Assuming you are returning tuple in this case
                outputs, attention = outputs

            outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)

            pred = outputs.detach()
            true = batch_y.detach()

            loss = criterion(pred, true)
            mae_loss = mae_metric(pred, true)

            # convert back to original input data range
            scaler = vali_data.scaler
            pred_shape = pred.shape
            true_shape = true.shape
            pred_back = pred.cpu().numpy().reshape(-1, pred_shape[-1])  # reshape to 2D
            true_back = true.cpu().numpy().reshape(-1, true_shape[-1])  # reshape to 2D
            pred_back = scaler.inverse_transform(pred_back)
            true_back = scaler.inverse_transform(true_back)
            pred_back = torch.from_numpy(pred_back.reshape(pred_shape)).to(accelerator.device)  # reshape back to 3D
            true_back = torch.from_numpy(true_back.reshape(true_shape)).to(accelerator.device)  # reshape back to 3D

            rmse_loss = torch.sqrt(torch.mean((pred_back - true_back) ** 2)) # lqr added

            # For each sample in the batch, process accordingly
            for b in range(pred_back.shape[0]):
                sample_id = i*args.batch_size + b
                pred = [round(x.item(), 1) for x in pred_back[b, :, 0]]  # Get all predictions for all time points
                true = round(true_back[b, 0, 0].item(), 1)  # Only take the first time point
                
                # Update prediction dictionary
                if sample_id not in predictions_dict:
                    predictions_dict[sample_id] = {
                        'true': true,
                        'preds': []
                    }
                predictions_dict[sample_id]['preds'].extend(pred)  # Store the entire prediction sequence
                
            if printout:
                # Write results to CSV file
                csv_file = f'prediction_results_{args.model_id}.csv'
                csv_path = os.path.join('./results', csv_file)
                os.makedirs('./results', exist_ok=True)
                if i == len(vali_loader) - 1:  # Last batch
                    with open(csv_path, mode='w', newline='') as f:
                        writer = csv.writer(f)
                        # Write header
                        writer.writerow(['Original_Index', 'Sample_ID', 'True_Value', 'Avg_Prediction', 'Avg_Error', 'Last_Prediction', 'Last_Error'])
                        print('\n')
                        for sample_id, data in predictions_dict.items():
                            true_val = data['true']
                            preds = data['preds']
                            original_pred = sample_id + args.seq_len + 2  # Prediction point index
                            avg_pred = round(sum(preds) / len(preds), 1)
                            last_pred = round(preds[-1], 1)
                            avg_error = round(abs(avg_pred - true_val), 1)
                            last_error = round(abs(last_pred - true_val), 1)
                            writer.writerow([
                                original_pred,  # Original_Index in the first column
                                sample_id,      # Sample_ID in the second column
                                true_val,
                                avg_pred,
                                avg_error,
                                last_pred,
                                last_error
                            ])
                            # Print detailed info to console
                            status_msg = (
                                f'\rSample {sample_id:4d} | '
                                f'True: {true_val:6.1f} | '
                                f'Avg_Pred: {avg_pred:6.1f} (Err: {avg_error:5.1f}) | '
                                f'Last_Pred: {last_pred:6.1f} (Err: {last_error:5.1f})'
                            )
                            if accelerator is not None:
                                accelerator.print(status_msg, end='')
                            else:
                                print(status_msg, end='')

            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())
            total_rmse_loss.append(rmse_loss.item())

    total_loss = np.average(total_loss)
    total_mae_loss = np.average(total_mae_loss)
    total_rmse_loss = np.average(total_rmse_loss)

    model.train()
    return total_loss, total_mae_loss, total_rmse_loss


def test(args, accelerator, model, train_loader, vali_loader, criterion):
    x, _ = train_loader.dataset.last_insample_window()
    y = vali_loader.dataset.timeseries
    x = torch.tensor(x, dtype=torch.float32).to(accelerator.device)
    x = x.unsqueeze(-1)

    model.eval()
    with torch.no_grad():
        B, _, C = x.shape
        dec_inp = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
        dec_inp = torch.cat([x[:, -args.label_len:, :], dec_inp], dim=1)
        outputs = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
        id_list = np.arange(0, B, args.eval_batch_size)
        id_list = np.append(id_list, B)
        for i in range(len(id_list) - 1):
            outputs[id_list[i]:id_list[i + 1], :, :] = model(
                x[id_list[i]:id_list[i + 1]],
                None,
                dec_inp[id_list[i]:id_list[i + 1]],
                None
            )
        accelerator.wait_for_everyone()
        outputs = accelerator.gather_for_metrics(outputs)
        f_dim = -1 if args.features == 'MS' else 0
        outputs = outputs[:, -args.pred_len:, f_dim:]
        pred = outputs
        true = torch.from_numpy(np.array(y)).to(accelerator.device)
        batch_y_mark = torch.ones(true.shape).to(accelerator.device)
        true = accelerator.gather_for_metrics(true)
        batch_y_mark = accelerator.gather_for_metrics(batch_y_mark)

        loss = criterion(x[:, :, 0], args.frequency_map, pred[:, :, 0], true, batch_y_mark)

    model.train()
    return loss


def load_content(args):
    file = 'data_desc'
    with open('./dataset/{0}.txt'.format(file), 'r') as f:
        content = f.read()
    return content