import pandas as pd
import time
import torch
import torch.nn.functional as F

N_MINI_BATCH_CHECK = 200


class Trainer:
    def __init__(self, model, n_epochs, optimizer, scheduler, criterion):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.n_epochs = n_epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion.to(self.device)

    def measure_performance(self, loader):
        running_loss = 0.0
        correct_count = 0.0
        total_count = 0.0
        for data in loader:
            input_ids = data[0].to(self.device)
            attn_masks = data[1].to(self.device)
            labels = data[2].to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attn_masks)[0]
                loss = self.criterion(outputs, labels)
                probas = F.softmax(outputs, dim=1)
                preds = torch.argmax(probas, axis=1)

                # Track stats
                running_loss += loss
                correct_count += torch.sum(preds == labels)
                total_count += len(labels)

        running_loss /= len(loader)
        acc = correct_count / total_count

        return running_loss, acc

    def train_loop(self, train_loader, val_loader, batch_size=128, batch_check=10):
        print('Starting training loop')

        print('Initial evaluating on validation dataset')
        train_loss, train_acc = self.measure_performance(train_loader)
        val_loss, val_acc = self.measure_performance(val_loader)
        epoch_summary = f'[Epoch 0] | Train acc: {train_acc:.4f} Train loss: {train_loss:.4f} Val acc: {val_acc:.4f} Val loss: {val_loss:.4f}'
        print(epoch_summary)

        epoch_history = [{'epoch': 0,
                          'train loss': train_loss,
                          'train accuracy': train_acc,
                          'validation loss': val_loss,
                          'validation accuracy': val_acc,
                          'epoch time': 0}]
        batch_history = [{'epoch': 0,
                          'batch': 0,
                          'train loss': train_loss,
                          'train accuracy': train_acc,
                          'validation loss': val_loss,
                          'validation accuracy': val_acc,
                          'batch time': 0}]

        for epoch in range(self.n_epochs):
            print(f'--- Epoch: {epoch+1} ---')
            epoch_start_time = time.time()
            batch_start_time = time.time()
            running_loss = 0.0
            running_acc = 0.0

            for i, data in enumerate(train_loader):
                input_ids = data[0].to(self.device)
                attn_masks = data[1].to(self.device)
                labels = data[2].to(self.device)

                self.optimizer.zero_grad()

                # Evaluation/optimization step
                outputs = self.model(input_ids=input_ids, attention_mask=attn_masks)[0]
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

                running_loss += loss.item()
                probas = F.softmax(outputs, dim=1)
                preds = torch.argmax(probas, axis=1)
                running_acc += torch.sum(preds == labels)

                # Print/log statistics periodically
                if i % batch_check == batch_check - 1:
                    batch_end_time = time.time()
                    total_batch_time = batch_end_time - batch_start_time
                    batch_loss = running_loss / batch_check
                    batch_acc = running_acc / batch_check * batch_size
                    batch_val_loss, batch_val_acc = self.measure_performance(val_loader)

                    batch_history.append({'epoch': epoch+1,
                                          'batch': i + 1,
                                          'train loss': batch_loss,
                                          'train accuracy': batch_acc,
                                          'validation loss': batch_val_loss,
                                          'validation accuracy': batch_val_acc,
                                          'batch time': total_batch_time})

                    print(
                        f'[E{epoch + 1:d} B{i + 1:d}] ',
                        f'Loss: {batch_loss:.5f} ',
                        f'Acc: {batch_acc} ',
                        f'Time: {total_batch_time:.2f} ',
                        f'LR: {self.scheduler.get_last_lr()}' if self.scheduler else '')

                    # Reset statistics
                    batch_start_time = time.time()
                    running_loss = 0.0
                    running_acc = 0.00

            epoch_end_time = time.time()
            total_epoch_time = epoch_end_time - epoch_start_time
            train_loss, train_acc = self.measure_performance(train_loader)
            val_loss, val_acc = self.measure_performance(val_loader)
            epoch_summary = f'[Epoch {epoch + 1}] {total_epoch_time} seconds'
            epoch_summary += f' | Train acc: {train_acc:.4f} Train loss: {train_loss:.4f} Val acc: {val_acc:.4f} Val loss: {val_loss:.4f}'

            epoch_history.append({'epoch': epoch + 1,
                                  'training loss': train_loss,
                                  'training accuracy': train_acc,
                                  'validation loss': val_loss,
                                  'validation accuracy': val_acc,
                                  'epoch time': total_epoch_time})

            print(epoch_summary)

        print('Finished training')

        return pd.DataFrame(epoch_history), pd.DataFrame(batch_history)
