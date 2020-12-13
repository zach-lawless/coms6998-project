import pandas as pd
import time
import torch
import torch.nn.functional as F
from trainer.utils.learning_scheme import get_learning_scheme

N_MINI_BATCH_CHECK = 200


class Trainer:
    def __init__(self, model, n_epochs, optimizer, scheduler, criterion, learning_scheme, learning_rate, adapter):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.n_epochs = n_epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion.to(self.device)
        self.learning_scheme = learning_scheme
        self.learning_rate = learning_rate
        self.adapter = adapter

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

    def train_loop(self, train_loader, val_loader, batch_logging=10):
        print('Starting training loop')

        print('Initial evaluating on validation dataset')
        train_loss, train_acc = self.measure_performance(train_loader)
        val_loss, val_acc = self.measure_performance(val_loader)
        epoch_summary = f'[Epoch 0] | Train acc: {train_acc:.4f} Train loss: {train_loss:.4f} Val acc: {val_acc:.4f} Val loss: {val_loss:.4f}'
        print(epoch_summary)

        epoch_history = [{'epoch': 0,
                          'train loss': train_loss.item(),
                          'train accuracy': train_acc.item(),
                          'validation loss': val_loss.item(),
                          'validation accuracy': val_acc.item(),
                          'epoch time': 0}]
        batch_history = [{'epoch': 0,
                          'batch': 0,
                          'train loss': train_loss.item(),
                          'train accuracy': train_acc.item(),
                          'validation loss': val_loss.item(),
                          'validation accuracy': val_acc.item(),
                          'batch time': 0}]

        for epoch in range(self.n_epochs):

            if self.learning_scheme == 'gradual-unfreeze':
                self.optimizer = get_learning_scheme(self.learning_scheme,
                                                     self.model,
                                                     self.learning_rate,
                                                     self.adapter,
                                                     epoch+1)

            print(f'--- Epoch: {epoch+1} ---')
            epoch_start_time = time.time()
            batch_start_time = time.time()
            running_loss = 0.0
            running_acc = 0.0
            total_count = 0.0

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
                running_acc += torch.sum(preds == labels).item()
                total_count += len(labels)

                # Print/log statistics periodically
                if i % batch_logging == batch_logging - 1:
                    batch_end_time = time.time()
                    total_batch_time = batch_end_time - batch_start_time
                    batch_loss = running_loss / batch_logging
                    batch_acc = running_acc / total_count
                    batch_val_loss, batch_val_acc = self.measure_performance(val_loader)

                    batch_history.append({'epoch': epoch+1,
                                          'batch': i + 1,
                                          'train loss': batch_loss,
                                          'train accuracy': batch_acc,
                                          'validation loss': batch_val_loss.item(),
                                          'validation accuracy': batch_val_acc.item(),
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
                    running_acc = 0.0
                    total_count = 0.0

            epoch_end_time = time.time()
            total_epoch_time = epoch_end_time - epoch_start_time
            train_loss, train_acc = self.measure_performance(train_loader)
            val_loss, val_acc = self.measure_performance(val_loader)
            epoch_summary = f'[Epoch {epoch + 1}] {total_epoch_time:.2f} seconds'
            epoch_summary += f' | Train acc: {train_acc:.4f} Train loss: {train_loss:.4f} Val acc: {val_acc:.4f} Val loss: {val_loss:.4f}'

            epoch_history.append({'epoch': epoch + 1,
                                  'train loss': train_loss.item(),
                                  'train accuracy': train_acc.item(),
                                  'validation loss': val_loss.item(),
                                  'validation accuracy': val_acc.item(),
                                  'epoch time': total_epoch_time})

            print(epoch_summary)

        print('Finished training')

        return pd.DataFrame(epoch_history), pd.DataFrame(batch_history)
