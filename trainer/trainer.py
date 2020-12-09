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
        # self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def measure_performance(self, loader):
        running_loss = 0.0
        correct_count = 0
        total_count = 0
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

    def train_loop(self, train_loader, val_loader=None):
        print('Starting training loop\n\n')

        if val_loader:
            print('Initial evaluating on validation dataset')
            val_loss, val_acc = self.measure_performance(val_loader)
            epoch_summary = f'[Epoch 0] | Val acc: {val_acc:.4f} Val loss: {val_loss:.4f}\n\n'
            print(epoch_summary)

        for epoch in range(self.n_epochs):
            print(f'--- Epoch: {epoch} ---')
            epoch_start_time = time.time()
            batch_start_time = time.time()
            running_loss = 0.0

            for i, data in enumerate(iter(train_loader)):
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

                # Print statistics periodically
                running_loss += loss.item()
                if i % N_MINI_BATCH_CHECK == N_MINI_BATCH_CHECK - 1:
                    batch_end_time = time.time()
                    total_batch_time = batch_end_time - batch_start_time

                    print(
                        f'[E{epoch + 1:d} B{i + 1:d}] ',
                        f'Loss: {running_loss / N_MINI_BATCH_CHECK:.5f} ',
                        f'Time: {total_batch_time:.2f} ',
                        f'LR: {self.scheduler.get_last_lr()}' if self.scheduler else '')

                    # Reset statistics
                    batch_start_time = time.time()
                    running_loss = 0.0

            epoch_end_time = time.time()
            total_epoch_time = epoch_end_time - epoch_start_time
            epoch_summary = '[Epoch {}] {} seconds'.format((epoch + 1), total_epoch_time)

            if val_loader:
                val_loss, val_acc = self.measure_performance(val_loader)
                epoch_summary += f' | Val acc: {val_acc:.4f} | Val loss: {val_loss:.4f}'

            print(epoch_summary)

        print('Finished training')
