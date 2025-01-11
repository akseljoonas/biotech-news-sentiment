import os
import pandas as pd
import random

class ProfitEvaluator:
    def __init__(self, model_predictions, test_file, test_size=0.15, random_state=42):
        self.predictions = model_predictions
        self.base_dir = os.getcwd()
        data_file = "../../data/processed/news+prices-new.csv"
        self.data_file = os.path.join(self.base_dir, data_file)
        self.test_file = os.path.join(self.base_dir, test_file)
        self.test_size = test_size
        self.random_state = random_state

        self.eval_profit_dataset = pd.read_csv(self.data_file)
        self.test_data = pd.read_csv(self.test_file)

        self.test_indices = self.test_data.index.tolist()

    def print_datasets(self):
        print("Training Set:")
        print(self.train_dataset)
        print("\nTesting Set:")
        print(self.test_dataset)

    def print_row_ids(self):
        print("\nRow IDs for Training Set:")
        print(self.train_indices)
        print("\nRow IDs for Testing Set:")
        print(self.test_indices)
    
    def increase_decrease(self, current_index):
        buy_in = self.eval_profit_dataset.loc[current_index, "buy_in_price"]
        sell = self.eval_profit_dataset.loc[current_index, "close"]
        increase_decrease = sell / buy_in - 1
        return(increase_decrease)




    def calculate_profit(self):
        total_profit = 0
        test_indices_copy = self.test_indices.copy()

        for prediction in self.predictions:
            if test_indices_copy:  # Check if there are still indices to process
                current_index = test_indices_copy[0]  # Get the current index
                price_move = self.increase_decrease(current_index)

                if prediction == 0:
                    total_profit -= price_move
                elif prediction == 2:
                    total_profit += price_move
                # For prediction == 1, no profit adjustment happens
                
                print(f"Total Profit: {total_profit} (Prediction: {prediction}, Price Move: {price_move})")
                
                # Remove the processed index from the list
                test_indices_copy.pop(0)
            else:
                print("No more indices left to process!")
                break

        print(f"\nFinal Total Profit: {total_profit}")
        print(f"Remaining rows in eval dataset: {len(test_indices_copy)}")



test_file = "../../data/processed/finetuning-3label.csv"
model_predictions = [random.choice([0, 1, 2]) for _ in range(1000)]

evaluator = ProfitEvaluator(model_predictions, test_file)
evaluator.print_datasets()
evaluator.print_row_ids()
evaluator.calculate_profit()