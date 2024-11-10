from training_traditional import MLPRegressorTrainer
from training_simple import SimpleNNTrainer
from training_marginalize import MixtureOfExpertsTrainer
from training_flows import NormalizingFlowsTrainer
from scm_model import generate_data
from confounding_advertising_env import AdvertisingEnv
import numpy as np
import pandas as pd
import torch
import random
import matplotlib.pyplot as plt

def reward_regression_experiment():
    data_lens = [100, 250, 500, 1000, 5000, 10000]
    mses = []
    r2s = []
    datas = [generate_data(data_len) for data_len in data_lens]

    for i, data in enumerate(datas):
        trainer = SimpleNNTrainer(data)
        trainer.preprocess_data()
        trainer.train_model()
        traditional_mse, traditional_r2 = trainer.evaluate_model()
        trainer.plot_results()
        traditional_y_test, traditional_y_pred_test = trainer.plotting_data()

        trainer = MixtureOfExpertsTrainer(data)
        trainer.preprocess_data()
        trainer.train_model()
        moe_mse, moe_r2 = trainer.evaluate_model()
        trainer.plot_results()
        moe_y_test, moe_y_pred_test = trainer.plotting_data()

        mses.append((traditional_mse, moe_mse))
        r2s.append((traditional_r2, moe_r2))
        
        plt.clf()
        plt.figure(figsize=(8, 6))
        plt.scatter(traditional_y_test, traditional_y_pred_test, alpha=0.5, label='Traditional')
        plt.scatter(moe_y_test, moe_y_pred_test, alpha=0.5, label='Marginalizing over U')
        plt.xlabel('Actual R')
        plt.ylabel('Predicted R')
        plt.title('Actual vs. Predicted R on Test Set')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.legend()
        plt.savefig(f'./comparison_results/actual_vs_predicted_{data_lens[i]}.png')
        plt.close()

        residuals_traditional = traditional_y_test - traditional_y_pred_test
        residuals_moe = moe_y_test - moe_y_pred_test

        plt.clf()
        plt.figure(figsize=(8, 6))
        plt.hist(residuals_traditional, bins=50, alpha=0.7, label='Traditional')
        plt.hist(residuals_moe, bins=50, alpha=0.7, label='Marginalizing over U')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residuals Distribution on Test Set')
        plt.legend()
        plt.savefig(f'./comparison_results/residuals_distribution_{data_lens[i]}.png')
        plt.close()
        plt.rcdefaults()

    mses_df = pd.DataFrame(mses, columns=['Traditional', 'Marginalizing over U'], index=data_lens)
    r2s_df = pd.DataFrame(r2s, columns=['Traditional', 'Marginalizing over U'], index=data_lens)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    mses_df.plot(ax=ax[0], kind='bar', title='MSE', ylabel='MSE', xlabel='Data Length')
    r2s_df.plot(ax=ax[1], kind='bar', title='R^2', ylabel='R^2', xlabel='Data Length')
    plt.tight_layout()
    plt.savefig('./metrics.png')

def data_aug_experiment():
    data = generate_data(2000)

    n_flows = NormalizingFlowsTrainer(data)
    X, R = n_flows.preprocess_data()
    n_flows.train_model()
    n_flows.evaluate_model()
    n_flows.plot_results()

    # train simple regressor on the same data
    regressor = MLPRegressorTrainer(data)
    regressor.preprocess_data()
    regressor.train_model()
    print("\nMSE and R^2 for the traditional model:")
    regressor.evaluate_model()
    before_r_test, before_r_pred = regressor.plotting_data()
    print()

    # Data Augmentation
    def intervention(a):
        action_space = np.round(np.arange(0.05, 1.05, 0.05), 2)
        action_space = action_space[action_space != a]
        new_actions = np.random.choice(action_space, size=4, replace=False)
        return new_actions
    
    # generate a new dataset with a different action using counterfactual outcomes
    data_aug = data.copy()
    for i in range(len(data_aug)):
        x = X[i].unsqueeze(0)
        r = R[i].unsqueeze(0)
        a_primes = intervention(data_aug.iloc[i]['A'])
        for a_prime in a_primes:
            r_prime = n_flows.counterfactual_outcome(x, r, a_prime)
            new_row = data_aug.iloc[i].copy()
            new_row['A'] = a_prime
            new_row['R'] = r_prime.item()
            data_aug = pd.concat([data_aug, pd.DataFrame([new_row])], ignore_index=True)
    
    # train simple regressor on the augmented data
    regressor_aug = MLPRegressorTrainer(data_aug)
    test_indices=range(0, len(data))
    regressor_aug.preprocess_data(test_indices)
    regressor_aug.train_model()
    print("\nMSE and R^2 for the traditional model on the augmented data:")
    regressor_aug.evaluate_model()
    after_r_test, after_r_pred = regressor_aug.plotting_data()
    print(f"\nOld Data length: {len(data)}, New Data length: {len(data_aug)}")

    data.to_csv('./counterfactual_results/data.csv', index=False)
    data_aug.to_csv('./counterfactual_results/data_aug.csv', index=False)

    plt.clf()
    plt.figure(figsize=(8, 6))
    plt.scatter(before_r_test, before_r_pred, alpha=0.5, label='Before Intervention')
    plt.scatter(after_r_test, after_r_pred, alpha=0.5, label='After Intervention')
    plt.xlabel('Actual R')
    plt.ylabel('Predicted R')
    plt.title('Actual vs. Predicted R on Test Set')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.legend()
    plt.savefig('./counterfactual_results/actual_vs_predicted.png')
    plt.close()

def counterfactual_experiment():
    data = generate_data(3000)

    n_flows = NormalizingFlowsTrainer(data)
    X, R = n_flows.preprocess_data()
    n_flows.train_model()
    n_flows.evaluate_model()
    n_flows.plot_results()

    env = AdvertisingEnv()
    context, _ = env.reset()
    done = False

    r_values = []
    n = 1000
    while n > 0:
        action = env.action_space.sample()  # Randomly select an action
        context, reward, done, _, _ = env.step(action)
        bid = env.bid_values[action]

        x = torch.cat([torch.tensor(context, dtype=torch.float32).unsqueeze(0), torch.tensor([bid], dtype=torch.float32).unsqueeze(0)], dim=1)
        r = torch.tensor([reward], dtype=torch.float32).unsqueeze(0)
        # get counterfactual outcome
        a_prime = np.random.choice([a for a in env.bid_values if a != bid])
        r_prime = n_flows.counterfactual_outcome(x, r, a_prime)

        action_prime = np.where(env.bid_values == a_prime)[0][0]
        context, reward, done, _, _ = env.step(action_prime)
        r_values.append((reward, r_prime.item()))

        if done:
            context, _ = env.reset()
        n -= 1

    r_values = np.array(r_values)
    residuals = r_values[:, 0] - r_values[:, 1]
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=50, alpha=0.7)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution')
    plt.savefig('./counterfactual_results/residuals_distribution.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(r_values[:, 0], r_values[:, 1], alpha=0.5)
    plt.xlabel('Actual R')
    plt.ylabel('Counterfactual R')
    plt.title('Actual vs. Counterfactual R')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.savefig('./counterfactual_results/actual_vs_counterfactual.png')
    plt.close()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    set_seed(0)

    # reward_regression_experiment()
    # print("Reward regression experiment completed.")
    # data_aug_experiment()
    # print("Data augmentation experiment completed.")
    counterfactual_experiment()
    print("Counterfactual experiment completed.")