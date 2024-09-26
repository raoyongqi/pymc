import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import os

# 读取 Excel 文件中的数据
def load_data(file_path, column_name):
    df = pd.read_excel(file_path)
    return df[column_name].dropna()  # 去除空值

# 绘制直方图以查看数据分布
def plot_data(data):
    sns.histplot(data, kde=True)
    plt.title('Data Histogram with KDE')
    plt.show()

# 贝叶斯分析：假设正态分布
def bayesian_normal_model(data):
    with pm.Model() as model:
        mu = pm.Normal('mu', mu=np.mean(data), sigma=np.std(data))
        sigma = pm.HalfNormal('sigma', sigma=10)
        likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=data)

        # 采样
        trace = pm.sample(1000, return_inferencedata=False)
        
        # 转换为 ArviZ 的 InferenceData
        az_trace = az.from_pymc(trace)
        
        # 绘制后验分布
        az.plot_posterior(az_trace)
        plt.title('Posterior Distribution for Normal Model')
        plt.savefig('pic/normal_model_posterior.png')  # 保存图形
        plt.show()
        
    return trace

# 贝叶斯分析：假设指数分布
def bayesian_exponential_model(data):
    with pm.Model() as model:
        lam = pm.Exponential('lam', lam=1.0)
        likelihood = pm.Exponential('y', lam=lam, observed=data)

        # 采样
        trace = pm.sample(1000, return_inferencedata=False)

        # 转换为 ArviZ 的 InferenceData
        az_trace = az.from_pymc(trace)

        # 绘制后验分布
        az.plot_posterior(az_trace)
        plt.title('Posterior Distribution for Exponential Model')
        plt.savefig('pic/exponential_model_posterior.png')  # 保存图形
        plt.show()
        
    return trace

# 贝叶斯分析：假设伽马分布
def bayesian_gamma_model(data):
    with pm.Model() as model:
        alpha = pm.HalfNormal('alpha', sigma=10)
        beta = pm.HalfNormal('beta', sigma=10)
        likelihood = pm.Gamma('y', alpha=alpha, beta=beta, observed=data)

        # 采样
        trace = pm.sample(1000, return_inferencedata=False)

        # 转换为 ArviZ 的 InferenceData
        az_trace = az.from_pymc(trace)

        # 绘制后验分布
        az.plot_posterior(az_trace)
        plt.title('Posterior Distribution for Gamma Model')
        plt.savefig('pic/gamma_model_posterior.png')  # 保存图形
        plt.show()
        
    return trace

# 主函数：执行贝叶斯模型比较
def main():
    # 创建保存图形的目录
    if not os.path.exists('pic'):
        os.makedirs('pic')

    # Excel 文件路径和要分析的列名
    file_path = 'data/output_data.xlsx'
    column_name = 'PL'
    
    # 读取数据
    data = load_data(file_path, column_name)
    
    # 绘制数据分布
    plot_data(data)

    # 运行贝叶斯分析
    print("Running Normal model...")
    normal_trace = bayesian_normal_model(data)
    
    print("Running Exponential model...")
    exponential_trace = bayesian_exponential_model(data)
    
    print("Running Gamma model...")
    gamma_trace = bayesian_gamma_model(data)
    
    # 模型比较可以使用 WAIC 或 LOO（留一交叉验证）
    # 比较不同模型的适用性
    with pm.Model() as model:
        normal_waic = pm.waic(normal_trace, model)
        exponential_waic = pm.waic(exponential_trace, model)
        gamma_waic = pm.waic(gamma_trace, model)
        
        print(f"Normal model WAIC: {normal_waic}")
        print(f"Exponential model WAIC: {exponential_waic}")
        print(f"Gamma model WAIC: {gamma_waic}")
        
if __name__ == '__main__':
    main()
