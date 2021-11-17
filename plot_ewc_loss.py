import  pandas as pd
import matplotlib.pyplot as plt

def read_log(logpath):

    df = pd.read_csv(logpath,
        sep='\t|:',
        skiprows=22, skipfooter = 1,
        engine = 'python',
        usecols=[2,4,6,8,10],
        names = ['Loss_D', 'Loss_G', 'D(x)', 'D(G(z))', 'Loss_EWC'])
    return df

df_ewc = read_log('log/Run 0.log')
df_no_ewc = read_log('log/Run 1.log')

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(2, 1, 1)
ax.plot(df_ewc['Loss_EWC'], label = 'With EWC')
ax.plot(df_no_ewc['Loss_EWC'],  label = 'No EWC')
ax.set_yscale('log')
ax.legend()
plt.savefig('results/ewc_loss.png', bbox_inches='tight', transparent=True, pad_inches=.1)
