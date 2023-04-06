def plot_daily_trend():
    ################################################### DAILY TREND
    sub_Y = np.array_split(Y, 6)
    fig1, axs1 = plt.subplots(2,3)
    fig1.suptitle('daily trend')
    subplots_day_D = [magnitude_D[n:n+int(len(magnitude_D)/6)] for n in range(0, len(magnitude_D), int(len(magnitude_D)/6))]
    subplots_day_ND = [magnitude_ND[n:n+int(len(magnitude_D)/6)] for n in range(0, len(magnitude_ND),int(len(magnitude_D)/6))]
    kek=-3
    for z in range(0,2):
        kek+=3
        for a in range(0,3):
            sub_Y_multiplied=[]
            for el in sub_Y[a+kek]:
                for u in range(sample_size):
                    sub_Y_multiplied.append(el)
        
            #axs1[z, a].scatter(list(range(len(subplots_day_D[a+kek]))), subplots_day_D[a+kek], c=sub_Y_multiplied, cmap= 'brg', s=5, alpha=0.5)
            #axs1[z, a].scatter(list(range(len(subplots_day_ND[a+kek]))), subplots_day_D[a+kek], c=sub_Y_multiplied, cmap= 'brg',s=5, alpha=0.5)          
            
            subplotD = [subplots_day_D[a+kek].iloc[n:n+sample_size] for n in range(0, len(subplots_day_D[a+kek]), sample_size)]
            subplotND = [subplots_day_ND[a+kek].iloc[n:n+sample_size] for n in range(0, len(subplots_day_ND[a+kek]), sample_size)]
            for s in range(len(subplotD)):
                if sub_Y[a+kek][s] == 0:
                    axs1[z, a].plot(subplotD[s], color='darkred', alpha=0.5)
                    axs1[z, a].plot(subplotND[s], color='red', alpha=0.5)
                elif sub_Y[a+kek][s] == 1:
                    axs1[z, a].plot(subplotD[s], color='darkgreen', alpha=0.5)
                    axs1[z, a].plot(subplotND[s], color='green', alpha=0.5)
                else:
                    axs1[z, a].plot(subplotD[s], color='darkblue', alpha=0.5)
                    axs1[z, a].plot(subplotND[s], color='blue', alpha=0.5)

    plt.show()
    plt.close()
    ################################################### DAILY TREND