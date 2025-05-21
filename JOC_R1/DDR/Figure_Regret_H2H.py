from scipy.stats import gaussian_kde
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np


class Regret_H2H:
    def __init__(self):
        pass

    def figure_plot_upleft(self,all_x, all_y, figure_name, size = (5, 5), move = [-0.12, 0.04, 0.35, 0.55], 
                        ysame = 0, yrange = [6,6], sublabel = '', ypio = 0):
        
        data = np.asarray([all_x,all_y])

        xmin, ymin = data.min(axis = 1)
        xmax, ymax = data.max(axis = 1)

        xmax, xmin = tuple(np.array([xmax, xmin]) + 0.25*(xmax - xmin)*np.array([1, -1]))
        ymax, ymin = tuple(np.array([ymax, ymin]) + 0.25*(ymax - ymin)*np.array([1, -1]))
        
        ####### Obtain KDE  
        #KDE for top marginal
        kde_X = gaussian_kde(data[0])
        #KDE for right marginal
        kde_Y = gaussian_kde(data[1])

        x = np.linspace(0, 100, 100)
        y = np.linspace(ymin, ymax, 100)
            
        dx = kde_X(x) # X-marginal density
        dy = kde_Y(y) # Y-marginal density

        #Define grid for subplots, ratio for axis
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 3], height_ratios = [1, 3]) 

        ####### Create scatter plot
        fig = plt.figure(figsize = size)
        ax = plt.subplot(gs[1, 1])
        cax = ax.scatter(data[0], data[1], s = 15, color='#003D7C', marker = "o", edgecolors = "#EF7C00")
        plt.xlabel('Head-to-head (%)')
        if ypio == 0:
            plt.ylabel('Mean cost reduction (%)')
        else:
            plt.ylabel('Regret reduction (%)') #pio
        
        
        if ysame == 0:
            plt.vlines(50, ymin, ymax, linestyle="dashed", alpha = 0.8,color = 'k')
        else:
            plt.vlines(50, yrange[0], yrange[1], linestyle="dashed", alpha = 0.8,color = 'k')
        
        if ypio == 0:
            plt.hlines(0, 0, 100, linestyle="dashed", alpha = 0.8,color = 'k')
            ax.annotate(sublabel, xy = (0.55,0.9), xycoords = 'axes fraction', bbox=dict(boxstyle="round", fc="w"), size = 10)
        elif ypio == 1: #(base - item)/(base - oracle)
            plt.hlines(0, 0, 100, linestyle="dashed", alpha = 0.8,color = 'k')
            ax.annotate(sublabel, xy = (0.05,0.9), xycoords = 'axes fraction', bbox=dict(boxstyle="round", fc="w"), size = 10)
        else: #(item - oracle)/(base - oracle)
            plt.hlines(100, 0, 100, linestyle="dashed", alpha = 0.8,color = 'k') #pio
            ax.annotate(sublabel, xy = (0.55,0.9), xycoords = 'axes fraction', bbox=dict(boxstyle="round", fc="w"), size = 10)
        


        ####### Create Y-marginal (right)
        axr = plt.subplot(gs[1, 0], xticks = [], yticks = [], frameon = False)
        axr.plot(dy, y, color = 'black')
        
        if ypio == 0:
            axr.fill_betweenx(y, 0, dy, where = y <= 0, alpha = 1, color='#003D7C')
            axr.fill_betweenx(y, 0, dy, where = y >= 0, alpha = 1, color='#EF7C00')

            leftarea = np.round( sum(n <= 0 for n in all_y)/len(all_y),2 )
            rightarea = np.round( sum(n > 0 for n in all_y)/len(all_y),2 )

            axr.annotate(leftarea, xy=(0.8, abs(ymin)/(ymax - ymin) + move[0]), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), size = 12)
            axr.annotate(rightarea, xy=(0.8, abs(ymin)/(ymax - ymin) + move[1]), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), size = 12)
        elif ypio == 1:
            axr.fill_betweenx(y, 0, dy, where = y <= 0, alpha = 1, color='#EF7C00')
            axr.fill_betweenx(y, 0, dy, where = y >= 0, alpha = 1, color='#003D7C')

            leftarea = np.round( sum(n <= 0 for n in all_y)/len(all_y),2 )
            rightarea = np.round( sum(n > 0 for n in all_y)/len(all_y),2 )

            axr.annotate(leftarea, xy=(0.8, abs(ymin)/(ymax - ymin) + move[0]), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), size = 12)
            axr.annotate(rightarea, xy=(0.8, abs(ymin)/(ymax - ymin) + move[1]), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), size = 12)
        else:
            axr.fill_betweenx(y, 0, dy, where = y <= 100, alpha = 1, color='#003D7C') #pio
            axr.fill_betweenx(y, 0, dy, where = y >= 100, alpha = 1, color='#EF7C00') #pio

            leftarea = np.round( sum(n <= 100 for n in all_y)/len(all_y),2 ) #pio
            rightarea = np.round( sum(n > 100 for n in all_y)/len(all_y),2 ) #pio

            axr.annotate(leftarea, xy=(0.8, (100 - ymin)/(ymax - ymin) + move[0]), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), size = 12)
            axr.annotate(rightarea, xy=(0.8, (100 - ymin)/(ymax - ymin) + move[1]), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), size = 12)

        axr.invert_xaxis()

        ####### Create X-marginal (top)
        axt = plt.subplot(gs[0,1], frameon = False, yticks = [], xticks = [])
        #base = pyplot.gca().transData
        #rot = transforms.Affine2D().rotate_deg(180)
        axt.plot(x, dx, color = 'black')
        axt.fill_between(x, 0, dx, where = x >= 49.9, alpha= 1, color = '#003D7C')
        axt.fill_between(x, 0, dx, where = x <= 50, alpha= 1, color = '#EF7C00')

    #     axt.invert_yaxis()

        leftarea = np.round( sum(n <= 50 for n in all_x)/len(all_x),2 )
        rightarea = np.round( sum(n > 50 for n in all_x)/len(all_x),2 )

        axt.annotate(leftarea, xy=(move[2], 0.15), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), size = 12)
        axt.annotate(rightarea, xy=(move[3], 0.15), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), size = 12)

        ####### Bring the marginals closer to the scatter plot and save eps file
        fig.tight_layout(pad = 1)
        plt.savefig(figure_name + '.eps', format='eps')
        plt.savefig(figure_name + '.pdf', format='pdf')

    def figure_plot_upright(self,all_x, all_y, figure_name, size = (5, 5), move = [-0.07, 0.07, 0.35, 0.55], 
                            ysame = 0, yrange = [6,6], sublabel = '', ypio = 0):
        
        data = np.asarray([all_x,all_y])

        xmin, ymin = data.min(axis = 1)
        xmax, ymax = data.max(axis = 1)

        xmax, xmin = tuple(np.array([xmax, xmin]) + 0.25*(xmax - xmin)*np.array([1, -1]))
        ymax, ymin = tuple(np.array([ymax, ymin]) + 0.25*(ymax - ymin)*np.array([1, -1]))

        ####### Obtain KDE  

        #KDE for top marginal
        kde_X = gaussian_kde(data[0])
        #KDE for right marginal
        kde_Y = gaussian_kde(data[1])

        x = np.linspace(0, 100, 100)
        y = np.linspace(ymin, ymax, 100)

        dx = kde_X(x) # X-marginal density
        dy = kde_Y(y) # Y-marginal density

        #Define grid for subplots
        gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios = [1, 3])

        ####### Create scatter plot
        fig = plt.figure(figsize = size)
        ax = plt.subplot(gs[1, 0])
        cax = ax.scatter(data[0], data[1], s = 15, color='#003D7C', marker = "o", edgecolors = "#EF7C00")
        plt.xlabel('Head-to-head (%)')
        if ypio == 0:
            plt.ylabel('Mean cost reduction (%)')
        else:
            plt.ylabel('Regret reduction (%)') #pio
        
        
        if ysame == 0:
            plt.vlines(50, ymin, ymax, linestyle="dashed", alpha = 0.8,color = 'k')
        else:
            plt.vlines(50, yrange[0], yrange[1], linestyle="dashed", alpha = 0.8,color = 'k')
        
        if ypio == 0:
            plt.hlines(0, 0, 100, linestyle="dashed", alpha = 0.8,color = 'k')
            ax.annotate(sublabel, xy = (0.55,0.9), xycoords = 'axes fraction', bbox=dict(boxstyle="round", fc="w"), size = 10)
        elif ypio == 1:
            plt.hlines(0, 0, 100, linestyle="dashed", alpha = 0.8,color = 'k')
            ax.annotate(sublabel, xy = (0.05,0.9), xycoords = 'axes fraction', bbox=dict(boxstyle="round", fc="w"), size = 10)
        else:
            plt.hlines(100, 0, 100, linestyle="dashed", alpha = 0.8,color = 'k') #pio
            ax.annotate(sublabel, xy = (0.55,0.9), xycoords = 'axes fraction', bbox=dict(boxstyle="round", fc="w"), size = 10)

        ####### Create Y-marginal (right)
        axr = plt.subplot(gs[1, 1], xticks = [], yticks = [], frameon = False)
        axr.plot(dy, y, color = 'black')

        if ypio == 0:
            axr.fill_betweenx(y, 0, dy, where = y <= 0.01, alpha = 1, color='#003D7C')
            axr.fill_betweenx(y, 0, dy, where = y >= 0, alpha = 1, color='#EF7C00')

            leftarea = np.round( sum(n <= 0 for n in all_y)/len(all_y),2 )
            rightarea = np.round( sum(n > 0 for n in all_y)/len(all_y),2 )

            axr.annotate(leftarea, xy=(0.15, abs(ymin)/(ymax - ymin) + move[0]), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), size = 12)
            axr.annotate(rightarea, xy=(0.15, abs(ymin)/(ymax - ymin) + move[1]), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), size = 12)
        elif ypio == 1:
            axr.fill_betweenx(y, 0, dy, where = y <= 0, alpha = 1, color='#EF7C00')
            axr.fill_betweenx(y, 0, dy, where = y >= 0, alpha = 1, color='#003D7C')

            leftarea = np.round( sum(n <= 0 for n in all_y)/len(all_y),2 )
            rightarea = np.round( sum(n > 0 for n in all_y)/len(all_y),2 )

            axr.annotate(leftarea, xy=(0.15, abs(ymin)/(ymax - ymin) + move[0]), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), size = 12)
            axr.annotate(rightarea, xy=(0.15, abs(ymin)/(ymax - ymin) + move[1]), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), size = 12)
        else:
            axr.fill_betweenx(y, 0, dy, where = y <= 100, alpha = 1, color='#003D7C') #pio
            axr.fill_betweenx(y, 0, dy, where = y >= 100, alpha = 1, color='#EF7C00') #pio

            leftarea = np.round( sum(n <= 100 for n in all_y)/len(all_y),2 ) #pio
            rightarea = np.round( sum(n > 100 for n in all_y)/len(all_y),2 ) #pio

            axr.annotate(leftarea, xy=(0.15, (100 - ymin)/(ymax - ymin) + move[0]), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), size = 12)
            axr.annotate(rightarea, xy=(0.15, (100 - ymin)/(ymax - ymin) + move[1]), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), size = 12)
    #     axr.invert_xaxis()


        ####### Create X-marginal (top)
        axt = plt.subplot(gs[0,0], frameon = False, yticks = [], xticks = [])
        #base = pyplot.gca().transData
        #rot = transforms.Affine2D().rotate_deg(180)
        axt.plot(x, dx, color = 'black')
        axt.fill_between(x, 0, dx, where = x >= 49.9, alpha= 1, color = '#003D7C')
        axt.fill_between(x, 0, dx, where = x <= 50, alpha= 1, color = '#EF7C00')

    #     axt.invert_yaxis()

        leftarea = np.round( sum(n <= 50 for n in all_x)/len(all_x),2 )
        rightarea = np.round( sum(n > 50 for n in all_x)/len(all_x),2 )

        axt.annotate(leftarea, xy=(move[2], 0.15), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), size = 12)
        axt.annotate(rightarea, xy=(move[3], 0.15), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"), size = 12)


        ####### Bring the marginals closer to the scatter plot and save eps file
        fig.tight_layout(pad = 1)
        plt.savefig(figure_name + '.eps', format='eps')
        plt.savefig(figure_name + '.pdf', format='pdf')