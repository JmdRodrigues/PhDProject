from pylab import rc, close, figure, axes, subplot, plot, axis, show, grid, savefig, text, gcf, gca, draw
from numpy import arange
import numpy as np
# added import - David Belo
from matplotlib import pyplot as plt
import pandas
import tkinter.filedialog as tkFileDialog

from matplotlib.font_manager import FontProperties
from matplotlib.figure import SubplotParams
from matplotlib.markers import MarkerStyle
import matplotlib.patheffects as pte
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as grid
import matplotlib as mpl


def plot_config():

    ratio = 16/4

    # color
    face_color_r = 248 / 255.0
    face_color_g = 247 / 255.0
    face_color_b = 249 / 255.0

    rc('lines', linewidth=2, color='k')

    rc('text', color='grey')

    rc('figure', figsize=(8, 8/ratio), dpi=96)

    rc('axes', grid=True, edgecolor='grey', labelsize=20)

    rc('grid', color='lightgrey')


def subplot_pars():
    # pars
    left = 0.05  # the left side of the subplots of the figure
    right = 0.95  # the right side of the subplots of the figure
    bottom = 0.05  # the bottom of the subplots of the figure
    top = 0.92  # the top of the subplots of the figure
    wspace = 0.2  # the amount of width reserved for blank space between subplots
    hspace = 2  # the amount of height reserved for white space between subplots

    pars = SubplotParams(left, bottom, right, top, wspace, hspace)

    return pars


def font_config():

    # specify Font properties with fontmanager---------------------------------------------------
    font0 = FontProperties()
    font0.set_weight('medium')
    font0.set_family('monospace')

    # Specify Font properties of Legends
    font1 = FontProperties()
    font1.set_weight('normal')
    font1.set_family('sans-serif')
    font1.set_style('italic')
    font1.set_size(12)

    # Specify font properties of Titles
    font2 = FontProperties()
    font2.set_size(15)
    font2.set_family('sans-serif')
    font2.set_weight('medium')
    font2.set_style('italic')

    return font0, font1, font2

def Cplot(y, x=0, ax=0):
    plot_config()
    normal_f, legends_f, titles_f = font_config()

    if(x == 0):
        x = np.linspace(0, len(y), len(y))

    if (ax == 0):
        plt.plot(x, y)
    else:
        ax.plot(x, y)

def Csubplot(n_rows, n_columns, graphs):
    if(len(graphs)!=n_rows*n_columns):
        return "error...number of groups of plots must be equal to the number of " \
               "subplots"
    else:
        fig, axs = plt.subplots(n_rows, n_columns)
        graph_group = 0
        for row in range(0, n_rows):
            print(row)
            for column in range(0, n_columns):
                print(column)
                for graph in graphs[graph_group]:
                    Cplot(y=graph, x=0, ax=axs[graph_group])
                graph_group+=1
    return axes

def zoom(event):
    ax = gca()
    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()

    # edit the scale if needed
    base_scale = 1.1

    xdata = event.xdata     # get event x location
    ydata = event.ydata     # get event y location

    # performs a prior check in order to not exceed figure margins
    if xdata != None and ydata != None:
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
            print(event.button)

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])

        ax.set_xlim([xdata - new_width * (1-relx), xdata + new_width * (relx)])
        ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * (rely)])
        ax.figure.canvas.draw()

    return zoom


def on_key_press(event):

    # keyboard zoom-in
    if event.key == '+':
        a = axis()
        w = a[1] - a[0]
        axis([a[0] + w * .2, a[1] - w * .2, a[2], a[3]])
        draw()

    # keyboard zoom-out
    if event.key in ['-', '\'']:
        a = axis()
        w = a[1] - a[0]
        axis([a[0] - w / 3.0, a[1] + w / 3.0, a[2], a[3]])
        draw()

    # right displacement
    if event.key in ['.', 'right']:
        a = axis()
        w = a[1] - a[0]
        axis([a[0] + w * .2, a[1] + w * .2, a[2], a[3]])
        draw()

    # left displacement
    if event.key in [',', 'left']:
        a = axis()
        w = a[1] - a[0]
        axis([a[0] - w * .2, a[1] - w * .2, a[2], a[3]])
        draw()

    # up displacement
    if event.key == 'up':
        a = axis()
        w = a[3] - a[2]
        axis([a[0], a[1], a[2] + w * .2, a[3] + w * .2])
        draw()

    # down displacement
    if event.key == 'down':
        a = axis()
        w = a[3] - a[2]
        axis([a[0], a[1], a[2] - w * .2, a[3] - w * .2])
        draw()

    # close figure
    if event.key == 'q':
        close()
        # NOTE: We should make the disconnect (mpl_disconect(cid)
        # But since the figure is destroyed we may keep this format
        # if implemented the mpl_connect should use the return cid

    # print('you pressed', event.key, event.xdata, event.ydata)
    # TODO: Reset zoom with an initial default value -> suggest 'r' key


def on_key_release(event):
    # print('you released', event.key, event.xdata, event.ydata)
    pass


def niplot():
    """
    This script extends the native matplolib keyboard bindings.
    This script allows to use the `up`, `down`, `left`, and `right` keys
    to move the visualization window. Zooming can be performed using the `+`
    and `-` keys. Finally, the scroll wheel can be used to zoom under cursor.

    Returns
    -------

    """
    fig = gcf()
    cid = fig.canvas.mpl_connect('key_press_event',  # @UnusedVariable
                                 on_key_press)
    cid = fig.canvas.mpl_connect('key_release_event',  # @UnusedVariable
                                 on_key_release)
    cid = fig.canvas.mpl_connect('scroll_event', zoom)





def load_data_dialog(path):
    # root = Tkinter.Tk()
    # root.withdraw()

    # Make it almost invisible - no decorations, 0 size, top left corner.
    # root.overrideredirect(True)
    # root.geometry('0x0+0+0')

    # Show window again and lift it to top so it can get focus,
    # otherwise dialogs will end up behind the terminal.
    # root.deiconify()
    # root.lift()
    # root.focus_force()

    # filename = tkFileDialog.askopenfile(parent=root) # Or some other dialog
    filename = tkFileDialog.askopenfile()  # Or some other dialog

    # Get rid of the top-level instance once to make it actually invisible.
    # root.destroy()
    return pandas.read_csv(filename, sep=' ', header=None)


##########
# Initial configurations
def pylabconfig():
    rc('lines', linewidth=2, color='k')
    # rc('lines', linewidth=1, color='k')

    rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    rc('font', style='italic', size=10)

    rc('text', color='grey')

    #       rc('text', usetex=True)

    rc('text', usetex=False)

    rc('figure', figsize=(8, 5), dpi=80)
    rc('axes', grid=True, edgecolor='grey', labelsize=10, )
    rc('grid', color='grey')
    rc('xtick', color='grey', labelsize=10)
    rc('ytick', color='grey', labelsize=10)

    close('all')


def plotwithhist(t, s, bins=50):
    from matplotlib.ticker import NullFormatter

    nullfmt = NullFormatter()
    figure()
    ax2 = axes([0.125 + 0.5, 0.1, 0.2, 0.8])
    ax1 = axes([0.125, 0.1, 0.5, 0.8])

    ax1.plot(t, s)
    ax1.set_xticks(ax1.get_xticks()[:-1])

    ax2.hist(s, bins, normed=True, facecolor='white',
             orientation='horizontal', lw=2)

    ax2.axis([0, 1, ax1.axis()[2], ax1.axis()[3]])

    ax2.yaxis.set_major_formatter(nullfmt)
    ax2.set_xticks([0, 0.5, 1])

    return ax1, ax2

    ###########


def plotwithstats(t, s):
    from matplotlib.ticker import NullFormatter

    nullfmt = NullFormatter()
    figure()
    ax2 = axes([0.125 + 0.5, 0.1, 0.2, 0.8])

    ax1 = axes([0.125, 0.1, 0.5, 0.8])

    ax1.plot(t, s)
    ax1.set_xticks(ax1.get_xticks()[:-1])

    meanv = s.mean()
    mi = s.min()
    mx = s.max()
    sd = s.std()

    ax2.bar(-0.5, mx - mi, 1, mi, lw=2, color='#f0f0f0')
    ax2.bar(-0.5, sd * 2, 1, meanv - sd, lw=2, color='#c0c0c0')
    ax2.bar(-0.5, 0.2, 1, meanv - 0.1, lw=2, color='#b0b0b0')
    ax2.axis([-1, 1, ax1.axis()[2], ax1.axis()[3]])

    ax2.yaxis.set_major_formatter(nullfmt)
    ax2.set_xticks([])

    return ax1, ax2


def multilineplot(signal, linesize=250, events=None, title='', dir='', step=1):
    from pylab import rc
    rc('axes', labelcolor='#a1a1a1', edgecolor='#a1a1a1', labelsize='xx-small')
    rc('xtick', color='#a1a1a1', labelsize='xx-small')

    grid('off')
    nplots = len(signal) // linesize + 1
    ma_x = max(signal)
    mi_x = min(signal)
    f = figure(figsize=(20, 1.5 * nplots), dpi=80)
    for i in range(nplots):
        ax = subplot(nplots, 1, i + 1)

        start = int(i * len(signal) / nplots)
        end = int((i + 1) * len(signal) / nplots)
        plot(arange(start, end, step), signal[start:end:step], 'k')
        axis((start, end, mi_x, ma_x))
        ax.set_yticks([])
        ax.set_xticks(ax.get_xticks()[1:-1])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_bounds(mi_x,ma_x)
        ax.xaxis.set_ticks_position('bottom')
        if events != None:
            e = events[(events >= start) & (events < end)]

            if len(e) > 0:
                plot.vlines(e, mi_x, ma_x - (ma_x - mi_x) / 4. * 3., lw=2)

        if title != None:
            text(start, ma_x, title)
        grid('off')
    f.tight_layout()
    # savefig(dir + title + '.pdf')
    # close()

