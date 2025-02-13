import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch

#import matplotlib.pyplot as plt
#import seaborn as sns

########################### General functions ###########################################

def Clear_patches(ax):
    for p in ax.patches:
        p.set_visible(False)
        p.remove()

def Clear_texts(ax):
    for p in ax.texts:
        p.set_visible(False)
        p.remove()

def Clear_current_axes():
    try:
        Clear_patches(plt.gca())
    except:
        pass
    try:
        Clear_texts(plt.gca())
    except:
        pass

############################ Visualisation Chaîne de Markov ##############################

def MCupdate(P, k):
    return np.random.choice(P.shape[0], 1, p=P[k, :])[0]

def MCanimate(nb_frame=100): #TODO: Use OOP to refactor all that garbage.
    global P
    P = np.array([[0, 1/2, 1/2, 0, 0], [1/2, 0, 1/4, 1/4, 0], [1/4, 1/2, 1/4, 0, 0], [0, 0, 0, 0, 1], [1/4, 0, 0, 1/4, 1/2]])

    fig, ax = plt.subplots()
    ax.set(xlim=[-1.3, np.sqrt(3)/2 + 0.5], ylim=[-0.3, 1.3], aspect="equal")
    ax.set_axis_off()

    R = 0.2
    Circle_centers = [(0, 1), (0, 0), (np.sqrt(3)/2, 1/2), (-1, 0), (-1, 1)]
    Circle_centers = [np.array(cc) for cc in Circle_centers]
    Circle_list = [Circle(cc, R, edgecolor="black", facecolor="white", lw=2) for cc in Circle_centers]
    for circ in Circle_list:
        ax.add_patch(circ)

    link_list = [(0, 1), (0, 2), (1, 2), (3, 4), (0, 4), (1, 3)]
    link_type = [2, 2, 2, 2, 0, 1]
    link_dir = [1, 1, 1, 1, -1, -1]
    ang_arr = 0.3
    dico_arr = dict({})
    for ll, lt, ld in zip(link_list, link_type, link_dir):
        v = Circle_centers[ll[1]] - Circle_centers[ll[0]]
        ang = np.angle(v[0] + v[1]* 1j)
        mov1 = np.array([np.cos(ang - 0.3), np.sin(ang - 0.3)])
        mov2 = np.array([np.cos(ang + 0.3), np.sin(ang + 0.3)])
        center = (Circle_centers[ll[1]] + Circle_centers[ll[0]])/2
        if lt in [0, 2]:
            if ld == 1:
                arrow = FancyArrowPatch(Circle_centers[ll[0]] + R * mov1, Circle_centers[ll[1]] - R * mov2, connectionstyle="arc3, rad=" + str(ang_arr), arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8", color="black")
                dico_arr["-".join([str(ll[0]), str(ll[1])])] = arrow
                lab = str(P[ll[0], ll[1]])
                loc = center + R * (mov1 - mov2)/2
                rot = ang * 180 / np.pi
            else:
                arrow = FancyArrowPatch(Circle_centers[ll[1]] - R * mov2, Circle_centers[ll[0]] + R * mov1, connectionstyle="arc3, rad=" + str(-ang_arr), arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8", color="black")
                dico_arr["-".join([str(ll[1]), str(ll[0])])] = arrow
                lab = str(P[ll[1], ll[0]])
                loc = center + R * (mov1 - mov2)/2
                rot = 180 + ang * 180 / np.pi
            ax.text(loc[0], loc[1], lab, rotation=rot, ha="center", va="center")
            ax.add_patch(arrow)
        if lt in [1, 2]:
            if ld == 1:
                arrow = FancyArrowPatch(Circle_centers[ll[1]] - R * mov1, Circle_centers[ll[0]] + R * mov2, connectionstyle="arc3, rad=" + str(ang_arr), arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8", color="black")
                dico_arr["-".join([str(ll[1]), str(ll[0])])] = arrow
                lab = str(P[ll[1], ll[0]])
                loc = center + R * (mov2 - mov1)/2
                rot = ang * 180 / np.pi
            else:
                arrow = FancyArrowPatch(Circle_centers[ll[0]] + R * mov2, Circle_centers[ll[1]] - R * mov1, connectionstyle="arc3, rad=" + str(-ang_arr), arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8", color="black")
                dico_arr["-".join([str(ll[0]), str(ll[1])])] = arrow
                lab = str(P[ll[0], ll[1]])
                loc = center + R * (mov2 - mov1)/2
                rot = 180 + ang * 180 / np.pi
            ax.text(loc[0], loc[1], lab, rotation=rot, ha="center", va="center")
            ax.add_patch(arrow)

    link_list = [2, 4]
    ang_list = [0, 3 * np.pi /4]
    ang_arr = 2
    for ll, ang in zip(link_list, ang_list):
        mov1 = np.array([np.cos(ang - 0.3), np.sin(ang - 0.3)])
        mov2 = np.array([np.cos(ang + 0.3), np.sin(ang + 0.3)])
        arrow = FancyArrowPatch(Circle_centers[ll] + R * mov1, Circle_centers[ll] + R * mov2, connectionstyle="arc3, rad=" + str(ang_arr), arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8", color="black")
        ax.add_patch(arrow)
        dico_arr["-".join([str(ll), str(ll)])] = arrow
        loc = Circle_centers[ll] + 1.8 * R * np.array([np.cos(ang), np.sin(ang)])
        ax.text(loc[0], loc[1], str(P[ll, ll]))

    def update(frame):
        global edge, vertex, P
        if frame == 0:
            try:
                dico_arr[vertex].set_color("black")
            except:
                pass
            edge = 0
            vertex = None
            Circle_list[edge].set_edgecolor("red")
        elif frame % 2 == 1:
            Circle_list[edge].set_edgecolor("black")
            new_edge = MCupdate(P, edge)
            vertex = "-".join([str(edge), str(new_edge)])
            edge = new_edge
            dico_arr[vertex].set_color("red")
        else:
            dico_arr[vertex].set_color("black")
            Circle_list[edge].set_edgecolor("red")

    ani = animation.FuncAnimation(fig=fig, func=update, frames=nb_frame, interval=240)
    return ani

############################ Collectionneur Coupons ########################################

def Vignette_Matrix(n):
    P = np.zeros((n+1, n+1))
    np.fill_diagonal(P, np.arange(0, n+1)/n)
    np.fill_diagonal(P[:-1, 1:], 1-np.arange(0, n)/n)
    return P

############################ Snake and ladders ########################################

def Snake_Ladder_Matrix(load=False):
    if load:
        return np.load("Data/SL_matrix.npy")
    
    P = np.zeros((100, 100))
    for i in range(94): # Lancer de dés avant la fin
        P[i, (i+1):(i+7)] = 1/6
    for i in range(94, 100): # Lancer de dés sur la fin
        P[i, i] = (i-93)/6
        P[i, (i+1):(i+7)] = 1/6
    for (i, j) in zip([2, 7, 27, 57, 74, 79, 89, 16, 51, 56, 61, 87, 94, 96], [20, 29, 83, 76, 85, 99, 90, 12, 28, 39, 21, 17, 50, 78]): # Echelles et serpents
        for k in range(100):
            if P[k, i] != 0:
                P[k, j] = P[k, i]
                P[k, i] = 0
    return P

def Make_board_SL(v):
    M = np.zeros((10, 10))
    for i in range(10):
        u = v[(100-10*(i+1)):(100-10*i)]
        if i % 2 == 0:
            u = np.flip(u)
        M[i, :] = u.flatten()
    return M

############################ Monopoly ########################################

def Monopoly_Matrix(load=False):
    if load:
        return np.load("Data/M_matrix.npy")
    
    board = np.zeros((40, 3, 40, 3))
    prison = np.zeros((3, 3))
    board_to_prison = np.zeros((40, 3, 3))
    prison_to_board = np.zeros((3, 40, 3))

    nb_comb = [i for i in range(1, 7)] + [6 - i for i in range(1, 6)] # [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]
    #nb_comb_not_double = [nb_comb[i] if i % 2 == 1 else nb_comb[i] - 1 for i in range(11)] # [0, 2, 2, 4, 4, 6, 4, 4, 2, 2, 0]
    #nb_comb_double = [1 if i % 2 == 0 else 0 for i in range(11)] # [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    def Create_Proba_Matrix(board, prison, board_to_prison, prison_to_board):
        P = np.zeros((123, 123))
        P[:40, :40] = board[:, 0, :, 0]
        P[:40, 40:80] = board[:, 0, :, 1]
        P[:40, 80:120] = board[:, 0, :, 2]
        P[:40, 120:] = board_to_prison[:, 0, :]
        P[40:80, :40] = board[:, 1, :, 0]
        P[40:80, 40:80] = board[:, 1, :, 1]
        P[40:80, 80:120] = board[:, 1, :, 2]
        P[40:80, 120:] = board_to_prison[:, 1, :]
        P[80:120, :40] = board[:, 2, :, 0]
        P[80:120, 40:80] = board[:, 2, :, 1]
        P[80:120, 80:120] = board[:, 2, :, 2]
        P[80:120, 120:] = board_to_prison[:, 2, :]
        P[120:, :40] = prison_to_board[:, :, 0]
        P[120:, 40:80] = prison_to_board[:, :, 1]
        P[120:, 80:120] = prison_to_board[:, :, 2]
        P[120:, 120:] = prison
        return P

    ### Pure dice proba
    x = np.arange(40)

    # Dealing with doubles
    for move in [2, 4, 6, 8, 10, 12]:
        board[x, 0, np.roll(x, -move), 1] = 1/36
        board[x, 1, np.roll(x, -move), 2] = 1/36
    board_to_prison[x, 2, 0] = 1/6
    move_list = [i for i in range(3, 12)]
    nb_comb_list = [2, 2, 4, 4, 6, 4, 4, 2, 2]

    # Dealing with non doubles
    for move, nb in zip(move_list, nb_comb_list):
        board[x, 0, np.roll(x, -move), 0] = nb/36
        board[x, 1, np.roll(x, -move), 0] = nb/36
        board[x, 2, np.roll(x, -move), 0] = nb/36
        
    # Dealing with the prison
    prison[0, 1] = 30/36
    prison[1, 2] = 30/36
    prison_to_board[0, [11, 13, 15, 17, 19, 21], 0] = 1/36
    prison_to_board[1, [11, 13, 15, 17, 19, 21], 0] = 1/36
    prison_to_board[2, 11:22, 0] = np.array(nb_comb)/36

    ### Special squares
    # Go to jail
    gtj_sq = np.sum(board[:, :, 30, :], axis=2)
    board_to_prison[:, :, 0] += gtj_sq
    board[:, :, 30, :] = 0

    ## Comunity chest (16 cards)
    cc_sq = [2, 17, 33]
    for sq in cc_sq:
        proba_sq = np.copy(board[:, :, sq, :])
        board[:, :, sq, :] = proba_sq * 14 / 16 # Not moving
        board[:, :, 0, :] += proba_sq * 1 / 16 # Advance to Go [0]
        board_to_prison[:, :, 0] += np.sum(proba_sq, axis=2) * 1 / 16 # Go to Jail [40]

    # Chance (16 cards)
    chance_sq = [7, 22, 36]
    nearest_rr = [15, 25, 5]
    nearest_ut = [12, 28, 12]
    for sq, rr, ut in zip(chance_sq, nearest_rr, nearest_ut):
        proba_sq = np.copy(board[:, :, sq, :])
        board[:, :, sq, :] = proba_sq * 7 / 16 # Not moving
        board[:, :, 0, :] += proba_sq * 1 / 16 # Advance to Go [0]
        board[:, :, 5, :] += proba_sq * 1 / 16 # Take a trip to Reading Railroad [5]
        board[:, :, 11, :] += proba_sq * 1 / 16 # Advance to St. Charles Place [11]
        board[:, :, 24, :] += proba_sq * 1 / 16 # Advance to Illinois Avenue [24]
        board[:, :, 39, :] += proba_sq * 1 / 16 # Advance to Boardwalk [39]
        board[:, :, sq - 3, :] += proba_sq * 1 / 16 # Go Back 3 Spaces [4, 19, 33]
        board[:, :, rr, :] += proba_sq * 1 / 16 # Advance to the nearest Railroad [15, 25, 5]
        board[:, :, ut, :] += proba_sq * 1 / 16 # Advance token to nearest Utility [12, 28, 12]
        board_to_prison[:, :, 0] += np.sum(proba_sq, axis=2) * 1 / 16 # Go to Jail [40]

    return Create_Proba_Matrix(board, prison, board_to_prison, prison_to_board)

def Make_board_M(v, cmap, debug=False):
    # Setting up colors
    n_col = len(cmap)
    v_clamp = (v - min(v))/max(v)
    id_col = np.floor(v_clamp * n_col)
    id_col[id_col == n_col] = n_col-1
    id_col = id_col.astype(int)
    col_sq = np.array(cmap)[id_col]
    col_sq[v == 0] = 1
    
    # Setting up labels
    lab = np.round(v, 2).astype(str)
    lab[lab == "0.0"] = "<0.01"
    lab = np.char.add(lab, "%")
    lab[v == 0] = ""
    lab_col = ["white" if p >= 0.5 else "black" for p in v_clamp]
    
    fig, ax = plt.subplots()
    shift1 = 0.85
    shift2 = 0.35
    ax.set_xlim([-0.1, 15.1])
    ax.set_ylim([-0.1, 15.1])
    ax.set_axis_off()
    for coord, id in zip([(0, 0), (12, 0), (0, 12), (12, 12)], [0, 30, 10, 20]):
        ax.add_patch(Rectangle(coord, 3, 3, fill=True, edgecolor="black", facecolor=col_sq[id]))
        if coord == (0, 12):
            ax.text(coord[0] + 1, 14.3, lab[id], color=lab_col[id])
        else:
            ax.text(coord[0] + 1, coord[1] + 1.3, lab[id], color=lab_col[id])
        if debug:
            ax.text(*coord, str(id))
    for i in range(3, 12):
        ax.add_patch(Rectangle((i, 0), 1, 3, fill=True, edgecolor="black", facecolor=col_sq[42 - i]))
        ax.add_patch(Rectangle((i, 12), 1, 3, fill=True, edgecolor="black", facecolor=col_sq[i + 8]))
        ax.text(i + shift2, shift1, lab[42 - i], rotation=90, color=lab_col[42 - i])
        ax.text(i + shift2, 12 + shift1, lab[i + 8], rotation=90, color=lab_col[i + 8])
        if debug:
            ax.text(i, 0, str(42 - i))
            ax.text(i, 12, str(i + 8))
    for j in range(3, 12):
        ax.add_patch(Rectangle((0, j), 3, 1, fill=True, edgecolor="black", facecolor=col_sq[j - 2]))
        ax.add_patch(Rectangle((12, j), 3, 1, fill=True, edgecolor="black", facecolor=col_sq[32 - j]))
        ax.text(shift1, j + shift2, lab[j - 2], color=lab_col[j - 2])
        ax.text(12 + shift1, j + shift2, lab[32 - j], color=lab_col[32 - j])
        if debug:
            ax.text(0, j, str(j - 2))
            ax.text(12, j, str(32 - j))
    ax.add_patch(Rectangle((1, 12), 2, 2, fill=True, edgecolor="black", facecolor=col_sq[40]))
    ax.text(1.5, 12.8, lab[40], color=lab_col[40])
    if debug:
            ax.text(1, 12, "40")
    ax.set_aspect("equal")
    return fig, ax

def Get_Stationnary(P):
    _, eigval = np.linalg.eig(P.T)
    v = eigval[:, 0]
    return abs(v / sum(v))