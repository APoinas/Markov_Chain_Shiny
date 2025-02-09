import os as os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from shiny import App, Inputs, Outputs, Session, render, ui, reactive
from utils import Snake_Ladder_Matrix, Make_board_SL, Vignette_Matrix, Monopoly_Matrix, Clear_current_axes, Make_board_M, Get_Stationnary, MCanimate

from IPython.display import HTML # For the animation

app_ui = ui.page_fluid(
    ui.tags.style(
        ".action-button { font-size: 30px; text-align: center; padding: 5px 5px; background-color: #e7e7e7; border-radius: 8px; border: 2px solid}"
    ),
    ui.navset_pill(
        ui.nav_panel("Echelles et serpents",
            ui.h1("Simulation du jeu des échelles et des serpents", align = "center"),
            ui.layout_columns(
                ui.output_image("Image_SL"),
                ui.output_plot("Probability_SL", height="550px"),
                col_widths=(-1, 5, 6),
                height="450px"
            ),
            ui.layout_columns(
                ui.card(
                    ui.layout_columns(
                        ui.input_action_button("Decrease_move_nb", "-"),
                        ui.div(
                            ui.output_ui("Display_move_nb"),
                            align="center"
                        ),
                        ui.input_action_button("Increase_move_nb", "+"),
                        col_widths=(3, 6, 3),
                        align="center"
                    )
                ),
                col_widths=(-4, 4),
                align="center"
            ),
        ),
        ui.nav_panel("Monopoly",
            ui.h1("Simulation du jeu de Monopoly", align = "center"),
            ui.layout_columns(
                ui.output_image("Image_M"),
                ui.output_plot("Probability_Monopoly", width="900px"),
                align="center",
                col_widths=(-1, 4, 3),
                height="600px"
            ),
            ui.layout_columns(
                ui.card(
                    ui.layout_columns(
                        ui.input_action_button("Decrease_move_nb_M", "-"),
                        ui.div(
                            ui.output_ui("Display_move_nb_M"),
                            align="center"
                        ),
                        ui.input_action_button("Increase_move_nb_M", "+"),
                        col_widths=(3, 6, 3),
                        align="center",
                    )
                ),
                col_widths=(-4, 4),
                align="center"
            ),
            ui.layout_columns(ui.card(ui.input_action_button("Infinity_M", "Infinity"), align="center"), col_widths=(-5, 2))
        ),
        ui.nav_panel("Collectionneur de vignettes",
            ui.h1("Simulation du problème du collectionneur de vignettes", align="center"),
            ui.layout_columns(
                ui.card(ui.input_slider("Move_nb_V", "Nombre total de vignettes:", min=1, max=15, value=10), align="center"),
                col_widths=(-5, 2, 5),
                align="center",
            ),
            ui.div(ui.output_plot("Probability_Vignette", width="1000px", height="200px"), align="center"),
            ui.layout_columns(
                ui.card(
                    ui.layout_columns(
                        ui.input_action_button("Decrease_move_nb_V", "-"),
                        ui.div(
                            ui.output_ui("Display_move_nb_V"),
                            align="center"
                        ),
                        ui.input_action_button("Increase_move_nb_V", "+"),
                        col_widths=(3, 6, 3),
                        align="center",
                    )
                ),
                col_widths=(-4, 4),
                align="center"
            ),
        ),
        ui.nav_panel("Visualisation",
            ui.h1("Visualisation de l'évolution d'une chaîne de Markov", align="center"),
            ui.div(ui.output_ui("Visualisation_CM"), align="center"),
        ),
    )
)

def server(input: Inputs, output: Outputs, session: Session):
    
    Move_nb = reactive.value(1)
    Move_nb_M = reactive.value(1)
    Move_nb_V = reactive.value(2)
    Infinity_M = reactive.value(False)
    
    @reactive.effect
    @reactive.event(input.Increase_move_nb)
    def Increase_move_nb():
        Move_nb.set(Move_nb() + 1)
        
    @reactive.effect
    @reactive.event(input.Decrease_move_nb)
    def Decrease_move_nb():
        if Move_nb() >= 1:
            Move_nb.set(Move_nb() - 1)
            
    @reactive.effect
    @reactive.event(input.Increase_move_nb_M)
    def Increase_move_nb():
        if not Infinity_M():
            Move_nb_M.set(Move_nb_M() + 1)
        
    @reactive.effect
    @reactive.event(input.Decrease_move_nb_M)
    def Decrease_move_nb():
        if Move_nb_M() >= 1 and not Infinity_M():
            Move_nb_M.set(Move_nb_M() - 1)
            
    @reactive.effect
    @reactive.event(input.Increase_move_nb_V)
    def Increase_move_nb():
        Move_nb_V.set(Move_nb_V() + 1)
        
    @reactive.effect
    @reactive.event(input.Decrease_move_nb_V)
    def Decrease_move_nb():
        if Move_nb_V() >= 1:
            Move_nb_V.set(Move_nb_V() - 1)
    
    @reactive.effect
    @reactive.event(input.Infinity_M)
    def Switch_Infinity_M():
        Infinity_M.set(not Infinity_M())

    @render.ui
    def Display_move_nb():
        return ui.HTML(f"Nombre de lancers de dés:<br> {Move_nb()}")
    
    @render.ui
    def Display_move_nb_M():
        s = Move_nb_M()
        if Infinity_M():
            s = "∞"
        return ui.HTML(f"Nombre de lancers de dés:<br> {s}")
    
    @render.ui
    def Display_move_nb_V():
        return ui.HTML(f"Nombre de vignettes achetées:<br> {Move_nb_V()}")
    
    @render.ui
    def Visualisation_CM():
        ani = MCanimate(50)
        return HTML(ani.to_jshtml())

    @render.plot
    def Probability_SL():
        move = Move_nb()
        v = np.zeros((1, 100))
        v[0, 0] = 1
        P = Snake_Ladder_Matrix()
        M = 100 * Make_board_SL(v.dot(np.linalg.matrix_power(P, move)))
        
        annot = np.zeros((10, 10))
        annot = np.array(["e" for _ in range(100)])
        annot = np.round(M, 2)
        annot = annot.astype("U")
        annot[annot == "0.0"] = "<0.01"
        annot = np.char.add(annot, "%")
        
        Clear_current_axes()
        sns.set_style(style='white')
        ax = sns.heatmap(M, annot=annot, cbar=False, linewidth=0.7, linecolor="black", fmt="",
                    xticklabels=False, yticklabels=False, mask=M==0, cmap=sns.color_palette("crest", as_cmap=True), annot_kws={"size": 10, "weight": "normal"})
        ax.set_xlim(-0.1, 10.1)
        ax.set_ylim(10.1, -0.1)
        ax.set_aspect(0.7)
    
    @render.plot
    def Probability_Vignette():
        move = Move_nb_V()
        n = input.Move_nb_V()
        v = np.zeros((1, n+1))
        v[0, 0] = 1
        P = Vignette_Matrix(n)
        p = 100 * v.dot(np.linalg.matrix_power(P, move))
        
        annot = np.round(p, 2).astype("U")
        annot[annot == "0.0"] = "<0.01"
        annot = np.char.add(annot, "%")
        
        Clear_current_axes()
        sns.set_style(style='white')
        ax = sns.heatmap(p, annot=annot, cbar=False, linewidth=.5, linecolor="black", fmt="",
                    xticklabels=False, yticklabels=False, mask=p==0, cmap=sns.color_palette("crest", as_cmap=True))
        ax.set_xlim(0, n + 1.1)
        ax.set_ylim(-1, 1.1)
        for i in range(n+1):
            ax.text(0.4 + i, -0.3, str(i))
        ax.set_aspect(0.7)
    
    @render.image
    def Image_SL():
        img = {"src": Path(dname + "/www") / "snakesandladders.png", "width": "600px"}  
        return img
    
    @render.plot
    def Probability_Monopoly():
        move = Move_nb_M()
        P = Monopoly_Matrix()
        if Infinity_M():
            M = 100 * Get_Stationnary(P)
        else:
            v = np.zeros((1, 123))
            v[0, 0] = 1
            M = 100 * v.dot(np.linalg.matrix_power(P, move))
            M = M.flatten()
        M2 = M[:40] + M[40:80] + M[80:120]
        M2 = np.append(M2, np.sum(M[120:]))

        cmap = sns.color_palette("crest", n_colors=40)
        Clear_current_axes()
        fig, ax = Make_board_M(M2, cmap, debug=False)
    
    @render.image
    def Image_M():
        img = {"src": Path(dname + "/www") / "Monopoly.jpg", "width": "600px"}  
        return img

app = App(app_ui, server)

