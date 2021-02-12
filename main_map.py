import numpy as np
import time
import cv2
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

def pathfind(my_img, start, end):

    arr = ['Main_entrance','entrance_nearXerox','entrance_nextXerox','entrance_nearAuditorium','Auditorium','Lab1','Lab2','dining']
    main_arr = [i.upper() for i in arr]

    if(start.upper() in main_arr and end.upper() in main_arr):

        matrix=[
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,1,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],

        ]
        

        # MAtrix dimension : 16x50 :: indexes- 15 : 49

        co_ordinates = [
            [ [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0] ],
            [ [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0] ],
            [ [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0] ],
            [ [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0] ],
            [ [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0] ],
            [ [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0] ],
            [ [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0] ],
            [ [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0] ],
            [ [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0] ],
            [ [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0] ],
            [ [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0] ],
            [ [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0] ],
            [ [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0] ],
            [ [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0] ],
            [ [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0] ],
            [ [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0] ]
        ]
        
        # DEfining all the 1 points:
        co_ordinates[15][49] = [1450,650]
        co_ordinates[15][48] = [1450,650]
        co_ordinates[15][47] = [1450,650]
        co_ordinates[15][46] = [1450,650]
        co_ordinates[15][45] = [1250,650]
        co_ordinates[14][45] = [1250,650]
        co_ordinates[13][45] = [1250,580]
        co_ordinates[15][44] = [1250,650]
        co_ordinates[15][43] = [1250,650]
        co_ordinates[15][42] = [1250,650]
        co_ordinates[15][41] = [1100,650]
        co_ordinates[14][41] = [1100,650]
        co_ordinates[13][41] = [1100,580]
        co_ordinates[15][40] = [1100,650]
        co_ordinates[15][39] = [1100,650]
        co_ordinates[15][38] = [1100,650]
        co_ordinates[15][37] = [1100,650]
        co_ordinates[15][36] = [850,650]
        co_ordinates[14][36] = [850,650]
        co_ordinates[13][36] = [850,650]
        co_ordinates[12][36] = [850,650]
        co_ordinates[11][36] = [850,650]
        co_ordinates[10][36] = [850,405]
        co_ordinates[10][37] = [850,405]
        co_ordinates[10][38] = [900,405]
        co_ordinates[15][35] = [850,650]
        co_ordinates[15][34] = [850,650]
        co_ordinates[15][33] = [850,650]
        co_ordinates[15][32] = [850,650]
        co_ordinates[15][31] = [720,650]
        co_ordinates[14][31] = [680,650]
        co_ordinates[13][31] = [680,510]
        co_ordinates[15][30] = [680,650]
        co_ordinates[15][29] = [680,650]
        co_ordinates[15][28] = [680,650]
        co_ordinates[15][27] = [450,650]
        co_ordinates[14][27] = [450,650]
        co_ordinates[13][27] = [450,650]
        co_ordinates[12][27] = [450,650]
        co_ordinates[12][26] = [450,500]
        co_ordinates[12][25] = [400,500]
        co_ordinates[15][26] = [400,650]
        co_ordinates[15][25] = [400,650]
        co_ordinates[15][24] = [400,650]
        co_ordinates[15][23] = [400,650]
        co_ordinates[15][22] = [400,650]
        co_ordinates[15][21] = [400,650]
        co_ordinates[15][20] = [400,650]
        co_ordinates[15][19] = [400,650]
        co_ordinates[15][18] = [400,650]
        co_ordinates[15][17] = [400,650]
        co_ordinates[15][16] = [400,650]
        co_ordinates[15][15] = [400,650]
        co_ordinates[15][14] = [400,650]
        co_ordinates[15][13] = [400,650]
        co_ordinates[15][12] = [400,650]
        co_ordinates[15][11] = [400,650]
        co_ordinates[15][10] = [400,650]
        co_ordinates[15][9] = [400,650]
        co_ordinates[15][8] = [400,650]
        co_ordinates[15][7] = [400,650]
        co_ordinates[15][6] = [400,650]
        co_ordinates[15][5] = [400,650]
        co_ordinates[15][4] = [400,650]
        co_ordinates[15][3] = [400,650]
        co_ordinates[15][2] = [400,650]
        co_ordinates[15][1] = [400,650]
        co_ordinates[15][0] = [20,650]
        co_ordinates[11][27] = [450,650]
        co_ordinates[10][27] = [450,500]
        co_ordinates[9][27] = [450,500]
        co_ordinates[8][27] = [450,500]
        co_ordinates[7][27] = [450,250]
        co_ordinates[7][26] = [450,250]
        co_ordinates[7][25] = [300,250]
        co_ordinates[6][25] = [300,250]
        co_ordinates[5][25] = [300,200]
        co_ordinates[5][24] = [300,200]
        co_ordinates[5][23] = [260,200]
        co_ordinates[6][27] = [450,200]
        co_ordinates[5][27] = [450,200]
        co_ordinates[5][28] = [450,200]
        co_ordinates[5][29] = [450,200]
        co_ordinates[5][30] = [450,200]
        co_ordinates[5][31] = [450,200]
        co_ordinates[5][32] = [450,200]
        co_ordinates[5][33] = [450,200]
        co_ordinates[5][34] = [450,200]
        co_ordinates[5][35] = [450,200]
        co_ordinates[5][36] = [850,200]
        co_ordinates[6][36] = [850,200]    
        co_ordinates[7][36] = [850,200]    
        co_ordinates[8][36] = [850,200]    
        co_ordinates[9][36] = [850,200]
        co_ordinates[10][36] = [850,405]    
        co_ordinates[10][37] = [850,405]    
        co_ordinates[10][38] = [900,405]
        co_ordinates[5][37] = [850,200]
        co_ordinates[5][38] = [850,200]
        co_ordinates[5][39] = [1000,200]
        co_ordinates[4][39] = [1000,200]
        co_ordinates[3][39] = [1000,100]
        co_ordinates[5][40] = [1000,200]
        co_ordinates[5][41] = [1100,200]
        co_ordinates[5][42] = [1110,200]
        co_ordinates[5][43] = [1140,200]
        co_ordinates[5][44] = [1160,200]
        co_ordinates[5][45] = [1170,200]
        co_ordinates[5][46] = [1180,200]
        co_ordinates[5][47] = [1190,200]
        co_ordinates[5][48] = [1200,200]
        co_ordinates[5][49] = [1450,200]
        co_ordinates[6][49] = [1450,210]
        co_ordinates[7][49] = [1450,230]
        co_ordinates[8][49] = [1450,250]
        co_ordinates[9][49] = [1450,280]
        co_ordinates[10][49] = [1450,380]
        co_ordinates[10][48] = [1450,380]
        co_ordinates[10][47] = [1350,380]
        co_ordinates[11][49] = [1450,400]
        co_ordinates[12][49] = [1450,440]
        co_ordinates[13][49] = [1450,460]
        co_ordinates[14][49] = [1450,480]
        co_ordinates[15][49] = [1450,650]


        grid = Grid(matrix=matrix)
        
        #routes:
        Main_entrance = grid.node(47,10)
        entrance_nearXerox = grid.node(45,13)
        entrance_nextXerox = grid.node(41,13)
        entrance_nearAuditorium = grid.node(38,10)
        Auditorium = grid.node(31,13)
        Lab1 = grid.node(25,12)
        Lab2 = grid.node(23,5)
        dining = grid.node(39,3)

        if(start.upper() == "main_entrance".upper()):
            start = Main_entrance
        elif(start.upper() == "entrance_nearxerox".upper()):
            start = entrance_nearXerox
        elif(start.upper() == "entrance_nextxerox".upper()):
            start = entrance_nextXerox
        elif(start.upper() == "entrance_nearauditorium".upper()):
            start = entrance_nearAuditorium
        elif(start.upper() == "auditorium".upper()):
            start = Auditorium
        elif(start.upper() == "lab1".upper()):
            start = Lab1
        elif(start.upper() == "lab2".upper()):
            start = Lab2
        elif(start.upper() == "dining".upper()):
            start = dining
        
        if(end.upper() == "main_entrance".upper()):
            end = Main_entrance
        elif(end.upper() == "entrance_nearxerox".upper()):
            end = entrance_nearXerox
        elif(end.upper() == "entrance_nextxerox".upper()):
            end = entrance_nextXerox
        elif(end.upper() == "entrance_nearauditorium".upper()):
            end = entrance_nearAuditorium
        elif(end.upper() == "auditorium".upper()):
            end = Auditorium
        elif(end.upper() == "lab1".upper()):
            end = Lab1
        elif(end.upper() == "lab2".upper()):
            end = Lab2
        elif(end.upper() == "dining".upper()):
            end = dining

        finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
        path, runs = finder.find_path(start, end, grid)

        for i in range(len(path)-1):
            my_img = cv2.line(my_img,tuple(co_ordinates[path[i][1]][path[i][0]]),tuple(co_ordinates[path[i+1][1]][path[i+1][0]]),(0,0,255),8)

        my_img = cv2.circle(my_img, tuple(co_ordinates[path[len(path)-1][1]][path[len(path)-1][0]]), 20, (0,255,0), -1)
        end_x = tuple(co_ordinates[path[len(path)-1][1]][path[len(path)-1][0]])[0]
        end_y = tuple(co_ordinates[path[len(path)-1][1]][path[len(path)-1][0]])[1]
        my_img = cv2.putText(my_img, 'END', (end_x-25,end_y-25), cv2.FONT_HERSHEY_SIMPLEX ,  
                    1, (255,0,0), 3, cv2.LINE_AA)

        my_img = cv2.circle(my_img, tuple(co_ordinates[path[0][1]][path[0][0]]), 20, (0,0,255), -1)
        start_x = tuple(co_ordinates[path[0][1]][path[0][0]])[0]
        start_y = tuple(co_ordinates[path[0][1]][path[0][0]])[1]
        my_img = cv2.putText(my_img, 'START', (start_x-25,start_y-25), cv2.FONT_HERSHEY_SIMPLEX ,
                    1, (255,0,0), 3, cv2.LINE_AA)

        print(path)
    else:
        print("please Enter valid Location name")

"""
my_img = np.ones((700, 1500, 3), dtype = "uint8")*255

my_img = cv2.line(my_img,(1450,50),(1450,700),(0,0,0),30)
my_img = cv2.line(my_img,(1450,650),(20,650),(0,0,0),30)
my_img = cv2.line(my_img,(1435,380),(1350,380),(0,0,0),30)
#class:
my_img = cv2.rectangle(my_img,(900,230),(1350,580),(0,0,0),5)
my_img = cv2.putText(my_img, 'Class', (1080,405), cv2.FONT_HERSHEY_SIMPLEX ,  
                    1, (0,0,0), 3, cv2.LINE_AA)

my_img = cv2.line(my_img,(1250,650),(1250,580),(0,0,0),30)
my_img = cv2.line(my_img,(1100,650),(1100,580),(0,0,0),30)
my_img = cv2.line(my_img,(680,650),(680,510),(0,0,0),30)
#auditorium:
my_img = cv2.rectangle(my_img,(500,230),(800,510),(0,0,0),5)
my_img = cv2.putText(my_img, 'Auditorium', (580,340), cv2.FONT_HERSHEY_SIMPLEX ,  
                    1, (0,0,0), 3, cv2.LINE_AA)

my_img = cv2.line(my_img,(850,650),(850,200),(0,0,0),30)
my_img = cv2.line(my_img,(900,405),(850,405),(0,0,0),30)
my_img = cv2.line(my_img,(450,650),(450,20),(0,0,0),30)
#Lab1:
my_img = cv2.rectangle(my_img,(40,600),(400,300),(0,0,0),5)
my_img = cv2.putText(my_img, 'Lab-1', (180,450), cv2.FONT_HERSHEY_SIMPLEX ,  
                    1, (0,0,0), 3, cv2.LINE_AA)

my_img = cv2.line(my_img,(450,500),(400,500),(0,0,0),30)
my_img = cv2.line(my_img,(450,250),(300,250),(0,0,0),30)
my_img = cv2.line(my_img,(300,250),(300,200),(0,0,0),30)
my_img = cv2.line(my_img,(300,200),(260,200),(0,0,0),30)
#lab2:
my_img = cv2.rectangle(my_img,(40,270),(260,20),(0,0,0),5)
my_img = cv2.putText(my_img, 'Lab-2', (130,145), cv2.FONT_HERSHEY_SIMPLEX ,  
                    1, (0,0,0), 3, cv2.LINE_AA)

my_img = cv2.line(my_img,(450,200),(1450,200),(0,0,0),30)
#Mess:
my_img = cv2.rectangle(my_img,(700,20),(1400,100),(0,0,0),5)
my_img = cv2.putText(my_img, 'Dining', (1000,60), cv2.FONT_HERSHEY_SIMPLEX ,  
                    1, (0,0,0), 3, cv2.LINE_AA)

my_img = cv2.line(my_img,(1000,200),(1000,100),(0,0,0),30)
#Class Rooms:
my_img = cv2.rectangle(my_img,(910,240),(980,310),(0,0,0),5)
my_img = cv2.putText(my_img, '1', (940,280), cv2.FONT_HERSHEY_SIMPLEX ,  
                    1, (0,0,0), 3, cv2.LINE_AA)

my_img = cv2.rectangle(my_img,(1000,240),(1070,310),(0,0,0),5)
my_img = cv2.putText(my_img, '2', (1025,280), cv2.FONT_HERSHEY_SIMPLEX ,  
                    1, (0,0,0), 3, cv2.LINE_AA)

my_img = cv2.rectangle(my_img,(1090,240),(1160,310),(0,0,0),5)
my_img = cv2.putText(my_img, '3', (1115,280), cv2.FONT_HERSHEY_SIMPLEX ,  
                    1, (0,0,0), 3, cv2.LINE_AA)

my_img = cv2.rectangle(my_img,(1180,240),(1250,310),(0,0,0),5)
my_img = cv2.putText(my_img, '4', (1205,280), cv2.FONT_HERSHEY_SIMPLEX ,  
                    1, (0,0,0), 3, cv2.LINE_AA)

my_img = cv2.rectangle(my_img,(1270,240),(1340,310),(0,0,0),5)
my_img = cv2.putText(my_img, 'Toilet', (1290,275), cv2.FONT_HERSHEY_SIMPLEX ,  
                    0.5, (0,0,0), 1, cv2.LINE_AA)

my_img = cv2.rectangle(my_img,(910,500),(980,570),(0,0,0),5)
my_img = cv2.putText(my_img, '9', (940,540), cv2.FONT_HERSHEY_SIMPLEX ,  
                    1, (0,0,0), 3, cv2.LINE_AA)

my_img = cv2.rectangle(my_img,(1000,500),(1070,570),(0,0,0),5)
my_img = cv2.putText(my_img, '8', (1030,540), cv2.FONT_HERSHEY_SIMPLEX ,  
                    1, (0,0,0), 3, cv2.LINE_AA)

my_img = cv2.rectangle(my_img,(1130,500),(1200,570),(0,0,0),5)
my_img = cv2.putText(my_img, '7', (1160,540), cv2.FONT_HERSHEY_SIMPLEX ,  
                    1, (0,0,0), 3, cv2.LINE_AA)

my_img = cv2.rectangle(my_img,(1270,500),(1340,570),(0,0,0),5)
my_img = cv2.putText(my_img, 'Xerox', (1290,540), cv2.FONT_HERSHEY_SIMPLEX ,  
                    0.5, (0,0,0), 1, cv2.LINE_AA)

"""
my_img = cv2.imread('campus.png')

print("\n\nWelcome to the PathFinding app:".upper())
time.sleep(2)
print("\nplease type your starting location and destination name from the list of available routes in our campus as shown below: ".upper())
print("\nMain_entrance , entrance_nearXerox , entrance_nextXerox , entrance_nearAuditorium , Auditorium , Lab1 , Lab2 , dining")
time.sleep(2)
#print("\nenter your stating point: ",end='')
starting_point = "main_entrance"
print('\ncurrent location : You are now in the MAIN ENTRANCE')
print("\nenter your destination: ",end='')
ending_point = input()

print(pathfind(my_img,starting_point,ending_point))
cv2.imshow('map',my_img)
cv2.waitKey(0)
cv2.destroyAllWindows()