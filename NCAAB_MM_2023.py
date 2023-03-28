## LIBRARY IMPORTS ##
import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import numpy as np

import plotly as ply
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from PIL import Image
import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# from streamlit_aggrid import AgGrid

# import matplotlib.pyplot as plt
# import seaborn as sns
# import dash as dash
# from dash import dash_table
# from dash import dcc
# from dash import html
# from dash.dependencies import Input, Output
# from dash.exceptions import PreventUpdate
# import dash_bootstrap_components as dbc

# import scipy.stats as stats
# import statistics

## DIRECTORY CONFIGURATION ##
abs_path = r'https://raw.githubusercontent.com/nehat312/march-madness-2023/main'
mm_database = abs_path + '/data/mm_2023_database.xlsx'

## DATA IMPORT ##
mm_database_2023 = pd.read_xlsx(mm_database, index_col='TEAM', header=0, sheet_name='TR')

## PRE-PROCESSING ##
# players_path.sort_values(by='disc_year', inplace=True)

# ## TIME INTERVALS ##
# today = datetime.date.today()
# before = today - datetime.timedelta(days=1095) #700
# start_date = '2003-01-01'
# end_date = today


## IMAGE IMPORT ##

## FINAL FOUR LOGOS ##
NCAA_logo = Image.open('images/NCAA_logo1.png')
FAU_logo = Image.open('images/FAU_Owls.png')
Miami_logo = Image.open('images/Miami_Canes.png')
UConn_logo = Image.open('images/UConn_Huskies.png')
SDSU_logo = Image.open('images/east/CHI-Bulls.png')



## FORMAT / STYLE ##

## COLOR SCALES ##
Tropic = px.colors.diverging.Tropic
Blackbody = px.colors.sequential.Blackbody
BlueRed = px.colors.sequential.Bluered

Sunsetdark = px.colors.sequential.Sunsetdark
Sunset = px.colors.sequential.Sunset
Temps = px.colors.diverging.Temps
Tealrose = px.colors.diverging.Tealrose

Ice = px.colors.sequential.ice
Ice_r = px.colors.sequential.ice_r
Dense = px.colors.sequential.dense
Deep = px.colors.sequential.deep
PuOr = px.colors.diverging.PuOr
Speed = px.colors.sequential.speed
# IceFire = px.colors.diverging.
# YlOrRd = px.colors.sequential.YlOrRd
# Mint = px.colors.sequential.Mint
# Electric = px.colors.sequential.Electric

# pd.options.display.float_format = '${:,.2f}'.format
# pd.set_option('display.max_colwidth', 200)

## VISUALIZATION LABELS ##

all_cols = [
            ]

viz_cols = ['SEED', 'KenPom RANK', 'NET RANK',
            'WIN', 'LOSS',
            'WIN% ALL GM', 'WIN% CLOSE GM',
            'KenPom EM', #'AdjO', 'AdjD', #'KP_Rank', #'AdjT', 'Luck',
            'KenPom SOS EM', #'SOS OppO', 'SOS OppD', 'NCSOS Adj EM'
            'OFF EFF', 'DEF EFF',
            'AVG MARGIN', #'opponent-average-scoring-margin',
            'PTS/GM', 'OPP PTS/GM',
            'eFG%', 'OPP eFG%', 'TS%', 'OPP TS%',
            'AST/TO%', #'NET AST/TOV RATIO',
            'STOCKS/GM', 'STOCKS-TOV/GM',
            ]

scale_cols = [
            ]

team_df_cols = [
                ]

chart_labels = {'':'',
                'Team': 'TEAM', 'TEAM_KP': 'TEAM', 'TEAM_TR': 'TEAM',
                'Conference': 'CONFERENCE', 'Seed': 'SEED',
                'Win': 'WIN', 'Loss': 'LOSS',
                'win-pct-all-games': 'WIN%', 'win-pct-close-games': 'WIN%_CLOSE',

                'points-per-game': 'PTS/GM', 'average-scoring-margin': 'AVG_MARGIN',
                'effective-field-goal-pct': 'EFG%', 'true-shooting-percentage': 'TS%',
                'effective-possession-ratio': 'POSS%',
                'three-point-pct': '3P%', 'two-point-pct': '2P%', 'free-throw-pct': 'FT%',
                'field-goals-made-per-game': 'FGM/GM', 'field-goals-attempted-per-game': 'FGA/GM',
                'three-pointers-made-per-game': '3PM/GM', 'three-pointers-attempted-per-game': '3PA/GM',
                'offensive-efficiency': 'O_EFF', 'defensive-efficiency': 'D_EFF',
                'total-rebounds-per-game': 'TRB/GM', 'offensive-rebounds-per-game': 'ORB/GM',
                'defensive-rebounds-per-game': 'DRB/GM',
                'offensive-rebounding-pct': 'ORB%', 'defensive-rebounding-pct': 'DRB%',
                'total-rebounding-percentage': 'TRB%',
                'blocks-per-game': 'B/GM', 'steals-per-game': 'S/GM', 'assists-per-game': 'AST/GM',
                'turnovers-per-game': 'TO/GM',
                'assist--per--turnover-ratio': 'AST/TO', 'possessions-per-game': 'POSS/GM',
                'personal-fouls-per-game': 'PF/GM',
                'opponent-points-per-game': 'OPP_PTS/GM', 'opponent-average-scoring-margin': 'OPP_AVG_MARGIN',
                'opponent-effective-field-goal-pct': 'OPP_EFG%', 'opponent-true-shooting-percentage': 'OPP_TS%',
                'opponent-three-point-pct': 'OPP_3P%', 'opponent-two-point-pct': 'OPP_2P%',
                'opponent-free-throw-pct': 'OPP_FT%', 'opponent-shooting-pct': 'OPP_FG%',
                'opponent-assists-per-game': 'OPP_AST/GM', 'opponent-turnovers-per-game': 'OPP_TO/GM',
                'opponent-assist--per--turnover-ratio': 'OPP_AST/TO',
                'opponent-offensive-rebounds-per-game': 'OPP_OREB/GM',
                'opponent-defensive-rebounds-per-game': 'OPP_DREB/GM',
                'opponent-total-rebounds-per-game': 'OPP_TREB/GM',
                'opponent-offensive-rebounding-pct': 'OPP_OREB%', 'opponent-defensive-rebounding-pct': 'OPP_DREB%',
                'opponent-blocks-per-game': 'OPP_BLK/GM', 'opponent-steals-per-game': 'OPP_STL/GM',
                'opponent-effective-possession-ratio': 'OPP_POSS%',
                'net-avg-scoring-margin': 'NET_AVG_MARGIN', 'net-points-per-game': 'NET_PTS/GM',
                'net-adj-efficiency': 'NET_ADJ_EFF',
                'net-effective-field-goal-pct': 'NET_EFG%', 'net-true-shooting-percentage': 'NET_TS%',
                'stocks-per-game': 'STOCKS/GM', 'total-turnovers-per-game': 'TTL_TO/GM',
                'net-assist--per--turnover-ratio': 'NET_AST/TO',
                'net-total-rebounds-per-game': 'NET_TREB/GM', 'net-off-rebound-pct': 'NET_OREB%',
                'net-def-rebound-pct': 'NET_DREB%',
                # 'win-pct-all-games':'WIN%', 'win-pct-close-games':'CLOSE GM WIN%',
                # 'offensive-efficiency':'OFF EFF', 'defensive-efficiency':'DEF EFF',
                'Adj EM': 'KenPom ADJ EM', 'SOS Adj EM': 'KenPom SOS ADJ EM',
                'points-per-game': 'PTS/GM', 'opponent-points-per-game': 'OPP PTS/GM',
                'average-scoring-margin': 'AVG SCORE MARGIN', 'opponent-average-scoring-margin': 'OPP AVG SCORE MARGIN',
                'effective-field-goal-pct': 'eFG%', 'true-shooting-percentage': 'TS%',
                'opponent-effective-field-goal-pct': 'OPP eFG%', 'opponent-true-shooting-percentage': 'OPP TS%',
                'assist--per--turnover-ratio': 'AST_TOV_RATIO', 'NET AST/TOV RATIO': 'NET AST/TOV%',
                'STOCKS-per-game': 'STOCKS/GM', 'STOCKS-TOV-per-game': 'STOCKS-TOV/GM',
                'MMS STOCKS-TOV-per-game': 'STOCKS-TOV/GM', 'MMS Adj EM': 'KenPom ADJ EM',

                }

team_logos_dict = {'FAU OWLS':FAU_logo,
                   'MIAMI HURRICANES':Miami_logo,
                   'UCONN HUSKIES':UConn_logo,
                   'SDSU AZTECS':SDSU_logo,
                   }

#https://nbacolors.com/team/washington-wizards-color
#https://teamcolorcodes.com/nba-team-color-codes/

# team_colors_dict = {'FAU OWLS':['#E03A3E','#C1D32F','#26282A'],
#                     'MIAMI HURRICANES':['#000000','#FFFFFF'],
#                     'UCONN HUSKIES':['#007A33','#007A33'],
#                     'SDSU AZTECS':['#000000','#FFFFFF'],
#                   }

## COURT BACKGROUND ##
# court_img_dict = dict(source=court_img_1, xref="paper", yref="paper", x=0.5, y=0.5, sizex=2.5, sizey=1,
#                       xanchor="center", yanchor="middle", #left right  #top bottom
#                       opacity=.35, visible=True, layer="below", # sizing="contain",
#                       )

## FEATURED VARIABLES ##

# team_list = list(champion_players['TEAM'].unique())

team_logos_list = []

eastconf_logos_list = []

westconf_logos_list = []


# champion_players['LOGO'] = champion_players.TEAM.map(team_logos_dict)


#%%
# # group_cols = ['']
# college_raptor = champion_players.groupby(['TEAM', 'COLLEGE']).mean()
# print(college_raptor)

# Keep? Functional?
# champion_players['LOGO'] = champion_players.TEAM.map(team_logos_dict)


## FILTER DATA ##
mm_database_2023 = mm_database_2023[viz_cols]

# champion_players = champion_players[champion_players['MP'] > 350] #175
# lebron_val_players = champion_players[champion_players['YEAR'] >= 2010]


## TO DO / NOTES

#**#value per dollar is the key AKA secret sauce *** #
#DK PTS/MIN
## TABLEAU ##
## NEO4J / MONGO ?? ##
## PAGES OR TABS FOR EACH ROSTER???
## LOG TRANSFORM??
## PCA From test code



# MinMaxScaler

# ss = StandardScaler()
# mms = MinMaxScaler()

# scaled_MP = ss.fit_transform(champion_players['MP'])
# scaled_MP = scaled_MP.reshape(-1,1)
# champion_players['ss_MP'] = scaled_MP
# print(champion_players['ss_MP'])


## GROUP BY CHAMPIONSHIP TEAM

# champion_teams = champion_players[team_df_cols]

champ_df_list = [
                 ]

for df in champ_df_list:
  df = df.drop(columns=['CHAMP', 'PLAYER'], inplace=True)

#%%
## AGG GRID
# grid = AgGrid(df)


#%%
## VISUALIZATIONS ##

####################################################################################################################

## BOX PLOTS ##

####################################################################################################################

## BAR CHARTS ##
bar_usg_salary = px.bar(data_frame=champion_players,
                              x=champion_players['CHAMP'],
                              y=champion_players['SALARY'],
                              color=champion_players['USG%'],     # EXPERIENCE AGE MP APE
                              color_continuous_scale=Tropic,
                              color_discrete_sequence=Tropic,
                              # color_continuous_midpoint=10,
                              # color_discrete_map=team_logos_dict,
                              hover_name=champion_players['PLAYER'],
                              hover_data=champion_players[['CHAMP', 'SALARY', 'MP']], #'WS/$',
                              barmode='group',
                              title='USG% RELATIVE TO TEAM SALARY',
                              labels=chart_labels,
                              # template='simple_white+gridon',
                              # range_x=[1991,2023],
                              # range_y=[0,200000000],
                              height=750,
                            # category_orders={"InternetService": ["DSL", "Fiber optic", "No"],
#                               "gender": ["Female", "Male"]})
                              )

bar_WS_salary = px.bar(data_frame=champion_players,
                              x=champion_players['CHAMP'],
                              y=champion_players['SALARY'],
                              color=champion_players['WS'],     # EXPERIENCE AGE MP APE
                              color_continuous_scale=Tropic,
                              color_discrete_sequence=Tropic,
                              color_continuous_midpoint=10,
                              # color_discrete_map=team_logos_dict,
                              hover_name=champion_players['PLAYER'],
                              hover_data=champion_players[['CHAMP', 'SALARY', 'MP', 'WS']], #'WS/$',
                              barmode='group',
                              title='WIN SHARES (WS) METRIC RELATIVE TO CHAMPIONSHIP TEAM SALARY',
                              labels=chart_labels,
                              # template='simple_white+gridon',
                              # range_x=[1991,2023],
                              # range_y=[0,200000000],
                              height=750,
                              # width=1000,
                              )

bar_raptor_salary = px.bar(data_frame=champion_players,
                              x=champion_players['CHAMP'],
                              y=champion_players['SALARY'],
                              color=champion_players['RAPTOR'],     # EXPERIENCE AGE MP APE
                              color_continuous_scale=Tropic,
                              color_discrete_sequence=Tropic,
                              color_continuous_midpoint=0,
                              # color_discrete_map=team_logos_dict,
                              hover_name=champion_players['PLAYER'],
                              hover_data=champion_players[['CHAMP', 'SALARY', 'MP', 'RAPTOR']],
                              barmode='group',
                              title='RAPTOR METRIC RELATIVE TO CHAMPIONSHIP TEAM SALARY',
                              labels=chart_labels,
                              # template='simple_white+gridon',
                              # range_x=[1991,2022],
                              # range_y=[0,200000000],
                              height=750,
                              # width=1000,
                              )

bar_lebron_salary = px.bar(data_frame=lebron_val_players,
                              x=lebron_val_players['CHAMP'],
                              y=lebron_val_players['SALARY'],
                              color=lebron_val_players['LEBRON'],     # EXPERIENCE AGE MP APE
                              color_continuous_scale=Tropic,
                              color_discrete_sequence=Tropic,
                              # color_continuous_midpoint=10,
                              # color_discrete_map=team_logos_dict,
                              hover_name=lebron_val_players['PLAYER'],
                              hover_data=lebron_val_players[['CHAMP', 'SALARY', 'MP', 'LEBRON']],
                              barmode='group',
                              title='LEBRON METRIC RELATIVE TO CHAMPIONSHIP TEAM SALARY',
                              labels=chart_labels,
                              # template='simple_white+gridon',
                              # range_x=[1991,2022],
                              # range_y=[0,200000000],
                              height=750,
                              # width=1000,
                              )

bar_eWINS_WS = px.bar(data_frame=champion_players,
                              x=champion_players['CHAMP'],
                              y=champion_players['SALARY'],
                              color=champion_players['$MM/PlrWS'],     # EXPERIENCE AGE MP APE
                              color_continuous_scale=Tropic,
                              color_discrete_sequence=Tropic,
                              # color_continuous_midpoint=10,
                              # color_discrete_map=team_logos_dict,
                              hover_name=champion_players['PLAYER'],
                              hover_data=champion_players[['CHAMP', 'SALARY', 'MP']], #'WS/$',
                              barmode='group',
                              title='$MM / PLAYER WS RELATIVE TO TEAM SALARY',
                              labels=chart_labels,
                              # template='simple_white+gridon',
                              # range_x=[1991,2023],
                              # range_y=[0,200000000],
                              height=750,
                            # category_orders={"InternetService": ["DSL", "Fiber optic", "No"],
#                               "gender": ["Female", "Male"]})
                              )



####################################################################################################################

## SCATTER MATRIX ##
scatter_matrix_metrics = px.scatter_matrix(champion_players,
                                         dimensions=['USG%', 'TS%', 'AST%/TO%', 'STOCK%', 'RAPTOR', 'LEBRON', 'WS', ],
                                         color=champion_players['WTD POS'],
                                         color_continuous_scale=Dense,
                                         color_discrete_sequence=Dense,
                                           # symbol=champion_players['TEAM'],
                                           # # symbol_sequence=team_logos_dict,
                                           # symbol_map=team_logos_dict,
                                         # color_discrete_map=team_logos_dict,
                                         hover_name=champion_players['PLAYER'],
                                         hover_data=champion_players[['MP', 'CHAMP']],
                                         title='PLAYER PERFORMANCE BY CHAMPIONSHIP TEAM',
                                         labels=chart_labels,
                                         # custom_data= [league_logo_list],
                                         height=800,
                                         # width=800,
                                         )

scatter_matrix_measurables = px.scatter_matrix(champion_players,
                                         dimensions=['BMI', 'APE', 'WS', 'RAPTOR', 'LEBRON'],
                                         color=champion_players['WTD POS'],
                                         color_continuous_scale=Dense,
                                         color_discrete_sequence=Dense,
                                         # color_discrete_map=team_logos_dict,
                                         hover_name=champion_players['PLAYER'],
                                         hover_data=champion_players[['MP', 'CHAMP']],
                                         title='PLAYER PERFORMANCE BY CHAMPIONSHIP TEAM',
                                         labels=chart_labels,
                                         # custom_data= [league_logo_list],
                                         height=800,
                                         # width=800,
                                         )


####################################################################################################################

## SCATTER TERNARY ##

scatter_ternary_stl_blk_ast_to = px.scatter_ternary(data_frame=champion_players,
                                       a=champion_players['STL%'],
                                       b=champion_players['BLK%'],
                                       c=champion_players['AST%/TO%'],
                                       color=champion_players['WTD POS'],
                                        color_discrete_sequence=Dense,
                                       color_continuous_scale=Dense,
                                        color_continuous_midpoint=3,
                                       # symbol=champion_players['TEAM'], #'RD POS'
                                       #              symbol_map=team_logos_dict,
                                       size=champion_players['TS%'],
                                       size_max=20,
                                       opacity=.8,
                                       title='NBA CHAMPIONS -- STL% - BLK% - AST%/TO%',
                                       hover_name=champion_players['PLAYER'],
                                       hover_data=champion_players[['CHAMP', 'SALARY', 'MP',]],
                                       labels=chart_labels,
                                       height=900,
                                       )

scatter_ternary_ast_to_usg = px.scatter_ternary(data_frame=champion_players,
                                       a=champion_players['AST%'],
                                       b=champion_players['TO%'],
                                       c=champion_players['USG%'],
                                       color=champion_players['WTD POS'],
                                                color_discrete_sequence=Dense,
                                                color_continuous_scale=Dense,
                                                color_continuous_midpoint=3,
                                                symbol=champion_players['RD POS'],
                                                size=champion_players['TS%'],
                                                size_max=20,
                                       opacity=.8,
                                       title='NBA CHAMPIONS -- TS% - USG% - STOCK%',
                                       hover_name=champion_players['PLAYER'],
                                       hover_data=champion_players[['CHAMP', 'SALARY', 'MP', 'STL%', 'BLK%']],
                                       labels=chart_labels,
                                       height=900,
                                       )


####################################################################################################################

## SCATTER 3D ##

scatter_3D_to_ast_usg = px.scatter_3d(data_frame=champion_players,
                                     x=champion_players['TO%'],
                                     y=champion_players['AST%'],
                                     z=champion_players['USG%'],
                                     color=champion_players['WTD POS'],
                                      symbol=champion_players['RD POS'],
                                     color_discrete_sequence=Dense,
                                     color_continuous_scale=Dense,
                                     color_continuous_midpoint=3,
                                     title='NBA CHAMPIONS -- TO% / AST% / USG% BY POSITION',
                                     hover_name=champion_players['PLAYER'],
                                     hover_data=champion_players[['CHAMP']], #'LOGO'
                                     # 'HEIGHT (IN)' 'WEIGHT (LBS)' 'BMI' 'W-SPAN (IN)'
                                     # custom_data=['LOGO'],
                                     # size=champion_players['WS'],
                                     # size_max=50,
                                     labels=chart_labels,
                                     # range_x=[0,360],
                                     # range_y=[-50,50],
                                     # range_z=[0,2500],
                                     # range_color=Sunsetdark,
                                     opacity=.8,
                                     height=1000,
                                     # width=1000,
                                     )

#%%

## HISTORICAL LINE CHARTS
box_eWINS_WS = px.bar(data_frame=champion_players,
                              x=champion_players['CHAMP'],
                              y=champion_players['$MM/eWIN'], #, '$MM/TmWIN']
                              color=champion_players['WTD POS'],     # EXPERIENCE AGE MP APE
                              color_discrete_sequence=Tropic,
                      color_continuous_scale=Dense,
                              # color_continuous_midpoint=10,
                              # color_discrete_map=team_logos_dict,
                              hover_name=champion_players['PLAYER'],
                              hover_data=champion_players[['CHAMP',]], #'WS/$',
                              title='$MM / PLAYER WS RELATIVE TO TEAM SALARY',
                              labels=chart_labels,
                              # template='simple_white+gridon',
                              # range_x=[1991,2023],
                              # range_y=[0,200000000],
                              height=750,
                            # category_orders={"InternetService": ["DSL", "Fiber optic", "No"],
#                               "gender": ["Female", "Male"]})
                              )

#%%

####################################################################################################################
## TRACE LINES ##

line_NBA_salary = go.Scatter(x=champion_players['CHAMP'], y=champion_players['NBA TM AVG SAL'],
                             line_color='#000000', mode='lines')

line_NBA_eWIN = go.Scatter(x=champion_players['CHAMP'], y=champion_players['$MM/eWIN'],
                             line_color='#000000', mode='lines')

####################################################################################################################

## LOGO OVERLAY
# for x,y, png in zip(fig.data[0].x, fig.data[0].y, Path.cwd().joinpath("nfl-logos").glob("*.png")):
#     fig.add_layout_image(
#         x=x,
#         y=y,
#         source=Image.open(png),
#         xref="x",
#         yref="y",
#         sizex=2,
#         sizey=2,
#         xanchor="center",
#         yanchor="middle",
#     )

####################################################################################################################

## DENSITY MAP
##

####################################################################################################################
#%%

#####################
### STREAMLIT APP ###
#####################

## CONFIGURATION ##
st.set_page_config(page_title='COMPOSITION OF NBA CHAMPIONS', layout='wide', initial_sidebar_state='auto') #, page_icon=":smirk:"

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """

st.markdown(hide_menu_style, unsafe_allow_html=True)

## CSS LAYOUT CUSTOMIZATION ##

th_props = [('font-size', '12px'),
            ('text-align', 'center'),
            ('font-weight', 'bold'),
            ('color', '#EBEDE9'), #6d6d6d #29609C
            ('background-color', '#29609C') #f7f7f9
            ]

td_props = [('font-size', '12px'),
            # ('text-align', 'center'),
            # ('font-weight', 'bold'),
            # ('color', '#EBEDE9'), #6d6d6d #29609C
            # ('background-color', '#29609C') #f7f7f9
            ]

df_styles = [dict(selector="th", props=th_props),
             dict(selector="td", props=td_props)]


col_format_dict = {
    # 'WTD POS': "{:.2}",
    #                'BMI': "{:.1}",
    #                'W-SPAN (IN)': "{:.1}",
    #                'APE': "{:.1}",
    #                'USG%': "{:.1}",
    #                'TS%': "{:.1}",
    #                'AST%/TO%': "{:.1}",
    #                'STOCK%': "{:.1}",
    #                # 'USG%': "{:.1}",
    #
    #                'WS': "{:,}",

#                    'O-WS': "{:,}",
#                    'D-WS': "{:,}",

#                    'TM_WS': "{:,}",
                   # #: "{:.1%}", #:"{:.1}x", "${:.2}", #"${:,}"
                   }


## SIDEBAR ##
# sidebar_header = st.sidebar.subheader('DIRECTORY:')

# sector_sidebar_select = st.sidebar.selectbox('SECTOR', (sector_list_of_names), help='SELECT CRE SECTOR')
# ticker_sidebar_select = st.sidebar.selectbox('TICKER', (sector_dict['apartment'])) #sector_sidebar_select

# sidebar_start = st.sidebar.date_input('START DATE', before)
# sidebar_end = st.sidebar.date_input('END DATE', today)
# if sidebar_start < sidebar_end:
#     st.sidebar.success('START DATE: `%s`\n\nEND DATE: `%s`' % (sidebar_start, sidebar_end))
# else:
#     st.sidebar.error('ERROR: END DATE BEFORE START DATE')



## HEADER ##
st.container()

st.title('CHAMPIONSHIP-CALIBER NBA ROSTER CONSTRUCTION')
st.write('*STATISTICAL BREAKDOWN OF HISTORICAL AND MODERN-DAY NBA CHAMPIONSHIP ROSTERS*')

## EAST LOGOS ##
MM_col_0, MM_col_1, MM_col_2, MM_col_3, MM_col_4, = st.columns(5)

MM_col_0.image(NCAA_logo, caption='BKN', width=35)
MM_col_1.image(FAU_logo, caption='BOS', width=35)
MM_col_2.image(Miami_logo, caption='NYK', width=35)
MM_col_3.image(UConn_logo, caption='PHI', width=35)
MM_col_4.image(SDSU_logo, caption='PHI', width=35)

## WEST LOGOS ##
WN_col_1, WN_col_2, WN_col_3, WN_col_4, WN_col_5, \
WP_col_1, WP_col_2, WP_col_3, WP_col_4, WP_col_5, \
WS_col_1, WS_col_2, WS_col_3, WS_col_4, WS_col_5 = st.columns(15)
WN_col_1.image(DEN_logo, caption='DEN', width=35)
WN_col_2.image(MIN_logo, caption='MIN', width=35)
WN_col_3.image(POR_logo, caption='POR', width=35)
WN_col_4.image(OKC_logo, caption='OKC', width=35)


tab_0, tab_1, tab_2, tab_3, tab_4, tab_5, tab_6, tab_7, tab_8, tab_9, tab_10, \
tab_11, tab_12, tab_13, tab_14, tab_15, tab_16, tab_17, tab_18, tab_19, tab_20, \
tab_21, tab_22, tab_23, tab_24, tab_25, tab_26, tab_27, tab_28, tab_29, tab_30, tab_31, tab_32, \
    = st.tabs(['NBA',
               '2022-GSW', '2021-MIL', '2020-LAL', '2019-TOR', '2018-GSW',
               '2017-GSW', '2016-CLE', '2015-GSW', '2014-SAS', '2013-MIA',
               '2012-MIA', '2011-DAL', '2010-LAL', '2009-LAL', '2008-BOS',
               '2007-SAS', '2006-MIA', '2005-SAS', '2004-DET', '2003-SAS',
               '2002-LAL', '2001-LAL', '2000-LAL', '1999-SAS', '1998-CHI',
               '1997-CHI', '1996-CHI', '1995-HOU', '1994-HOU', '1993-CHI',
               '1992-CHI', '1991-CHI'
               ])

# 'ATL', 'BKN', 'BOS', 'CHA', 'CHI', 'CLE', 'DET', 'IND', 'MIA', 'MIL', 'NYK', 'ORL', 'PHI', 'TOR', 'WAS',
#                'DAL', 'DEN', 'HOU', 'GSW', 'LAC', 'LAL', 'MEM', 'MIN', 'NOP', 'OKC', 'PHX', 'POR', 'SAC', 'SAS', 'UTA',


with tab_0:
    # st.subheader('ALL SECTORS')
    st.plotly_chart(scatter_matrix_metrics, use_container_width=True, sharing="streamlit")

    ## BAR - WS SALARY ##
    st.plotly_chart(bar_WS_salary.add_traces(line_NBA_salary).add_layout_image(court_img_dict), use_container_width=True, sharing="streamlit")

    st.image(capstone_court, width=1000, use_column_width=True)

    ## SCATTER TERNARY ##

    # left, right = st.columns(2)
    # with left:
    #     st.plotly_chart(scatter_ternary_ast_to_usg, use_container_width=True, sharing="streamlit") #.add_layout_image(court_img_dict)
    # with right:
    #     st.plotly_chart(scatter_ternary_stl_blk_ast_to, use_container_width=True, sharing="streamlit") #.add_layout_image(court_img_dict)



    ## SCATTER MATRIX ##
    # st.plotly_chart(scatter_matrix_teams, use_container_width=True, sharing="streamlit")

    # st.plotly_chart(scatter_matrix_measurables, use_container_width=True, sharing="streamlit")


    # st.plotly_chart(box_eWINS_WS.add_traces(line_NBA_eWIN).add_layout_image(court_img_dict), use_container_width=True, sharing="streamlit")

    ## 3D SCATTER ##
    # st.plotly_chart(scatter_3D_to_ast_usg.add_layout_image(court_img_dict_3D), use_container_width=True, sharing="streamlit")



    ## LEAGUE LOGOS ##
    east_col_1, nba_col_2, west_col_3 = st.columns(3)
    east_col_1.image(East_logo, width=250) # caption='WESTERN CONFERENCE'
    nba_col_2.image(nba_logo_1, width=300) # caption='NATIONAL BASKETBALL ASSOCIATION'
    west_col_3.image(West_logo, width=250) # caption='EASTERN CONFERENCE'



    ## FORM FUNCTIONS ##
    # @st.cache(persist=True, allow_output_mutation=True, suppress_st_warning=True)


    ## DISCOVERY INFORMATION ##
    # st.plotly_chart(disc_info_1.update_yaxes(categoryorder='total ascending'), use_container_width=True, sharing="streamlit")


    ## EXTERNAL LINKS ##

    github_link = '[GITHUB REPOSITORY](https://github.com/nehat312/NBA-championship-caliber/)'
    nba_site_link = '[NBA.com](https://www.nba.com/)'
    bbref_site_link = '[BASKETBALL REFERENCE](https://www.basketball-reference.com/)'

    link_col_1, link_col_2, link_col_3 = st.columns(3)
    ext_link_1 = link_col_1.markdown(github_link, unsafe_allow_html=True)
    ext_link_2 = link_col_2.markdown(nba_site_link, unsafe_allow_html=True)
    ext_link_3 = link_col_3.markdown(bbref_site_link, unsafe_allow_html=True)


with tab_1:
    st.subheader('2021-2022 GOLDEN STATE WARRIORS')
    st.image(GSW_logo)
    st.dataframe(gsw_warriors_2022.style.format(col_format_dict).set_table_styles(df_styles))

    ## LEAGUE LOGOS ##
    east_col_1, nba_col_2, west_col_3 = st.columns(3)
    east_col_1.image(East_logo, width=250)  # caption='WESTERN CONFERENCE'
    nba_col_2.image(nba_logo_1, width=300)  # caption='NATIONAL BASKETBALL ASSOCIATION'
    west_col_3.image(West_logo, width=250)  # caption='EASTERN CONFERENCE'

with tab_2:
    st.subheader('2020-2021 MILWAUKEE BUCKS')
    st.image(MIL_logo)
    st.dataframe(mil_bucks_2021.style.format(col_format_dict).set_table_styles(df_styles))

    ## LEAGUE LOGOS ##
    east_col_1, nba_col_2, west_col_3 = st.columns(3)
    east_col_1.image(East_logo, width=250)  # caption='WESTERN CONFERENCE'
    nba_col_2.image(nba_logo_1, width=300)  # caption='NATIONAL BASKETBALL ASSOCIATION'
    west_col_3.image(West_logo, width=250)  # caption='EASTERN CONFERENCE'

with tab_3:
    st.subheader('2019-2020 LOS ANGELES LAKERS')
    st.image(LAL_logo)
    st.dataframe(lal_lakers_2020.style.format(col_format_dict).set_table_styles(df_styles))

    ## LEAGUE LOGOS ##
    east_col_1, nba_col_2, west_col_3 = st.columns(3)
    east_col_1.image(East_logo, width=250)  # caption='WESTERN CONFERENCE'
    nba_col_2.image(nba_logo_1, width=300)  # caption='NATIONAL BASKETBALL ASSOCIATION'
    west_col_3.image(West_logo, width=250)  # caption='EASTERN CONFERENCE'

with tab_4:
    st.subheader('2018-2019 TORONTO RAPTORS')
    st.image(TOR_logo)
    st.dataframe(tor_raptors_2019.style.format(col_format_dict).set_table_styles(df_styles))

    ## LEAGUE LOGOS ##
    east_col_1, nba_col_2, west_col_3 = st.columns(3)
    east_col_1.image(East_logo, width=250)  # caption='WESTERN CONFERENCE'
    nba_col_2.image(nba_logo_1, width=300)  # caption='NATIONAL BASKETBALL ASSOCIATION'
    west_col_3.image(West_logo, width=250)  # caption='EASTERN CONFERENCE'

with tab_5:
    st.subheader('2017-2018 GOLDEN STATE WARRIORS')
    st.image(GSW_logo)
    st.dataframe(gsw_warriors_2018.style.format(col_format_dict).set_table_styles(df_styles))

    ## LEAGUE LOGOS ##
    east_col_1, nba_col_2, west_col_3 = st.columns(3)
    east_col_1.image(East_logo, width=250)  # caption='WESTERN CONFERENCE'
    nba_col_2.image(nba_logo_1, width=300)  # caption='NATIONAL BASKETBALL ASSOCIATION'
    west_col_3.image(West_logo, width=250)  # caption='EASTERN CONFERENCE'

with tab_6:
    st.subheader('2016-2017 GOLDEN STATE WARRIORS')
    st.image(GSW_logo)
    st.dataframe(gsw_warriors_2017.style.format(col_format_dict).set_table_styles(df_styles))

    ## LEAGUE LOGOS ##
    east_col_1, nba_col_2, west_col_3 = st.columns(3)
    east_col_1.image(East_logo, width=250)  # caption='WESTERN CONFERENCE'
    nba_col_2.image(nba_logo_1, width=300)  # caption='NATIONAL BASKETBALL ASSOCIATION'
    west_col_3.image(West_logo, width=250)  # caption='EASTERN CONFERENCE'

with tab_7:
    st.subheader('2015-2016 CLEVELAND CAVALIERS')
    st.image(CLE_logo)
    st.dataframe(cle_cavs_2016.style.format(col_format_dict).set_table_styles(df_styles))

    ## LEAGUE LOGOS ##
    east_col_1, nba_col_2, west_col_3 = st.columns(3)
    east_col_1.image(East_logo, width=250)  # caption='WESTERN CONFERENCE'
    nba_col_2.image(nba_logo_1, width=300)  # caption='NATIONAL BASKETBALL ASSOCIATION'
    west_col_3.image(West_logo, width=250)  # caption='EASTERN CONFERENCE'

with tab_8:
    st.subheader('2014-2015 GOLDEN STATE WARRIORS')
    st.image(GSW_logo)
    st.dataframe(gsw_warriors_2015.style.format(col_format_dict).set_table_styles(df_styles))

    ## LEAGUE LOGOS ##
    east_col_1, nba_col_2, west_col_3 = st.columns(3)
    east_col_1.image(East_logo, width=250)  # caption='WESTERN CONFERENCE'
    nba_col_2.image(nba_logo_1, width=300)  # caption='NATIONAL BASKETBALL ASSOCIATION'
    west_col_3.image(West_logo, width=250)  # caption='EASTERN CONFERENCE'

with tab_9:
    st.subheader('2013-2014 SAN ANTONIO SPURS')
    st.image(SAS_logo)
    st.dataframe(sas_spurs_2014.style.format(col_format_dict).set_table_styles(df_styles))

    ## LEAGUE LOGOS ##
    east_col_1, nba_col_2, west_col_3 = st.columns(3)
    east_col_1.image(East_logo, width=250)  # caption='WESTERN CONFERENCE'
    nba_col_2.image(nba_logo_1, width=300)  # caption='NATIONAL BASKETBALL ASSOCIATION'
    west_col_3.image(West_logo, width=250)  # caption='EASTERN CONFERENCE'

with tab_10:
    st.subheader('2012-2013 MIAMI HEAT')
    st.image(MIA_logo)
    st.dataframe(mia_heat_2013.style.format(col_format_dict).set_table_styles(df_styles))

    ## LEAGUE LOGOS ##
    east_col_1, nba_col_2, west_col_3 = st.columns(3)
    east_col_1.image(East_logo, width=250)  # caption='WESTERN CONFERENCE'
    nba_col_2.image(nba_logo_1, width=300)  # caption='NATIONAL BASKETBALL ASSOCIATION'
    west_col_3.image(West_logo, width=250)  # caption='EASTERN CONFERENCE'

with tab_11:
    st.subheader('2011-2012 MIAMI HEAT')
    st.image(MIA_logo)
    st.dataframe(mia_heat_2012.style.format(col_format_dict).set_table_styles(df_styles))
    ## LEAGUE LOGOS ##
    east_col_1, nba_col_2, west_col_3 = st.columns(3)
    east_col_1.image(East_logo, width=250)  # caption='WESTERN CONFERENCE'
    nba_col_2.image(nba_logo_1, width=300)  # caption='NATIONAL BASKETBALL ASSOCIATION'
    west_col_3.image(West_logo, width=250)  # caption='EASTERN CONFERENCE'

with tab_12:
    st.subheader('2010-2011 DALLAS MAVERICKS')
    st.image(DAL_logo)
    st.dataframe(dal_mavs_2011.style.format(col_format_dict).set_table_styles(df_styles))

    ## LEAGUE LOGOS ##
    east_col_1, nba_col_2, west_col_3 = st.columns(3)
    east_col_1.image(East_logo, width=250)  # caption='WESTERN CONFERENCE'
    nba_col_2.image(nba_logo_1, width=300)  # caption='NATIONAL BASKETBALL ASSOCIATION'
    west_col_3.image(West_logo, width=250)  # caption='EASTERN CONFERENCE'

with tab_13:
    st.subheader('2009-2010 LOS ANGELES LAKERS')
    st.image(LAL_logo)
    st.dataframe(lal_lakers_2010.style.format(col_format_dict).set_table_styles(df_styles))

    ## LEAGUE LOGOS ##
    east_col_1, nba_col_2, west_col_3 = st.columns(3)
    east_col_1.image(East_logo, width=250)  # caption='WESTERN CONFERENCE'
    nba_col_2.image(nba_logo_1, width=300)  # caption='NATIONAL BASKETBALL ASSOCIATION'
    west_col_3.image(West_logo, width=250)  # caption='EASTERN CONFERENCE'

with tab_14:
    st.subheader('2008-2009 LOS ANGELES LAKERS')
    st.image(LAL_logo)
    st.dataframe(lal_lakers_2009.style.format(col_format_dict).set_table_styles(df_styles))

    ## LEAGUE LOGOS ##
    east_col_1, nba_col_2, west_col_3 = st.columns(3)
    east_col_1.image(East_logo, width=250)  # caption='WESTERN CONFERENCE'
    nba_col_2.image(nba_logo_1, width=300)  # caption='NATIONAL BASKETBALL ASSOCIATION'
    west_col_3.image(West_logo, width=250)  # caption='EASTERN CONFERENCE'

with tab_15:
    st.subheader('2007-2008 BOSTON CELTICS')
    st.image(BOS_logo)
    st.dataframe(bos_celtics_2008.style.format(col_format_dict).set_table_styles(df_styles))

    ## LEAGUE LOGOS ##
    east_col_1, nba_col_2, west_col_3 = st.columns(3)
    east_col_1.image(East_logo, width=250)  # caption='WESTERN CONFERENCE'
    nba_col_2.image(nba_logo_1, width=300)  # caption='NATIONAL BASKETBALL ASSOCIATION'
    west_col_3.image(West_logo, width=250)  # caption='EASTERN CONFERENCE'

with tab_16:
    st.subheader('2006-2007 SAN ANTONIO SPURS')
    st.image(SAS_logo)
    st.dataframe(sas_spurs_2007.style.format(col_format_dict).set_table_styles(df_styles))

    ## LEAGUE LOGOS ##
    east_col_1, nba_col_2, west_col_3 = st.columns(3)
    east_col_1.image(East_logo, width=250)  # caption='WESTERN CONFERENCE'
    nba_col_2.image(nba_logo_1, width=300)  # caption='NATIONAL BASKETBALL ASSOCIATION'
    west_col_3.image(West_logo, width=250)  # caption='EASTERN CONFERENCE'

with tab_17:
    st.subheader('2005-2006 MIAMI HEAT')
    st.image(MIA_logo)
    st.dataframe(mia_heat_2006.style.format(col_format_dict).set_table_styles(df_styles))

    ## LEAGUE LOGOS ##
    east_col_1, nba_col_2, west_col_3 = st.columns(3)
    east_col_1.image(East_logo, width=250)  # caption='WESTERN CONFERENCE'
    nba_col_2.image(nba_logo_1, width=300)  # caption='NATIONAL BASKETBALL ASSOCIATION'
    west_col_3.image(West_logo, width=250)  # caption='EASTERN CONFERENCE'

with tab_18:
    st.subheader('2004-2005 SAN ANTONIO SPURS')
    st.image(SAS_logo)
    st.dataframe(sas_spurs_2005.style.format(col_format_dict).set_table_styles(df_styles))

    ## LEAGUE LOGOS ##
    east_col_1, nba_col_2, west_col_3 = st.columns(3)
    east_col_1.image(East_logo, width=250)  # caption='WESTERN CONFERENCE'
    nba_col_2.image(nba_logo_1, width=300)  # caption='NATIONAL BASKETBALL ASSOCIATION'
    west_col_3.image(West_logo, width=250)  # caption='EASTERN CONFERENCE'

with tab_19:
    st.subheader('2003-2004 DETROIT PISTONS')
    st.image(DET_logo)
    st.dataframe(det_pistons_2004.style.format(col_format_dict).set_table_styles(df_styles))

    ## LEAGUE LOGOS ##
    east_col_1, nba_col_2, west_col_3 = st.columns(3)
    east_col_1.image(East_logo, width=250)  # caption='WESTERN CONFERENCE'
    nba_col_2.image(nba_logo_1, width=300)  # caption='NATIONAL BASKETBALL ASSOCIATION'
    west_col_3.image(West_logo, width=250)  # caption='EASTERN CONFERENCE'

with tab_20:
    st.subheader('2002-2003 SAN ANTONIO SPURS')
    st.image(SAS_logo)
    st.dataframe(sas_spurs_2003.style.format(col_format_dict).set_table_styles(df_styles))

    ## LEAGUE LOGOS ##
    east_col_1, nba_col_2, west_col_3 = st.columns(3)
    east_col_1.image(East_logo, width=250)  # caption='WESTERN CONFERENCE'
    nba_col_2.image(nba_logo_1, width=300)  # caption='NATIONAL BASKETBALL ASSOCIATION'
    west_col_3.image(West_logo, width=250)  # caption='EASTERN CONFERENCE'

with tab_21:
    st.subheader('2001-2002 LOS ANGELES LAKERS')
    st.image(LAL_logo)
    st.dataframe(lal_lakers_2002.style.format(col_format_dict).set_table_styles(df_styles))

    ## LEAGUE LOGOS ##
    east_col_1, nba_col_2, west_col_3 = st.columns(3)
    east_col_1.image(East_logo, width=250)  # caption='WESTERN CONFERENCE'
    nba_col_2.image(nba_logo_1, width=300)  # caption='NATIONAL BASKETBALL ASSOCIATION'
    west_col_3.image(West_logo, width=250)  # caption='EASTERN CONFERENCE'

with tab_22:
    st.subheader('2000-2001 LOS ANGELES LAKERS')
    st.image(LAL_logo)
    st.dataframe(lal_lakers_2001.style.format(col_format_dict).set_table_styles(df_styles))

    ## LEAGUE LOGOS ##
    east_col_1, nba_col_2, west_col_3 = st.columns(3)
    east_col_1.image(East_logo, width=250)  # caption='WESTERN CONFERENCE'
    nba_col_2.image(nba_logo_1, width=300)  # caption='NATIONAL BASKETBALL ASSOCIATION'
    west_col_3.image(West_logo, width=250)  # caption='EASTERN CONFERENCE'

with tab_23:
    st.subheader('1999-2000 LOS ANGELES LAKERS')
    st.image(LAL_logo)
    st.dataframe(lal_lakers_2000.style.format(col_format_dict).set_table_styles(df_styles))

    ## LEAGUE LOGOS ##
    east_col_1, nba_col_2, west_col_3 = st.columns(3)
    east_col_1.image(East_logo, width=250)  # caption='WESTERN CONFERENCE'
    nba_col_2.image(nba_logo_1, width=300)  # caption='NATIONAL BASKETBALL ASSOCIATION'
    west_col_3.image(West_logo, width=250)  # caption='EASTERN CONFERENCE'

with tab_24:
    st.subheader('1998-1999 SAN ANTONIO SPURS')
    st.image(SAS_logo)
    st.dataframe(sas_spurs_1999.style.format(col_format_dict).set_table_styles(df_styles))

    ## LEAGUE LOGOS ##
    east_col_1, nba_col_2, west_col_3 = st.columns(3)
    east_col_1.image(East_logo, width=250)  # caption='WESTERN CONFERENCE'
    nba_col_2.image(nba_logo_1, width=300)  # caption='NATIONAL BASKETBALL ASSOCIATION'
    west_col_3.image(West_logo, width=250)  # caption='EASTERN CONFERENCE'

with tab_25:
    st.subheader('1997-1998 CHICAGO BULLS')
    st.image(CHI_logo)
    st.dataframe(chi_bulls_1998.style.format(col_format_dict).set_table_styles(df_styles))

    ## LEAGUE LOGOS ##
    east_col_1, nba_col_2, west_col_3 = st.columns(3)
    east_col_1.image(East_logo, width=250)  # caption='WESTERN CONFERENCE'
    nba_col_2.image(nba_logo_1, width=300)  # caption='NATIONAL BASKETBALL ASSOCIATION'
    west_col_3.image(West_logo, width=250)  # caption='EASTERN CONFERENCE'

with tab_26:
    st.subheader('1996-1997 CHICAGO BULLS')
    st.image(CHI_logo)
    st.dataframe(chi_bulls_1997.style.format(col_format_dict).set_table_styles(df_styles))

    ## LEAGUE LOGOS ##
    east_col_1, nba_col_2, west_col_3 = st.columns(3)
    east_col_1.image(East_logo, width=250)  # caption='WESTERN CONFERENCE'
    nba_col_2.image(nba_logo_1, width=300)  # caption='NATIONAL BASKETBALL ASSOCIATION'
    west_col_3.image(West_logo, width=250)  # caption='EASTERN CONFERENCE'

with tab_27:
    st.subheader('1995-1996 CHICAGO BULLS')
    st.image(CHI_logo)
    st.dataframe(chi_bulls_1996.style.format(col_format_dict).set_table_styles(df_styles))

    ## LEAGUE LOGOS ##
    east_col_1, nba_col_2, west_col_3 = st.columns(3)
    east_col_1.image(East_logo, width=250)  # caption='WESTERN CONFERENCE'
    nba_col_2.image(nba_logo_1, width=300)  # caption='NATIONAL BASKETBALL ASSOCIATION'
    west_col_3.image(West_logo, width=250)  # caption='EASTERN CONFERENCE'

with tab_28:
    st.subheader('1994-1995 HOUSTON ROCKETS')
    st.image(HOU_logo)
    st.dataframe(hou_rockets_1995.style.format(col_format_dict).set_table_styles(df_styles))

    ## LEAGUE LOGOS ##
    east_col_1, nba_col_2, west_col_3 = st.columns(3)
    east_col_1.image(East_logo, width=250)  # caption='WESTERN CONFERENCE'
    nba_col_2.image(nba_logo_1, width=300)  # caption='NATIONAL BASKETBALL ASSOCIATION'
    west_col_3.image(West_logo, width=250)  # caption='EASTERN CONFERENCE'

with tab_29:
    st.subheader('1993-1994 HOUSTON ROCKETS')
    st.image(HOU_logo)
    st.dataframe(hou_rockets_1994.style.format(col_format_dict).set_table_styles(df_styles))

    ## LEAGUE LOGOS ##
    east_col_1, nba_col_2, west_col_3 = st.columns(3)
    east_col_1.image(East_logo, width=250)  # caption='WESTERN CONFERENCE'
    nba_col_2.image(nba_logo_1, width=300)  # caption='NATIONAL BASKETBALL ASSOCIATION'
    west_col_3.image(West_logo, width=250)  # caption='EASTERN CONFERENCE'

with tab_30:
    st.subheader('1992-1993 CHICAGO BULLS')
    st.image(CHI_logo)
    st.dataframe(chi_bulls_1993.style.format(col_format_dict).set_table_styles(df_styles))

    ## LEAGUE LOGOS ##
    east_col_1, nba_col_2, west_col_3 = st.columns(3)
    east_col_1.image(East_logo, width=250)  # caption='WESTERN CONFERENCE'
    nba_col_2.image(nba_logo_1, width=300)  # caption='NATIONAL BASKETBALL ASSOCIATION'
    west_col_3.image(West_logo, width=250)  # caption='EASTERN CONFERENCE'

with tab_31:
    st.subheader('1991-1992 CHICAGO BULLS')
    st.image(CHI_logo)
    st.dataframe(chi_bulls_1992.style.format(col_format_dict).set_table_styles(df_styles))

    ## LEAGUE LOGOS ##
    east_col_1, nba_col_2, west_col_3 = st.columns(3)
    east_col_1.image(East_logo, width=250)  # caption='WESTERN CONFERENCE'
    nba_col_2.image(nba_logo_1, width=300)  # caption='NATIONAL BASKETBALL ASSOCIATION'
    west_col_3.image(West_logo, width=250)  # caption='EASTERN CONFERENCE'

with tab_32:
    st.subheader('1990-1991 CHICAGO BULLS')
    st.image(CHI_logo)
    st.dataframe(chi_bulls_1991.style.format(col_format_dict).set_table_styles(df_styles))

    ## LEAGUE LOGOS ##
    east_col_1, nba_col_2, west_col_3 = st.columns(3)
    east_col_1.image(East_logo, width=250)  # caption='WESTERN CONFERENCE'
    nba_col_2.image(nba_logo_1, width=300)  # caption='NATIONAL BASKETBALL ASSOCIATION'
    west_col_3.image(West_logo, width=250)  # caption='EASTERN CONFERENCE'



## SCRIPT TERMINATION ##
st.stop()




### INTERPRETATION ###





### SCRATCH NOTES ###


# top-level filters
# job_filter = st.selectbox (“Select the status”, pd.unique (df(“job”)))
# df = df [df [“status”] == status_filter]

## METRICS - TOP of each team?
# kpix.metric
# {
# label = “Fitness”,
# value = f”$ { round ( balance, 4) }”,
# delta = - round (balance / count_fitness) * 100,
# }


## PAGE BACKGROUND ##

# def add_bg_from_url():
#     st.markdown(
#          f"""
#          <style>
#          .stApp {{
#              background-image: url("images/Court1.png");
#              background-attachment: fixed;
#              background-size: cover
#          }}
#          </style>
#          """,
#          unsafe_allow_html=True
#      )
#
# add_bg_from_url()


# CONFIG TEMPLATE
    # st.set_page_config(page_title="CSS hacks", page_icon=":smirk:")
    #
    # c1 = st.container()
    # st.markdown("---")
    # c2 = st.container()
    # with c1:
    #     st.markdown("Hello")
    #     st.slider("World", 0, 10, key="1")
    # with c2:
    #     st.markdown("Hello")
    #     st.slider("World", 0, 10, key="2")

# STYLE WITH CSS THROUGH MARKDOWN
    # st.markdown("""
    # <style>
    # div[data-testid="stBlock"] {
    #     padding: 1em 0;
    #     border: thick double #32a1ce;
    # }
    # </style>
    # """, unsafe_allow_html=True)


# STYLE WITH JS THROUGH HTML IFRAME
    # components.html("""
    # <script>
    # const elements = window.parent.document.querySelectorAll('div[data-testid="stBlock"]')
    # console.log(elements)
    # elements[0].style.backgroundColor = 'paleturquoise'
    # elements[1].style.backgroundColor = 'lightgreen'
    # </script>
    # """, height=0, width=0)


# st.markdown("""
#             <style>
#             div[data-testid="stBlock"] {padding: 1em 0; border: thick double #32a1ce; color: blue}
#             </style>
#             """,
#             unsafe_allow_html=True)

# style={'textAlign': 'Center', 'backgroundColor': 'rgb(223,187,133)',
#                                            'color': 'black', 'fontWeight': 'bold', 'fontSize': '24px',
#                                            'border': '4px solid black', 'font-family': 'Arial'}),

#pattern_shape = "nation", pattern_shape_sequence = [".", "x", "+"]

            # fig = px.bar(df, x="sex", y="total_bill", color="smoker", barmode="group", facet_row="time", facet_col="day",
            #        category_orders={"day": ["Thur", "Fri", "Sat", "Sun"], "time": ["Lunch", "Dinner"]})

            # fig = px.scatter_matrix(df, dimensions=["sepal_width", "sepal_length", "petal_width", "petal_length"], color="species")

            # fig = px.parallel_categories(df, color="size", color_continuous_scale=px.colors.sequential.Inferno)

            # fig = px.parallel_coordinates(df, color="species_id", labels={"species_id": "Species",
            #                   "sepal_width": "Sepal Width", "sepal_length": "Sepal Length",
            #                   "petal_width": "Petal Width", "petal_length": "Petal Length", },
            #                     color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=2)


# from IPython.display import HTML
# import base64
# # convert your links to html tags
# def path_to_image_html(path):
#     return '<img src="'+ path + '" width="60" >'
# HTML(champion_players[['LOGO']].to_html(escape=False, formatters=dict(image=path_to_image_html)))