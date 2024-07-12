# Import necessary libraries
from openpyxl import load_workbook  # For loading Excel files
import pandas as pd  # Data manipulation library
import numpy as np  # Numerical operations library
import matplotlib.pyplot as plt  # Plotting library
from sklearn.linear_model import LinearRegression  # Linear regression model
from sklearn.metrics import r2_score  # R-squared score metric


def fit_model(x1_values, x2_values, y_values):
    # Function to fit a linear regression model

    # Stack x1_values and x2_values horizontally into a matrix X
    X = np.column_stack((x1_values, x2_values))

    # Initialize LinearRegression model
    model = LinearRegression()

    # Fit the model using X and y_values
    model.fit(X, y_values)

    # Extract coefficients alpha and beta from the model
    alpha = model.coef_[0]
    beta = model.coef_[1]

    # Make predictions using the model
    predictions = model.predict(X)

    # Calculate R-squared score
    r_squared = r2_score(y_values, predictions)

    # Print results
    print("Optimal consts:", alpha, beta)
    print("R2 = ", r_squared)
    print(predictions)


def plot(x_h, y_h, x_a, y_a, o_v):
    # Function to plot data and regression lines for home and away teams

    # Convert input lists to numpy arrays
    xh = np.array(x_h)
    yh = np.array(y_h)
    xa = np.array(x_a)
    ya = np.array(y_a)

    # Fit a linear regression line for home team data
    a, b = np.polyfit(xh, yh, 1)
    r = np.corrcoef(xh, yh)[0, 1]
    print("yt =", a, "* x +", b)
    print("r^2 =", r * r)
    print("yt =", (a * o_v + b) / 1)

    # Plot home team data and regression line
    plt.scatter(xh, yh, color='black')
    plt.plot(xh, a * xh + b, color='steelblue', linestyle="--", linewidth=2)
    plt.text(1, 17, 'y = ' + '{:.2f}'.format(b) + ' + {:.2f}'.format(a) + 'x', size=14)

    # Fit a linear regression line for away team data
    a, b = np.polyfit(xa, ya, 1)
    r = np.corrcoef(xa, ya)[0, 1]
    print("ys =", a, "* x +", b)
    print("r^2 =", r * r)
    print("predicted shots =", (a * o_v + b) / 1)

    # Plot away team data and regression line
    plt.scatter(xa, ya, color='purple')
    plt.plot(xa, a * xa + b, color='pink', linestyle="--", linewidth=2)
    plt.text(1, 17, 'y = ' + '{:.2f}'.format(b) + ' + {:.2f}'.format(a) + 'x', size=14)

    # Show the plot
    plt.show()


def shots_each_game(xls, team):
    # Function to extract shot data for each game for a specific team from Excel

    # Read specific sheet and columns from Excel file into a DataFrame
    df = pd.read_excel(xls, sheet_name=team, usecols="E, F, N")

    # Extract relevant columns and filter out NaN values
    unf_game_loc = df["LOC"].tolist()
    unf_shot_list = df["S"].tolist()
    unf_opp_list = df["OPP"].tolist()

    game_loc = [x for x in unf_game_loc if x == x]
    shot_list = [x for x in unf_shot_list if x == x]
    opp_list = [x for x in unf_opp_list if x == x]

    return game_loc, shot_list, opp_list


def calc(xls, home, team, opp):
    # Function to calculate and plot shot data based on game location (home or away)

    # Dictionary mapping teams to their average shots allowed per (SA_GP) (home or away)
    SA_GP = {'MTL': [34.1563, 32.3125], 'TOR': [30.3125, 29.125], 'BOS': [29.6571, 33.1875], 'NYR': [28.2581, 30.5455], 'CHI': [32.3939, 32.9375], 'DET': [30.9697, 33.8387], 'LAK': [27.3438, 29.5], 'DAL': [28.1613, 30.6], 'PHI': [26.5, 28.8788], 'PIT': [30.2188, 30.0645], 'STL': [32.4, 32.2571], 'BUF': [27.2121, 31], 'VAN': [29.1333, 29.3056], 'CGY': [28.4194, 31.5758], 'NYI': [32.2188, 33.9375], 'NJD': [28.9706, 30.0645], 'WSH': [30.2188, 30.871], 'EDM': [29.2333, 27.8485], 'CAR': [23.8182, 27.1935], 'COL': [28.4063, 30.7879], 'ARI': [30.625, 33.0606], 'SJS': [35, 35.8387], 'OTT': [28.7576, 30.931], 'TBL': [29.0909, 29.75], 'ANA': [32.3529, 33.7333], 'VGK': [30.1563, 31.7188], 'FLA': [28.1563, 27.1818], 'CBJ': [34.8788, 32.1935], 'MIN': [29.4375, 30.9394], 'NSH': [30.1765, 30.5625], 'WPG': [29.1875, 29.25], 'SEA': [27.69, 29.39]}

    # Extract shot data for the specified team and game location (home or away)
    game_loc, shot_list, opp_list = shots_each_game(xls, team)
    home_x_values = []  # List for opponent shots allowed for home games
    home_y_values = []  # List for shots taken in each home game
    away_x_values = []  # List for opponent shots allowed for away games
    away_y_values = []  # List for shots taken in each away game

    # Iterate through each game's data
    for i in range(len(opp_list)):
        if game_loc[i] == "vs":  # Home game
            home_x_values.append(SA_GP[opp_list[i]][1])  # Append opponent's shots allowed at home
            home_y_values.append(shot_list[i])  # Append shots taken in this home game
        else:  # Away game
            away_x_values.append(SA_GP[opp_list[i]][0])  # Append opponent's shots allowed away
            away_y_values.append(shot_list[i])  # Append shots taken in this away game

    # Determine opponent's shots allowed value based on game location (home or away)
    if home:
        opp_value = SA_GP[opp][1]  # Home game, use opponent's shots allowed at home
        print("OPP VALUE:", opp_value)
        plot(home_x_values + away_x_values, home_y_values + away_y_values, home_x_values, home_y_values, opp_value)
    else:
        opp_value = SA_GP[opp][0]  # Away game, use opponent's shots allowed away
        print("OPP VALUE:", opp_value)
        plot(home_x_values + away_x_values, home_y_values + away_y_values, away_x_values, away_y_values, opp_value)


def main():
    # Main function to orchestrate the analysis

    # Load Excel file
    xls = pd.ExcelFile("stats_NHL.xlsx")

    # Define home and away teams
    home = "STL"
    away = "LAK"

    # Perform analysis for home team
    print("HOME:", home)
    calc(xls, True, home, away)

    # Perform analysis for away team
    print("AWAY:", away)
    calc(xls, False, away, home)

main()
