import streamlit as st
import pickle
import pandas as pd

teams = ['Rajasthan Royals', 'Royal Challengers Bangalore',
       'Sunrisers Hyderabad', 'Delhi Capitals', 'Chennai Super Kings',
       'Gujarat Titans', 'Lucknow Super Giants', 'Kolkata Knight Riders',
       'Punjab Kings', 'Mumbai Indians']

cities = ['Mumbai', 'Navi Mumbai', 'Chennai', 'Dubai', 'Jaipur', 'Bangalore',
       'Delhi', 'Chandigarh', 'Ahmedabad', 'Kimberley', 'Kolkata',
       'Abu Dhabi', 'Indore', 'Bengaluru', 'Bloemfontein', 'Cuttack',
       'Dharamsala', 'Ranchi', 'Hyderabad', 'Visakhapatnam', 'Sharjah',
       'Port Elizabeth', 'Pune', 'Kanpur', 'Durban', 'Nagpur',
       'Johannesburg', 'Centurion', 'East London', 'Cape Town', 'Kochi',
       'Rajkot', 'Raipur']

pipe = pickle.load(open('model.pkl','rb'))
st.title('IPL Win Predictor')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team',sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team',sorted(teams))

selected_city = st.selectbox('Select host city',sorted(cities))

target = st.number_input('Target')

col3,col4,col5 = st.columns(3)

with col3:
    score = st.number_input('Score')
with col4:
    overs = st.number_input('Overs completed')
with col5:
    wickets = st.number_input('Wickets out')

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets = 10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left

    input_df = pd.DataFrame({'BattingTeam':[batting_team],'BallingTeam':[bowling_team],'City':[selected_city],'runs_left':[runs_left],'balls_left':[balls_left],'wicket_left':[wickets],'total_run_x':[target],'crr':[crr],'rrr':[rrr]})

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + "- " + str(round(win*100)) + "%")
    st.header(bowling_team + "- " + str(round(loss*100)) + "%")