# Voice-Guardian Fix Plan
1. [x] Fix Plotly titlefont deprecation in visuals/score_gauge.py (replace titlefont -> title_font)
2. [x] Fixed app.py indentation/Pylance errors for st.plotly_chart calls
3. [ ] Update score_mfcc to non-monotonic human range 80-320
4. [ ] Add score_mfcc_delta2 and score_spectral_flux functions
5. [ ] Update compute_trust_score weights and component_scores to 7 features
6. [ ] Update app.py metric cards and "How It Works" table to 7 features
7. [ ] Restart Streamlit server
8. [ ] Test with audio upload
