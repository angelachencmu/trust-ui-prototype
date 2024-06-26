from types import SimpleNamespace

import pandas as pd
import streamlit as st

import iris_toy

st.set_page_config(page_title='Trust UI Prototype')
st.write('# KNN Model on Iris Dataset')

################################################################################
# Main content
################################################################################
main_content = SimpleNamespace()
main_content.features = st.multiselect(
    'Select feature(s)',
    iris_toy.available_features,
    [iris_toy.available_features[0], iris_toy.available_features[2]],
    key='main_content_features',
)

if len(main_content.features) > 0:
    main_content.n_neighbors = list(range(1, 10 + 1))
    main_content.accuracies = [
        iris_toy.train_knn_iris(main_content.features, n)
        for n in main_content.n_neighbors
    ]
    st.line_chart(
        pd.DataFrame({
            'n_neighbors': main_content.n_neighbors,
            'accuracies': main_content.accuracies,
        }),
        x='n_neighbors', y='accuracies'
    )
else:
    st.warning('Please select at least one feature')


################################################################################
# Calculator sidebar
################################################################################
calculator = SimpleNamespace()
with st.sidebar:
    st.write('## Calculator')

    calculator.features = st.multiselect(
        'Select feature(s)',
        iris_toy.available_features,
        [iris_toy.available_features[0], iris_toy.available_features[2]],
        key='calculator_features',
    )
    calculator.n_neighbors = st.slider('Number of neighbors', 1, 10, 5)

    st.write('---')

    is_train = st.button('Train model')
    is_auto_train = st.checkbox('Auto train', value=True)

    if is_train or is_auto_train:
        if len(calculator.features) == 0:
            st.warning('Please select at least one feature')
        accuracy = iris_toy.train_knn_iris(calculator.features, calculator.n_neighbors)
        st.write(f'Accuracy: {accuracy:.6f}')
