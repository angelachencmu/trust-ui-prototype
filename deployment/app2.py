from types import SimpleNamespace

import pandas as pd
import streamlit as st
import iris_toy_2

st.set_page_config(page_title='Trust UI Prototype')

st.write('# Decision Tree Model on Iris Dataset')

################################################################################
# Main content
################################################################################

main_content = SimpleNamespace()

main_content.features = st.multiselect(
    'Select feature(s)',
    iris_toy_2.available_features,
    [iris_toy_2.available_features[0], iris_toy_2.available_features[2]],
    key='main_content_features',
)

if len(main_content.features) > 0:
    main_content.accuracies = [
        iris_toy_2.train_decision_tree_iris(main_content.features)
    ]

    st.line_chart(
        pd.DataFrame({
            'accuracies': main_content.accuracies,
        }),
        y='accuracies'
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
        iris_toy_2.available_features,
        [iris_toy_2.available_features[0], iris_toy_2.available_features[2]],
        key='calculator_features',
    )

    st.write('---')

    is_train = st.button('Train model')
    is_auto_train = st.checkbox('Auto train', value=True)

    if is_train or is_auto_train:
        if len(calculator.features) == 0:
            st.warning('Please select at least one feature')

        accuracy = iris_toy_2.train_decision_tree_iris(calculator.features)
        st.write(f'Accuracy: {accuracy:.6f}')
