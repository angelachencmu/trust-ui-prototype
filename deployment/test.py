# ... (previous code remains the same)

# Initialize the interactions list in the session state if it doesn't exist
if 'interactions' not in st.session_state:
    st.session_state.interactions = []

# ... (rest of the code remains the same)

if st.button('Train Model', key=f'train_model_{iteration_counter}'):
    start_time = st.session_state.start_time  # Get the start time from session state
    end_time = time.time()  # Record the end time when the "Train Model" button is clicked
    duration = round(end_time - start_time, 2)

    # ... (model training code remains the same)

    start_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
    end_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))
    st.session_state.interactions.append([user_id, start_time_str, end_time_str, duration, ','.join(selected_features), classifier, acc])  # Store the interaction in the session state
    st.write('Accuracy: ', acc)

    # Write interactions to CSV file
    csv_file = f"{user_id}_interactions.csv"
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['User ID', 'Start Time', 'End Time', 'Duration (seconds)', 'Selected Features', 'Algorithm', 'Accuracy'])
        writer.writerows(st.session_state.interactions)  # Write interactions from the session state

    model_trained = True  # Set the flag to indicate that a model has been trained
    del st.session_state.start_time  # Remove the start time from session state

# ... (rest of the code remains the same)

# Display the interaction log as a table if the checkbox is selected
if st.checkbox('Show interaction log'):
    if len(st.session_state.interactions) > 0:
        st.subheader('Interaction Log')
        log_df = pd.DataFrame(st.session_state.interactions, columns=['User ID', 'Start Time', 'End Time', 'Duration (seconds)', 'Selected Features', 'Algorithm', 'Accuracy'])
        st.table(log_df)

        # Allow users to download the CSV file
        csv_data = log_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"{user_id}_interactions.csv",
            mime='text/csv'
        )
    else:
        st.write("No interactions recorded.")
