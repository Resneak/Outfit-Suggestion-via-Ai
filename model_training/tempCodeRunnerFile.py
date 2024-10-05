        # Update the path to save the model within the current project folder
    model_save_dir = 'models'
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    model.save(os.path.join(model_save_dir, 'clothing_classifier.h5'))
    print("Model trained and saved to disk.")