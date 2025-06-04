from model_inference import ResumeFitModel

def test_predict_fit_score():
    model = ResumeFitModel()
    resume = "Experienced software engineer with Python and ML background."
    job_desc = "Looking for a Python developer with machine learning experience."
    score = model.predict_fit_score(resume, job_desc)
    print("Predicted fit score:", score)

if __name__ == "__main__":
    test_predict_fit_score()
