from transformers import T5Tokenizer, T5ForConditionalGeneration

class ResumeFitModel:
    def __init__(self, model_dir="/home/ubuntu/models/t5-resume-fit"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_dir)
        self.model = T5ForConditionalGeneration.from_pretrained(model_dir)

    def predict_fit_score(self, resume, job_desc):
        input_text = f"Resume: {resume}\nJob Description: {job_desc}\n"
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model.generate(**inputs, max_length=32)
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result

# Example usage:
model = ResumeFitModel()
score = model.predict_fit_score("""John Doe  
Email: john.doe@email.com  
Phone: (123) 456-7890  
Location: San Francisco, CA  

Professional Summary:  
Data analyst with 3 years of experience in generating business insights using SQL, Excel, and Tableau. Passionate about using data to solve real-world problems.

Experience:

Data Analyst  
ABC Retail Inc. | San Francisco, CA | Jan 2021 – Present  
- Designed and maintained Tableau dashboards to track key sales metrics.  
- Wrote complex SQL queries to analyze customer behavior and product trends.  
- Collaborated with cross-functional teams to improve reporting accuracy.  
- Reduced weekly reporting time by 40% through automation.

Business Intelligence Intern  
XYZ Tech | San Jose, CA | Jun 2020 – Dec 2020  
- Assisted in developing Power BI reports for internal stakeholders.  
- Cleaned and transformed raw data using Excel and SQL.  

Education:  
B.S. in Statistics, University of California, Berkeley | 2020  

Skills:  
- SQL, Excel, Tableau, Power BI  
- Data cleaning, exploratory analysis, visualization  
- Python (basic)
""", """Position: Data Scientist – E-commerce Analytics  
Company: GlobalMart  

We are looking for a Data Scientist to join our e-commerce analytics team. This individual will work closely with product, marketing, and engineering teams to deliver insights that drive strategic decisions.

Responsibilities:
- Build predictive models using Python and machine learning libraries.  
- Design A/B experiments and analyze test results.  
- Automate data pipelines using SQL and Python.  
- Create intuitive visualizations using tools like Tableau or Looker.

Requirements:
- 3+ years of experience in data analysis or data science.  
- Proficient in Python and SQL.  
- Experience with machine learning (e.g., scikit-learn, XGBoost).  
- Strong communication and storytelling skills.  
- Bachelor’s or Master’s degree in a quantitative field (e.g., Statistics, Computer Science).

Nice to Have:
- Experience with cloud platforms (AWS, GCP).  
- Prior work in e-commerce or retail analytics.
""")
print(score)
