import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class SkillMatcher:
    """
    Extracts and matches skills between resume and job description.
    Generates probing questions to uncover hidden experience.
    """

    def __init__(self, llm_service: Any):
        self.llm_service = llm_service

    def extract_skills(self, text: str, context: str = "resume") -> Dict[str, Any]:
        """Extract skills from resume or job description."""
        prompt = f"""Extract all technical and professional skills from the following {context}. 
Categorize them as: Technical Skills, Soft Skills, Tools & Platforms, Certifications, Languages.
Return a CONCISE list with no explanations.

{context.upper()}:
{text}

Format your response EXACTLY as:
Technical Skills: [skill1, skill2, ...]
Soft Skills: [skill1, skill2, ...]
Tools & Platforms: [skill1, skill2, ...]
Certifications: [skill1, skill2, ...]
Languages: [skill1, skill2, ...]
"""
        result = self.llm_service.generate_response(prompt, max_length=512)
        return self._parse_skills_response(result["response"])

    def _parse_skills_response(self, response: str) -> Dict[str, List[str]]:
        """Parse the LLM skill extraction response."""
        skills = {
            "Technical Skills": [],
            "Soft Skills": [],
            "Tools & Platforms": [],
            "Certifications": [],
            "Languages": [],
        }
        
        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue
            for category in skills.keys():
                if line.startswith(category + ":"):
                    skills_str = line.replace(category + ":", "").strip()
                    skills[category] = [s.strip() for s in skills_str.split(",")]
                    break
        
        return skills

    def match_skills(
        self,
        resume_skills: Dict[str, List[str]],
        job_skills: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """Match resume skills against job requirements."""
        prompt = f"""Analyze the skill match between resume and job description.

RESUME SKILLS:
{self._format_skills(resume_skills)}

REQUIRED JOB SKILLS:
{self._format_skills(job_skills)}

For each category:
1. List MATCHED skills (skills that appear in both or are directly equivalent)
2. List MISSING skills (required in job but not in resume)
3. List BONUS skills (in resume but not in job description - assess if they're valuable to the role)

For each MISSING or BONUS skill category, rate relevance (High/Medium/Low).

Format your response EXACTLY as:
TECHNICAL SKILLS:
Matched: [skills]
Missing: [skills]
Bonus: [skills]

SOFT SKILLS:
Matched: [skills]
Missing: [skills]
Bonus: [skills]

TOOLS & PLATFORMS:
Matched: [skills]
Missing: [skills]
Bonus: [skills]

CERTIFICATIONS:
Matched: [skills]
Missing: [skills]
Bonus: [skills]

Overall Match Score (0-100): [score]
Explanation: [brief explanation]
"""
        result = self.llm_service.generate_response(prompt, max_length=1024)
        return self._parse_match_response(result["response"])

    def generate_probing_questions(
        self,
        resume_text: str,
        job_description: str,
        missing_skills: List[str],
    ) -> Dict[str, Any]:
        """Generate probing questions about experience related to missing skills."""
        prompt = f"""Based on the user's resume and the job requirements, generate 3-5 intelligent probing questions.

These questions should:
1. Explore if the user has experience with the missing required skills
2. Identify if existing experience in the resume could translate to required skills
3. Uncover any relevant projects or work that weren't explicitly mentioned
4. Be open-ended to encourage detailed answers

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

MISSING SKILLS IN RESUME:
{', '.join(missing_skills)}

Generate the questions in this format:
Question 1: [question]
Motivation: [why this matters for the role]

Question 2: [question]
Motivation: [why this matters for the role]

Question 3: [question]
Motivation: [why this matters for the role]

(Continue for 3-5 questions total)
"""
        result = self.llm_service.generate_response(prompt, max_length=768)
        return self._parse_questions_response(result["response"])

    def evaluate_response_to_questions(
        self,
        user_responses: Dict[int, str],
        missing_skills: List[str],
    ) -> Dict[str, Any]:
        """Evaluate user's responses to probing questions."""
        responses_text = "\n".join(
            [f"Q{i}: {resp}" for i, resp in user_responses.items()]
        )
        
        prompt = f"""Based on the user's responses to probing questions, assess:
1. Which of the missing skills are now covered or partially covered by their experience
2. New capabilities revealed through their answers
3. Recommended way to articulate this experience in their resume

MISSING SKILLS BEING ASSESSED:
{', '.join(missing_skills)}

USER RESPONSES:
{responses_text}

Format your response as:
NEWLY IDENTIFIED SKILLS:
[List skills now covered with High/Medium/Low confidence]

GAPS STILL REMAINING:
[List skills still not covered]

RECOMMENDED RESUME IMPROVEMENTS:
[Suggest how to articulate their experience to match missing skills - NO new bullet points, just rephrasing guidance]

UPDATED MATCH SCORE: [0-100]
EXPLANATION: [Why the score changed]
"""
        result = self.llm_service.generate_response(prompt, max_length=1024)
        return self._parse_evaluation_response(result["response"])

    def _format_skills(self, skills: Dict[str, List[str]]) -> str:
        """Format skills dictionary into readable text."""
        lines = []
        for category, skill_list in skills.items():
            if skill_list:
                lines.append(f"{category}: {', '.join(skill_list)}")
        return "\n".join(lines)

    def _parse_match_response(self, response: str) -> Dict[str, Any]:
        """Parse the skill match response."""
        result = {
            "matched": {},
            "missing": {},
            "bonus": {},
            "overall_score": 0,
            "explanation": "",
        }
        
        current_category = None
        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue
            
            if "Overall Match Score" in line:
                try:
                    score_str = line.split(":")[-1].strip().split("/")[0]
                    result["overall_score"] = int(score_str)
                except ValueError:
                    result["overall_score"] = 0
            elif "Explanation:" in line:
                result["explanation"] = line.split(":", 1)[-1].strip()
            elif "TECHNICAL SKILLS:" in line:
                current_category = "Technical"
            elif "SOFT SKILLS:" in line:
                current_category = "Soft"
            elif "TOOLS & PLATFORMS:" in line:
                current_category = "Tools"
            elif "CERTIFICATIONS:" in line:
                current_category = "Certifications"
            elif current_category and line.startswith("Matched:"):
                skills = line.replace("Matched:", "").strip().split(",")
                result["matched"][current_category] = [s.strip() for s in skills if s.strip()]
            elif current_category and line.startswith("Missing:"):
                skills = line.replace("Missing:", "").strip().split(",")
                result["missing"][current_category] = [s.strip() for s in skills if s.strip()]
            elif current_category and line.startswith("Bonus:"):
                skills = line.replace("Bonus:", "").strip().split(",")
                result["bonus"][current_category] = [s.strip() for s in skills if s.strip()]
        
        return result

    def _parse_questions_response(self, response: str) -> Dict[int, Dict[str, str]]:
        """Parse the probing questions response."""
        questions = {}
        current_q = None
        
        for line in response.split("\n"):
            line_stripped = line.strip()
            if line_stripped.startswith("Question "):
                try:
                    q_num = int(line_stripped.split(":")[0].replace("Question", "").strip())
                    question_text = line_stripped.split(":", 1)[-1].strip()
                    current_q = q_num
                    questions[current_q] = {"question": question_text, "motivation": ""}
                except ValueError:
                    pass
            elif line_stripped.startswith("Motivation:") and current_q:
                questions[current_q]["motivation"] = line_stripped.split(":", 1)[-1].strip()
        
        return questions

    def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """Parse the evaluation response."""
        result = {
            "newly_identified_skills": [],
            "gaps_remaining": [],
            "resume_improvements": "",
            "updated_score": 0,
            "explanation": "",
        }
        
        section = None
        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue
            
            if "NEWLY IDENTIFIED SKILLS:" in line:
                section = "newly_identified"
            elif "GAPS STILL REMAINING:" in line:
                section = "gaps"
            elif "RECOMMENDED RESUME IMPROVEMENTS:" in line:
                section = "improvements"
            elif "UPDATED MATCH SCORE:" in line:
                try:
                    score_str = line.split(":")[-1].strip().split("/")[0]
                    result["updated_score"] = int(score_str)
                except ValueError:
                    pass
                section = None
            elif "EXPLANATION:" in line:
                result["explanation"] = line.split(":", 1)[-1].strip()
            elif section == "newly_identified" and line:
                result["newly_identified_skills"].append(line)
            elif section == "gaps" and line:
                result["gaps_remaining"].append(line)
            elif section == "improvements" and line:
                result["resume_improvements"] += line + "\n"
        
        return result
