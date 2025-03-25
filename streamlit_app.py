import os
import streamlit as st
from typing_extensions import TypedDict
from typing import Dict, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel,Field
from typing import Dict, List
from typing import Literal

class State(TypedDict):
    user_requirements: str
    user_stories: List[str]
    design_docs: Dict[List[str],List[str]]
    code: str
    test_cases: List[str]
    feedback:str
    status:str

class UserStories(BaseModel):
    stories: List[str]

class DesignDocs(BaseModel):
    functional:List[str]=Field(
        description="Functional Documents",
    )
    technical:List[str]=Field(
        description="Technical Documents",
    )
class TestCases(BaseModel):
    cases:List[str]

class Review(BaseModel):
    review: str = Field(
        description="Detailed feedback that provides specific, actionable insights about strengths and weaknesses. Should include concrete suggestions for improvement with clear reasoning behind each point. For code reviews, include comments on quality, readability, performance, and adherence to best practices. For design documents, address completeness, clarity, and technical feasibility."
    )
    status: Literal["Approved", "Not Approved"]

def user_inputs_requirements(state: State):
    return state

def auto_generate_user_stories(state: State):
    if not state["user_requirements"]:
        st.error("Please enter requirements before generating user stories.")
        return state

    with st.container():
        st.write("🔄 Generating User Stories...")
        userstories_prompt = PromptTemplate(
        template="""You are an expert agile product manager with expertise in user story creation.

                    Task: Based on the following requirement: "{user_requirements}", generate exactly 5 well-structured user stories.

                    Each user story must:
                    - Follow the format: As a <specific user type>, I want to <specific action/feature> so that <clear benefit>
                    - Be concise yet descriptive
                    - Focus on user value, not implementation details
                    - Be testable with clear acceptance criteria
                    - Be independent of each other (no dependencies between stories)
                    - Cover different aspects of the application functionality

                    Your user stories should address the core functionality described in the requirements while considering different user perspectives.
                    """,
                    input_variables=["user_requirements"]
                    )

        userstory_chain = userstories_prompt | llm.with_structured_output(UserStories)
        response = userstory_chain.invoke({"user_requirements": state["user_requirements"]})
        state["user_stories"] = response.stories
        
        with tabs_dict["User Stories"]:
            st.subheader("Generated User Stories")
            for story in state["user_stories"]:
                st.write(f"- {story}")

    return state

def product_owner_review(state: State):
    with st.container():
        st.write("🔄 Product Owner Review in Progress...")
        review_prompt = PromptTemplate(
        template="""You are a senior product owner with 10+ years of experience reviewing user stories.

                    Task: Review the following user stories based on INVEST criteria (Independent, Negotiable, Valuable, Estimable, Small, Testable):

                    {user_stories}

                    Provide a comprehensive evaluation addressing:
                    1. Clarity and structure of each story
                    2. Whether each story provides clear user value
                    3. Whether stories collectively cover the main functionality needed
                    4. Whether acceptance criteria are implied or need clarification
                    5. Suggestions for improvements where needed

                    Respond in the following format:
                    - Status: Approved / Not Approved
                    - Feedback: [Detailed evaluation with specific recommendations for improvement]
                    """,
                        input_variables=["user_stories"]
                    )

        review_chain = review_prompt | llm.with_structured_output(Review)
        response = review_chain.invoke({"user_stories": "\n".join(state["user_stories"])})
        state['status']=response.status
        state['feedback']=response.review
        
        with tabs_dict["User Stories"]:
            st.write(f'Product Owner Approval status')
            st.write(f"**Status:** {state['status']}")
            st.write(f"**Feedback:** {state['feedback']}")
    return state
  
def decision(state):
    """Returns the next step based on status and feedback."""
    if state['status']=="Approved":
        return "Approved"
    else:
        return "Feedback"

def revise_user_stories(state: State):
    with st.container():
        st.write("🔄 Revising User Stories...")
        feedback_summary = state["feedback"]
        old_stories_context = "\n".join(state["user_stories"])

        regeneration_prompt = PromptTemplate(
            template="""You are an expert in generating user stories. The Product Owner has provided feedback.

            Below is the feedback:
            {feedback_summary}

            These are the previously generated user stories:
            {old_stories_context}

            Based on this feedback, regenerate exactly 5 user stories that incorporate these improvements.
            Format:
            As a <user>, I want to <action> so that <benefit>.""",
            input_variables=["feedback_summary", "old_stories_context"]
        )

        regeneration_chain = regeneration_prompt | llm.with_structured_output(UserStories)
        response = regeneration_chain.invoke({
            "feedback_summary": feedback_summary,
            "old_stories_context": old_stories_context
        })

        state['user_stories']= response.stories
        
        with tabs_dict["User Stories"]:
            st.subheader("Revised User stories after Product Owner Review")
            for story in state["user_stories"]:
                st.write(f"- {story}")
    return state

def create_design_documents(state: State):
    """Generates functional & technical design docs based on user stories."""
    with st.container():
        st.write("🔄 Creating Design Documents...")
        # Functional Design Prompt
        prompt = PromptTemplate(
        template="""You are a senior software architect with expertise in both functional and technical specifications.
    
        Task: Based on the provided user stories, create comprehensive design documents that will guide the implementation.

        User Stories:
        {user_stories}

        Provide comprehensive but clear and concise documentation that would enable a developer to build the system without further clarification.
        """,
            input_variables=["user_stories"]
        )

        # Generate Functional Design
        chain = prompt | llm.with_structured_output(DesignDocs)
        response = chain.invoke({"user_stories": "\n".join(state["user_stories"])})
        
        with tabs_dict["Design Docs"]:
            st.subheader("Design Docs")
            st.write(f"**Functional documents:**")
            for doc in response.functional:
                st.write(f"- {doc}")
            st.write(f"**Technical documents:**")
            for doc in response.technical:
                st.write(f"- {doc}")
    return {
        "design_docs": {
            "functional": response.functional,
            "technical": response.technical
        }
    }

def design_review(state: State):
    with st.container():
        st.write("🔄 Design Review in Progress...")
        review_prompt = PromptTemplate(
           template="""You are a senior technical architect reviewing functional and technical design documents.
            Below is the design documentation:
            {design_docs}
                   
            Respond in the following format:
            - Status: Approved / Not Approved
            - Feedback: Provide feedback on Design Docs

            """,
            input_variables=["design_docs"]
        )

        review_chain = review_prompt | llm.with_structured_output(Review)
        response = review_chain.invoke({
            "design_docs": state["design_docs"]
        })
        state['status']=response.status
        state['feedback']=response.review
        
        with tabs_dict["Design Docs"]:
            st.write(f'Design Review')
            st.write(f"**Status:** {state['status']}")
            st.write(f"**Feedback:** {state['feedback']}")
    return state
    
def revise_design_docs(state: State):
    """Revises design documents based on code review feedback."""
    with st.container():
        st.write("🔄 Revising Design Documents...")
        revision_prompt = PromptTemplate(
            template="""You are a senior technical architect correcting docs based on feedback.
            The following feedback was provided after deisgn review:

           {design_review_feedback}

           these are old docs

           {old_docs}

            **Update the design documents to address the issues while maintaining clarity and structure. **""",
            input_variables=["design_review_feedback","old_docs"]
        )

        revision_chain = revision_prompt | llm.with_structured_output(DesignDocs)
        response = revision_chain.invoke({'design_review_feedback':state['feedback'],"old_docs":state["design_docs"]})

        with tabs_dict["Design Docs"]:
            st.subheader("Revised Design Docs")
            st.write(f"**Functional documents:** {response.functional}")
            st.write(f"**Technical documents:** {response.technical}")
    return {
        "design_docs": {
            "functional": response.functional,
            "technical": response.technical
        }
    }
        
def generate_code(state: State):
    """Generates executable code based on design documents."""
    with st.container():
        st.write("🔄 Generating Code...")
        code_prompt = PromptTemplate(
        template="""You are an expert software developer with deep knowledge of modern programming practices, patterns, and best practices.

        Task: Generate production-ready, fully functional code based on the provided design documents:

        {design_documents}

        Your code should:
        1. Follow clean code principles (readability, maintainability, SOLID principles)
        2. Include proper error handling and edge cases
        3. Be secure against common vulnerabilities
        4. Be optimized for performance where appropriate
        5. Include comments for complex logic
        6. Follow standard naming conventions and code organization
        7. Be modular and well-structured
        8. Include all necessary imports and dependencies
       

        Choose the most appropriate language and framework based on the requirements. Implement all functionality described in the design documents, ensuring that business logic is correctly reflected in the code.

        Return only the code with proper indentation, no explanations.
        """,
            input_variables={"design_documents"}
        )

        code_chain = code_prompt | llm 
        code_response = code_chain.invoke({"design_documents":state['design_docs']})
        state['code']=code_response.content
        
        with tabs_dict["Code"]:
            st.subheader("Generated Code")
            st.code(state['code'])
    return state

def code_review(state: State):
    """Reviews generated code based on design documents and provides feedback."""
    with st.container():
        st.write("🔄 Code Review in Progress...")
        review_prompt = PromptTemplate(
            template="""You are a senior software engineer conducting a code review.
            Analyze the following code and provide feedback with Approved/Not Approved
            {generated_code}
           """,
            input_variables=["generated_code"]
        )

        review_chain = review_prompt | llm.with_structured_output(Review)
        response = review_chain.invoke(state['code'])

        state['status']=response.status
        state['feedback']=response.review
        
        with tabs_dict["Code"]:
            st.write("Code Review status")
            st.write(f'**Status:** {response.status}')
            st.write(f'**Feedback:** {response.review}')
    return state

def fix_code_after_code_review(state: State):
    """Fixes code based on feedback from the Code Review process."""
    with st.container():
        st.write("🔄 Fixing Code After Review...")
        fix_prompt = PromptTemplate(
            template="""You are an expert software engineer responsible for fixing code issues.
            The following code was reviewed, and feedback was provided:

            **Original Code:**
            {generated_code}

            **Code Review Feedback:**
            {code_review_feedback}

            Fix the code to address all issues, including security vulnerabilities, performance optimizations, 
            and best practices. Return only the corrected code, with proper indentation and structure, 
            and without any explanations.""",
            input_variables=["generated_code", "code_review_feedback"]
        )

        fix_chain = fix_prompt | llm | StrOutputParser()
        fixed_code = fix_chain.invoke({
            "generated_code": state["code"],
            "code_review_feedback": state["feedback"]
        })

        state["code"] = fixed_code
        
        with tabs_dict["Code"]:
            st.subheader("Fixed Code After Code Review")
            st.code(state['code'])
    return state

def security_review(state: State):
    """Conducts a security review of the code to check for vulnerabilities."""
    with st.container():
        st.write("🔄 Security Review in Progress...")
        security_prompt = PromptTemplate(
            template="""You are a senior cybersecurity expert specializing in secure coding practices and vulnerability assessment.

            Task: Conduct a thorough security review of the following code:

            **Code:**
            {generated_code}

            Provide structured feedback, including detected issues and suggested fixes.
            Format:
            - Status: Approved / Needs Fixes
            - Feedback: (Explain security risks and provide recommended changes)
            
            """,
            input_variables="generated_code"
        )

        security_chain = security_prompt | llm.with_structured_output(Review)
        response = security_chain.invoke({
            "generated_code": state["code"]
        })
        state['status']=response.status
        state['feedback']=response.review
        
        with tabs_dict["Security"]:
            st.subheader("Security Review")
            st.write(f'**Status:** {response.status}')
            st.write(f'**Feedback:** {response.review}')
    return state

def fix_code_after_security_review(state: State):
    """Fixes code based on security review feedback to eliminate vulnerabilities."""
    with st.container():
        st.write("🔄 Fixing Security Issues...")
        security_fix_prompt = PromptTemplate(
            template="""You are a cybersecurity expert and software engineer fixing security vulnerabilities.
            The following code was reviewed, and security concerns were identified:

            **Original Code:**
            {generated_code}

            **Security Review Feedback:**
            {security_review_feedback}

            Fix all security issues

            Return only the corrected code. 
            Do not include explanations.""",
            input_variables=["generated_code", "security_review_feedback"]
        )

        security_fix_chain = security_fix_prompt | llm | StrOutputParser()
        fixed_code = security_fix_chain.invoke({
            "generated_code": state["code"],
            "security_review_feedback": state["feedback"]
        })

        state["code"] = fixed_code
        
        with tabs_dict["Security"]:
            st.subheader("Fixed Code After Security Review")
            st.code(state['code'])
        
        with tabs_dict["Code"]:
            st.subheader("Updated Code After Security Fixes")
            st.code(state['code'])
    return state

def write_test_cases(state: State):
    """Generates test cases for the code based on functional and technical design documents."""
    with st.container():
        st.write("🔄 Writing Test Cases...")
        test_case_prompt = PromptTemplate(
            template="""You are a senior QA engineer with expertise in comprehensive test coverage and test-driven development.

            Task: Create a comprehensive test suite for the following code and design specifications:

            **Code:**
            {generated_code}

            **Functional Design Document:**
            {functional_design}

            **Technical Design Document:**
            {technical_design}

            Generate a structured list of **unit tests, integration tests, and edge cases**.
            Use the following format:

            - **Test Case Name:** <Descriptive Name>
            - **Description:** <What the test validates>
            - **Test Steps:** <Step-by-step execution>
            - **Expected Result:** <Expected output>

           """,
            input_variables={"generated_code", "functional_design", "technical_design"}
        )

        test_case_chain = test_case_prompt | llm.with_structured_output(TestCases)
        test_cases = test_case_chain.invoke({
            "generated_code": state["code"],
            "functional_design": state["design_docs"].get("functional", "No functional design available."),
            "technical_design": state["design_docs"].get("technical", "No technical design available.")
        })

        state["test_cases"] = test_cases.cases
        
        with tabs_dict["Testing"]:
            st.subheader("Test Cases")
            for i, case in enumerate(state["test_cases"]):
                st.markdown(f"**Test Case {i+1}:**")
                st.markdown(case)
                st.divider()
    return state

def test_case_review(state):
    """Conducts a Testcase review of test cases."""
    with st.container():
        st.write("🔄 Test Case Review in Progress...")
        prompt = PromptTemplate(
            template="""You are a senior test strategy expert reviewing the following test cases:

            **testcases:**
            {testcases}

            Provide structured feedback
            Format:
            - Status: Approved / Needs Fixes
            - Feedback: (Explain improvements)
            
            """,
            input_variables={"testcases"}
        )

        chain = prompt | llm.with_structured_output(Review)
        response = chain.invoke({
            "testcases": state["test_cases"]
        })
        state['status']=response.status
        state['feedback']=response.review
        
        with tabs_dict["Testing"]:
            st.subheader("Test Cases Review")
            st.write(f'**Status:** {response.status}')
            st.write(f'**Feedback:** {response.review}')
    return state

def fix_testcases_after_review(state):
    """Fixes testcases based on review feedback """
    with st.container():
        st.write("🔄 Fixing Test Cases...")
        prompt = PromptTemplate(
            template="""You are a Test case review expert fixing test cases.
            The following test cases was reviewed, and feedback is provided:

            **Original test cases:**
            {testcases}

            **testcases Review Feedback:**
            {feedback}

            Fix all issues

            Return only the corected test cases. 
            Do not include explanations.""",
            input_variables={"testcases", "feedback"}
        )

        chain = prompt | llm.with_structured_output(TestCases)
        fixed_testcases = chain.invoke({
            "testcases": state["test_cases"],
            "feedback": state["feedback"]
        })

        state["test_cases"] = fixed_testcases.cases
        
        with tabs_dict["Testing"]:
            st.subheader("Revised Test Cases")
            for i, case in enumerate(state['test_cases']):
                st.markdown(f"**Test Case {i+1}:**")
                st.markdown(case)
                st.divider()
    return state
    
def qa_testing(state):
    """Conducts QA testing."""
    with st.container():
        st.write("🔄 QA Testing in Progress...")
        prompt = PromptTemplate(
            template="""You are a seasoned QA engineer with expertise in thorough testing and quality validation.

            Task: Perform a comprehensive QA evaluation of the following code and test cases:
            Perform QA testing on code {code} with test cases {testcases}
            provide status(Approved/Not Approved) and feedback
            and provide test case execution/results in feed back
            """,
            input_variables={"code","testcases"}
        )
    
        chain = prompt | llm.with_structured_output(Review)
        response = chain.invoke({"code":state['code'],"testcases":state['test_cases']})
        state['status'] = response.status
        state['feedback'] = response.review
        
        with tabs_dict["QA"]:
            st.subheader("QA Testing Results")
            st.write(f'**Status:** {response.status}')
            st.write(f'**Feedback:** {response.review}')
    return state
    
def decision_qa(state):
    """Returns the next step based on status and feedback."""
    if state['status']=="Approved":
        return "Passed"
    else:
        return "Failed"

def fix_code_after_QA_feedback(state):
    """ Fixing code after QA testing"""
    with st.container():
        st.write("🔄 Fixing Code After QA Feedback...")
        prompt = PromptTemplate(
            template="""You are an expert software engineer responsible for fixing code based on QA Feedback.
            The following is code,test cases and QA testing feedback:

            **Original Code:**
            {code}

            **testcases:**
            {testcases}

            ** qa feedback**
            {qa_feedback}

            Fix the code to address all issues 
            Return only the corrected code, with proper indentation and structure, 
            and without any explanations.""",
            input_variables={"code", "testcases","qa_feedback"}
        )

        chain = prompt | llm 
        qa_code = chain.invoke({
            "code": state["code"],
            "testcases": state["test_cases"],
            "qa_feedback":state['feedback']})

        state["code"] = qa_code.content
        
        with tabs_dict["QA"]:
            st.subheader("Updated Code After QA Feedback")
            st.code(state['code'])
        
        with tabs_dict["Code"]:
            st.subheader("Final Code After QA Fixes")
            st.code(state['code'])
    return state


# Streamlit UI setup
st.set_page_config(page_title="Software Development Workflow", layout="wide")

st.title("📌 AI-Powered Software Development Workflow")
st.write("This app allows you to define requirements, generate user stories, and automate the entire development process.")

# Create tabbed interface
tab_names = ["Overview", "User Stories", "Design Docs", "Code", "Security", "Testing", "QA"]
tabs = st.tabs(tab_names)

# Create a dictionary to easily access tabs
tabs_dict = {name: tab for name, tab in zip(tab_names, tabs)}

# Overview tab content
with tabs_dict["Overview"]:
    st.header("Software Development Workflow")
    st.write("""
    This application automates the entire software development lifecycle using AI:
    
    1. **Requirements Gathering**: Enter your software requirements
    2. **User Stories Generation**: AI generates user stories from requirements
    3. **Product Owner Review**: AI reviews user stories like a product owner
    4. **Design Documents**: AI creates functional and technical design documents
    5. **Code Generation**: AI generates the code based on the design docs
    6. **Code Review**: AI reviews the code for quality and issues
    7. **Security Review**: AI identifies potential security vulnerabilities
    8. **Test Case Generation**: AI creates comprehensive test cases
    9. **QA Testing**: AI tests the code against the test cases
    
    Each step includes feedback loops to ensure quality throughout the process.
    """)
    
    requirements=st.text_area("Enter Requirements:", "", height=150)
    with st.sidebar:
                st.header("GROQ API")
                api=st.text_input("Enter your Groq API key", type="password")
                st.subheader("Workflow Diagram")
                st.image("workflow_diagram.png", caption="Workflow Execution")
                
    os.environ["GROQ_API_KEY"] = api
    llm = ChatGroq(model="gemma2-9b-it")
    
    # Display the start button in the Overview tab
    if st.button("Start Workflow"):
        initial_state = {
            "user_requirements": requirements,
        }
        
        # Setup the graph
        graph_builder = StateGraph(State)
        
        graph_builder.add_node("UI User Inputs Requirements", user_inputs_requirements)
        graph_builder.add_node("Auto-generate User Stories", auto_generate_user_stories)
        graph_builder.add_node("Product Owner Review", product_owner_review)
        graph_builder.add_node("Revise User Stories", revise_user_stories)
        graph_builder.add_node("Create Design Documents Functional and Technical", create_design_documents)
        graph_builder.add_node("Design Review", design_review)
        graph_builder.add_node("Revise Design Documents", revise_design_docs)
        graph_builder.add_node("Generate Code", generate_code)
        graph_builder.add_node("Code Review", code_review)
        graph_builder.add_node("Fix Code after Code Review", fix_code_after_code_review)
        graph_builder.add_node("Security Review", security_review)
        graph_builder.add_node("Fix Code after Security Review", fix_code_after_security_review)
        graph_builder.add_node("Write Test Cases", write_test_cases)
        graph_builder.add_node("Test Case Review", test_case_review)
        graph_builder.add_node("Fix Test Cases After Review", fix_testcases_after_review)
        graph_builder.add_node("QA Testing", qa_testing)
        graph_builder.add_node("Fix Code After QA Feedback", fix_code_after_QA_feedback)
        
        graph_builder.add_edge(START, "UI User Inputs Requirements")
        graph_builder.add_edge("UI User Inputs Requirements", "Auto-generate User Stories")
        graph_builder.add_edge("Auto-generate User Stories", "Product Owner Review")
        graph_builder.add_conditional_edges("Product Owner Review", decision, {"Approved":"Create Design Documents Functional and Technical", "Feedback":"Revise User Stories"})
        graph_builder.add_edge("Revise User Stories", "Product Owner Review")
        graph_builder.add_edge("Create Design Documents Functional and Technical", "Design Review")
        graph_builder.add_conditional_edges("Design Review", decision, {"Approved":"Generate Code", "Feedback":"Revise Design Documents"})
        graph_builder.add_edge("Revise Design Documents", "Design Review")
        graph_builder.add_edge("Generate Code", "Code Review")
        graph_builder.add_conditional_edges("Code Review", decision, {"Approved":"Security Review", "Feedback":"Fix Code after Code Review"})
        graph_builder.add_edge("Fix Code after Code Review", "Code Review")
        graph_builder.add_conditional_edges("Security Review", decision, {"Approved":"Write Test Cases", "Feedback":"Fix Code after Security Review"})
        graph_builder.add_edge("Fix Code after Security Review", "Security Review")
        graph_builder.add_edge("Write Test Cases", "Test Case Review")
        graph_builder.add_conditional_edges("Test Case Review", decision, {"Approved":"QA Testing", "Feedback":"Fix Test Cases After Review"})
        graph_builder.add_edge("Fix Test Cases After Review", "Test Case Review")
        graph_builder.add_conditional_edges("QA Testing", decision_qa, {"Passed":END, "Failed":"Fix Code After QA Feedback"})
        graph_builder.add_edge("Fix Code After QA Feedback", "Code Review")
        
        graph = graph_builder.compile()
       
        
        with st.status("Executing workflow...", expanded=True) as status:
            st.write("Starting software development workflow")
           
            final_state = graph.invoke(initial_state)
            status.update(label="Workflow completed!", state="complete", expanded=False)
            
            with tabs_dict["Overview"]:
                st.success("✅ Workflow completed successfully!")
                
                # Final workflow summary
                st.subheader("Workflow Summary")
                st.write("The AI has completed the entire software development lifecycle:")
                st.write("1. ✅ User Stories generated and approved")
                st.write("2. ✅ Design Documents created and reviewed")
                st.write("3. ✅ Code generated, reviewed, and secured")
                st.write("4. ✅ Test Cases created and validated")
                st.write("5. ✅ QA Testing completed")
