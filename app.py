 
import pickle
import streamlit as st
 
# loading the trained model
pickle_in = open('model2.p', 'rb') 
classifier = pickle.load(pickle_in)
 
@st.cache()
  
# defining the function which will make the prediction using the data which the user inputs 
def prediction(Gender, Married, Dependents, Education, Self_Employed):   
 
    # Pre-processing user input    
    if Gender == "Male":
        Gender = 0
    else:
        Gender = 1
 
    if Married == "Unmarried":
        Married = 0
    else:
        Married = 1
 
#     if Credit_History == "Unclear Debts":
#         Credit_History = 0
#     else:
#         Credit_History = 1  
 
#     LoanAmount = LoanAmount / 1000
 
    # Making predictions 
#     prediction = classifier.predict( 
#         [[Gender, Married, ApplicantIncome, LoanAmount, Credit_History]])
     
    data = {
        'Gender':Gender,           
        'Married':Married,         
        'Dependents':Dependents,       
        'Education':Education,        
        'Self_Employed':Self_Employed,    
        'ApplicantIncome':3406,  
        'CoapplicantIncome':4417.0,
        'LoanAmount':4.812184,       
        'Loan_Amount_Term':360.0, 
        'Credit_History':1.0,  
        'Property_Area': '1',    
        'total_income':8.964823
    }
    
    import requests
    URL = "http://127.0.0.1:5000/scoring"
    r = requests.post(url = URL, json = data) 
    
#     if prediction == 0:
#         pred = 'Rejected'
#     else:
#         pred = 'Approved'
    return r.json()
      
  
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Loan Prediction ML App</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    Gender = st.selectbox('Gender',("Male","Female"))
    Married = st.selectbox('Marital Status',("Unmarried","Married")) 
    Dependents = st.number_input("Applicants monthly income") 
    Education = st.number_input("Total loan amount")
    Self_Employed = st.selectbox('Credit_History',("Unclear Debts","No Unclear Debts"))
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(Gender, Married, Dependents, Credit_History,Self_Employed) 
        st.success('Your loan is {}'.format(result))
        print(LoanAmount)
     
if __name__=='__main__': 
    main()
