from langchain import PromptTemplate
from langchain import LLMChain
from langchain.llms import CTransformers
import time

start_time = time.time()
'''
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

DEFAULT_SYSTEM_PROMPT="""\
You are an Expert NER tagger for Organisation names and Institution names like banks and financial corporations, Dont include Person names or Location into this list."""

instruction = "Please identify all the institutions and organisations names in the format {{ORG:[List of all Organisation names]}} from the below text. Please dont include any locations like city, disrtict or country to this list. : \n\n {text}"

SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS

template = B_INST + SYSTEM_PROMPT + instruction + E_INST

print(template)
prompt = PromptTemplate(template=template, input_variables=["text"])

'''

template = """Please identify all the company,organisation names or corporation names present in the below text. Organisation names can be banks, Financial institution, corportations ending with corp or inc, ltd or limited.
If there are no entity of this type return empty list and nothing else

Text: {text}

Question: Please return the company,organisation names or corporation names present as dictionary in this format {{'ORG': [list of Orgs present]}}

Answer: """

prompt = PromptTemplate(template=template, input_variables=["text"])

llm = CTransformers(model='./llama-2-13b-chat.ggmlv3.q4_0.bin',
                    model_type='llama',
                    config={'max_new_tokens': 128,
                            'temperature': 0.01}
                   )

LLM_Chain=LLMChain(prompt=prompt, llm=llm)


print(LLM_Chain.run('''[Execution] , ABL CREDIT AGREEMENT, dated as of, April 30, 2015, among EXIDE TECHNOLOGIES,   as US Borrower, EXIDE TECHNOLOGIES CANADA CORPORATION,  as Canadian Borrower, EXIDE TECHNOLOGIES [(]TRANSPORTATION[)] LIMITED and  GNB INDUSTRIAL POWER [(]UK[)] LIMITED,  as UK Borrowers, EXIDE HOLDING NETHERLANDS B.V.,   as Dutch Borrower, the other Borrowers from time to time party hereto, the Lenders from time to time party hereto, BANK OF AMERICA, N.A.,  as Administrative Agent, BANK OF AMERICA, N.A.,  PNC BANK, NATIONAL ASSOCIATION and  BANK OF MONTREAL,  as Co-Collateral Agents, BANK OF AMERICA, N.A.,  as Issuing Bank, and BANK OF AMERICA, N.A.,  as Swingline Lender, BANK OF AMERICA, N.A.,   as Syndication Agent, and BANK OF AMERICA, N.A.,  PNC CAPITAL MARKETS LLC and  BMO CAPITAL MARKETS CORP.  as Lead Arrangers and Joint Bookrunners, 3614380.16
1.1 Name. The name of the Joint Venture shall be .the Comstock/Skanska, A Joint Venture, which shall be formed under the .laws of the State of New York, and the business of the Joint Venture shall be  carried on under that name and under no other name.
BETWEEN: BMO First Canadian Capital Partners (GP) Inc., a corporation existing under the Canada Business Corporations Act (the "General Partner") - and - BMO First Canadian Capital Partners (Founder Partner) LP, a limited partnership existing under the laws of Ontario (the "Founder LP") - and - Iain Munro (the "Initial LP") - and - Each Person who, from time to time, becomes a limited partner in accordance with the terms of this Agreement (individually, a "Limited Partner" and collectively with the Founder LP, the "Limited Partners")'''))

end_time =time.time()
print(f"Time taken: {end_time-start_time:.6f} seconds")