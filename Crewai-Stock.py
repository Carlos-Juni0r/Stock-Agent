

#Importações de bibliotecas
import json
import os #Biblioteca padrão de python
from datetime import datetime

import yfinance as yf
from crewai import Agent, Task, Crew, Process

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
import streamlit as st


#Criando ferramenta - Yahoo Finance Tool
def fetch_stock_price(ticket):
    stock = yf.download(ticket, start="2023-08-08", end="2024-08-08") #vai trazer histórico de preço das ações da Apple de 08 de agosto de 2023 até 08 de agosto de 2024
    return stock


yahoo_finance_tool = Tool(
    name = "Yahoo finance Tool",
    description = "Fetches stocks price for {ticket} from last year about a specific stock from Yahoo Finance API",
    func = lambda ticket: fetch_stock_price(ticket),
)


#Importando OpenAI LLM - GPT

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY'] #colocar a key da api do gpt

llm =ChatOpenAI(model="gpt-3.5-turbo")

#Criando o agente - Analista de preço de ações

stock_price_analytics = Agent(
    role="Senior stock price Analyst",        
    goal="Find the {ticket} stock price and analyses trends",        
    backstory="""You're a highly experienced in analyzing the price of an specific stock 
    and make predictions about its future price """,   
    verbose=True,   
    llm = llm,      
    max_iter= 5,    
    memory= True,   
    tools= [yahoo_finance_tool],
    allow_delegation=False        
)


#Criando tarefa - pegar preços da ação

get_stock_price = Task(
    description="Analyze the stock {ticket} price history and crate a trend analyses of up, down or sideways",
    expected_output="""Specify the current trend stock price -up, down or sideway. 
    example. stock='APPL, price up' 
    """,
    Agent = stock_price_analytics
)

#Criando Ferramenta - Ferramenta de pesquisa

#Importando ferramenta de pesquisa -> duck duck go
search_tool = DuckDuckGoSearchResults(backend='news', num_results=10)


#Criando o Agente - Analista de notícias
news_analyst = Agent(
    
    role="Stock News Analyst",        
    goal="""Creat a short summary of the market news related to the stock {ticket} company. Specify the current trend - up, down or 
    sideways with the news context. For each request stock asset, specify a number between 0 and 100, where 0 is extreme fear and 100 is extreme greed.""",
    backstory="""You're highly experienced in analyzing the market trends and news and have tracked assest for more then 10 years.
    
    You're also master level analyts in the tradicional markets and have deep understanding of human psychology.
    
    You understand news, their tittles and information, but you look at thoose with a health dose of skepticism. 
    You consider also the source of the news articles. 
    """,   
    verbose=True,   
    llm = llm,      
    max_iter= 10,    
    memory= True,   
    tools= [search_tool],
    allow_delegation=False       
)


#Criando tarefa para o agente - Pegar notícias e analisar

get_news = Task(
    description= f"""Take the stock and always include BTC to it(if not request).
    Use the search tool to search each one individually.
    
    The current date is {datetime.now()}.
    
    Compose the results into a helpfull report.
    """,         
    expected_output="""A summary of the overall market and one sentence summary for each request asset.
    Include a fear/greed score for each asset based on the news. Use format: 
    <STOCK ASSET>
    <SUMMARY BASED ON NEWS>
    <TREND PREDICTION>
    <FEAR/GREED SCORE>
    
    """,     
    Agent = news_analyst           
    
)


#Criando o Agente - Analista Final
stock_analyst_write = Agent(
    role="Senior Stock Analyst Writer",        
    goal="""Analyze the trend price and news and write an insighfull compelling and informative 3 paragraph long newsletter based on the stock report and price trend.""",        
    backstory="""You"re widely accepted as the best stock analyst in the market. 
    You understand complex concepts and create compelling stories and narratives that resonate with wider audiences.
    
    You understand macro factors and combine multiple theories - example: cycle theory and fundamental analyses.
    You're able to hold multple opinions when analyzing anything.
    """,   
    
    verbose=True,   
    llm = llm,      
    max_iter= 5,    
    memory= True,   
    allow_delegation=True
)

#Criando tarefa para o Agente - Resposta do analista final

write_analyzes = Task(
    description="""Use the stock price trend and the stock news report to create an analyses and write the newsletter about the {ticket} company
    that is brief and highlights the most important points.
    
    Focus on the stock price trend, news and fear/greed score. what are the near future considerations?
    Include the previous analyses  of stock trend and news summary.
    """,         
    expected_output="""An eloquent 3 paragraphs newsletter formated as markdown in an easy readable manner. It Should contain:
    
    - 3 bullet executive summary
    - Introduction - set the overall picture and spike up the interest
    - Main part provides the meat of the analysis including the news summary and fear/greed scores
    - Summary - key facts and concrete future trend prediction - up,down or sideways.
    
    """,     
    Agent = stock_analyst_write,
    context = [get_news, get_stock_price]
                  
    )


#Criando o grupo de agentes

crews = Crew(
    agents=[stock_price_analytics, news_analyst, stock_analyst_write],
    tasks=[get_stock_price, get_news, write_analyzes],
    verbose= True,
    process= Process.hierarchical,
    full_output=True,
    share_crew=False,
    manager_llm=llm,
    max_inter=15   
    
)


#results= crews.kickoff(inputs={'ticket': 'AAPL'})
#results['final_output']

with st.sidebar:
    st.header("Enter the stock to research")
    
    with st.form(key='research_form'):
        topic = st.text_input("Select the ticket")
        submit_button = st.form_submit_button(label = "Run Research")
        
if submit_button:
    if not topic:
         st.error("Please fill the ticket field")
    else:
        results= crews.kickoff(inputs={'ticket': topic})       
        
        st.subheader("Results of your research:")
        
        st.write(results.raw)
        
        