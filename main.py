import os
import json
from textwrap import dedent
from enum import Enum
from dotenv import load_dotenv, find_dotenv
import openai
from openai import OpenAI
from pydantic import BaseModel

# Carica le variabili dal file .env
load_dotenv(find_dotenv())

# Configura il client per Groq oppure OpenAI ufficiale
# Opzione 1: Groq
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)
MODEL = "openai/gpt-oss-120b"

# Opzione 2: OpenAI ufficiale (decommenta per usare OpenAI)
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# MODEL = "gpt-4o"


# Esempio 1: Math tutor con Pydantic e parse

math_tutor_prompt = """
    Sei un tutor di matematica. 
    Ti verrà fornito un problema matematico,
    e il tuo obiettivo sarà quello di fornire una soluzione 
    passo dopo passo, insieme a una risposta finale.
    Per ogni passaggio, è sufficiente fornire l'output come equazione 
    e utilizzare il campo della spiegazione per dettagliare il ragionamento.
"""

class MathReasoning(BaseModel):
    class Step(BaseModel):
        explanation: str
        output: str

    steps: list[Step]
    final_answer: str

def get_math_solution(question: str):
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": dedent(math_tutor_prompt)},
            {"role": "user", "content": question},
        ],
        response_format=MathReasoning,
    )
    return completion.choices[0].message

question = "come posso risolvere 8 x + 7 = -23"
function_result = get_math_solution(question)
result = function_result.parsed

print("Esempio 1 - Math tutor:")
for step in result.steps:
    print("Spiegazione:", step.explanation)
    print("Output:", step.output)
    print()

print("Risposta finale:", result.final_answer)
print()


# Esempio 2: Personal shopper con tool calling

product_search_prompt = """
    Sei un personal shopper, specializzato nel trovare l'abbinamento perfetto per un utente.
    Ti viene fornito un input dell'utente e un contesto aggiuntivo,
    come il sesso e la fascia d'età dell'utente e la stagione.
    Sei dotato di uno strumento per cercare in un database gli abiti che corrispondono al profilo 
    e alle preferenze dell'utente.
    In base all'input dell'utente e al contesto, determinate il valore più probabile 
    dei parametri da utilizzare per la ricerca nel database.
    
    Ecco le diverse categorie disponibili sul sito:
    - shoes: boots, sneakers, sandals
    - jackets: winter coats, cardigans, parkas, rain jackets
    - tops: shirts, blouses, t-shirts, crop tops, sweaters
    - bottoms: jeans, skirts, trousers, joggers    
    
    È disponibile un'ampia gamma di colori, ma cerca di attenerti a nomi di colori comuni.
"""

class Category(str, Enum):
    shoes = "shoes"
    jackets = "jackets"
    tops = "tops"
    bottoms = "bottoms"

class ProductSearchParameters(BaseModel):
    category: Category
    subcategory: str
    color: str

def get_response(user_input, context):
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": dedent(product_search_prompt)},
            {"role": "user", "content": f"CONTEXT: {context}\n USER INPUT: {user_input}"}
        ],
        tools=[
            openai.pydantic_function_tool(
                ProductSearchParameters,
                name="product_search",
                description="Search for a match in the product database"
            )
        ]
    )
    return response.choices[0].message.tool_calls

user_input = "Sto cercando un nuovo cappotto. Ho sempre freddo, quindi per favore qualcosa di caldo! L'ideale sarebbe qualcosa che si abbini ai miei occhi."
context = "Gender: female, Age group: 40-50, Physical appearance: blue eyes"

tool_calls = get_response(user_input, context)

print("Esempio 2 - Personal shopper (Tool calling):")
for tool_call in tool_calls:
    args = json.loads(tool_call.function.arguments)
    print("Funzione:", tool_call.function.name)
    print("Parametri estratti:")
    for key, value in args.items():
        print(f"- {key}: {value}")
