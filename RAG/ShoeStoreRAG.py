import time
import openai
from pinecone import Pinecone
from pinecone import ServerlessSpec


class ShoeStoreRAG:

    def __init__(self,pinecone_api_key=None,openai_api_key=None):
        """
        Initializes pinecone and openAI classes. API keys are set in CLI using
        $env:OPENAI_API_KEY="value" and $env:PINECONE_API_KEY="value"
        """
        self.pc = Pinecone(API_KEY=pinecone_api_key)
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.index=None
        self.index_name='shoe-store-sp-2'

        self.setup_pinecone_index()
        self.populate_knowledge_base()

    def setup_pinecone_index(self):
        """
        Creates index with index name if it is not existing already with pinecone
        :return: None
        """
        if self.index:
            print(f'skipping index creation')
        if not self.pc.has_index(self.index_name):
            print(f'creating index {self.index_name}')
            self.pc.create_index(name=self.index_name,
                                 metric='cosine',
                                 dimension=1536,
                                 spec=ServerlessSpec(
                                     cloud='aws',
                                     region='us-east-1'
                                 )
                                 )
            time.sleep(10)
        self.index=self.pc.Index(self.index_name)
        if self.index:
            print('index identified successfully')

    def get_embedding(self,text):
        """
        This method creates vector embedding using Open AI embedding for text
        :param text:
        :return: Embedding vector for text passed
        """
        try:
            response=self.openai_client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            print(e)
            return None

    def populate_knowledge_base(self):
        """
        populates (upserts) the knowledge base with the vector embeddings of documents below
        :return: None
        """
        documents = [
            {
                "id": "policy_returns",
                "text": "We offer a 30-day full refund policy at no extra cost for all shoes. No questions asked if you're not satisfied with your purchase.",
                "metadata": {"category": "returns", "topic": "refund_policy"}
            },
            {
                "id": "policy_shipping",
                "text": "Free shipping is available on all orders over $75. Standard shipping takes 3-5 business days, express shipping takes 1-2 business days for an additional $15.",
                "metadata": {"category": "shipping", "topic": "delivery_options"}
            },
            {
                "id": "inventory_sizes",
                "text": "Our shoe sizes range from US 5 to US 15 in both men's and women's styles. We also carry wide and narrow width options for most models.",
                "metadata": {"category": "inventory", "topic": "sizes_availability"}
            },
            {
                "id": "warranty_athletic",
                "text": "All our athletic shoes come with a 1-year warranty against manufacturing defects. This covers sole separation, stitching issues, and material defects.",
                "metadata": {"category": "warranty", "topic": "product_warranty"}
            },
            {
                "id": "inventory_brands",
                "text": "We carry popular brands including Nike, Adidas, New Balance, Converse, Vans, and our exclusive store brand ComfortWalk.",
                "metadata": {"category": "inventory", "topic": "brands"}
            },
            {
                "id": "store_hours",
                "text": "Our store hours are Monday-Saturday 9 AM to 9 PM, Sunday 11 AM to 7 PM. We're located at 123 Main Street, downtown shopping district.",
                "metadata": {"category": "store_info", "topic": "hours_location"}
            },
            {
                "id": "services_fitting",
                "text": "We offer professional shoe fitting services. Our certified fitters can measure your feet and recommend the best size and width for optimal comfort.",
                "metadata": {"category": "services", "topic": "fitting_service"}
            },
            {
                "id": "discounts_student",
                "text": "Student discounts are available - show your student ID for 15% off your purchase. Military personnel receive 20% discount with valid military ID.",
                "metadata": {"category": "discounts", "topic": "student_military"}
            },
            {
                "id": "loyalty_rewards",
                "text": "We have a loyalty program called SoleRewards. Earn 1 point for every dollar spent, get $5 off for every 100 points earned.",
                "metadata": {"category": "loyalty", "topic": "rewards_program"}
            },
            {
                "id": "services_custom",
                "text": "Custom shoe orders are available for select brands. Custom orders typically take 4-6 weeks to complete and require a 50% deposit upfront.",
                "metadata": {"category": "services", "topic": "custom_orders"}
            }
        ]
        vectors_to_upsert=[]
        for doc in documents:
            embedding=self.get_embedding(doc['text'])
            vectors_to_upsert.append(
                {
                    "id": doc['id'],
                    "values": embedding,
                    "metadata": {
                        **doc['metadata'],
                        "text": doc['text']
                    }
                }
            )
        if vectors_to_upsert:
            self.index.upsert(vectors_to_upsert,
                              namespace="shoe-store-test-data")

        print('upserting vectors to be available')
        time.sleep(5)

    def retrieve_context(self,query,topk=1):
        retrieved_docs=[]
        query_embedding=self.get_embedding(query)
        results=self.index.query(namespace='shoe-store-test-data',
                         vector=query_embedding,
                         top_k=topk,
                         include_metadata=True
                         )
        for match in results['matches']:
            if 'text' in match['metadata']:
                retrieved_docs.append(match['metadata']['text'])
        return retrieved_docs if retrieved_docs else []

    def generate_answers(self,query,context):
        if not context:
            return "No relevant context found to answer the question."
        context_str='\n\n'.join(context)
        prompt=f"""
        You are a helpful customer service assistant for a shoe store. Use provided context to 
        answer customer's question accurately and in a friendly manner.
        context: {context_str}
        customer question:{query}
        Please provide a helpful answer.If context does not fully answer the question, say so politely.
        """
        response=self.openai_client.chat.completions.create(
            model='gpt-4o',
            messages=[
                {"role": "system", "content": "You are a friendly shoe store customer service assistant"},
                {"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()



