from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

#  创建 Prompt 模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个得力的助手。你可以使用工具来回答问题。"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])