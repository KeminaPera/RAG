import os
os.chdir(r"e:\idea_workspace\RAG")

from memory_manager import ShortTermMemory, EntityMemory, LongTermMemory, MemoryManager

print("Testing ShortTermMemory...")
sm = ShortTermMemory()
sm.add_message('user', 'Hello')
sm.add_message('assistant', 'Hi there')
print("Short-term memory history:")
print(sm.get_history())
print()

print("Testing EntityMemory...")
em = EntityMemory()
em.add_entity('test_user', {'name': 'Test', 'preference': 'local model', 'tool': 'Obsidian'})
entity = em.get_entity('test_user')
print(f"Entity retrieved: {entity}")
print(f"Formatted entity prompt:\n{em.format_entity_prompt(['test_user'])}")
print()

print("Testing LongTermMemory...")
lm = LongTermMemory()
lm.add_memory('Test knowledge about RAG', {'type': 'test', 'topic': 'RAG'})
lm.add_memory('Another knowledge about LLM', {'type': 'test', 'topic': 'LLM'})
results = lm.retrieve('RAG')
print(f"Retrieved {len(results)} memories")
print(f"Formatted long-term prompt:\n{lm.format_retrieved_memory('RAG')}")
print()

print("Testing MemoryManager...")
mm = MemoryManager('test_session')
mm.add_short_term('user', 'What is RAG?')
mm.add_short_term('assistant', 'RAG is Retrieval-Augmented Generation')
mm.update_entity('user1', {'name': 'John', 'interest': 'AI'})
mm.add_long_term('RAG combines retrieval with generation', {'type': 'knowledge'})

system_prompt, user_query = mm.build_prompt('Tell me more about RAG', 'user1')
print("Built system prompt:")
print(system_prompt)
print()

print("All tests passed!")