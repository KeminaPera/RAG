"""
查看已存储的数据 - 快速检查工具
"""
import os
import json
import sqlite3

os.chdir(r"e:\idea_workspace\RAG")

print("=" * 70)
print("📊 RAG 系统数据存储检查")
print("=" * 70)

# 1. 检查 ChromaDB
print("\n1️⃣ ChromaDB 向量数据库")
print("-" * 70)

try:
    import chromadb
    
    client = chromadb.PersistentClient(path="./data/chromadb")
    
    # 列出所有 collection
    collections = client.list_collections()
    print(f"✅ Collection 数量: {len(collections)}")
    
    for coll in collections:
        print(f"\n📁 Collection: {coll.name}")
        
        # 获取数据
        collection = client.get_collection(coll.name)
        count = collection.count()
        print(f"   📄 文档总数: {count}")
        
        if count > 0:
            # 获取样本
            sample = collection.get(limit=2)
            
            print(f"\n   📝 示例文档:")
            for i in range(min(2, len(sample['ids']))):
                print(f"   ┌─ 文档 {i+1}")
                print(f"   │ ID: {sample['ids'][i]}")
                print(f"   │ 内容: {sample['documents'][i][:100]}...")
                print(f"   │ 元数据: {sample['metadatas'][i]}")
                print(f"   └─")
        
        # 获取磁盘占用
        import subprocess
        result = subprocess.run(
            ['powershell', '-Command', 
             f'(Get-ChildItem -Path data/chromadb -Recurse | Measure-Object -Property Length -Sum).Sum / 1KB'],
            capture_output=True, text=True
        )
        try:
            size_kb = float(result.stdout.strip())
            print(f"\n   💾 磁盘占用: {size_kb:.2f} KB")
        except:
            pass
            
except Exception as e:
    print(f"❌ 读取 ChromaDB 失败: {e}")

# 2. 检查 Entity Memory
print("\n\n2️⃣ Entity Memory 实体记忆")
print("-" * 70)

try:
    db_path = "./data/memory/entity_memory.db"
    
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM entities")
        count = cursor.fetchone()[0]
        print(f"✅ 实体数量: {count}")
        
        if count > 0:
            cursor.execute("SELECT * FROM entities LIMIT 2")
            rows = cursor.fetchall()
            
            print(f"\n📝 示例实体:")
            for i, row in enumerate(rows, 1):
                print(f"   ┌─ 实体 {i}")
                print(f"   │ 名称: {row[1]}")
                print(f"   │ 属性: {row[2]}")
                print(f"   │ 创建时间: {row[3]}")
                print(f"   │ 更新时间: {row[4]}")
                print(f"   └─")
        
        conn.close()
    else:
        print(f"⚠️  数据库文件不存在: {db_path}")
        
except Exception as e:
    print(f"❌ 读取 Entity Memory 失败: {e}")

# 3. 检查 Long-Term Memory
print("\n\n3️⃣ Long-Term Memory 长期记忆")
print("-" * 70)

try:
    memory_file = "./data/memory/long_term/memory.json"
    
    if os.path.exists(memory_file):
        with open(memory_file, "r", encoding="utf-8") as f:
            memories = json.load(f)
        
        print(f"✅ 记忆条目数: {len(memories)}")
        
        if len(memories) > 0:
            print(f"\n📝 示例记忆:")
            for i, mem in enumerate(memories[:2], 1):
                content = mem.get('content', '')
                metadata = mem.get('metadata', {})
                created_at = mem.get('created_at', 'N/A')
                
                print(f"   ┌─ 记忆 {i}")
                print(f"   │ 内容: {content[:100]}...")
                print(f"   │ 元数据: {metadata}")
                print(f"   │ 创建时间: {created_at}")
                print(f"   └─")
        
        # 文件大小
        size_kb = os.path.getsize(memory_file) / 1024
        print(f"\n   💾 文件大小: {size_kb:.2f} KB")
    else:
        print(f"⚠️  记忆文件不存在: {memory_file}")
        
except Exception as e:
    print(f"❌ 读取 Long-Term Memory 失败: {e}")

# 4. 存储总览
print("\n\n4️⃣ 存储总览")
print("-" * 70)

try:
    import subprocess
    result = subprocess.run(
        ['powershell', '-Command', 
         'Get-ChildItem -Path data -Recurse | Measure-Object -Property Length -Sum | Select-Object -ExpandProperty Sum'],
        capture_output=True, text=True
    )
    
    if result.returncode == 0:
        total_bytes = float(result.stdout.strip())
        print(f"📊 总存储占用:")
        print(f"   • {total_bytes / 1024:.2f} KB")
        print(f"   • {total_bytes / 1024 / 1024:.2f} MB")
except:
    pass

print("\n" + "=" * 70)
print("✅ 检查完成！")
print("=" * 70)
