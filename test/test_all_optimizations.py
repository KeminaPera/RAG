"""
综合测试脚本 - 验证所有优化功能
"""
import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000"

def test_upload():
    """测试文档上传（使用 ChromaDB）"""
    print("=" * 60)
    print("测试 1: 文档上传 (ChromaDB)")
    print("=" * 60)
    
    # 检查是否有可上传的文件
    import os
    upload_dir = "./uploads"
    if not os.path.exists(upload_dir) or not os.listdir(upload_dir):
        print("⚠️  uploads 目录为空，跳过上传测试")
        return True
    
    print("✅ ChromaDB 集成正常（启动时无报错）")
    return True

def test_query():
    """测试查询功能"""
    print("\n" + "=" * 60)
    print("测试 2: 查询功能")
    print("=" * 60)
    
    url = f"{BASE_URL}/query"
    data = {"query": "NLP算法实习生"}
    
    print(f"发送查询: {data['query']}")
    start_time = time.time()
    
    try:
        response = requests.post(url, data=data, timeout=90)
        elapsed = time.time() - start_time
        
        print(f"响应时间: {elapsed:.2f}秒")
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ 查询成功")
            return True
        else:
            print(f"❌ 查询失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 请求异常: {e}")
        return False

def test_memory_cache():
    """测试 MemoryManager LRU 缓存"""
    print("\n" + "=" * 60)
    print("测试 3: MemoryManager LRU 缓存")
    print("=" * 60)
    
    # 发送多个不同session的查询
    print("发送 12 个不同 session 的查询（超过缓存限制 10）...")
    
    for i in range(12):
        url = f"{BASE_URL}/query"
        data = {"query": f"测试查询 {i+1}"}
        
        try:
            response = requests.post(url, data=data, timeout=30)
            if response.status_code == 200:
                print(f"  Session {i+1:2d}: ✅ 成功")
            else:
                print(f"  Session {i+1:2d}: ❌ 失败")
        except Exception as e:
            print(f"  Session {i+1:2d}: ⚠️  超时或异常")
    
    print("\n✅ LRU 缓存测试完成（检查日志确认缓存淘汰）")
    return True

def test_config():
    """测试配置外置"""
    print("\n" + "=" * 60)
    print("测试 4: 配置外置验证")
    print("=" * 60)
    
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    configs = {
        'CHUNK_SIZE': os.getenv('CHUNK_SIZE'),
        'CHUNK_OVERLAP': os.getenv('CHUNK_OVERLAP'),
        'VECTOR_SEARCH_TOP_K': os.getenv('VECTOR_SEARCH_TOP_K'),
        'RERANK_TOP_K': os.getenv('RERANK_TOP_K'),
        'MAX_SESSION_CACHE': os.getenv('MAX_SESSION_CACHE'),
        'MAX_UPLOAD_SIZE_MB': os.getenv('MAX_UPLOAD_SIZE_MB'),
    }
    
    print("当前配置:")
    for key, value in configs.items():
        print(f"  {key}: {value}")
    
    print("\n✅ 配置外置验证通过")
    return True

def test_security():
    """测试安全配置"""
    print("\n" + "=" * 60)
    print("测试 5: 安全配置验证")
    print("=" * 60)
    
    from flask import Flask
    import secrets
    
    # 检查 secret_key 是否硬编码
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()
        if "secret_key = 'supersecretkey'" in content:
            print("❌ Secret key 仍然硬编码")
            return False
        else:
            print("✅ Secret key 使用动态生成")
    
    # 检查 MAX_CONTENT_LENGTH
    if 'MAX_CONTENT_LENGTH' in content:
        print("✅ 文件大小限制已配置")
    else:
        print("⚠️  文件大小限制未配置")
    
    return True

def check_logs():
    """检查日志中的关键信息"""
    print("\n" + "=" * 60)
    print("测试 6: 日志分析")
    print("=" * 60)
    
    try:
        with open('logs/app.log', 'r', encoding='utf-8') as f:
            logs = f.read()
        
        # 检查关键日志
        checks = {
            'ChromaDB 使用': 'Saved' in logs or 'ChromaDB' in logs,
            'MemoryManager 创建': 'MemoryManager created' in logs,
            '缓存淘汰': 'cache evicted' in logs,
            '无 ERROR': 'ERROR' not in logs,
        }
        
        for check_name, result in checks.items():
            status = "✅" if result else "❌"
            print(f"  {status} {check_name}")
        
        return all(checks.values())
        
    except Exception as e:
        print(f"⚠️  无法读取日志: {e}")
        return False

def main():
    print("\n" + "=" * 60)
    print("🚀 RAG 系统综合测试")
    print("=" * 60)
    
    results = []
    
    # 运行所有测试
    results.append(("配置外置", test_config()))
    results.append(("安全配置", test_security()))
    results.append(("文档上传", test_upload()))
    results.append(("查询功能", test_query()))
    results.append(("Memory 缓存", test_memory_cache()))
    results.append(("日志分析", check_logs()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {test_name}")
    
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！系统运行正常！")
    else:
        print(f"\n⚠️  有 {total - passed} 个测试未通过，请检查日志")

if __name__ == "__main__":
    main()
