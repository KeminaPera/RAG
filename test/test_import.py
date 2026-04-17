try:
    import app
    print("App imported successfully")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()