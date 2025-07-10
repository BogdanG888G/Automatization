from sqlalchemy import text

def call_stored_procedure(network: str, engine):
    try:
        proc_name = f"{network}.upload_to_stage"
        with engine.connect() as conn:
            conn.execute(text(f"EXEC {proc_name}"))
            conn.commit()
        print(f"[INFO] Процедура {proc_name} успешно выполнена.")
        return True
    except Exception as e:
        print(f"[ERROR] Ошибка при вызове процедуры: {e}")
        raise

    