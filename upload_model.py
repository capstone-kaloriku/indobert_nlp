import os
from pathlib import Path
try:
    from huggingface_hub import HfApi, login
except ImportError:
    print("Menginstall huggingface_hub...")
    os.system("pip install huggingface_hub")
    from huggingface_hub import HfApi, login

def upload():
    print("="*50)
    print(" 🤗 UPLOADER MODEL INDOBERT KE HUGGING FACE 🤗")
    print("="*50)
    
    # 1. Minta Token
    token = input("Masukkan Hugging Face Access Token (tipe WRITE): ").strip()
    if not token:
        print("Error: Token tidak boleh kosong.")
        return
        
    try:
        login(token=token)
        # Ambil username otomatis dari token agar tidak salah namespace (403 Forbidden)
        from huggingface_hub import whoami
        user_info = whoami(token=token)
        username = user_info.get("name")
        if not username:
            raise ValueError("Gagal mendapatkan username dari token.")
        print(f"Login sukses! Terdeteksi akun: {username}")
    except Exception as e:
        print(f"Gagal login ke Hugging Face: {e}")
        return
        
    # 2. Minta Nama Repo saja
    repo_name = input("Masukkan nama repo baru (default: indobert-intent-classifier): ").strip()
    if not repo_name:
        repo_name = "indobert-intent-classifier"
        
    repo_id = f"{username}/{repo_name}"
    
    # 3. Path Model Lokal
    model_dir = Path(__file__).parent / "models" / "indobert_intent_classifier"
    if not model_dir.exists():
        print(f"Error: Folder model tidak ditemukan di {model_dir}")
        print("Pastikan lo sudah running training (Step 3).")
        return
        
    print(f"\nMemulai upload folder {model_dir} ke repo {repo_id}...")
    
    try:
        api = HfApi()
        
        # Buat repo jika belum ada
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        
        # Upload folder model (mengabaikan checkpoint)
        api.upload_folder(
            folder_path=str(model_dir),
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=["checkpoint-*", "training_args.bin"]
        )
        print("\n" + "="*50)
        print("🎉 SUCCESS! Model berhasil di-upload ke Hugging Face!")
        print(f"Repo ID: {repo_id}")
        print("="*50)
        print("\nSekarang, kabarin gue Repo ID lo di atas biar gue ubah kodenya.")
        
    except Exception as e:
        print(f"\nTerjadi error saat upload: {e}")

if __name__ == "__main__":
    upload()
