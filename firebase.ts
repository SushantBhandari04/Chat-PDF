import { getApp, getApps, initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";
import { getStorage } from "firebase/storage";


const firebaseConfig = {
    apiKey: "AIzaSyDB8XuTny3mp-StAdWg--Vna-vL6DQJBqw",
    authDomain: "chat-with-pdf-37e41.firebaseapp.com",
    projectId: "chat-with-pdf-37e41",
    storageBucket: "chat-with-pdf-37e41.firebasestorage.app",
    messagingSenderId: "396753183794",
    appId: "1:396753183794:web:9def3d9b0d6cc9b132c82b"
  };

  const app = getApps().length===0 ? initializeApp(firebaseConfig) : getApp();

  const db = getFirestore(app);
  const storage = getStorage(app);

  export { db, storage };