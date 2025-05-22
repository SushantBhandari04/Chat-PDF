'use server'

import { generateEmbeddingsInPineconeVectoreStore } from "@/lib/langchain";
import { auth } from "@clerk/nextjs/server";
import { revalidatePath } from "next/cache";

export async function generateEmbeddings(docId: string){
    auth.protect(); // Protect this route with clerk

    // turn a PDF into embeddings
    await generateEmbeddingsInPineconeVectoreStore(docId);

    revalidatePath("/dashboard");

    return { completed: true };

}