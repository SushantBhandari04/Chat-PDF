import { ChatOpenAI } from "@langchain/openai";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import pineconeClient from "./pinecone";
import { PineconeStore } from "@langchain/pinecone";
import { Index, RecordMetadata } from "@pinecone-database/pinecone";
import { adminDb } from "@/firebaseAdmin";
import { auth } from "@clerk/nextjs/server";
import fetch from "node-fetch";

import { GoogleGenAI } from "@google/genai";
import { CohereClient } from "cohere-ai";

const cohere = new CohereClient({
    token: process.env.COHERE_API_KEY!,
});

const ai = new GoogleGenAI({
    apiKey: process.env.GOOGLE_API_KEY!,
});

export const indexName = "chat-ai";



export async function fetchMessageFromDB(docId: string) {
    const { userId } = await auth();
    if (!userId) throw new Error("User not found.");

    const chatsSnapshot = await adminDb
        .collection("users")
        .doc(userId)
        .collection("files")
        .doc(docId)
        .collection("chat")
        .orderBy("createdAt", "desc")
        .get();

    return chatsSnapshot.docs.map((doc) =>
        doc.data().role === "human"
            ? new HumanMessage(doc.data().message)
            : new AIMessage(doc.data().message)
    );
}

export async function generateDocs(docId: string) {
    const { userId } = await auth();
    if (!userId) throw new Error("User not found");

    const fileDoc = await adminDb
        .collection("users")
        .doc(userId)
        .collection("files")
        .doc(docId)
        .get();

    const downloadUrl = fileDoc.data()?.downloadUrl;
    if (!downloadUrl) throw new Error("No download URL found in Firebase.");

    // Fetch PDF as blob
    const response = await fetch(downloadUrl);
    const data = await response.blob();

    // Load PDF and split
    const loader = new PDFLoader(data);
    const docs = await loader.load();

    const splitter = new RecursiveCharacterTextSplitter();
    return splitter.splitDocuments(docs);
}

async function nameSpaceExists(index: Index<RecordMetadata>, namespace: string) {
    const { namespaces } = await index.describeIndexStats();
    return namespaces?.[namespace] !== undefined;
}

// Adapter to satisfy PineconeStore expected interface
class CohereEmbeddingsAdapter {
    embeddings: number[][];

    constructor(embeddings: number[][]) {
        this.embeddings = embeddings;
    }

    async embedDocuments(): Promise<number[][]> {
        return this.embeddings;
    }

    async embedQuery(): Promise<number[]> {
        return this.embeddings[0];
    }
}

export async function generateEmbeddingsInPineconeVectoreStore(docId: string) {
    const { userId } = await auth();
    if (!userId) throw new Error("User not found");

    const index = await pineconeClient.index(indexName);
    const exists = await nameSpaceExists(index, docId);

    if (exists) {
        // Namespace already exists, load from Pinecone
        return PineconeStore.fromExistingIndex(new CohereEmbeddingsAdapter([]), {
            pineconeIndex: index,
            namespace: docId,
        });
    }

    // Load and split document pages
    const splitDocs = await generateDocs(docId);
    const docTexts = splitDocs.map((doc) => doc.pageContent);

    // Generate Cohere embeddings
    const embedResponse = await cohere.v2.embed({
        model: "embed-v4.0",
        inputType: "search_document",
        embeddingTypes: ["float"],
        texts: docTexts,
    });

    if (!embedResponse || !embedResponse.embeddings) {
        throw new Error("Failed to generate embeddings from Cohere.");
    }

    // Extract float embeddings array
    const embeddingsArray: number[][] =
        (embedResponse.embeddings as any).float ?? embedResponse.embeddings;

    const adapter = new CohereEmbeddingsAdapter(embeddingsArray);

    return PineconeStore.fromDocuments(splitDocs, adapter, {
        pineconeIndex: index,
        namespace: docId,
    });
}

// Generate text using Gemini (Google GenAI)
export const generateTextWithGemini = async (
    prompt: string,
    config?: {
        model?: "gemini-2.0-flash" | "gemini-2.0-pro";
        temperature?: number;
        maxOutputTokens?: number;
        systemInstruction?: string;
    }
) => {
    const {
        model = "gemini-2.0-pro",
        temperature,
        maxOutputTokens,
        systemInstruction,
    } = config || {};

    const response = await ai.models.generateContent({
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: prompt }] }],
        config: {
            ...(systemInstruction && { systemInstruction }),
            ...(temperature !== undefined && { temperature }),
            ...(maxOutputTokens !== undefined && { maxOutputTokens }),
        },
    });

    return response.text;
};

// Main LangChain completion function with retrieval
export const generateLangchainCompletion = async (
    docId: string,
    question: string
) => {
    const pineconeStore = await generateEmbeddingsInPineconeVectoreStore(docId);
    if (!pineconeStore) throw new Error("Vector store not found.");

    const index = await pineconeClient.index(indexName);
    const chatHistory = await fetchMessageFromDB(docId);

    // Generate concise search query based on chat history + question
    const historyAwarePrompt = `
Based on this chat history:
${chatHistory.map((msg) => msg.content).join("\n")}

Generate a concise search query for: "${question}"
Respond with only the query.
`;

    const searchQuery = await generateTextWithGemini(historyAwarePrompt, {
        model: "gemini-2.0-flash",
        temperature: 0.7,
        maxOutputTokens: 300,
    });

    if (!searchQuery || searchQuery.trim().length === 0) {
        return "Sorry, I couldn't generate a valid search query.";
    }

    // Generate query embedding with Cohere
    let queryVector: number[];
    try {
        const queryEmbedResponse = await cohere.v2.embed({
            model: "embed-v4.0",
            inputType: "search_query",
            embeddingTypes: ["float"],
            texts: [searchQuery],
        });

        if (!queryEmbedResponse || !queryEmbedResponse.embeddings) {
            throw new Error("Failed to generate query embedding.");
        }

        const queryEmbeddingsArray: number[][] =
            (queryEmbedResponse.embeddings as any).float ?? queryEmbedResponse.embeddings;

        queryVector = queryEmbeddingsArray[0];
    } catch (error) {
        console.error("Error generating query vector with Cohere:", error);
        return "Error creating search vector.";
    }

    // Query Pinecone namespace for top 5 matches
    let retrievedDocs;
    try {
        const queryResponse = await index.namespace(docId).query({
            vector: queryVector,
            topK: 5,
            includeMetadata: true,
        });

        retrievedDocs = queryResponse.matches.map((match) => ({
            id: match.id,
            score: match.score,
            metadata: match.metadata,
            pageContent: match.metadata?.pageContent || "No content available",
        }));
    } catch (error) {
        console.error("Pinecone query error:", error);
        return "Error retrieving documents.";
    }

    if (!retrievedDocs || retrievedDocs.length === 0) {
        return "No relevant documents found.";
    }

    // Construct prompt with context and chat history
    const answerPrompt = `
You are an intelligent assistant.

Chat History:
${chatHistory.map((msg) => `- ${msg.content}`).join("\n")}

Document Excerpts:
${retrievedDocs
            .map((doc) => `- ${doc.metadata?.text || doc.pageContent || "No content"}`)
            .join("\n")}

Question:
"${question}"

Give a clear, helpful response using the above context. Do not repeat the question.
`;

    // Generate answer with Gemini
    const answer = await generateTextWithGemini(answerPrompt);
    return answer;
};
