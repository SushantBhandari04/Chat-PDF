import { Pinecone } from "@pinecone-database/pinecone";

if(!process.env.PINECONE_API_KEY){
    throw new Error("PINECONE_API_KEY is not defined");
}

const pineconeClient = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
});

export default pineconeClient