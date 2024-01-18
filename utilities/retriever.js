import { config } from "dotenv";
config({
  path: '.env'
})

import { createClient } from "@supabase/supabase-js";
import { SupabaseVectorStore } from 'langchain/vectorstores/supabase'
import { OpenAIEmbeddings } from 'langchain/embeddings/openai'


const supabaseApiKey = process.env.SUPABASE_API_KEY
const supabaseProjectUrl = process.env.PROJECT_URL
const openAiToken = process.env.OPEN_AI_TOKEN
const embeddings = new OpenAIEmbeddings({
    openAIApiKey: openAiToken
})

const client = createClient(supabaseProjectUrl, supabaseApiKey)

//retriver ke supabase

const vectors = new SupabaseVectorStore(embeddings, {
    client: client,
    tableName: 'mentality',
    queryName: 'match_mentality'
  })
  
const retriever = vectors.asRetriever(/* 10 */)

export {retriever}