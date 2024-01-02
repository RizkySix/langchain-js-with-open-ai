import { config } from "dotenv";
config({
  path: '.env'
})

import { RecursiveCharacterTextSplitter } from "langchain/text_splitter"
import { TextLoader } from "langchain/document_loaders/fs/text";
import { ChatOpenAI } from "langchain/chat_models/openai"
import { PromptTemplate } from "langchain/prompts"
import { StringOutputParser } from "langchain/schema/output_parser"
import { retriever } from "./utilities/retriever.js";

const openAiToken = process.env.OPEN_AI_TOKEN

const llm = new ChatOpenAI({
  openAIApiKey: openAiToken
})

const standAloneTemplate = "Given a question, convert it into standalone question. question: {question} standalone question:"

const standAlonePrompt = PromptTemplate.fromTemplate(standAloneTemplate)

const answerTemplate = `You are a helpful and enthusiastic support bot who can answer a given question about Scrimba based on the context provided. Try to find the answer in the context. If you really don't know the answer, say "I'm sorry, I don't know the answer to that." And direct the questioner to email rizky@gmail.com. Don't try to make up an answer. Always speak as if you were chatting to a friend.
context: {context}
question: {question}
answer: 
`
const answerPrompt = PromptTemplate.fromTemplate(answerTemplate)

const combineDocuments = (docs) => {
  return docs.map((doc) => doc.pageContent).join('\n\n')
} 

const chain = standAlonePrompt.pipe(llm).pipe(new StringOutputParser()).pipe(retriever).pipe(combineDocuments)


const response = await chain.invoke({
  question: "Some people doesnt want to contribute for nature, but is Surya decided to contribute by restoring warmth?, because surya is a kind person"
})


console.log(response)

