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
import { RunnableSequence, RunnablePassthrough } from "langchain/schema/runnable"

const openAiToken = process.env.OPEN_AI_TOKEN

const llm = new ChatOpenAI({
  openAIApiKey: openAiToken
})

//membuat standalone quest (pertanyaan yang straight to the point)
const standAloneTemplate = "Given a question, convert it into standalone question. question: {question} standalone question:"

//buat menjadi propmt agar dapat di pipe/chaining
const standAlonePrompt = PromptTemplate.fromTemplate(standAloneTemplate)

//template pernyataan untuk jawaban nantinya
const answerTemplate = `You are a helpful and enthusiastic support bot who can answer a given question about Mental health based on the context provided. Try to find the answer in the context. If you really don't know the answer, say "I'm sorry, I don't know the answer to that." And direct the questioner to email rizky@gmail.com. Don't try to make up an answer. Always speak as if you were chatting to a friend.
context: {context}
question: {question}
answer: 
`
//buat menjadi propmt agar dapat di pipe/chaining
const answerPrompt = PromptTemplate.fromTemplate(answerTemplate)

//kombinasikan chunk dari documents menjadi satu line of string
const combineDocuments = (docs) => {
  return docs.map((doc) => doc.pageContent).join('\n\n')
} 


const standAloneChain = standAlonePrompt.pipe(llm).pipe(new StringOutputParser())
const retrieverChain = RunnableSequence.from([
  prevResult => prevResult.standalone_question,
  retriever,
  combineDocuments
])
const answerChain = answerPrompt.pipe(llm).pipe(new StringOutputParser())

//jadikan runablesequence
const chain = RunnableSequence.from([
  {
    standalone_question: standAloneChain,
    original_input: new RunnablePassthrough()
  },
  {
    context: retrieverChain,
    question: ({original_input}) => original_input.question
  },
  answerChain
])


const response = await chain.invoke({
  question: "Setiap orang memiliki mental dan mood yang berubah-ubah, faktor utama terjadinya perubahan mental secara garis besar apa?"
})


console.log(response)

