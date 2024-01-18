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
import { formatConvertationHistory } from "./utilities/formatConvertationHistory.js";

import { createRequire } from "module";
const require = createRequire(import.meta.url);
const fs = require('fs');

const openAiToken = process.env.OPEN_AI_TOKEN

const llm = new ChatOpenAI({
  openAIApiKey: openAiToken,
  modelName: "gpt-3.5-turbo-1106"
})

//membuat standalone quest (pertanyaan yang straight to the point)
const standAloneTemplate = `Given some conversation history (if any) and a question, convert the question to a standalone question. 
conversation history: {conv_history}
question: {question} 
standalone question:`

//buat menjadi propmt agar dapat di pipe/chaining
const standAlonePrompt = PromptTemplate.fromTemplate(standAloneTemplate)

//template pernyataan untuk jawaban nantinya
const answerTemplate = `You are a helpful and enthusiastic support bot who can answer a given question about Mental health document in Indonesian language, based on the context provided and the conversation history. Try to find the answer in the context. If the answer is not given in the context, find the answer in the conversation history if possible. If you really don't know the answer, say "I'm sorry, I don't know the answer to that." And direct the questioner to email help@scrimba.com. Don't try to make up an answer. Always speak as if you were chatting to a friend.
context: {context}
conversation history: {conv_history}
question: {question}
answer: `

//buat menjadi propmt agar dapat di pipe/chaining
const answerPrompt = PromptTemplate.fromTemplate(answerTemplate)

const convertation = []

//kombinasikan chunk dari documents menjadi satu line of string
const combineDocuments = (docs) => {
  return docs.map((doc) => doc.pageContent).join('\n\n')
} 

//stand alone question digunakan untuk mencari data2/chunk2 yang paling sesuai pada supabase
const standAloneChain = standAlonePrompt.pipe(llm).pipe(new StringOutputParser())
const retrieverChain = RunnableSequence.from([
  prevResult => prevResult.standalone_question,
  retriever,
  combineDocuments,
])


const answerChain = answerPrompt.pipe(llm).pipe(new StringOutputParser())

//jadikan runablesequence
const chain = RunnableSequence.from([
  {
    standalone_question: standAloneChain,
    original_input: new RunnablePassthrough(),
  },
  {
    context: retrieverChain,
    question: ({original_input}) => original_input.question,
    conv_history: ({ original_input }) => original_input.conv_history
  },
  answerChain,
  
])
const question = "what ages that people easly get mental health issue?"

let jsonConvertation = {
  messageArr: []
}

const oldConv = fs.readFileSync('convertation.json', 'utf8')
let resfack = []
if(oldConv.trim() !== ''){
 resfack = JSON.parse(oldConv)

jsonConvertation.messageArr.push(...resfack.messageArr)
}

const response = await chain.invoke({
  question: question,
  conv_history: resfack.messageArr ? resfack.messageArr.join(', ') : ''
})

// convertation.push(question)
// convertation.push(response)

jsonConvertation.messageArr.push(question)
jsonConvertation.messageArr.push(response)

fs.writeFileSync('convertation.json', JSON.stringify(jsonConvertation))

convertation.push(...jsonConvertation.messageArr)

//console.log(jsonConvertation)
 //console.log(convertation)

console.log(response)

