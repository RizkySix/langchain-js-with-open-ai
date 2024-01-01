import { OpenAI } from "langchain/llms/openai";
import { loadQAMapReduceChain } from "langchain/chains";
import { Document } from "langchain/document";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { TokenTextSplitter } from "langchain/text_splitter";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

// Optionally limit the number of concurrent requests to the language model.
const model = new OpenAI({ temperature: 0, maxConcurrency: 10,  openAIApiKey: "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", });
const chain = loadQAMapReduceChain(model);

const loader = new PDFLoader("src/Mental.pdf", {
  splitPages: true,
});

const parentDocuments = await loader.load();
const docs = await splitter.splitDocuments(parentDocuments);

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 10000,
  chunkOverlap: 20,
});


const res = await chain.call({
  input_documents: docs,
  question: "Siapa saja penulisnya",
});
console.log({ res });