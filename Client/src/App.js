import './App.css';
import Main from "./Object/Main";

function App() {
  return (
    <Main/>
  );
}

export default App;

global.method = {
  cosine:{
    select:'Embedding method',
    caption: 'Cosine similarity method',
    key:[
    'Bert',
    'BM25',
  ]},
  t5:{
    select:'Number cluster',
    caption: 'T5 similarity method',
    key:[
      1,
      2,
      3,
    ]},
}