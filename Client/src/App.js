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
    caption: 'Method',
    key:[
    'Bert',
    'BM25',
  ]},
  t5:{
    select:'Number cluster',
    caption: 'Method',
    key:[
      1,
      2,
      3,
    ]},
  bert:{
    select: "No parameter",
    caption: 'Methods',
    key: []
  }
}