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
    select:'chose embedding',
    caption: 'use cosine method',
    key:[
    'bert',
    'bm25',
  ]},
  t5:{
    select:'chose top k',
    caption: 'use t5 method',
    key:[
      1,
      2,
      3,
    ]},
}