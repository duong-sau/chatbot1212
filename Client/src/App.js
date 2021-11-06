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
    'Bert',
    'BM25',
  ]},
  t5:{
    select:'number cluster',
    caption: 'use t5 method',
    key:[
      1,
      2,
      3,
    ]},
}