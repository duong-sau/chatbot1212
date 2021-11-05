import './App.css';
import Welcome from "./Object/main";

function App() {
  return (
    <Welcome/>
  );
}

export default App;

global.method = {
  cosine:{
    select:'chose embedding',
    caption: 'use cosine method',
    key:[
    'bert high embedding',
    'bert low embedding',
    'word to vector embedding',
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