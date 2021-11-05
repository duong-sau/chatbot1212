import React, { Component } from "react";
import axios from 'axios';
import TextField from "@material-ui/core/TextField";
import Box from '@material-ui/core/Box'
import Container from '@material-ui/core/Container';
import '../main.css'
import Message from "./Message";
import {Grid} from "@material-ui/core";
import Controller from "./Controller";
let Self
class Main extends Component {
    constructor(props) {
        super(props);
        Self = this;
        this.setT5TopP = this.setT5TopP.bind(this);
        this.setMethod = this.setMethod.bind(this);
        this.setTopK = this.setTopK.bind(this);
        this.setCosineLevel = this.setCosineLevel.bind(this);
        this.child = React.createRef();
    }
    state = { t5_answer: ['Every Things In Iqtree You Need'],cosine_answer:[], link: "", question: "", play:false, key:0}

    question = (question,method, top_k, t5_top_p, cosine_embedding)=>{
        Self.setState({key: this.state.key + 1, play:true})
        if(this.state.link.trim() === ""){
            // eslint-disable-next-line react/no-direct-mutation-state
            this.state.link =  "http://127.0.0.1:5000";
        }
        console.log(this.state.link +'/question?question='+question +  '/')
        axios
            .get(this.state.link +'/question', {
                 params: {
                     question:question,
                     method: method,
                     top_k: top_k,
                     t5_top_p: t5_top_p,
                     cosine_embedding: cosine_embedding,
                 }
                }
            )
            .then(function (response) {
                let t5_json = JSON.parse(response.data.answer.t5_answer)
                let t5_answer = Object.values(t5_json);
                let cosine_json = JSON.parse(response.data.answer.cosine_answer)
                let cosine_answer = Object.values(cosine_json);
                Self.setState({t5_answer:t5_answer, cosine_answer: cosine_answer})
                this.child.current.setState({})
            })
            .catch(function (error) {
                console.log(error);
                Self.setState({key: Self.state.key + 1, play:false})
            });
    }

    method = 'both'
    top_k = 5
    t5_top_p = global.method.t5.key[0]
    cosine_level = global.method.cosine.key[0]
    setMethod(method){
        Self.method = method;
    }
    setTopK(top_k){
        Self.top_k = top_k;
    }
    setT5TopP(t5_top_p){
        Self.t5_top_p = t5_top_p;
        console.log(t5_top_p)
    }
    setCosineLevel(cosine_level){
        Self.cosine_level = cosine_level;
    }
    render() {
        return (
            <Container  display="flex" sx={{ flexDirection: 'column' }}>
                <TextField id="server" label="Server URL" size="small"
                           fullWidth={true} variant="standard" onChange={(e) => this.setState({link: e.target.value}) }/>
                < Box display="flex" sx={{ alignItems: 'center', flexDirection: 'column' }}>
                    <TextField id="message" autoFocus={true} fullWidth={true} multiline={true}
                               label="Enter question here" variant="standard" onChange={(e) => this.setState({question: e.target.value}) }/>
                </Box>
                <Controller onSubmit={()=>{this.question(this.state.question, Self.method, Self.top_k, Self.t5_top_p, Self.cosine_level)}} play={false} key={1} set={this.setTopK}/>
                < Box display="flex" sx={{ alignItems: 'center', flexDirection: 'column' }}>
                    <Grid container spacing={2} direction="row">
                        <Grid item xs={6} direction="column">
                            <Message ref={this.child} answer={Self.state.t5_answer} name={'T5'} method={global.method.t5.key} caption={global.method.t5.caption} select={global.method.t5.select} set={Self.setT5TopP}/>
                        </Grid>
                        <Grid item xs={6} direction="row">
                            <Message answer={Self.state.cosine_answer} name={'cosine'} method={global.method.cosine.key} caption={global.method.cosine.caption} select={global.method.cosine.select} set={Self.setCosineLevel}/>
                        </Grid>
                    </Grid>
                </Box>
            </Container >
        );
    }
}
export default Main;
