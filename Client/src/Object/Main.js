import React, { Component } from "react";
import axios from 'axios';
import TextField from "@material-ui/core/TextField";
import Box from '@material-ui/core/Box'
import Container from '@material-ui/core/Container';
import '../main.css'
import Message from "./Message";
import {AppBar, Grid, Typography} from "@material-ui/core";
import Controller from "./Controller";
let Self
const left = 'left'
const right = 'right'
const T5 = 'T5'
const Cosine = 'Cosine'
class Main extends Component {
    constructor(props) {
        super(props);
        Self = this;
        this.setParameter = this.setParameter.bind(this);
        this.setMethod = this.setMethod.bind(this);
        this.setTopK = this.setTopK.bind(this);
        this.left_child = React.createRef();
        this.right_child = React.createRef();
        this.controller = React.createRef();
        this.state = { left: global.method.t5,right: global.method.cosine,
        left_answer: [],right_answer:[], link: "", question: ""}
    }

    question = (question, top_k, left_method, right_method, left_parameter, right_parameter)=>{
        this.controller.current.setState({loading:true})
        this.left_child.current.reset();
        this.right_child.current.reset();
        if(this.state.link.trim() === ""){
            // eslint-disable-next-line react/no-direct-mutation-state
            this.state.link =  "https://chatbot1211.herokuapp.com/";
        }
        axios
            .get(this.state.link +'/question', {
                 params: {
                     question:question,
                     top_k: top_k,
                     left_method: left_method,
                     right_method: right_method,
                     left_parameter: left_parameter,
                     right_parameter:right_parameter,
                 }
                }
            )
            .then(function (response) {
                let left_answer = Object.values(JSON.parse(response.data.answer.left_answer));
                let right_answer = Object.values(JSON.parse(response.data.answer.right_answer));
                Self.setState({left_answer:left_answer, right_answer: right_answer})
                Self.controller.current.setState({loading:false})
            })
            .catch(function (error) {
                Self.controller.current.setState({loading:false})
                console.log(error);
            });
    }

    top_k = 1
    left_method = T5
    right_method = Cosine
    left_parameter = global.method.t5.key[0]
    right_parameter = global.method.cosine.key[0]
    setTopK(top_k){
        Self.top_k = top_k;
    }
    setMethod(method, position){
        if(position === left){
            Self.left_method = method;
        }
        else if(position === right){
            Self.right_method = method;
        }
        else {
            console.log('set method error')
        }
    }
    setParameter(parameter, position){
        if(position === left){
            Self.left_parameter = parameter;
        }
        else if(position === right){
            Self.right_parameter = parameter;
        }
        else {
            console.log('set parameter error')
        }
    }
    left_change(event) {
        let l = Self.get_method_parameter(event.target.value)
        Self.setState({left:l});
        Self.setMethod(event.target.value, left)
    }
    right_change(event){
        let r = Self.get_method_parameter(event.target.value)
        Self.setState({right:r});
        Self.setMethod(event.target.value, right)
    }
    get_method_parameter(method){
        if(method === T5){
            return global.method.t5;
        }
        if(method === Cosine){
            return global.method.cosine;
        }
    }
    render() {
        return (
            <Box>
                <AppBar position="static">
                    <Typography variant="h6" color="inherit" component="div">
                        IQ-TREE
                    </Typography>
                </AppBar>
                <Container  display="flex" sx={{ flexDirection: 'column' }}>
                    <TextField id="server" label="Server URL" size="small"
                               fullWidth={true} variant="standard" onChange={(e) => this.setState({link: e.target.value}) }/>
                    <br/>
                    <br/>
                    < Box display="flex" sx={{ alignItems: 'center', flexDirection: 'row' }}>
                        <TextField id="message" autoFocus={true} fullWidth={true} multiline={true} rows={8}
                                   label="Enter your issue here" variant="outlined" onChange={(e) => this.setState({question: e.target.value}) }/>
                        <Controller ref={this.controller} onSubmit={()=>{this.question(this.state.question, Self.top_k, Self.left_method, Self.right_method, Self.left_parameter, Self.right_parameter)}} play={false} key={1} set={this.setTopK}/>
                    </Box>
                    < Box display="flex" sx={{ alignItems: 'center', flexDirection: 'column' }}>
                        <Grid container spacing={2} direction="row">
                            <Grid item xs={6} direction="column">
                                <Message position={left} ref={this.left_child} answer={Self.state.left_answer} method_change={this.left_change} value={Self.left_method} method={Self.state.left.key} caption={Self.state.left.caption} select={Self.state.left.select} setParameter={Self.setParameter}/>
                            </Grid>
                            <Grid item xs={6} direction="row">
                                <Message position={right} ref={this.right_child} answer={Self.state.right_answer} method_change={this.right_change} value={Self.right_method} method={Self.state.right.key} caption={Self.state.right.caption} select={Self.state.right.select} setParameter={Self.setParameter}/>
                            </Grid>
                        </Grid>
                    </Box>
                </Container >
            </Box>
        );
    }
}
export default Main;
