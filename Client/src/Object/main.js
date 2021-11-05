import React, { Component } from "react";
import Button from '@material-ui/core/Button';
import axios from 'axios';
import TextField from "@material-ui/core/TextField";
import Box from '@material-ui/core/Box'
import Container from '@material-ui/core/Container';
import { CountdownCircleTimer } from 'react-countdown-circle-timer';
import '../main.css'
import Message from "./Message";
import {Grid} from "@material-ui/core";
import Collapse from '@material-ui/core/Collapse';
let Self
class Welcome extends Component {
    constructor(props) {
        super(props);
        Self = this;
    }
    state = { answer: ['Every Things In Iqtree You Need'], link: "", question: "", play:false, key:0}

    question = (question,method, top_k, t5_top_p, cosine_level)=>{
        Self.setState({key: this.state.key + 1, play:true})
        if(this.state.link.trim() === ""){
            // eslint-disable-next-line react/no-direct-mutation-state
            this.state.link =  "http://127.0.0.1:5000";
        }
        console.log(this.state.link +'/question?question='+question +  '/')
        axios
            .get(this.state.link +'/question_t5', {
                 params: {
                     question:question,
                     method: method,
                     top_k: top_k,
                     t5_top_p: t5_top_p,
                     cosine_level: cosine_level,
                 }
                }
            )
            .then(function (response) {
                let json = JSON.parse(response.data.answer);
                let list = Object.values(json);
                Self.setState({answer:list})
            })
            .catch(function (error) {
                console.log(error);
                Self.setState({key: Self.state.key + 1, play:false})
            });
    }

    render() {
        return (
            <Container  display="flex" sx={{ flexDirection: 'column' }}>
                <Collapse in={1} timeout="auto" unmountOnExit={true}>
                    <p>POOP</p>
                </Collapse>
                <TextField id="server" label="Server URL" size="small"
                           fullWidth={true} variant="standard" onChange={(e) => this.setState({link: e.target.value}) }/>
                < Box display="flex" sx={{ alignItems: 'center', flexDirection: 'column' }}>
                    <TextField id="message" autoFocus={true} fullWidth={true} multiline={true}
                               label="Enter question here" variant="standard" onChange={(e) => this.setState({question: e.target.value}) }/>
                    <Box display="flex" sx={{ flexDirection: 'row' }}>
                        <Button onClick={()=>{this.question(this.state.question)}} sx={{ alignItems: 'center' }}>
                            what
                        </Button>
                        <CountdownCircleTimer
                            key={this.state.key}
                            isPlaying={this.state.play}
                            size = {50}
                            duration={20}
                            colors={[
                                ['#004777', 0.33],
                                ['#F7B801', 0.33],
                                ['#A30000', 0.33],
                            ]}
                        >
                            {({ remainingTime }) => remainingTime}
                        </CountdownCircleTimer>
                    </Box>
                </Box>
                < Box display="flex" sx={{ alignItems: 'center', flexDirection: 'column' }}>
                    <Grid container spacing={2} direction="row">
                        <Grid item xs={6} direction="column">
                            <Message message={''} name={'T5'} method={global.method.t5.key} caption={global.method.t5.caption} select={global.method.t5.select}/>
                        </Grid>
                        <Grid item xs={6} direction="row">
                            <Message name={'cosine'} method={global.method.cosine.key} caption={global.method.cosine.caption} select={global.method.cosine.select}/>
                        </Grid>
                    </Grid>
                </Box>
            </Container >
        );
    }
}
export default Welcome;
