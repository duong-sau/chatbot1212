import React, { Component } from "react";
import Button from '@material-ui/core/Button';
import axios from 'axios';
import HTMLRenderer from 'react-html-renderer'
import TextField from "@material-ui/core/TextField";
import Box from '@material-ui/core/Box'
import Container from '@material-ui/core/Container';
import { CountdownCircleTimer } from 'react-countdown-circle-timer';

let Self
class Welcome extends Component {
    constructor(props) {
        super(props);
        Self = this;
    }
    state = { answer: '<div>Every Things In Iqtree You Need</div>', link: "", question: "", play:false, key:0}
    question = (question)=>{
        Self.setState({key: this.state.key + 1, play:true})
        axios
            .get(this.state.link +'/question' + '?question='+question)
            .then(function (response) {
                console.log(response.data.answer);
                Self.setState({answer:response.data.ans})
            })
            .catch(function (error) {
                console.log(error);
                Self.setState({key: Self.state.key + 1, play:false})
            });
    }
    componentDidMount = () =>{
        return;
    }
    render() {
        return (
            <Container  display="flex" sx={{ flexDirection: 'column' }}>
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
                            duration={7}
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
                <Box>
                    <p></p>
                    <br/>
                    <br/>
                </Box>
                <Box display="flex" sx={{ alignItems: 'center', flexDirection: 'column' }}>
                    <HTMLRenderer
                    html={this.state.answer}/>
                </Box>
            </Container >
        );
    }
}
export default Welcome;