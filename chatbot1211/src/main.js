import React, { Component } from "react";
import Button from '@mui/material/Button';
import {TextField} from "@mui/material";
class Welcome extends Component {

    state = { answer: '<div>all in iqtree</div>', link: "" }
    question = (question)=>{
        const requestOptions = {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: question })
        };
        fetch('https://reqres.in/api/posts', requestOptions)
            .then(response => this.setState({ answer: response }))
    }

    componentDidMount = () =>{
        let link = "https://drive.google.com/file/d/1-zMukGnnG5vqjR7G6bouYN7yf6b6eUTQ/view?usp=sharing"
        fetch(link).then((r)=>{
            if(r.status !== 200){
                alert('init fail')
            }
            this.setState({link: r.text})
        })
    }
    render() {
        return (
            <div>
                <TextField id="standard-basic" label="Standard" variant="standard" ref="q"/>
                <Button onClick={()=>{this.question(this.refs.q.getValue())}}>
                    <h1>send </h1>
                </Button>
                <div>
                    {this.state.answer}
                </div>
            </div>
        );
    }
}
export default Welcome;