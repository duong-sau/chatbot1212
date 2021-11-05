import React, {Component} from 'react';
import Box from "@material-ui/core/Box";
import { MenuItem, Select} from "@material-ui/core";


class Message extends Component {
    constructor(props){
        super(props);
        this.state = {
            selection : ""
        };
        this.handleChange = this.handleChange.bind(this);
        console.log(this.props.method);
    }
    handleChange(event) {
        //set selection to the value selected
        this.setState({ selection : event.target.value });

    }
    static defaultProps = {
        caption: 'caption',
        answer:[{
            answer: 'answer',
            intent: 'intent',
            highlight: 'highlight sentence',
            first: 'first sentence'
        }]
    }
    render() {
        return (
            <Box  style={{ border: '2px solid black', borderRightColor: 'black', height:'100%', padding:5 }}>
                <Box>{this.props.caption}</Box>
                <Box display="flex" sx={{ padding: 10, flexDirection: 'row' }}>
                    <Box>{this.props.select}</Box>
                    <Select
                        value={this.state.selection}
                        label="selection"
                        onChange={this.handleChange}
                    >
                    {
                        this.props.method.map(v=>
                        (<MenuItem key={v} value={v}>{v}</MenuItem>)
                    )}
                    </Select>
                </Box>
            <Box>
                <br/>
                <br/>
                <br/>
            </Box>
            <Box display="flex" sx={{ padding: 10, flexDirection: 'column' }}>
                <div>
                    {this.props.answer.map(a =>
                        <div>
                            <a href={a.answer}>
                                <h2>
                                    {a.intent}
                                </h2>
                            </a>
                            <div key={a}> {a.first + "..................."}<span  className={'highlight'}>
                                        {a.highlight}
                                    </span> </div>

                            <br/>
                        </div>
                    )}
                </div>
            </Box>
            </Box>
        );
    }
}



export default Message;