import React, {Component} from 'react';
import Button from "@material-ui/core/Button";
import {CountdownCircleTimer} from "react-countdown-circle-timer";
import Box from "@material-ui/core/Box";
import {MenuItem, Select} from "@material-ui/core";

class Controller extends Component {
    constructor(props){
        super(props);
        this.state = {
            selection : 1
        };
        this.handleChange = this.handleChange.bind(this);
    }
    handleChange(event) {
        //set selection to the value selected
        this.setState({ selection : event.target.value });
        this.props.set(event.target.value);
    }
    render() {
        return (
            <div>
                <Box display="flex" sx={{ flexDirection: 'row' }}>
                    <Button onClick={()=>{this.props.onSubmit()}} sx={{ alignItems: 'center' }}>
                        what
                    </Button>
                    <CountdownCircleTimer
                        key={this.props.key}
                        isPlaying={this.props.play}
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
                    <Box display="flex" sx={{ padding: 10, flexDirection: 'row' }}>
                        <Box>chose top k level</Box>
                        <Select
                            value={this.state.selection}
                            label="selection"
                            onChange={this.handleChange}
                        >
                            <MenuItem key={1} value={1}>1</MenuItem>
                            <MenuItem key={2} value={2}>2</MenuItem>
                            <MenuItem key={3} value={3}>3</MenuItem>
                            <MenuItem key={4} value={4}>4</MenuItem>
                            <MenuItem key={5} value={5}>5</MenuItem>
                        </Select>
                    </Box>
                </Box>
            </div>
        );
    }
}

export default Controller;