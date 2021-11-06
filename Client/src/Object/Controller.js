import React, {Component} from 'react';
import LoadingButton from '@mui/lab/LoadingButton';
import Box from "@material-ui/core/Box";
import {MenuItem, Select} from "@material-ui/core";
import SendIcon from '@mui/icons-material/Send';
class Controller extends Component {
    constructor(props){
        super(props);
        this.state = {
            selection : 1, play:false, key:0
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
                <Box display="flex" sx={{ padding: 10, flexDirection: 'column' }}>
                    <Box display="flex" alignItems="center"  sx={{ padding: 5, flexDirection: 'row' }}>
                        <Box style={{ padding:10, width:100}}>number_result</Box>
                        <Select style={{ padding:5, width:80}}
                            value={this.state.selection}
                            label="number of result"
                            onChange={this.handleChange}
                        >
                            <MenuItem key={1} value={1}>1</MenuItem>
                            <MenuItem key={2} value={2}>2</MenuItem>
                            <MenuItem key={3} value={3}>3</MenuItem>
                            <MenuItem key={4} value={4}>4</MenuItem>
                            <MenuItem key={5} value={5}>5</MenuItem>
                            <MenuItem key={6} value={6}>6</MenuItem>
                            <MenuItem key={7} value={7}>7</MenuItem>
                            <MenuItem key={8} value={8}>8</MenuItem>
                            <MenuItem key={9} value={9}>9</MenuItem>
                            <MenuItem key={10} value={10}>10</MenuItem>
                        </Select>
                    </Box>
                    <LoadingButton loading={this.state.loading} loadingPosition="end" variant="contained" onClick={()=>{this.props.onSubmit()}} sx={{ alignItems: 'center' }} endIcon={<SendIcon />}>
                        Send
                    </LoadingButton>
                </Box>
            </div>
        );
    }
}

export default Controller;