import React, {Component} from 'react';
import Box from "@material-ui/core/Box";
import {Grid, MenuItem, Select, Typography} from "@material-ui/core";
import {Alert} from "@mui/lab";


class Message extends Component {
    constructor(props){
        super(props);
        this.state = {
            reset: false,
            selection : "",
            reload: false
        };
        this.handleChange = this.handleChange.bind(this);
        this.reset = this.reset.bind(this);
    }
    handleChange(event) {
        this.setState({ selection : event.target.value });
        this.props.setParameter(event.target.value, this.props.position);

    }
    componentWillReceiveProps(nextProps, nextContext) {
        this.setState({reload:true})
    }
    reset = () => {
        this.setState({reset:true})
    }
    render() {
        console.log(this.props.answer)
        if(this.state.reset){
            this.state.reset = false;
        }
        if(this.state.error){
            this.state.error = false;
            Alert("An error occurred")
        }
        return (
            <Box  style={{  height:'100%', padding:5 }}>
                <Box  sx={{border: '2px solid black', borderRightColor: 'black', borderRadius:'5px', padding: 2, flexDirection: 'row' }}>
                    <Grid container spacing={2} direction="row" display="flex" alignItems="center"  sx={{ padding: 5, flexDirection: 'row' }}>
                        <Grid item xs={8} direction="column">
                            <Typography variant="h5" color="primary">
                                {this.props.caption}
                            </Typography>
                       </Grid>
                        <Grid item xs={4} direction="row">
                            <Select value={this.props.value} label="left method selection" onChange={this.props.method_change} style={{ padding:2, width:120}}>
                                <MenuItem value={'T5'}>T5</MenuItem>
                                <MenuItem value={'Cosine'}>Cosine</MenuItem>
                                <MenuItem value={'bert'}>Bert</MenuItem>
                            </Select>
                        </Grid>
                    </Grid>
                    <Grid container spacing={2} direction="row" display="flex" alignItems="center"  sx={{ padding: 5, flexDirection: 'row' }}>
                        <Grid item xs={8} direction="column">
                            <Typography variant="h5" color="primary">
                                {this.props.select}
                            </Typography>
                        </Grid>
                        <Grid item xs={4} direction="row">
                            <Select
                                value={this.state.selection}
                                label="selection"
                                onChange={this.handleChange}
                                style={{ padding:5, width:120}}
                            >
                                {
                                    this.props.method.map(v=>
                                        (<MenuItem key={v} value={v}>{v}</MenuItem>)
                                    )}
                            </Select>
                        </Grid>
                    </Grid>
                </Box>
            <Box>
                <br/>
            </Box>
            <Box display="flex" sx={{ padding: 10, flexDirection: 'column' }} style={{backgroundColor: "#E5E5E5", borderRadius:'5px'}}>
                <div>
                    {this.props.answer.map(a =>
                        <div>
                            <a href={a.answer}>
                                <h2>
                                    {a.label}
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