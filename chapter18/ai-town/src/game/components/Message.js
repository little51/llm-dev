import { makeStyles } from '@material-ui/core/styles';

const useStyles = makeStyles((theme) => ({
    dialogMessage: ({ multiplier }) => ({
        fontFamily: '"宋体"',
        fontSize: `${8 * multiplier}px`,
        textTransform: 'uppercase',
        lineHeight: '1.5em'
    }),
}));

const Message = ({
    message = [],
    trail = 35,
    multiplier = 1,
    onMessageEnded = () => { },
    forceShowFullMessage = false,
}) => {
    const classes = useStyles({ multiplier });
    return (
        <div className={classes.dialogMessage}>
            <span dangerouslySetInnerHTML={{ __html: message }}></span>
        </div>
    );
};

export default Message;
