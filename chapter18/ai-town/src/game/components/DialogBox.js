import { useCallback, useEffect, useState } from 'react';
import { makeStyles } from '@material-ui/core/styles';

// Images
import dialogBorderBox from '../assets/images/dialog_borderbox.png';

// Components
import Message from './Message';

const useStyles = makeStyles((theme) => ({
    dialogWindow: ({ width, height, multiplier }) => {
        const messageBoxHeight = Math.ceil((height / 5) * multiplier);
        return {
            imageRendering: 'pixelated',
            fontFamily: '"Press Start 2P"',
            textTransform: 'uppercase',
            backgroundColor: '#e2b27e',
            border: 'solid',
            borderImage: `url("${dialogBorderBox}") 6 / ${6 * multiplier}px ${6 * multiplier}px ${6 * multiplier}px ${6 * multiplier}px stretch`,
            padding: `${8 * multiplier}px`,
            position: 'absolute',
            top: `${Math.ceil((height * multiplier) - (messageBoxHeight + messageBoxHeight * 0.4))}px`,
            width: `${Math.ceil(width * 0.5 * multiplier)}px`,
            left: '50%',
            transform: 'translate(-50%, 0%)',
            minHeight: `${messageBoxHeight}px`,
        };
    },
    dialogTitle: ({ multiplier }) => ({
        fontSize: `${6 * multiplier}px`,
        marginBottom: `${6 * multiplier}px`,
        fontWeight: 'bold',
    }),
    dialogFooter: ({ multiplier }) => ({
        fontSize: `${6 * multiplier}px`,
        cursor: 'pointer',
        textAlign: 'end',
        position: 'absolute',
        right: `${6 * multiplier}px`,
        bottom: `${6 * multiplier}px`,
    }),
}));

const DialogBox = ({
    messages,
    characterName,
    onDone,
    gameSize,
}) => {
    const {
        width,
        height,
        multiplier,
    } = gameSize;

    const [currentMessage, setCurrentMessage] = useState(0);
    const [messageEnded, setMessageEnded] = useState(false);
    const classes = useStyles({
        width,
        height,
        multiplier,
    });

    const handleClick = useCallback(() => {
        if (messageEnded) {
            setMessageEnded(false);
            if (currentMessage < messages.length - 1) {
                setCurrentMessage(currentMessage + 1);
            } else {
                setCurrentMessage(0);
                onDone();
            }
        } else {
            setMessageEnded(true);
        }
    }, [currentMessage, messageEnded, messages.length, onDone]);

    useEffect(() => {
        const handleKeyPressed = (e) => {
            if (['Enter', 'Space', 'Escape'].includes(e.code)) {
                handleClick();
            }
        };

        window.addEventListener('keydown', handleKeyPressed);

        return () => {
            window.removeEventListener('keydown', handleKeyPressed);
        };
    }, [handleClick]);

    return (
        <div className={classes.dialogWindow}>
            <div className={classes.dialogTitle}>
                {characterName}
            </div>
            <Message
                action={messages[currentMessage].action}
                message={messages[currentMessage].message}
                key={currentMessage}
                multiplier={multiplier}
                forceShowFullMessage={true}
                onMessageEnded={() => {
                    setMessageEnded(true);
                }}
            />
            <div
                onClick={handleClick}
                className={classes.dialogFooter}
            >
                {'Ok'}
            </div>
        </div>
    );
};

export default DialogBox;
