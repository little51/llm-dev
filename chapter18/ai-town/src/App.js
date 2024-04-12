import { useCallback, useEffect, useState } from 'react';
import Phaser from 'phaser';
import GridEngine from 'grid-engine';
import BootScene from './game/scenes/BootScene';
import GameScene from './game/scenes/GameScene';
import { makeStyles } from '@material-ui/core/styles';
import RexUIPlugin from 'phaser3-rex-plugins/templates/ui/ui-plugin'
import dialogBorderBox from './game/assets/images/dialog_borderbox.png';
import DialogBox from "./game/components/DialogBox";
import './App.css';
import { calculateGameSize } from "./game/utils";
import GameHint from "./game/components/GameHint";

const { width, height, multiplier } = calculateGameSize();

const useStyles = makeStyles((theme) => ({
  modal: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  paper: {
    backgroundColor: theme.palette.background.paper,
    border: '2px solid #000',
    boxShadow: theme.shadows[5],
    padding: theme.spacing(2, 4, 3),
    overflow: 'auto',
  },
  postContainer: {
    maxWidth: '90%',
    maxHeight: '90%',
  },
  gameContentWrapper: {
    width: `${width * multiplier}px`,
    height: `${height * multiplier}px`,
    margin: 'auto',
    padding: 0,
    overflow: 'hidden',
    '& canvas': {
      imageRendering: 'pixelated',
      '-ms-interpolation-mode': 'nearest-neighbor',
      boxShadow: '0px 0px 0px 3px rgba(0,0,0,0.75)',
    },
  },
  pageWrapper: {
    background: theme.palette.background.paper,
    padding: 0,
    margin: 0,
  },
  loadingText: {
    fontFamily: '"Press Start 2P"',
    marginTop: '30px',
    marginLeft: '30px',
  },
  preLoadDialogImage: {
    backgroundImage: `url("${dialogBorderBox}")`,
    backgroundSize: '1px',
    backgroundRepeat: 'no-repeat',
  },
  gameWrapper: {
    color: '#FFFFFF',
  },
  gameGif: {
    width: '100%',
    position: 'absolute',
    imageRendering: 'pixelated',
    top: 0,
  },
}));

function App() {
  const classes = useStyles();
  const [messages, setMessages] = useState([]);
  const [characterName, setCharacterName] = useState('');
  const [gameHintText, setGameHintText] = useState("https://github.com/git-cloner/ai-town");

  const handleMessageIsDone = useCallback(() => {
    const customEvent = new CustomEvent(`${characterName}-dialog-finished`, {
      detail: {},
    });
    window.dispatchEvent(customEvent);

    setMessages([]);
    setCharacterName('');
  }, [characterName]);

  useEffect(() => {
    const game = new Phaser.Game({
      type: Phaser.AUTO,
      title: 'ai-town',
      parent: 'game-content',
      orientation: Phaser.Scale.LANDSCAPE,
      localStorageName: 'ai-town',
      width,
      height,
      autoRound: true,
      pixelArt: true,
      scale: {
        autoCenter: Phaser.Scale.CENTER_BOTH,
        mode: Phaser.Scale.ENVELOP,
      },
      scene: [
        BootScene,
        GameScene,
      ],
      physics: {
        default: 'arcade',
      },
      dom: {
        createContainer: true
      },
      plugins: {
        scene: [
          {
            key: 'gridEngine',
            plugin: GridEngine,
            mapping: 'gridEngine',
          },
          {
            key: 'rexUI',
            plugin: RexUIPlugin,
            mapping: 'rexUI'
          }
        ],
      },
      backgroundColor: '#000000',
    });
    window.phaserGame = game;
  }, []);

  useEffect(() => {
    //show dialogs event
    const showdialogBoxEventListener = ({ detail }) => {
      setCharacterName(detail.characterName);
      setMessages([{
        "message": detail.message
      }]);
    };
    window.addEventListener('show-dialog', showdialogBoxEventListener);
    //close dialogs event
    const closedialogBoxEventListener = ({ detail }) => {
      setCharacterName(detail.characterName);
      setMessages([]);
      const timer = setInterval(() => {
        clearInterval(timer);
        handleMessageIsDone();
      }, 2000);
    };
    window.addEventListener('close-dialog', closedialogBoxEventListener);
    //game hint event
    const gameHintEventListener = ({ detail }) => {
      var hint = detail.hintText;
      if (hint === "") {
        hint = "ai-town";
      }
      setGameHintText(hint);
    };
    window.addEventListener('game-hint', gameHintEventListener);
    //remove listeners
    return () => {
      window.removeEventListener('show-dialog', showdialogBoxEventListener);
      window.removeEventListener('close-dialog', closedialogBoxEventListener);
      window.removeEventListener('game-hint', gameHintEventListener);
    };
  }, [setCharacterName, setMessages, handleMessageIsDone]);

  return (
    <div>
      <div className={classes.gameWrapper}>
        <div
          id="game-content"
          className={classes.gameContentWrapper}
        >
        </div>
        <GameHint
          gameSize={{
            width,
            height,
            multiplier,
          }}
          hintText={gameHintText}
        />
        {messages.length > 0 && (
          <DialogBox
            onDone={handleMessageIsDone}
            characterName={characterName}
            messages={messages}
            gameSize={{
              width,
              height,
              multiplier,
            }}
          />
        )}
      </div>
    </div>
  );
}

export default App;
