import { Scene } from 'phaser';

// Map files
import townMap from '../assets/sprites/maps/tilesets/town.json';

// Characters files
import heroJson from '../assets/sprites/atlas/hero.json';
import slimeJson from '../assets/sprites/atlas/slime.json';
import heartJson from '../assets/sprites/atlas/heart.json';
import coinJson from '../assets/sprites/atlas/coin.json';

// NPC jsons
import npc01Json from '../assets/sprites/atlas/npc_01.json';
import npc02Json from '../assets/sprites/atlas/npc_02.json';
import npc03Json from '../assets/sprites/atlas/npc_03.json';
import npc04Json from '../assets/sprites/atlas/npc_04.json';

// Images
import heroImage from '../assets/sprites/atlas/hero.png';
import slimeImage from '../assets/sprites/atlas/slime.png';
import heartImage from '../assets/sprites/atlas/heart.png';
import coinImage from '../assets/sprites/atlas/coin.png';
import mainMenuBackgroundImage from '../assets/images/main_menu_background.png';
import gameOverBackgroundImage from '../assets/images/game_over_background.png';
import gameLogoImage from '../assets/images/game_logo.png';
import heartContainerImage from '../assets/images/heart_container.png';
import swordImage from '../assets/images/sword.png';
import pushImage from '../assets/images/push.png';
import townImage from '../assets/sprites/maps/tilesets/town.png';

// NPC images
import npc01Image from '../assets/sprites/atlas/npc_01.png';
import npc02Image from '../assets/sprites/atlas/npc_02.png';
import npc03Image from '../assets/sprites/atlas/npc_03.png';
import npc04Image from '../assets/sprites/atlas/npc_04.png';

export default class BootScene extends Scene {
    constructor() {
        super('BootScene');
    }

    preload() {
        const fontSize = 16;

        // setup loading bar
        const progressBar = this.add.graphics();
        const progressBox = this.add.graphics();
        const { width: gameWidth, height: gameHeight } = this.cameras.main;

        const barPositionX = Math.ceil((gameWidth - (gameWidth * 0.7)) / 2);
        progressBox.fillStyle(0x222222, 0.8);
        progressBox.fillRect(
            barPositionX,
            Math.ceil(gameHeight / 6),
            Math.ceil(gameWidth * 0.7),
            Math.ceil(gameHeight / 10)
        );

        const loadingText = this.add.text(
            gameWidth / 2,
            Math.ceil(gameHeight / 10),
            'loading...',
            {
                fontFamily: '"Press Start 2P"',
                fontSize: `${fontSize}px`,
                size: `${fontSize}px`,
                fill: '#ffffff',
                color: '#ffffff',
            }
        );

        loadingText.setOrigin(0.5);
        loadingText.setResolution(30);

        const percentText = this.add.text(
            gameWidth / 2,
            Math.ceil((gameHeight / 6) + (fontSize / 2) + (gameHeight / 60)),
            '0%',
            {
                fontFamily: '"Press Start 2P"',
                fontSize: `${fontSize}px`,
                size: `${fontSize}px`,
                fill: '#ffffff',
                color: '#ffffff',
            }
        );

        percentText.setOrigin(0.5);
        percentText.setResolution(30);

        const assetText = this.add.text(
            gameWidth / 2,
            Math.ceil(gameHeight / 3),
            '',
            {
                fontFamily: '"Press Start 2P"',
                fontSize: `${fontSize / 2}px`,
                size: `${fontSize / 2}px`,
                fill: '#ffffff',
                color: '#ffffff',
            }
        );

        assetText.setOrigin(0.5);
        assetText.setResolution(30);

        this.load.on('progress', (value) => {
            progressBar.clear();
            progressBar.fillStyle(0xFFFFFF, 1);
            progressBar.fillRect(
                barPositionX,
                Math.ceil(gameHeight / 6),
                Math.ceil(gameWidth * 0.7) * value,
                Math.ceil(gameHeight / 10)
            );
            percentText.setText(`${Number.parseInt(value * 100, 10)}%`);
        });

        this.load.on('fileprogress', (file) => {
            assetText.setText(`loading: ${file.key}`);
        });

        this.load.on('complete', () => {
            progressBar.destroy();
            progressBox.destroy();
            percentText.destroy();
            assetText.destroy();
        });

        // Maps
        this.load.tilemapTiledJSON('town',townMap);

        // Atlas
        this.load.atlas('hero', heroImage, heroJson);
        this.load.atlas('slime', slimeImage, slimeJson);
        this.load.atlas('heart', heartImage, heartJson);
        this.load.atlas('coin', coinImage, coinJson);

        // NPCs
        this.load.atlas('npc_01', npc01Image, npc01Json);
        this.load.atlas('npc_02', npc02Image, npc02Json);
        this.load.atlas('npc_03', npc03Image, npc03Json);
        this.load.atlas('npc_04', npc04Image, npc04Json);

        // Tilesets
        this.load.image('town',townImage);

        // Images
        this.load.image('main_menu_background', mainMenuBackgroundImage);
        this.load.image('game_over_background', gameOverBackgroundImage);
        this.load.image('game_logo', gameLogoImage);
        this.load.image('heart_container', heartContainerImage);
        this.load.image('sword', swordImage);
        this.load.image('push', pushImage);
    }

    create() {
        this.scene.start('GameScene', {
            heroStatus: {
                position: { x: 29, y: 22 },
                previousPosition: { x: 29, y: 22 },
                frame: 'hero_idle_down_01',
                facingDirection: 'down'
            },
            mapKey: 'town',
        })
    }
}
