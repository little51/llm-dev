import { Scene } from 'phaser';
import { createInteractiveGameObject } from '../utils';
import {
    NPC_MOVEMENT_RANDOM,
    SCENE_FADE_TIME,
} from '../constants';
import { callGpt, getChatHistory } from '../ChatUtils';
import ModelDialog from '../components/ModelDialog';

var topic = "天气";

export default class GameScene extends Scene {
    constructor() {
        super('GameScene');
    }

    cursors = {};
    isConversationing = 0;
    isMoveRandomly = false;
    topicElement = undefined;

    init(data) {
        this.initData = data;
    }

    createPlayerWalkingAnimation(assetKey, animationName) {
        this.anims.create({
            key: `${assetKey}_${animationName}`,
            frames: [
                { key: assetKey, frame: `${assetKey}_${animationName}_01` },
                { key: assetKey, frame: `${assetKey}_${animationName.replace('walking', 'idle')}_01` },
                { key: assetKey, frame: `${assetKey}_${animationName}_02` },
            ],
            frameRate: 4,
            repeat: -1,
            yoyo: true,
        });
    }

    getStopFrame(direction, spriteKey) {
        switch (direction) {
            case 'up':
                return `${spriteKey}_idle_up_01`;
            case 'right':
                return `${spriteKey}_idle_right_01`;
            case 'down':
                return `${spriteKey}_idle_down_01`;
            case 'left':
                return `${spriteKey}_idle_left_01`;
            default:
                return null;
        }
    }

    getOppositeDirection(direction) {
        switch (direction) {
            case 'up':
                return 'down';
            case 'right':
                return 'left';
            case 'down':
                return 'up';
            case 'left':
                return 'right';
            default:
                return null;
        }
    }

    updateGameHint(hintText) {
        const customEvent = new CustomEvent('game-hint', {
            detail: {
                hintText,
            },
        });

        window.dispatchEvent(customEvent);
    }


    extractNpcDataFromTiled(data) {
        const [npcKey, config] = data.trim().split(':');
        const [movementType, delay, area, direction] = config.split(';');

        return {
            npcKey,
            movementType,
            facingDirection: direction,
            delay: Number.parseInt(delay, 10),
            area: Number.parseInt(area, 10),
        };
    }

    preload() {
        if (window.location.host.indexOf("localhost") >= 0) {
            this.load.html('topicform', 'ai-town/topicform.html');
        } else {
            this.load.html('topicform', 'topicform.html');
        }
    }

    showTopicDialog() {
        var element = this.topicElement;
        if (!element) {
            element = this.add.dom(80, 100).createFromCache('topicform');
            this.topicElement = element;
        }
        element.setVisible(true);
        element.setPerspective(100);
        element.addListener('click');
        element.on('click', function (event) {
            if (event.target.name === 'okButton') {
                var s = this.getChildByName('topic').value;
                if (s === "") {
                    s = "天气";
                }
                topic = s;
                this.removeListener('click');
                element.setVisible(false);
            } else if (event.target.name === 'cancelButton') {
                this.removeListener('click');
                element.setVisible(false);
            }
        });
    }

    create() {
        const camera = this.cameras.main;
        const { game } = this.sys;
        const isDebugMode = this.physics.config.debug;
        const { heroStatus, mapKey } = this.initData;
        const {
            position: initialPosition,
            frame: initialFrame
        } = heroStatus;

        camera.fadeIn(SCENE_FADE_TIME);

        this.cursors = this.input.keyboard.createCursorKeys();
        this.input.on('gameobjectdown', (pointer, gameObject) => {
            this.updateGameHint(gameObject.name);
        });
        // 鼠标弹起时，将hero移动到鼠标位置
        this.input.on("pointerup", (pointer) => {
            var pt = this.gridEngine.gridTilemap.tilemap.worldToTileXY(pointer.worldX, pointer.worldY);
            var newMapX = pt.x;
            var newMapY = pt.y;
            this.gridEngine.moveTo("hero", { x: newMapX, y: newMapY });
            this.heroActionCollider.update();
        });
        // hero角色随机移动事件
        const heroRandomMoveEventListener = () => {
            this.isMoveRandomly = !this.isMoveRandomly;
            if (this.isMoveRandomly) {
                this.gridEngine.moveRandomly("hero", 1500, 3);
            } else {
                this.gridEngine.stopMovement("hero");
            }
        };
        window.addEventListener('heroRandomMove', heroRandomMoveEventListener);
        // 显示话题事件
        const heroTopicDialogEventListener = () => {
            this.showTopicDialog();
        };
        window.addEventListener('topicDialog', heroTopicDialogEventListener);
        // 显示对话历史事件
        const chatHistoryEventListener = () => {
            ModelDialog(this, getChatHistory())
                .layout()
                .setDepth(10)
                .modalPromise()
                .then(function () {

                });
        };
        window.addEventListener('chatHistory', chatHistoryEventListener);
        // 加载地图
        const map = this.make.tilemap({ key: mapKey });
        const tileset = map.addTilesetImage('town', 'town');

        if (isDebugMode) {
            window.phaserGame = game;
            this.map = map;
        }

        // hero Sprite
        this.heroSprite = this.physics.add
            .sprite(0, 0, 'hero', initialFrame)
            .setDepth(1);

        this.heroSprite.body.setSize(14, 14);
        this.heroSprite.body.setOffset(9, 13);
        this.heroSprite.name = "hero";
        this.heroSprite.setInteractive();

        this.heroActionCollider = createInteractiveGameObject(
            this,
            this.heroSprite.x + 9,
            this.heroSprite.y + 36,
            14,
            8,
            'hero',
            isDebugMode
        );
        this.heroPresenceCollider = createInteractiveGameObject(
            this,
            this.heroSprite.x + 16,
            this.heroSprite.y + 20,
            320,
            320,
            'presence',
            isDebugMode,
            { x: 0.5, y: 0.5 }
        );
        this.heroObjectCollider = createInteractiveGameObject(
            this,
            this.heroSprite.x + 16,
            this.heroSprite.y + 20,
            24,
            24,
            'object',
            isDebugMode,
            { x: 0.5, y: 0.5 }
        );

        // 加载地图图层
        const elementsLayers = this.add.group();
        for (let i = 0; i < map.layers.length; i++) {
            const layer = map.createLayer(i, tileset, 0, 0);
            layer.layer.properties.forEach((property) => {
                const { value, name } = property;

                if (name === 'type' && value === 'elements') {
                    elementsLayers.add(layer);
                }
            });

            this.physics.add.collider(this.heroSprite, layer);
        }

        // 解析NPC的属性
        const npcsKeys = [];
        const dataLayer = map.getObjectLayer('actions');
        dataLayer.objects.forEach((data) => {
            const { properties, x, y } = data;

            properties.forEach((property) => {
                const { name, value } = property;

                switch (name) {
                    case 'npcData': {
                        const {
                            facingDirection,
                            movementType,
                            npcKey,
                            delay,
                            area,
                        } = this.extractNpcDataFromTiled(value);

                        npcsKeys.push({
                            facingDirection,
                            movementType,
                            npcKey,
                            delay,
                            area,
                            x,
                            y,
                        });
                        break;
                    }
                    default: {
                        break;
                    }
                }
            });
        });

        // 摄像机跟随
        camera.startFollow(this.heroSprite, true);
        camera.setFollowOffset(-this.heroSprite.width, -this.heroSprite.height);
        camera.setBounds(
            0,
            0,
            Math.max(map.widthInPixels, game.scale.gameSize.width),
            Math.max(map.heightInPixels, game.scale.gameSize.height)
        );

        if (map.widthInPixels < game.scale.gameSize.width) {
            camera.setPosition(
                (game.scale.gameSize.width - map.widthInPixels) / 2
            );
        }

        if (map.heightInPixels < game.scale.gameSize.height) {
            camera.setPosition(
                camera.x,
                (game.scale.gameSize.height - map.heightInPixels) / 2
            );
        }

        const gridEngineConfig = {
            characters: [
                {
                    id: 'hero',
                    sprite: this.heroSprite,
                    startPosition: initialPosition,
                    offsetY: 4
                },
            ],
        };

        // NPC移动
        const npcSprites = this.add.group();
        npcsKeys.forEach((npcData) => {
            const { npcKey, x, y, facingDirection = 'down' } = npcData;
            const npc = this.physics.add.sprite(0, 0, npcKey, `${npcKey}_idle_${facingDirection}_01`);
            npc.body.setSize(14, 14);
            npc.body.setOffset(9, 13);
            npc.name = npcKey;
            npcSprites.add(npc);
            npc.setInteractive();

            this.createPlayerWalkingAnimation(npcKey, 'walking_up');
            this.createPlayerWalkingAnimation(npcKey, 'walking_right');
            this.createPlayerWalkingAnimation(npcKey, 'walking_down');
            this.createPlayerWalkingAnimation(npcKey, 'walking_left');

            gridEngineConfig.characters.push({
                id: npcKey,
                sprite: npc,
                startPosition: { x: x / 16, y: (y / 16) - 1 },
                speed: 1,
                offsetY: 4,
            });
        });

        // 移动动画
        this.createPlayerWalkingAnimation('hero', 'walking_up');
        this.createPlayerWalkingAnimation('hero', 'walking_right');
        this.createPlayerWalkingAnimation('hero', 'walking_down');
        this.createPlayerWalkingAnimation('hero', 'walking_left');

        this.gridEngine.create(map, gridEngineConfig);

        // NPC 随机移动
        npcsKeys.forEach((npcData) => {
            const {
                movementType,
                npcKey,
                delay,
                area,
            } = npcData;
            if (movementType === NPC_MOVEMENT_RANDOM) {
                this.gridEngine.moveRandomly(npcKey, delay, area);
            }
        });

        // 动画播放
        this.gridEngine.movementStarted().subscribe(({ charId, direction }) => {
            if (charId === 'hero') {
                this.heroSprite.anims.play(`hero_walking_${direction}`);
            } else {
                const npc = npcSprites.getChildren().find((npcSprite) => npcSprite.texture.key === charId);
                if (npc) {
                    npc.anims.play(`${charId}_walking_${direction}`);
                    return;
                }
            }
        });
        // 订阅事件
        this.gridEngine.movementStopped().subscribe(({ charId, direction }) => {
            if (charId === 'hero') {
                this.heroSprite.anims.stop();
                this.heroSprite.setFrame(this.getStopFrame(direction, charId));
            } else {
                const npc = npcSprites.getChildren().find((npcSprite) => npcSprite.texture.key === charId);
                if (npc) {
                    npc.anims.stop();
                    npc.setFrame(this.getStopFrame(direction, charId));
                    return;
                }
            }
        });

        this.gridEngine.directionChanged().subscribe(({ charId, direction }) => {
            if (charId === 'hero') {
                this.heroSprite.setFrame(this.getStopFrame(direction, charId));
            } else {
                const npc = npcSprites.getChildren().find((npcSprite) => npcSprite.texture.key === charId);
                if (npc) {
                    npc.setFrame(this.getStopFrame(direction, charId));
                    return;
                }
            }
        });

        // Hero 角色位置更新
        this.heroActionCollider.update = () => {
            const facingDirection = this.gridEngine.getFacingDirection('hero');
            this.heroPresenceCollider.setPosition(
                this.heroSprite.x + 16,
                this.heroSprite.y + 20
            );

            this.heroObjectCollider.setPosition(
                this.heroSprite.x + 16,
                this.heroSprite.y + 20
            );

            switch (facingDirection) {
                case 'down': {
                    this.heroActionCollider.setSize(14, 8);
                    this.heroActionCollider.body.setSize(14, 8);
                    this.heroActionCollider.setX(this.heroSprite.x + 9);
                    this.heroActionCollider.setY(this.heroSprite.y + 36);

                    break;
                }

                case 'up': {
                    this.heroActionCollider.setSize(14, 8);
                    this.heroActionCollider.body.setSize(14, 8);
                    this.heroActionCollider.setX(this.heroSprite.x + 9);
                    this.heroActionCollider.setY(this.heroSprite.y + 12);

                    break;
                }

                case 'left': {
                    this.heroActionCollider.setSize(8, 14);
                    this.heroActionCollider.body.setSize(8, 14);
                    this.heroActionCollider.setX(this.heroSprite.x);
                    this.heroActionCollider.setY(this.heroSprite.y + 21);

                    break;
                }

                case 'right': {
                    this.heroActionCollider.setSize(8, 14);
                    this.heroActionCollider.body.setSize(8, 14);
                    this.heroActionCollider.setX(this.heroSprite.x + 24);
                    this.heroActionCollider.setY(this.heroSprite.y + 21);

                    break;
                }

                default: {
                    break;
                }
            }
        };

        // heor 与 NPC 碰撞
        this.physics.add.overlap(this.heroObjectCollider, npcSprites, (objA, objB) => {
            if (this.isConversationing > 0) {
                return;
            }
            this.isConversationing = 1;
            this.conversation(objB, npcsKeys);
        });
    }

    async genConversationByGPT(characterName) {
        for (var i = 0; i < 4; i++) {
            await callGpt(characterName, topic, i);
        }
    }

    async conversation(npc, npcsKeys) {
        const characterName = npc.texture.key;
        this.gridEngine.stopMovement(characterName);
        this.gridEngine.stopMovement("hero");
        this.updateGameHint("与" + characterName + "聊天中...");
        await this.genConversationByGPT(characterName);
        const timer = setInterval(() => {
            clearInterval(timer);
            this.time.delayedCall(2000, () => {
                //close dialog
                window.dispatchEvent(new CustomEvent('close-dialog', {
                    detail: {
                        "characterName": characterName
                    },
                }));
            });
            //Stop Conversation
            const dialogBoxFinishedEventListener = () => {
                window.removeEventListener(`
                            ${characterName}-dialog-finished`, dialogBoxFinishedEventListener);
                const { delay, area } = npcsKeys.find((npcData) => npcData.npcKey === characterName);
                this.gridEngine.moveRandomly(characterName, delay, area);
                if (this.isMoveRandomly) {
                    this.gridEngine.moveRandomly("hero", 1500, 3);
                }
                this.time.delayedCall(3000, () => {
                    this.isConversationing = 0;
                    this.updateGameHint(" ");
                });
            };
            window.addEventListener(`${characterName}-dialog-finished`, dialogBoxFinishedEventListener);
            const facingDirection = this.gridEngine.getFacingDirection('hero');
            npc.setFrame(this.getStopFrame(this.getOppositeDirection(facingDirection), characterName));
        }, 1000);
    }

    update() {
        this.heroActionCollider.update();
        if (this.cursors.left.isDown) {
            this.gridEngine.move('hero', 'left');
        } else if (this.cursors.right.isDown) {
            this.gridEngine.move('hero', 'right');
        } else if (this.cursors.up.isDown) {
            this.gridEngine.move('hero', 'up');
        } else if (this.cursors.down.isDown) {
            this.gridEngine.move('hero', 'down');
        }
    }
}
