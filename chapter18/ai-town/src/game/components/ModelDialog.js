const COLOR_PRIMARY = 0x4e342e;
const COLOR_LIGHT = 0x7b5e57;
const COLOR_DARK = 0x260e04;

var ModelDialog = function (scene, content) {
    var heroX = scene.heroSprite.x + 50;
    var heroY = scene.heroSprite.y + 50;
    if(heroX<250){
        heroX = 250;
    };
    if(heroX>700){
        heroX = 700
    };
    if(heroY<220){
        heroY = 220;
    };
    if(heroY>550){
        heroY = 550
    };
    
    return scene.rexUI.add.textArea({
        x: heroX,
        y: heroY,
        width: 300,
        height: 400,
        
        background: scene.rexUI.add.roundRectangle({
            color: COLOR_PRIMARY,
            radius: 20
        }),
        
        text: scene.rexUI.add.BBCodeText(0, 0, '', {
            fontSize: '12px',
            wrap: {
                mode: 'char',
                width: 200
            }
        }),
        
        slider: {
            track: scene.rexUI.add.roundRectangle(0, 0, 20, 10, 10, COLOR_DARK),
            thumb: scene.rexUI.add.roundRectangle(0, 0, 0, 0, 13, COLOR_LIGHT),
        },

        space: {
            left: 10, right: 10, top: 20, bottom: 20,

            text: 10,
            header: 20,
            footer: 20,
        },

        scroller: {
            
        },

        mouseWheelScroller: {
            focus: false,
            speed: 0.1
        },

        header: scene.rexUI.add.label({
            space: { left: 10, right: 10, top: 10, bottom: 10 },

            orientation: 0,
            background: scene.rexUI.add.roundRectangle(0, 0, 20, 20, 0, COLOR_DARK),
            text: scene.add.text(0, 0, '对话历史'),
        }),

        footer: scene.rexUI.add.label({
            space: { left: 10, right: 10, top: 10, bottom: 10 },

            orientation: 0,
            background: scene.rexUI.add.roundRectangle({
                radius: 10,
                color: COLOR_DARK,
                strokeColor: COLOR_LIGHT
            }),
            text: scene.add.text(0, 0, '关闭'),
        }).onClick(function (button, gameObject, pointer, event) {
            gameObject.getTopmostSizer().modalClose();
        }),

        content: content,

        expand: {
            footer: false
        }
    })
}

export default ModelDialog;