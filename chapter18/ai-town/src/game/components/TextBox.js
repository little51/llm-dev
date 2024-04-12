import Phaser from 'phaser';

const COLOR_PRIMARY = 0x4e342e;
const COLOR_LIGHT = 0x7b5e57;
const COLOR_DARK = 0x260e04;

const GetValue = Phaser.Utils.Objects.GetValue;
const TextBox = (scene, x, y, config) => {
    var wrapWidth = GetValue(config, 'wrapWidth', 0);
    var fixedWidth = GetValue(config, 'fixedWidth', 0);
    var fixedHeight = GetValue(config, 'fixedHeight', 0);
    var titleText = GetValue(config, 'title', undefined);

    var textBox = scene.rexUI.add.textBox({
        x: x,
        y: y,
        background: scene.rexUI.add.roundRectangle({ radius: 5, color: COLOR_PRIMARY, strokeColor: COLOR_LIGHT, strokeWidth: 2 }),
        icon: scene.rexUI.add.roundRectangle({ radius: 5, color: COLOR_DARK }),
        text: getBBcodeText(scene, wrapWidth, fixedWidth, fixedHeight),
        //action: scene.add.image(0, 0, 'nextPage').setTint(COLOR_LIGHT).setVisible(false),
        title: (titleText) ? scene.add.text(0, 0, titleText, { fontSize: '9px', }) : undefined,
        separator: (titleText) ? scene.rexUI.add.roundRectangle({ height: 3, color: COLOR_DARK }) : undefined,
        space: {
            left: 5, right: 5, top: 5, bottom: 5,
            icon: 10, text: 10,
            separator: 6,
        },
        align: {
            title: 'center'
        }
    })
        .setOrigin(0)
        .layout();

    textBox
        .setInteractive()
        .on('pointerdown', function () {
            var icon = this.getElement('action').setVisible(false);
            this.resetChildVisibleState(icon);
            if (this.isTyping) {
                this.stop(true);
            } else if (!this.isLastPage) {
                this.typeNextPage();
            } else {
                // Next actions
            }
        }, textBox)
        .on('pageend', function () {
            if (this.isLastPage) {
                return;
            }
        }, textBox)
        .on('complete', function () {
            console.log('all pages typing complete')
        })

    return textBox;
}
var getBBcodeText = function (scene, wrapWidth, fixedWidth, fixedHeight) {
    return scene.rexUI.add.BBCodeText(0, 0, '', {
        fixedWidth: fixedWidth,
        fixedHeight: fixedHeight,
        fontSize: '9px',
        wrap: {
            mode: 'word',
            width: wrapWidth
        },
        maxLines: 3
    })
}

export default TextBox;