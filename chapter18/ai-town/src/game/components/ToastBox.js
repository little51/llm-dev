const COLOR_PRIMARY = 0x4e342e;

const ToastBox = (scene,x, y) => {
    var toast = scene.rexUI.add.toast({
            x: x,
            y: y,
            background: scene.rexUI.add.roundRectangle(0, 0, 2, 2, 5, COLOR_PRIMARY),
            text: scene.add.text(0, 0, '', {
                font: '9px Arial'
            }),
            space: {
                left: 10,
                right: 10,
                top: 5,
                bottom: 5,
            },
        }) ;
    return toast ;
}

export default ToastBox;