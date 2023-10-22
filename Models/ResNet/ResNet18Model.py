from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, ReLU, BatchNormalization, Add, AveragePooling2D, Flatten


def relu_batch_normalization() -> list:
    return [ReLU(), BatchNormalization()]


def residual_block(downsample: bool, filters: int, kernel_size: int = 3) -> dict:
    """
    Create ResNet residual block.
    """
    block = {
        'conv_layers': [],
        'downsample_layer': None,
        'add_layer': None,
        'output_layers': [],
    }

    block['conv_layers'].append(
        Conv2D(kernel_size=kernel_size, strides=(1 if not downsample else 2), filters=filters, padding='same'))
    block['conv_layers'] = block['conv_layers'] + relu_batch_normalization()
    block['conv_layers'].append(Conv2D(kernel_size=kernel_size, strides=1, filters=filters, padding='same'))

    # Skip layer
    if downsample:
        block['downsample_layer'] = Conv2D(kernel_size=1, strides=2, filters=filters, padding='same')

    # Merge layers inside block
    block['add_layer'] = Add()
    block['output_layers'] = relu_batch_normalization()

    return block


def create_residual_blocks(num_blocks_list: list, num_filters: int) -> list:
    residual_blocks = []
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            residual_blocks.append(residual_block(downsample=(j == 0 and i != 0), filters=num_filters))
        num_filters *= 2

    return residual_blocks


class ResNet18Model(Model):
    def __init__(self, num_filters: int = 64, num_blocks_list: list = None):
        super(ResNet18Model, self).__init__()
        if num_blocks_list is None:
            num_blocks_list = [2, 2, 2, 2]

        self.bn_1 = BatchNormalization()
        self.conv_1 = Conv2D(kernel_size=3, strides=1, filters=num_filters, padding='same')
        self.relu_bn_1 = relu_batch_normalization()

        # Residual blocks
        self.residual_blocks = create_residual_blocks(num_blocks_list, num_filters)

        self.pooling = AveragePooling2D(4)
        self.flatten = Flatten()

    def call(self, inputs):
        x = self.bn_1(inputs)
        x = self.conv_1(x)

        # Apply ReLU and Batch Normalization
        for layer in self.relu_bn_1:
            x = layer(x)

        # Residual blocks
        for block in self.residual_blocks:
            block_input = x

            if block['downsample_layer'] is not None:
                block_input = block['downsample_layer'](block_input)

            for layer in block['conv_layers']:
                x = layer(x)

            if block['add_layer'] is not None:
                x = block['add_layer']([x, block_input])

            for layer in block['output_layers']:
                x = layer(x)

        x = self.pooling(x)
        x = self.flatten(x)

        return x
