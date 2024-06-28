import numpy as np
from Subfunctions.geometry import geographic_to_cartesian
from scipy.ndimage import gaussian_filter
from amulet.api.block import Block
from amulet.api.errors import ChunkDoesNotExist
from amulet.api.chunk import Chunk, Biomes
from amulet.utils.world_utils import chunk_coords_to_block_coords
from amulet import load_level
import os
import shutil

MINECRAFT_SETTINGS = {
    'HEIGHT_DIFF': 100,
    'WORLD_RESOLUTION': (1000, 500),
    'LOWEST_POINT': 50,
    'BLUR_FACTOR': 2
}


def generate_block_references(level, blocks):
    """
    From a list of blocks and a minecraft level return a dictionary of the blocks with their ids.
    """

    block_dictionary = {}
    for block in blocks:
        (
            universal_block_1,
            universal_block_entity_1,
            universal_extra_1,
        ) = level.translation_manager.get_version("java", (1, 19, 4)).block.to_universal(
            Block("minecraft", block)
        )
        block_dictionary[block] = level.block_palette.get_add_block(
            universal_block_1)
    return block_dictionary


def chunk_based_generation(level, resolution, grid, biomes):
    """
    Generate the Minecraft world from a level to a specificed resolution.
    Requires a biomes object and a grid object.
    """

    print(list(level.biome_palette))
    chunk_grid = np.array(resolution)//16
    height_min = np.min(grid.height)
    height_dispersion = (np.max(grid.height)-height_min)
    blocks = generate_block_references(
        level, ['stone', 'dirt', 'grass_block', 'gravel', 'water', 'sand', 'snow_block', 'packed_ice'])
    processed_sea_level = MINECRAFT_SETTINGS['LOWEST_POINT']+MINECRAFT_SETTINGS['HEIGHT_DIFF']*(
        grid.settings['SEA_LEVEL']-height_min)/height_dispersion
    xx, yy = np.meshgrid(np.linspace(0, 2*np.pi, chunk_grid[0]*16), np.linspace(
        0, np.pi, chunk_grid[1]*16))
    indices = grid.grid_tree.query(geographic_to_cartesian(np.array([xx.flatten(
    ), yy.flatten()]).transpose()))[1].reshape(chunk_grid[0]*16, chunk_grid[1]*16)
    print(np.shape(indices))
    height_map = gaussian_filter(MINECRAFT_SETTINGS['LOWEST_POINT']+MINECRAFT_SETTINGS['HEIGHT_DIFF']*(
        grid.height[indices]-height_min)/height_dispersion, MINECRAFT_SETTINGS['BLUR_FACTOR'])
    water_level = grid.water_level[indices] * MINECRAFT_SETTINGS['HEIGHT_DIFF']/height_dispersion
    biome = grid.biome[indices]
    minecraft_biome = biomes.biome_correspond[biome]
    for cx in range(chunk_grid[0]-1):
        for cz in range(chunk_grid[1]-1):
            try:
                chunk = level.get_chunk(cx, cz, "minecraft:overworld")
            except ChunkDoesNotExist:
                chunk = Chunk(cx, cz)
                level.put_chunk(chunk, "minecraft:overworld")
                chunk = level.get_chunk(cx, cz, "minecraft:overworld")
            chunk_x, chunk_z = chunk_coords_to_block_coords(cx, cz)
            tem_biomes = minecraft_biome[chunk_x:chunk_x +
                                         16, chunk_z:chunk_z + 16]
            if np.shape(tem_biomes) == (16, 16):
                chunk.biomes = Biomes(tem_biomes)
            for x in range(16):
                for z in range(16):
                    height_value = int(height_map[chunk_x+x, chunk_z+z])
                    chunk.blocks[x, -63:height_value-3, z] = blocks['stone']
                    if height_value > processed_sea_level:
                        if water_level[chunk_x + x, chunk_z+z] >= 1:
                            chunk.blocks[x, height_value -
                                         3:height_value+1, z] = blocks['gravel']
                            chunk.blocks[x, height_value+1:height_value +
                                         1+int(water_level[chunk_x+x, chunk_z+z]), z]
                        elif biome[chunk_x+x, chunk_z + z] in [23, 31]:
                            chunk.blocks[x, height_value -
                                         3:height_value+1, z] = blocks['sand']
                        elif biome[chunk_x + x, chunk_z + z] in [8, 7]:
                            chunk.blocks[x, height_value -
                                         3:height_value, z] = blocks['dirt']
                            chunk.blocks[x, height_value,
                                         z] = blocks['snow_block']
                        else:
                            chunk.blocks[x, height_value -
                                         3:height_value, z] = blocks['dirt']
                            chunk.blocks[x, height_value,
                                         z] = blocks['grass_block']
                        # add lakes
                    elif height_value == processed_sea_level:
                        chunk.blocks[x, height_value -
                                     3:height_value+1, z] = blocks['gravel']
                    else:
                        chunk.blocks[x, height_value -
                                     3:height_value, z] = blocks['gravel']
                        chunk.blocks[x, height_value:processed_sea_level,
                                     z] = blocks['water']
                        if biome[x, z] == 6:
                            chunk.blocks[x, int(
                                processed_sea_level)-1, z] = blocks['packed_ice']
            chunk.changed = True
    return level


def export_to_minecraft_world(self, file_path, biomes):
    """
    From the file path and a biomes function.
    Requires a minecraft world in a folder called biome_template and a folder called current_world.
    """

    shutil.rmtree(os.path.join(file_path, '.\\current_world'))
    shutil.copytree(os.path.join(file_path, '.\\biome_template'),
                    os.path.join(file_path, './current_world'))
    level = load_level('current_world')
    for biome in biomes.minecraft_biomes:
        level.biome_palette.register(f'universal_minecraft:{biome}')
    level = chunk_based_generation(
        level, MINECRAFT_SETTINGS['WORLD_RESOLUTION'], self, biomes)
    level.save()
    level.close()
