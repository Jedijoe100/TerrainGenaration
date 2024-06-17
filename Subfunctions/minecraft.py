import numpy as np
from geometry import geographic_to_cartesian
from scipy.ndimage import gaussian_filter
from amulet.api.block import Block
from amulet.api.errors import ChunkDoesNotExist
from amulet.api.chunk import Chunk


MINECRAFT_SETTINGS = {
    'HEIGHT_DIFF': 100,
    'WORLD_RESOLUTION': (1000, 500),
    'LOWEST_POINT': 50
}

def generate_block_references(level, blocks):
    block_dictionary = {}
    for block in blocks:
        (
            universal_block_1,
            universal_block_entity_1,
            universal_extra_1,
        ) = level.translation_manager.get_version("java", (1, 19, 4)).block.to_universal(
            Block("minecraft", block)
        )
        block_dictionary[block] = level.block_palette.get_add_block(universal_block_1) 
    return block_dictionary

def chunk_based_generation(level, resolution, grid):
    blur_factor = 2 #have this depend on the detail of grid and the 
    chunk_grid = resolution//16
    height_min = np.min(grid.height)
    height_dispersion = (np.max(grid.height)-height_min)
    blocks = generate_block_references(level, ['stone', 'dirt', 'grass_block','gravel', 'water'])
    processed_sea_level = MINECRAFT_SETTINGS['LOWEST_POINT']+MINECRAFT_SETTINGS['HEIGHT_DIFF']*(grid.settings['SEA_LEVEL']-height_min)/height_dispersion
    for cx in range(chunk_grid[x]):
        for cz in range(chunk_grid[1]):
            xx, yy = np.meshgrid(np.linspace(
                (cx-1)*2*np.pi/chunk_grid[0], (cx+2)*2*np.pi/chunk_grid[0], 48), np.linspace(
                (cx-1)*2*np.pi/chunk_grid[0], (cz+2)*2*np.pi/chunk_grid[1], 48))
            indices = grid.grid_tree.query(geographic_to_cartesian(np.array([xx.flatten(), yy.flatten()]).transpose())).reshape((48, 48))
            height_map = MINECRAFT_SETTINGS['LOWEST_POINT']+MINECRAFT_SETTINGS['HEIGHT_DIFF']*(gaussian_filter(grid.height[indices], sigma=blur_factor)-height_min)/height_dispersion
            try:
                chunk = level.get_chunk(cx, cz, "minecraft:overworld")
            except ChunkDoesNotExist:
                chunk = Chunk(cx, cz)
                level.put_chunk(chunk, "minecraft:overworld")
                chunk = level.get_chunk(cx, cz, "minecraft:overworld")
            chunk.block()
            for x in range(16):
                for z in range(16):
                    chunk.block
                    if height_map[x, z] > self.settings['SEA_LEVEL']:
                        chunk.blocks[x, int(height_map[16+x,16+z])-3:int(height_map[16+x,16+z]), z] = blocks['dirt']
                        chunk.blocks[x, int(height_map[16+x,16+z]), z] = blocks['grass_block']
                    else:
                        chunk.blocks[x, int(height_map[16+x,16+z])-3:int(height_map[16+x,16+z]), z] = blocks['gravel']
                        chunk.blocks[x, int(height_map[16+x,16+z]):processed_sea_level, z] = blocks['water']
            chuck.changed = True
    level.save()
