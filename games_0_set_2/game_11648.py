import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:17:49.588892
# Source Brief: brief_01648.md
# Brief Index: 1648
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Match falling tiles to create teleporting portals. Guide blocks through the portals "
        "to build the target structure before time runs out."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to select or "
        "deselect matching tiles to create portals."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 60 * self.FPS # 60 seconds episode

        # Colors
        self.COLOR_BG = (10, 5, 25)
        self.COLOR_TEXT = (220, 220, 255)
        self.COLOR_CURSOR = (255, 255, 100)
        self.COLOR_CURSOR_SELECTED = (100, 255, 100)
        self.COLOR_TARGET_OUTLINE = (100, 100, 180, 100)
        self.PORTAL_COLORS = [(60, 60, 255), (150, 80, 255)]
        self.BLOCK_COLORS = {
            'A': (255, 215, 0),  # Gold
            'B': (192, 192, 192), # Silver
            'C': (0, 255, 127),  # Jade
            'D': (255, 105, 180)  # Pink
        }
        self.TILE_BG_COLOR = (30, 20, 60)

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_tile = pygame.font.Font(None, 24)
        self.font_game_over = pygame.font.Font(None, 72)

        # Game state variables
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.last_space_held = False

        self.tiles = []
        self.tile_lanes = 8
        self.tile_width = self.WIDTH // self.tile_lanes
        self.tile_height = 30
        self.tile_fall_speed = 1.0
        self.tile_spawn_timer = 0
        self.tile_spawn_interval = 60

        self.cursor_pos = [0, 0] # [col, row]
        self.grid_rows = 6
        self.grid_height = self.HEIGHT // self.grid_rows
        self.selected_tile_1 = None

        self.portals = []
        self.blocks = []
        self.block_spawn_timer = 0
        self.block_spawn_interval = 90
        self.block_fall_speed = 2.0
        self.misplaced_blocks = 0
        self.max_misplaced_blocks = 5

        self.target_structure = {}
        self.built_structure = {}
        
        self.particles = []
        self.nebula = self._create_nebula(100)

        # self.reset() is called by the environment wrapper
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.last_space_held = False
        
        self.tiles.clear()
        self.portals.clear()
        self.blocks.clear()
        self.particles.clear()
        
        self.cursor_pos = [self.tile_lanes // 2, self.grid_rows // 2]
        self.selected_tile_1 = None
        self.misplaced_blocks = 0
        
        self.tile_fall_speed = 1.0
        self.tile_spawn_timer = 0
        self.block_spawn_timer = 0
        
        self._generate_target_structure()
        self.built_structure.clear()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        reward = 0.0
        self.steps += 1

        # --- Handle Input ---
        movement = action[0]
        space_held = action[1] == 1
        
        space_press = space_held and not self.last_space_held
        self.last_space_held = space_held
        
        reward += self._handle_input(movement, space_press)

        # --- Update Game Logic ---
        self._update_tiles()
        self._update_blocks()
        self._update_portals()
        self._update_particles()
        
        reward += self._check_block_placement()

        # --- Difficulty Scaling ---
        if self.steps % 500 == 0:
            self.tile_fall_speed += 0.05

        # --- Check Termination ---
        terminated = False
        if self.steps >= self.MAX_STEPS:
            terminated = True
            reward = -50.0 # Time out
            self.game_over_message = "TIME UP"
        elif len(self.built_structure) == len(self.target_structure):
            terminated = True
            reward = 50.0 # Victory
            self.game_over_message = "STRUCTURE COMPLETE"
        elif self.misplaced_blocks >= self.max_misplaced_blocks:
            terminated = True
            reward = -50.0 # Failure
            self.game_over_message = "TOO MUCH DEBRIS"
        
        if terminated:
            self.game_over = True

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_press):
        # Move cursor
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1) # Up
        elif movement == 2: self.cursor_pos[1] = min(self.grid_rows - 1, self.cursor_pos[1] + 1) # Down
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1) # Left
        elif movement == 4: self.cursor_pos[0] = min(self.tile_lanes - 1, self.cursor_pos[0] + 1) # Right
        
        # Handle selection
        if space_press:
            cursor_rect = self._get_cursor_rect()
            tile_under_cursor = None
            for tile in self.tiles:
                if cursor_rect.colliderect(tile['rect']):
                    tile_under_cursor = tile
                    break
            
            if tile_under_cursor:
                # sound_effect: "select_tile"
                if not self.selected_tile_1:
                    self.selected_tile_1 = tile_under_cursor
                else:
                    if self.selected_tile_1 != tile_under_cursor and self.selected_tile_1['word'] == tile_under_cursor['word']:
                        # Match found, create portal
                        # sound_effect: "create_portal"
                        self._create_portal(self.selected_tile_1, tile_under_cursor)
                        self.tiles.remove(self.selected_tile_1)
                        self.tiles.remove(tile_under_cursor)
                        self.selected_tile_1 = None
                        return 0.1 # Reward for matching tiles
                    else:
                        # sound_effect: "cancel_selection"
                        self.selected_tile_1 = None # Cancel selection
            else:
                # sound_effect: "cancel_selection"
                self.selected_tile_1 = None # Clicked empty space
        return 0.0

    def _update_tiles(self):
        self.tile_spawn_timer += 1
        if self.tile_spawn_timer >= self.tile_spawn_interval:
            self.tile_spawn_timer = 0
            self._spawn_tile()

        for tile in self.tiles[:]:
            tile['y'] += self.tile_fall_speed
            tile['rect'].y = int(tile['y'])
            if tile['rect'].top > self.HEIGHT:
                self.tiles.remove(tile)
                if tile == self.selected_tile_1:
                    self.selected_tile_1 = None

    def _update_blocks(self):
        self.block_spawn_timer += 1
        if self.block_spawn_timer >= self.block_spawn_interval and len(self.target_structure) > len(self.built_structure):
            self.block_spawn_timer = 0
            self._spawn_block()

        for block in self.blocks[:]:
            block['y'] += self.block_fall_speed
            block['rect'].y = int(block['y'])

            # Teleportation
            for portal in self.portals:
                if portal['in_rect'].colliderect(block['rect']):
                    # sound_effect: "teleport"
                    block['x'], block['y'] = portal['out_rect'].center
                    block['rect'].center = (int(block['x']), int(block['y']))
                    # Add particles at both ends
                    self._create_particles(portal['in_rect'].center, self.BLOCK_COLORS[block['word']])
                    self._create_particles(portal['out_rect'].center, self.BLOCK_COLORS[block['word']])
                    break

            # Out of bounds
            if block['rect'].top > self.HEIGHT:
                self.blocks.remove(block)
                self.misplaced_blocks += 1

    def _update_portals(self):
        for portal in self.portals[:]:
            portal['life'] -= 1
            portal['angle'] = (portal['angle'] + 5) % 360
            if portal['life'] <= 0:
                self.portals.remove(portal)

    def _update_particles(self):
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_block_placement(self):
        reward = 0.0
        for block in self.blocks[:]:
            for pos, word in self.target_structure.items():
                if pos not in self.built_structure:
                    target_rect = pygame.Rect(pos[0] * self.tile_width, self.HEIGHT - (pos[1] + 1) * self.tile_height, self.tile_width, self.tile_height)
                    if target_rect.colliderect(block['rect']) and block['word'] == word:
                        # sound_effect: "place_block_success"
                        self.built_structure[pos] = word
                        self.blocks.remove(block)
                        self._create_particles(target_rect.center, self.BLOCK_COLORS[word], 20)
                        reward += 0.5 # Reward for correct placement
                        
                        # Check for layer completion
                        layer_y = pos[1]
                        layer_complete = True
                        for t_pos, _ in self.target_structure.items():
                            if t_pos[1] == layer_y and t_pos not in self.built_structure:
                                layer_complete = False
                                break
                        if layer_complete:
                            # sound_effect: "layer_complete"
                            reward += 2.0
                        return reward # Process one block per step for stability
        return reward

    def _get_observation(self):
        # --- Background ---
        self.screen.fill(self.COLOR_BG)
        self._render_nebula()
        
        # --- Game Elements ---
        self._render_target_structure()
        self._render_portals()
        self._render_tiles()
        self._render_blocks()
        self._render_particles()
        self._render_cursor()
        
        # --- UI ---
        self._render_ui()

        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_nebula(self):
        for star in self.nebula:
            star['pos'][0] = (star['pos'][0] + star['vel'][0]) % self.WIDTH
            star['pos'][1] = (star['pos'][1] + star['vel'][1]) % self.HEIGHT
            alpha = int(128 + 127 * math.sin(self.steps * star['freq'] + star['phase']))
            color = (*star['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(star['pos'][0]), int(star['pos'][1]), star['size'], color)

    def _render_target_structure(self):
        # Draw outline
        for pos, word in self.target_structure.items():
            if pos not in self.built_structure:
                rect = pygame.Rect(pos[0] * self.tile_width, self.HEIGHT - (pos[1] + 1) * self.tile_height, self.tile_width, self.tile_height)
                pygame.draw.rect(self.screen, self.COLOR_TARGET_OUTLINE, rect, 1, border_radius=3)
        
        # Draw built parts
        for pos, word in self.built_structure.items():
            rect = pygame.Rect(pos[0] * self.tile_width, self.HEIGHT - (pos[1] + 1) * self.tile_height, self.tile_width, self.tile_height)
            self._draw_glowing_rect(rect, self.BLOCK_COLORS[word])

    def _render_portals(self):
        for portal in self.portals:
            self._draw_vortex(self.screen, portal['in_rect'].center, portal['angle'])
            self._draw_vortex(self.screen, portal['out_rect'].center, -portal['angle'])
    
    def _render_tiles(self):
        for tile in self.tiles:
            # Glow for selected tile
            if tile == self.selected_tile_1:
                glow_rect = tile['rect'].inflate(10, 10)
                self._draw_glowing_rect(glow_rect, self.COLOR_CURSOR_SELECTED, alpha=80)

            pygame.draw.rect(self.screen, self.TILE_BG_COLOR, tile['rect'], border_radius=4)
            pygame.draw.rect(self.screen, self.BLOCK_COLORS[tile['word']], tile['rect'], 2, border_radius=4)
            
            text_surf = self.font_tile.render(tile['word'], True, self.BLOCK_COLORS[tile['word']])
            text_rect = text_surf.get_rect(center=tile['rect'].center)
            self.screen.blit(text_surf, text_rect)

    def _render_blocks(self):
        for block in self.blocks:
            self._draw_glowing_rect(block['rect'], self.BLOCK_COLORS[block['word']])
    
    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), size, color)

    def _render_cursor(self):
        cursor_rect = self._get_cursor_rect()
        color = self.COLOR_CURSOR_SELECTED if self.selected_tile_1 else self.COLOR_CURSOR
        pygame.draw.rect(self.screen, color, cursor_rect, 2, border_radius=5)

    def _render_ui(self):
        # Score
        score_text = f"Score: {self.score:.1f}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Time
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        time_text = f"Time: {max(0, time_left):.1f}"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_TEXT)
        time_rect = time_surf.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(time_surf, time_rect)

        # Target preview
        preview_surf = pygame.Surface((self.tile_width * 3, self.tile_height * 3), pygame.SRCALPHA)
        for pos, word in self.target_structure.items():
            color = self.BLOCK_COLORS[word] if (pos in self.built_structure) else self.COLOR_TARGET_OUTLINE
            pygame.draw.rect(preview_surf, color, (pos[0]*10, (2-pos[1])*10, 10, 10), 0 if (pos in self.built_structure) else 1)
        preview_surf = pygame.transform.scale(preview_surf, (self.tile_width, self.tile_height))
        self.screen.blit(preview_surf, (self.WIDTH - 10 - self.tile_width, self.HEIGHT - 10 - self.tile_height))


    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        text_surf = self.font_game_over.render(self.game_over_message, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    # --- Helper Functions ---
    def _create_nebula(self, count):
        nebula = []
        for _ in range(count):
            nebula.append({
                'pos': [random.uniform(0, self.WIDTH), random.uniform(0, self.HEIGHT)],
                'vel': [random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2)],
                'size': random.randint(10, 50),
                'color': random.choice([(20, 10, 40), (40, 10, 20), (10, 20, 40)]),
                'freq': random.uniform(0.005, 0.02),
                'phase': random.uniform(0, 2 * math.pi)
            })
        return nebula

    def _spawn_tile(self):
        lane = random.randint(0, self.tile_lanes - 1)
        word = random.choice(list(self.BLOCK_COLORS.keys()))
        x = lane * self.tile_width
        self.tiles.append({
            'rect': pygame.Rect(x, -self.tile_height, self.tile_width, self.tile_height),
            'y': float(-self.tile_height),
            'word': word
        })
    
    def _spawn_block(self):
        # Spawn a block type that is still needed
        needed_words = [w for p, w in self.target_structure.items() if p not in self.built_structure]
        if not needed_words: return
        
        word = random.choice(needed_words)
        x = random.uniform(self.WIDTH * 0.4, self.WIDTH * 0.6)
        block_size = self.tile_width // 2
        self.blocks.append({
            'rect': pygame.Rect(x - block_size/2, -block_size, block_size, block_size),
            'x': x,
            'y': float(-block_size),
            'word': word
        })

    def _create_portal(self, tile1, tile2):
        self.portals.append({
            'in_rect': tile1['rect'].copy(),
            'out_rect': tile2['rect'].copy(),
            'life': 10 * self.FPS, # 10 seconds
            'angle': 0
        })
        self._create_particles(tile1['rect'].center, random.choice(self.PORTAL_COLORS), 30)
        self._create_particles(tile2['rect'].center, random.choice(self.PORTAL_COLORS), 30)

    def _create_particles(self, pos, color, count=10):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'x': pos[0], 'y': pos[1],
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'life': random.randint(15, 30), 'max_life': 30,
                'color': color, 'size': random.randint(2, 5)
            })

    def _generate_target_structure(self):
        self.target_structure.clear()
        # Simple 3-block pyramid
        words = list(self.BLOCK_COLORS.keys())
        self.target_structure[(3, 0)] = random.choice(words)
        self.target_structure[(4, 0)] = random.choice(words)
        self.target_structure[(3, 1)] = random.choice(words)

    def _get_cursor_rect(self):
        return pygame.Rect(
            self.cursor_pos[0] * self.tile_width,
            self.cursor_pos[1] * self.grid_height,
            self.tile_width,
            self.grid_height
        )
    
    def _draw_glowing_rect(self, rect, color, alpha=128):
        # Draw main rect
        pygame.draw.rect(self.screen, color, rect, border_radius=4)
        # Draw glow
        glow_surf = pygame.Surface((rect.width * 2, rect.height * 2), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*color, alpha), glow_surf.get_rect(), border_radius=8)
        glow_surf = pygame.transform.smoothscale(glow_surf, rect.size)
        self.screen.blit(glow_surf, rect.topleft, special_flags=pygame.BLEND_RGBA_ADD)

    def _draw_vortex(self, surface, center, angle):
        max_radius = self.tile_width // 2
        for i in range(max_radius, 0, -3):
            alpha = int(255 * (1 - i / max_radius))
            color_idx = (i // 3) % 2
            color = (*self.PORTAL_COLORS[color_idx], alpha)
            
            num_points = 12
            points = []
            for j in range(num_points):
                theta = (j / num_points) * 2 * math.pi + math.radians(angle + i*2)
                r = i * (1 + 0.2 * math.sin(theta * 3))
                x = center[0] + r * math.cos(theta)
                y = center[1] + r * math.sin(theta)
                points.append((int(x), int(y)))
            
            if len(points) > 2:
                pygame.gfxdraw.aapolygon(surface, points, color)

    def close(self):
        pygame.quit()
    
    def render(self):
        return self._get_observation()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Un-dummy the video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    pygame.quit() # Quit the dummy instance
    pygame.init() # Re-init with a display
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Dream Structure Builder")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0.0
    
    # Mapping keys to MultiDiscrete actions
    # actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    # actions[1]: Space button (0=released, 1=held)
    # actions[2]: Shift button (0=released, 1=held)
    
    action = [0, 0, 0] # no-op, space released, shift released

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0.0

        keys = pygame.key.get_pressed()
        
        # Movement
        action[0] = 0 # none
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        # Space
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        
        # Shift
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}")
            # Wait for a moment before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0.0

        clock.tick(env.FPS)
        
    env.close()