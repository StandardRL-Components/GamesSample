
# Generated: 2025-08-27T22:34:45.649101
# Source Brief: brief_03170.md
# Brief Index: 3170

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper class for isometric transformations and drawing
class IsometricConverter:
    def __init__(self, screen_width, screen_height, grid_dims, tile_dims):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.grid_width, self.grid_height = grid_dims
        self.tile_width, self.tile_height = tile_dims
        self.tile_width_half = self.tile_width // 2
        self.tile_height_half = self.tile_height // 2
        self.origin_x = self.screen_width // 2
        self.origin_y = self.screen_height // 2 - (self.grid_height * self.tile_height_half) // 2 + 50

    def to_screen(self, iso_x, iso_y):
        screen_x = self.origin_x + (iso_x - iso_y) * self.tile_width_half
        screen_y = self.origin_y + (iso_x + iso_y) * self.tile_height_half
        return int(screen_x), int(screen_y)

    def draw_cube(self, surface, iso_pos, color, height_offset=0):
        x, y = self.to_screen(iso_pos[0], iso_pos[1])
        y -= height_offset

        top_face = [
            (x, y - self.tile_height),
            (x + self.tile_width_half, y - self.tile_height_half),
            (x, y),
            (x - self.tile_width_half, y - self.tile_height_half),
        ]
        left_face = [
            (x - self.tile_width_half, y - self.tile_height_half),
            (x, y),
            (x, y + self.tile_height),
            (x - self.tile_width_half, y + self.tile_height_half),
        ]
        right_face = [
            (x + self.tile_width_half, y - self.tile_height_half),
            (x, y),
            (x, y + self.tile_height),
            (x + self.tile_width_half, y + self.tile_height_half),
        ]
        
        darker_color = tuple(max(0, c - 40) for c in color)
        darkest_color = tuple(max(0, c - 60) for c in color)

        pygame.gfxdraw.filled_polygon(surface, top_face, color)
        pygame.gfxdraw.aapolygon(surface, top_face, color)
        pygame.gfxdraw.filled_polygon(surface, left_face, darker_color)
        pygame.gfxdraw.aapolygon(surface, left_face, darker_color)
        pygame.gfxdraw.filled_polygon(surface, right_face, darkest_color)
        pygame.gfxdraw.aapolygon(surface, right_face, darkest_color)

# Entity classes
class Block:
    def __init__(self, iso_pos, max_health=50):
        self.iso_pos = iso_pos
        self.max_health = max_health
        self.health = max_health

class Enemy:
    def __init__(self, iso_pos, speed, health, color, target_pos):
        self.iso_pos = tuple(iso_pos)
        self.pixel_pos = [float(iso_pos[0]), float(iso_pos[1])] # For smooth movement
        self.speed = speed
        self.health = health
        self.color = color
        self.target_pos = target_pos
        
    def update(self, blocks):
        direction_vector = (self.target_pos[0] - self.pixel_pos[0], self.target_pos[1] - self.pixel_pos[1])
        dist = math.hypot(*direction_vector)
        
        if dist > 0:
            dx = direction_vector[0] / dist * self.speed
            dy = direction_vector[1] / dist * self.speed
            
            next_pos = (self.pixel_pos[0] + dx, self.pixel_pos[1] + dy)
            colliding_block = None
            for block in blocks:
                if math.hypot(block.iso_pos[0] - next_pos[0], block.iso_pos[1] - next_pos[1]) < 0.8:
                    colliding_block = block
                    break
            
            if colliding_block:
                return colliding_block # Return block it's attacking
            
            self.pixel_pos[0] += dx
            self.pixel_pos[1] += dy
            self.iso_pos = (int(round(self.pixel_pos[0])), int(round(self.pixel_pos[1])))
        return None

class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.vx = random.uniform(-1.5, 1.5)
        self.vy = random.uniform(-2.5, -0.5)
        self.life = random.randint(10, 20)
        self.radius = random.uniform(2, 4)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1 # Gravity
        self.life -= 1
        self.radius -= 0.1
        return self.life > 0 and self.radius > 0

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to build a defensive block."
    )
    game_description = (
        "An isometric base defense game. Place blocks to build a fortress and defend your core "
        "against waves of incoming enemies. Survive all 10 waves to win."
    )
    auto_advance = True
    
    # --- CONSTANTS ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 22, 22
    TILE_DIMS = (32, 16)
    
    COLOR_BG = (30, 35, 40)
    COLOR_GRID = (50, 55, 60)
    COLOR_PLAYER_BLOCK = (40, 200, 120)
    COLOR_FORTRESS = (60, 150, 255)
    COLOR_ENEMY_A = (255, 80, 80)
    COLOR_ENEMY_B = (255, 160, 80) # Faster enemy
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_CURSOR = (255, 255, 100, 100) # RGBA for transparency
    
    FORTRESS_POS = (GRID_WIDTH // 2, GRID_HEIGHT // 2)
    FORTRESS_MAX_HEALTH = 100
    BLOCK_MAX_HEALTH = 50
    
    MAX_WAVES = 10
    WAVE_DURATION_STEPS = 400
    MAX_EPISODE_STEPS = (MAX_WAVES + 1) * WAVE_DURATION_STEPS

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)

        self.iso = IsometricConverter(
            self.SCREEN_WIDTH, self.SCREEN_HEIGHT,
            (self.GRID_WIDTH, self.GRID_HEIGHT),
            self.TILE_DIMS
        )
        
        self.blocks = []
        self.enemies = []
        self.particles = []
        self.cursor_pos = [0,0]
        self.prev_space_held = False
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.fortress_health = self.FORTRESS_MAX_HEALTH
        
        self.blocks = []
        self.enemies = []
        self.particles = []
        
        self.current_wave = 0
        self.wave_timer = self.WAVE_DURATION_STEPS - 50 # Start first wave quickly
        
        self.cursor_pos = [self.GRID_WIDTH // 2, 5]
        self.prev_space_held = True # Prevent placing block on first frame
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        step_reward = 0.0
        
        self._handle_input(movement, space_held)
        
        step_reward += self._update_game_state()
        
        self.steps += 1
        terminated = self._check_termination()

        # Calculate final reward on terminal state
        if terminated:
            if self.fortress_health <= 0:
                step_reward = -100.0
            elif self.current_wave > self.MAX_WAVES:
                 step_reward = 100.0 # Capped win reward per spec
        else:
             step_reward += 0.1 # Survival reward per step

        self.score += step_reward
        
        return (
            self._get_observation(),
            step_reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Move cursor
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # Place block on key press (not hold)
        if space_held and not self.prev_space_held:
            can_place = True
            if tuple(self.cursor_pos) == self.FORTRESS_POS:
                can_place = False
            if can_place:
                for block in self.blocks:
                    if block.iso_pos == tuple(self.cursor_pos):
                        can_place = False
                        break
            if can_place:
                self.blocks.append(Block(tuple(self.cursor_pos), max_health=self.BLOCK_MAX_HEALTH))
                # sfx: block_place.wav
                self._create_particles(*self.iso.to_screen(*self.cursor_pos), self.COLOR_PLAYER_BLOCK, 10)

        self.prev_space_held = space_held

    def _update_game_state(self):
        reward = 0
        
        # Update wave logic
        self.wave_timer += 1
        if self.wave_timer >= self.WAVE_DURATION_STEPS and self.current_wave < self.MAX_WAVES:
            self.wave_timer = 0
            self.current_wave += 1
            self._spawn_wave()
            if self.current_wave > 1:
                reward += 100 # Reward for surviving previous wave
        
        # Update enemies
        destroyed_blocks_this_step = []
        for enemy in self.enemies[:]:
            attacked_block = enemy.update(self.blocks)
            if attacked_block:
                # sfx: hit_block.wav
                attacked_block.health -= 1
                self._create_particles(*self.iso.to_screen(*attacked_block.iso_pos), (200,200,200), 2)
                if attacked_block.health <= 0 and attacked_block not in destroyed_blocks_this_step:
                    self.blocks.remove(attacked_block)
                    destroyed_blocks_this_step.append(attacked_block)
                    reward -= 5 # Penalty for losing a block
            
            # Check collision with fortress
            if math.hypot(self.FORTRESS_POS[0] - enemy.pixel_pos[0], self.FORTRESS_POS[1] - enemy.pixel_pos[1]) < 1.0:
                # sfx: fortress_hit.wav
                self.fortress_health -= 10
                self._create_particles(*self.iso.to_screen(*self.FORTRESS_POS), self.COLOR_FORTRESS, 30)
                self.enemies.remove(enemy)
                reward += 1 # Reward for enemy "death"

        # Update particles
        self.particles = [p for p in self.particles if p.update()]
        
        return reward

    def _spawn_wave(self):
        # sfx: wave_start.wav
        num_enemies = self.current_wave
        for _ in range(num_enemies):
            # Spawn at random edge
            side = self.np_random.choice(['top', 'left'])
            if side == 'top':
                x, y = self.np_random.integers(0, self.GRID_WIDTH), -2
            else: # left
                x, y = -2, self.np_random.integers(0, self.GRID_HEIGHT)
            
            # Introduce faster enemy at wave 5
            if self.current_wave >= 5 and self.np_random.random() < 0.4:
                speed = 0.04 + (self.current_wave - 5) * 0.005 # Scale speed
                enemy = Enemy((x, y), speed, 20, self.COLOR_ENEMY_B, self.FORTRESS_POS)
            else:
                speed = 0.02
                enemy = Enemy((x, y), speed, 30, self.COLOR_ENEMY_A, self.FORTRESS_POS)
            self.enemies.append(enemy)

    def _check_termination(self):
        if self.fortress_health <= 0:
            return True
        if self.current_wave > self.MAX_WAVES: # Win condition
            return True
        if self.steps >= self.MAX_EPISODE_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for i in range(self.GRID_WIDTH + 1):
            start = self.iso.to_screen(i, -1)
            end = self.iso.to_screen(i, self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for i in range(self.GRID_HEIGHT + 1):
            start = self.iso.to_screen(-1, i)
            end = self.iso.to_screen(self.GRID_WIDTH, i)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

        # Entities need to be sorted by Y-index for correct isometric overlap
        render_queue = []
        render_queue.append(('fortress', self.FORTRESS_POS, self.FORTRESS_POS[0] + self.FORTRESS_POS[1]))
        for block in self.blocks:
            render_queue.append(('block', block, block.iso_pos[0] + block.iso_pos[1]))
        for enemy in self.enemies:
            render_queue.append(('enemy', enemy, enemy.iso_pos[0] + enemy.iso_pos[1]))
        
        render_queue.sort(key=lambda item: item[2])

        for item_type, item_data, _ in render_queue:
            if item_type == 'fortress':
                self.iso.draw_cube(self.screen, item_data, self.COLOR_FORTRESS)
            elif item_type == 'block':
                self.iso.draw_cube(self.screen, item_data.iso_pos, self.COLOR_PLAYER_BLOCK)
            elif item_type == 'enemy':
                self.iso.draw_cube(self.screen, item_data.iso_pos, item_data.color)

        # Draw cursor
        cursor_screen_pos = self.iso.to_screen(*self.cursor_pos)
        cursor_rect = pygame.Rect(
            cursor_screen_pos[0] - self.iso.tile_width_half, 
            cursor_screen_pos[1] - self.iso.tile_height, 
            self.iso.tile_width, 
            self.iso.tile_height * 2
        )
        cursor_surface = pygame.Surface(cursor_rect.size, pygame.SRCALPHA)
        self.iso.draw_cube(cursor_surface, (1, 2), self.COLOR_CURSOR) # Draw at local origin for correct positioning
        self.screen.blit(cursor_surface, cursor_rect.topleft)

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p.color, (int(p.x), int(p.y)), int(p.radius))

    def _render_ui(self):
        # Wave counter
        wave_text = f"WAVE: {min(self.current_wave, self.MAX_WAVES)}/{self.MAX_WAVES}"
        text_surf = self.font_large.render(wave_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (10, 10))

        # Fortress Health
        health_text = "FORTRESS HEALTH"
        text_surf = self.font_small.render(health_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (self.SCREEN_WIDTH - 160, 15))
        
        health_pct = max(0, self.fortress_health / self.FORTRESS_MAX_HEALTH)
        health_bar_rect = pygame.Rect(self.SCREEN_WIDTH - 162, 35, 154, 20)
        health_bar_fill = pygame.Rect(self.SCREEN_WIDTH - 160, 37, int(150 * health_pct), 16)
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, health_bar_rect, 1)
        pygame.draw.rect(self.screen, self.COLOR_FORTRESS, health_bar_fill)
        
        # Game Over / Win Text
        if self._check_termination() and self.steps > 1:
            if self.fortress_health <= 0:
                msg = "GAME OVER"
                color = self.COLOR_ENEMY_A
            else:
                msg = "VICTORY!"
                color = self.COLOR_PLAYER_BLOCK
            
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def _create_particles(self, x, y, color, count):
        for _ in range(count):
            self.particles.append(Particle(x, y, color))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "fortress_health": self.fortress_health,
            "enemies": len(self.enemies),
            "blocks": len(self.blocks),
        }

    def close(self):
        pygame.quit()
    
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Isometric Fortress Defense")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Pygame event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
        
        # --- Render to screen ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']:.2f}, Wave: {info['wave']}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()

        clock.tick(30) # Run at 30 FPS
        
    env.close()