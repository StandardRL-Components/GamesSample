
# Generated: 2025-08-27T18:10:35.473310
# Source Brief: brief_01754.md
# Brief Index: 1754

        
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


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move cursor. Space to place a reinforcing block."
    )

    game_description = (
        "Defend your fortress from waves of enemies by strategically placing reinforcing blocks."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 32, 20
    CELL_WIDTH = SCREEN_WIDTH // GRID_WIDTH
    CELL_HEIGHT = SCREEN_HEIGHT // GRID_HEIGHT

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 45, 60)
    COLOR_CURSOR = (0, 180, 255, 150)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_TEXT = (240, 240, 240)
    
    BLOCK_COLORS = {
        3: (50, 205, 50),   # Full health (Lawn Green)
        2: (255, 215, 0),   # Mid health (Gold)
        1: (255, 69, 0),    # Low health (OrangeRed)
    }
    COLOR_DESTROYED = (80, 70, 60)

    # Game parameters
    MAX_TURNS = 20
    INITIAL_BLOCKS = 40
    BLOCK_MAX_HEALTH = 3
    
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
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        self.fortress_blocks = {}
        self.destroyed_locations = set()
        self.enemies = []
        self.particles = []
        self.cursor_pos = (0, 0)
        self.turn = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.reset()

        # self.validate_implementation() # Optional self-check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.turn = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.fortress_blocks.clear()
        self.destroyed_locations.clear()
        self.enemies.clear()
        self.particles.clear()
        
        self._generate_fortress()
        
        self.cursor_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        
        return self._get_observation(), self._get_info()

    def _generate_fortress(self):
        start_x = self.np_random.integers(self.GRID_WIDTH // 4, self.GRID_WIDTH * 3 // 4)
        start_y = self.np_random.integers(self.GRID_HEIGHT // 2, self.GRID_HEIGHT * 3 // 4)
        
        current_pos = [start_x, start_y]
        
        for _ in range(self.INITIAL_BLOCKS):
            if tuple(current_pos) not in self.fortress_blocks:
                 self.fortress_blocks[tuple(current_pos)] = self.BLOCK_MAX_HEALTH
            
            # Random walk
            direction = self.np_random.integers(0, 4)
            if direction == 0: current_pos[0] += 1 # Right
            elif direction == 1: current_pos[0] -= 1 # Left
            elif direction == 2: current_pos[1] += 1 # Down
            elif direction == 3: current_pos[1] -= 1 # Up

            current_pos[0] = np.clip(current_pos[0], 2, self.GRID_WIDTH - 3)
            current_pos[1] = np.clip(current_pos[1], self.GRID_HEIGHT // 2, self.GRID_HEIGHT - 2)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        movement = action[0]
        space_pressed = action[1] == 1
        
        self._move_cursor(movement)
        
        reward = -0.01 # Small cost for taking a step
        
        if space_pressed:
            if self._is_valid_placement(self.cursor_pos):
                # --- Player Action: Place Block ---
                self.fortress_blocks[self.cursor_pos] = self.BLOCK_MAX_HEALTH
                self._create_particles(self.cursor_pos, self.BLOCK_COLORS[3], 20)
                # sfx: place_block.wav
                
                # --- Turn Advances ---
                self.turn += 1
                
                # --- Enemy Phase ---
                self._update_enemies()
                
                # --- Scoring & Termination Check ---
                reward = self._calculate_reward()
                terminated = self._check_termination()
                
                if terminated:
                    if self.game_won:
                        reward += 50.0 # Victory bonus
                    else:
                        reward -= 50.0 # Defeat penalty
                    self.game_over = True
            else:
                # --- Invalid Placement ---
                reward = -0.1 # Penalty for invalid action
                terminated = False
        else:
            # --- No Action Taken ---
            terminated = False

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _move_cursor(self, movement):
        x, y = self.cursor_pos
        if movement == 1: y -= 1 # Up
        elif movement == 2: y += 1 # Down
        elif movement == 3: x -= 1 # Left
        elif movement == 4: x += 1 # Right
        
        self.cursor_pos = (
            np.clip(x, 0, self.GRID_WIDTH - 1),
            np.clip(y, 0, self.GRID_HEIGHT - 1)
        )

    def _is_valid_placement(self, pos):
        return pos not in self.fortress_blocks and pos not in self.destroyed_locations

    def _update_enemies(self):
        # 1. Move existing enemies and attack
        next_enemies = []
        for ex, ey in self.enemies:
            ey += 1
            target_pos = (ex, ey)
            
            if target_pos in self.fortress_blocks:
                # sfx: hit_block.wav
                self.fortress_blocks[target_pos] -= 1
                self._create_particles(target_pos, (255, 255, 255), 10, speed=2)
                if self.fortress_blocks[target_pos] <= 0:
                    # sfx: destroy_block.wav
                    del self.fortress_blocks[target_pos]
                    self.destroyed_locations.add(target_pos)
                    self._create_particles(target_pos, self.COLOR_DESTROYED, 30, speed=1.5, life=40)
            elif ey >= self.GRID_HEIGHT:
                pass # Enemy moved off-screen
            else:
                next_enemies.append(target_pos)
        self.enemies = next_enemies

        # 2. Spawn new enemies
        num_to_spawn = 2 + (self.turn // 5)
        spawn_cols = self.np_random.choice(self.GRID_WIDTH, num_to_spawn, replace=False)
        for col in spawn_cols:
            self.enemies.append((col, 0))

    def _calculate_reward(self):
        # Reward for surviving blocks
        return len(self.fortress_blocks) * 0.1

    def _check_termination(self):
        if len(self.fortress_blocks) == 0 and self.turn > 0:
            self.game_won = False
            return True
        if self.turn >= self.MAX_TURNS:
            self.game_won = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._update_and_draw_particles()
        self._draw_grid()
        self._draw_destroyed_blocks()
        self._draw_fortress()
        self._draw_enemies()
        self._draw_cursor()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "turn": self.turn,
            "remaining_blocks": len(self.fortress_blocks),
        }

    # --- Rendering Methods ---

    def _draw_grid(self):
        for x in range(self.GRID_WIDTH):
            pygame.draw.line(self.screen, self.COLOR_GRID, 
                             (x * self.CELL_WIDTH, 0), 
                             (x * self.CELL_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.GRID_HEIGHT):
            pygame.draw.line(self.screen, self.COLOR_GRID, 
                             (0, y * self.CELL_HEIGHT), 
                             (self.SCREEN_WIDTH, y * self.CELL_HEIGHT))

    def _draw_destroyed_blocks(self):
        for x, y in self.destroyed_locations:
            rect = pygame.Rect(x * self.CELL_WIDTH, y * self.CELL_HEIGHT, self.CELL_WIDTH, self.CELL_HEIGHT)
            pygame.draw.rect(self.screen, self.COLOR_DESTROYED, rect.inflate(-4, -4))

    def _draw_fortress(self):
        for pos, health in self.fortress_blocks.items():
            x, y = pos
            color = self.BLOCK_COLORS.get(health, self.BLOCK_COLORS[1])
            rect = pygame.Rect(x * self.CELL_WIDTH, y * self.CELL_HEIGHT, self.CELL_WIDTH, self.CELL_HEIGHT)
            
            # Draw block with a slight 3D effect
            highlight = tuple(min(255, c + 20) for c in color)
            shadow = tuple(max(0, c - 20) for c in color)
            pygame.draw.rect(self.screen, shadow, rect)
            pygame.draw.rect(self.screen, color, rect.inflate(-2, -2))
            pygame.draw.rect(self.screen, highlight, rect.inflate(-6, -6))


    def _draw_enemies(self):
        for x, y in self.enemies:
            cx = int((x + 0.5) * self.CELL_WIDTH)
            cy = int((y + 0.5) * self.CELL_HEIGHT)
            radius = self.CELL_WIDTH // 3
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, radius, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, cx, cy, radius, self.COLOR_ENEMY)

    def _draw_cursor(self):
        x, y = self.cursor_pos
        rect = pygame.Rect(x * self.CELL_WIDTH, y * self.CELL_HEIGHT, self.CELL_WIDTH, self.CELL_HEIGHT)
        
        cursor_surface = pygame.Surface((self.CELL_WIDTH, self.CELL_HEIGHT), pygame.SRCALPHA)
        color = self.COLOR_CURSOR[:3] if self._is_valid_placement(self.cursor_pos) else self.COLOR_ENEMY
        alpha = self.COLOR_CURSOR[3]
        
        pygame.draw.rect(cursor_surface, color + (alpha,), cursor_surface.get_rect(), border_radius=3)
        pygame.draw.rect(cursor_surface, color + (255,), cursor_surface.get_rect(), 2, border_radius=3)
        self.screen.blit(cursor_surface, rect.topleft)

    def _render_ui(self):
        # Turn counter
        turn_text = self.font_small.render(f"TURN: {self.turn}/{self.MAX_TURNS}", True, self.COLOR_TEXT)
        self.screen.blit(turn_text, (10, 10))

        # Block counter
        block_text = self.font_small.render(f"BLOCKS: {len(self.fortress_blocks)}", True, self.COLOR_TEXT)
        self.screen.blit(block_text, (self.SCREEN_WIDTH - block_text.get_width() - 10, 10))

        # Game Over / Victory text
        if self.game_over:
            message = "VICTORY" if self.game_won else "FORTRESS LOST"
            color = self.BLOCK_COLORS[3] if self.game_won else self.COLOR_ENEMY
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            # Draw a semi-transparent background for readability
            bg_surf = pygame.Surface(text_rect.size, pygame.SRCALPHA)
            bg_surf.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 200))
            self.screen.blit(bg_surf, text_rect.topleft)
            self.screen.blit(end_text, text_rect)

    # --- Particle System for Visual Effects ---

    def _create_particles(self, grid_pos, color, count, speed=3.0, life=20):
        px, py = (grid_pos[0] + 0.5) * self.CELL_WIDTH, (grid_pos[1] + 0.5) * self.CELL_HEIGHT
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            s = self.np_random.uniform(0.5, 1.0) * speed
            vel = [math.cos(angle) * s, math.sin(angle) * s]
            self.particles.append([
                [px, py], # pos
                vel,      # velocity
                self.np_random.integers(2, 5), # radius
                life,     # lifetime
                color
            ])

    def _update_and_draw_particles(self):
        for p in self.particles:
            p[0][0] += p[1][0] # pos.x += vel.x
            p[0][1] += p[1][1] # pos.y += vel.y
            p[3] -= 1          # life -= 1
            p[1][1] += 0.05    # gravity
            
            pos = (int(p[0][0]), int(p[0][1]))
            radius = int(p[2] * (p[3] / 20.0))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, p[4])
        
        self.particles = [p for p in self.particles if p[3] > 0]

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
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Fortress Defense")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # No-op
        space_pressed = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]:
            space_pressed = 1
            
        action = [movement, space_pressed, 0] # Shift is not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Turns: {info['turn']}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()

        clock.tick(30) # Limit to 30 FPS for consistent manual play
        
    env.close()