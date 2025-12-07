
# Generated: 2025-08-27T18:27:45.667338
# Source Brief: brief_01842.md
# Brief Index: 1842

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrows to move cursor, Space to place a defensive block."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of enemies by building a fortress one block at a time."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Game Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 32
    GRID_HEIGHT = 20
    CELL_SIZE = 20

    MAX_WAVES = 10
    MAX_STEPS = 2000
    ACTIONS_PER_ENEMY_MOVE = 3
    INITIAL_BASE_HEALTH = 10

    # --- Colors ---
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 70)
    COLOR_BASE = (255, 50, 50)
    COLOR_BASE_GLOW = (255, 100, 100)
    COLOR_WALL = (50, 150, 255)
    COLOR_WALL_GLOW = (100, 200, 255)
    COLOR_ENEMY = (50, 255, 100)
    COLOR_ENEMY_GLOW = (150, 255, 200)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_TEXT = (220, 220, 220)
    COLOR_UI_BG = (0, 0, 0, 128)

    # --- Grid Cell States ---
    CELL_EMPTY = 0
    CELL_WALL = 1
    CELL_BASE = 2
    
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
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # Initialize state variables (will be properly set in reset)
        self.grid = None
        self.enemies = []
        self.particles = []
        self.cursor_pos = [0, 0]
        self.base_pos = [0, 0]
        self.base_health = 0
        self.current_wave = 0
        self.actions_until_enemy_move = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.full((self.GRID_WIDTH, self.GRID_HEIGHT), self.CELL_EMPTY, dtype=np.int8)
        
        self.base_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT - 2]
        self.grid[self.base_pos[0], self.base_pos[1]] = self.CELL_BASE

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        self.enemies = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.base_health = self.INITIAL_BASE_HEALTH
        self.current_wave = 0
        self.actions_until_enemy_move = self.ACTIONS_PER_ENEMY_MOVE
        self.game_over = False
        self.win = False

        self._start_new_wave()
        
        return self._get_observation(), self._get_info()

    def _start_new_wave(self):
        self.current_wave += 1
        if self.current_wave > self.MAX_WAVES:
            self.win = True
            self.game_over = True
            return 0

        self.enemies = []
        spawn_locations = list(range(1, self.GRID_WIDTH - 1))
        self.np_random.shuffle(spawn_locations)
        
        for i in range(self.current_wave):
            if not spawn_locations: break
            spawn_x = spawn_locations.pop()
            self.enemies.append({'pos': [spawn_x, 0]})
        
        return 1.0 # Reward for clearing a wave

    def step(self, action):
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, _ = action[0], action[1] == 1, action[2] == 1
        
        action_taken = False

        # 1. Player Action Phase
        if movement == 1:  # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            action_taken = True
        elif movement == 2:  # Down
            self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
            action_taken = True
        elif movement == 3:  # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            action_taken = True
        elif movement == 4:  # Right
            self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)
            action_taken = True

        if space_pressed:
            if self.grid[self.cursor_pos[0], self.cursor_pos[1]] == self.CELL_EMPTY:
                self.grid[self.cursor_pos[0], self.cursor_pos[1]] = self.CELL_WALL
                # SFX: block_place.wav
                self._create_particles(self.cursor_pos, self.COLOR_WALL, 10)
                action_taken = True

        # 2. Update Turn Counter
        if action_taken:
            self.actions_until_enemy_move -= 1

        # 3. Enemy Turn Phase
        if self.actions_until_enemy_move <= 0:
            self.actions_until_enemy_move = self.ACTIONS_PER_ENEMY_MOVE
            enemy_reward, wave_cleared_reward = self._update_enemies()
            reward += enemy_reward
            if wave_cleared_reward:
                reward += self._start_new_wave()
                self.score += 10 # Bonus score for clearing a wave

        # 4. Update Game State
        self._update_particles()
        self.steps += 1
        self.score += reward

        # 5. Check Termination Conditions
        terminated = self._check_termination()
        if self.win:
            reward += 50
            self.score += 500
        elif self.base_health <= 0:
            reward -= 50
            self.score -= 50

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_enemies(self):
        block_reward = 0
        enemies_to_remove = []

        for i, enemy in enumerate(self.enemies):
            # Simple pathfinding: move towards base
            dx = self.base_pos[0] - enemy['pos'][0]
            dy = self.base_pos[1] - enemy['pos'][1]

            primary_move, secondary_move = [0, 0], [0, 0]

            if abs(dx) > abs(dy): # Primary horizontal
                primary_move[0] = np.sign(dx)
                secondary_move[1] = np.sign(dy)
            else: # Primary vertical
                primary_move[1] = np.sign(dy)
                secondary_move[0] = np.sign(dx)

            moved = False
            for move in [primary_move, secondary_move]:
                if move[0] == 0 and move[1] == 0: continue
                
                target_pos = [enemy['pos'][0] + move[0], enemy['pos'][1] + move[1]]
                
                if not (0 <= target_pos[0] < self.GRID_WIDTH and 0 <= target_pos[1] < self.GRID_HEIGHT):
                    continue

                cell_content = self.grid[target_pos[0], target_pos[1]]

                if cell_content == self.CELL_EMPTY:
                    # SFX: enemy_move.wav
                    enemy['pos'] = target_pos
                    moved = True
                    break
                elif cell_content == self.CELL_BASE:
                    # SFX: base_hit.wav
                    self.base_health -= 1
                    enemies_to_remove.append(i)
                    self._create_particles(self.base_pos, self.COLOR_BASE, 30, life=20, speed=3)
                    moved = True
                    break
                elif cell_content == self.CELL_WALL:
                    # SFX: enemy_hit_wall.wav
                    if np.array_equal(move, primary_move):
                        block_reward += 0.1 # Reward for blocking primary path
                    continue # Try secondary move
            
        # Remove enemies that hit the base
        for i in sorted(enemies_to_remove, reverse=True):
            del self.enemies[i]
        
        wave_cleared = len(self.enemies) == 0 and not self.win
        return block_reward, wave_cleared

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
        if self.win:
            self.game_over = True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_WIDTH + 1):
            px = x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.SCREEN_HEIGHT))
        for y in range(self.GRID_HEIGHT + 1):
            py = y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.SCREEN_WIDTH, py))
        
        # Draw grid objects
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                cell_type = self.grid[x, y]
                rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                
                if cell_type == self.CELL_BASE:
                    self._draw_glowing_rect(rect, self.COLOR_BASE, self.COLOR_BASE_GLOW)
                elif cell_type == self.CELL_WALL:
                    self._draw_glowing_rect(rect, self.COLOR_WALL, self.COLOR_WALL_GLOW)
        
        # Draw enemies
        for enemy in self.enemies:
            rect = pygame.Rect(enemy['pos'][0] * self.CELL_SIZE, enemy['pos'][1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            self._draw_glowing_rect(rect, self.COLOR_ENEMY, self.COLOR_ENEMY_GLOW)
            
        # Draw particles
        for p in self.particles:
            p_pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, p_pos[0], p_pos[1], int(p['radius']), p['color'])

        # Draw cursor
        cursor_px = self.cursor_pos[0] * self.CELL_SIZE
        cursor_py = self.cursor_pos[1] * self.CELL_SIZE
        cs = self.CELL_SIZE
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cursor_px, cursor_py), (cursor_px + cs, cursor_py + cs), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cursor_px + cs, cursor_py), (cursor_px, cursor_py + cs), 2)

    def _draw_glowing_rect(self, rect, color, glow_color):
        glow_rect = rect.inflate(self.CELL_SIZE * 0.5, self.CELL_SIZE * 0.5)
        s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, (*glow_color, 50), s.get_rect(), border_radius=5)
        self.screen.blit(s, glow_rect.topleft)
        pygame.draw.rect(self.screen, color, rect, border_radius=3)
        
    def _render_ui(self):
        ui_bg_surface = pygame.Surface((self.SCREEN_WIDTH, 35), pygame.SRCALPHA)
        ui_bg_surface.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_bg_surface, (0, 0))

        self._render_text(f"WAVE: {self.current_wave}/{self.MAX_WAVES}", (10, 5), self.font_ui, self.COLOR_TEXT)
        self._render_text(f"SCORE: {int(self.score)}", (self.SCREEN_WIDTH / 2, 5), self.font_ui, self.COLOR_TEXT, center_x=True)
        health_text = f"BASE HP: {'♥' * self.base_health}{' ' * (self.INITIAL_BASE_HEALTH - self.base_health)}"
        self._render_text(health_text, (self.SCREEN_WIDTH - 10, 5), self.font_ui, self.COLOR_BASE_GLOW, right_align=True)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_ENEMY if self.win else self.COLOR_BASE
            self._render_text(msg, (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 30), self.font_game_over, color, center_x=True, center_y=True)

    def _render_text(self, text, position, font, color, center_x=False, center_y=False, right_align=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center_x: text_rect.centerx = position[0]
        elif right_align: text_rect.right = position[0]
        else: text_rect.left = position[0]
        if center_y: text_rect.centery = position[1]
        else: text_rect.top = position[1]
        self.screen.blit(text_surface, text_rect)

    def _create_particles(self, grid_pos, color, count, life=10, speed=2):
        px = (grid_pos[0] + 0.5) * self.CELL_SIZE
        py = (grid_pos[1] + 0.5) * self.CELL_SIZE
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            s = self.np_random.uniform(0.5, speed)
            vel = [math.cos(angle) * s, math.sin(angle) * s]
            self.particles.append({
                'pos': [px, py],
                'vel': vel,
                'radius': self.np_random.uniform(2, 5),
                'color': (*color, self.np_random.integers(100, 200)),
                'life': life
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] -= 0.1
        self.particles = [p for p in self.particles if p['life'] > 0 and p['radius'] > 0]
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "base_health": self.base_health,
            "win": self.win
        }

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Fortress")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        movement = 0  # No-op
        space_pressed = 0
        shift_pressed = 0 # Unused in this game

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if not terminated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space_pressed = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_pressed = 1
            
            action = [movement, space_pressed, shift_pressed]
            obs, reward, terminated, truncated, info = env.step(action)

        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(15) # Limit frame rate for manual play
        
        if terminated:
            # Wait a bit before resetting on game over
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False

    pygame.quit()