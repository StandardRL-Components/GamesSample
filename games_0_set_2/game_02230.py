
# Generated: 2025-08-27T19:42:32.658880
# Source Brief: brief_02230.md
# Brief Index: 2230

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the cursor. Press Space to place a block."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from enemy waves by strategically placing blocks to create a maze."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_WIDTH = 20
        self.GRID_HEIGHT = 12
        self.CELL_SIZE = 30
        
        # Center the grid
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.GRID_OFFSET_Y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) + 10

        # Colors
        self.COLOR_BG = (32, 32, 32)
        self.COLOR_GRID = (64, 64, 64)
        self.COLOR_BLOCK = (0, 255, 128)
        self.COLOR_BASE = (0, 170, 255)
        self.COLOR_ENEMY = (255, 64, 64)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_WIN = (128, 255, 128)
        self.COLOR_LOSE = (255, 128, 128)

        # Game parameters
        self.MAX_WAVES = 10
        self.MAX_STEPS = 1000
        self.INITIAL_BLOCKS = 50
        self.BLOCK_HEALTH = 5
        self.BASE_HEALTH = 20
        self.ENEMY_ATTACK_COOLDOWN = 10 # steps

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 60, bold=True)

        # --- Game State (initialized in reset) ---
        self.grid = None
        self.block_health = None
        self.base_pos = None
        self.cursor_pos = None
        self.enemies = None
        self.particles = None
        self.wave_number = 0
        self.blocks_remaining = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self._place_block_flag = False
        
        # Initialize state variables
        self.reset()
        
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Grid: 0=empty, 1=block, 2=base
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.block_health = {}

        self.base_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT - 2)
        self.grid[self.base_pos] = 2
        self.block_health[self.base_pos] = self.BASE_HEALTH

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.enemies = []
        self.particles = []

        self.wave_number = 0
        self.blocks_remaining = self.INITIAL_BLOCKS
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self._start_new_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.001  # Small penalty per step to encourage efficiency

        if not self.game_over:
            # 1. Unpack and handle player action
            movement, space_held = action[0], action[1] == 1
            self._handle_player_action(movement, space_held)
            if self._place_block_flag:
                reward += 0.01
                self.score += 0.01
                self._place_block_flag = False

            # 2. Update enemies (AI, movement, attacks)
            enemy_reward, base_destroyed = self._update_enemies()
            reward += enemy_reward
            self.score += enemy_reward
            
            if base_destroyed:
                self.game_over = True
                reward = -100.0
                self.score = -100.0

            # 3. Check for wave completion
            if not self.enemies and not self.game_over:
                wave_clear_reward = 10.0
                reward += wave_clear_reward
                self.score += wave_clear_reward
                
                if self.wave_number >= self.MAX_WAVES:
                    self.game_over = True
                    win_reward = 100.0
                    reward += win_reward
                    self.score += win_reward
                else:
                    self._start_new_wave()
        
        # 4. Update visual effects
        self._update_particles()
        
        # 5. Check termination conditions
        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over: # Terminated by steps
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _start_new_wave(self):
        self.wave_number += 1
        num_enemies = self.wave_number
        
        # Spawn enemies at random non-overlapping points on the top row
        spawn_points = self.np_random.choice(self.GRID_WIDTH, size=num_enemies, replace=False)
        for x in spawn_points:
            self.enemies.append({
                "pos": [float(x), 0.0],
                "path": [],
                "attack_cooldown": 0
            })

    def _handle_player_action(self, movement, space_held):
        # Move cursor
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # Place block
        self._place_block_flag = False
        if space_held:
            if self._place_block():
                self._place_block_flag = True

    def _place_block(self):
        x, y = self.cursor_pos
        if self.blocks_remaining > 0 and self.grid[x, y] == 0:
            self.grid[x, y] = 1
            self.block_health[(x, y)] = self.BLOCK_HEALTH
            self.blocks_remaining -= 1
            self._create_particle(x, y, self.COLOR_BLOCK, 'place')
            # sound: block_place.wav
            return True
        return False

    def _update_enemies(self):
        reward = 0
        base_destroyed = False
        
        for i in range(len(self.enemies) - 1, -1, -1):
            enemy = self.enemies[i]
            
            if enemy["attack_cooldown"] > 0:
                enemy["attack_cooldown"] -= 1
                continue

            # Pathfinding
            if not enemy["path"]:
                enemy["path"] = self._bfs(tuple(map(int, enemy["pos"])), self.base_pos)
            
            if not enemy["path"]:
                self._create_particle(enemy['pos'][0], enemy['pos'][1], self.COLOR_TEXT, 'fade')
                self.enemies.pop(i)
                reward += 5.0 # Reward for trapping an enemy
                # sound: enemy_trapped.wav
                continue
            
            # Check for adjacent targets to attack
            int_pos = tuple(map(int, enemy['pos']))
            target_pos = None
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                check_pos = (int_pos[0] + dx, int_pos[1] + dy)
                if 0 <= check_pos[0] < self.GRID_WIDTH and 0 <= check_pos[1] < self.GRID_HEIGHT:
                    if self.grid[check_pos] > 0:
                        target_pos = check_pos
                        break
            
            if target_pos:
                # Attack
                enemy['attack_cooldown'] = self.ENEMY_ATTACK_COOLDOWN
                self.block_health[target_pos] -= 1
                reward -= 0.1
                self._create_particle(target_pos[0], target_pos[1], self.COLOR_ENEMY, 'hit')
                # sound: block_hit.wav
                
                if self.block_health[target_pos] <= 0:
                    if target_pos == self.base_pos:
                        base_destroyed = True
                        # sound: base_destroyed.wav
                    else:
                        self.grid[target_pos] = 0
                        # sound: block_break.wav
                    del self.block_health[target_pos]
            else:
                # Move
                if enemy["path"]:
                    next_pos = enemy["path"].pop(0)
                    enemy["pos"] = list(next_pos)
        
        return reward, base_destroyed

    def _bfs(self, start, end):
        q = deque([[start]])
        visited = {start}
        
        while q:
            path = q.popleft()
            x, y = path[-1]
            
            if (x, y) == end:
                return path[1:]
            
            # Use a shuffled order for neighbors to break ties randomly
            neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            self.np_random.shuffle(neighbors)
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and \
                   (nx, ny) not in visited and (self.grid[nx, ny] == 0 or (nx, ny) == end):
                    visited.add((nx, ny))
                    new_path = list(path)
                    new_path.append((nx, ny))
                    q.append(new_path)
        return None

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = pygame.Rect(
                    self.GRID_OFFSET_X + x * self.CELL_SIZE,
                    self.GRID_OFFSET_Y + y * self.CELL_SIZE,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw blocks and base
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[x, y] > 0:
                    color = self.COLOR_BASE if self.grid[x, y] == 2 else self.COLOR_BLOCK
                    rect = pygame.Rect(
                        self.GRID_OFFSET_X + x * self.CELL_SIZE + 2,
                        self.GRID_OFFSET_Y + y * self.CELL_SIZE + 2,
                        self.CELL_SIZE - 4, self.CELL_SIZE - 4
                    )
                    pygame.draw.rect(self.screen, color, rect, border_radius=4)
                    
                    if (x,y) in self.block_health:
                        max_hp = self.BASE_HEALTH if self.grid[x,y] == 2 else self.BLOCK_HEALTH
                        hp = self.block_health[(x,y)]
                        if hp < max_hp and max_hp > 0:
                            hp_ratio = hp / max_hp
                            hp_color = (255 * (1-hp_ratio), 255 * hp_ratio, 0)
                            bar_w = (self.CELL_SIZE - 8) * hp_ratio
                            bar_rect = pygame.Rect(rect.left + 2, rect.bottom - 6, bar_w, 3)
                            pygame.draw.rect(self.screen, hp_color, bar_rect, border_radius=2)
        
        # Draw particles
        self._render_particles()

        # Draw enemies
        for enemy in self.enemies:
            rect = pygame.Rect(
                self.GRID_OFFSET_X + enemy['pos'][0] * self.CELL_SIZE + 5,
                self.GRID_OFFSET_Y + enemy['pos'][1] * self.CELL_SIZE + 5,
                self.CELL_SIZE - 10, self.CELL_SIZE - 10
            )
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, rect, border_radius=12)
            pygame.draw.rect(self.screen, (255, 150, 150), rect, 2, border_radius=12)

        # Draw cursor
        cursor_surf = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(cursor_surf, self.COLOR_CURSOR + (100,), (0, 0, self.CELL_SIZE, self.CELL_SIZE), border_radius=4)
        pygame.draw.rect(cursor_surf, self.COLOR_CURSOR + (200,), (0, 0, self.CELL_SIZE, self.CELL_SIZE), 2, border_radius=4)
        self.screen.blit(cursor_surf, (
            self.GRID_OFFSET_X + self.cursor_pos[0] * self.CELL_SIZE,
            self.GRID_OFFSET_Y + self.cursor_pos[1] * self.CELL_SIZE
        ))

    def _render_ui(self):
        # Top bar background
        pygame.draw.rect(self.screen, (20, 20, 20), (0, 0, self.SCREEN_WIDTH, 35))
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, 35), (self.SCREEN_WIDTH, 35))

        # Wave
        wave_text = self.font_main.render(f"Wave: {min(self.wave_number, self.MAX_WAVES)}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 5))

        # Blocks
        block_text = self.font_main.render(f"Blocks: {self.blocks_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(block_text, (180, 5))

        # Base HP
        base_hp = self.block_health.get(self.base_pos, 0)
        hp_color = self.COLOR_TEXT if base_hp > self.BASE_HEALTH * 0.3 else self.COLOR_LOSE
        base_text = self.font_main.render(f"Base HP: {max(0, base_hp)}/{self.BASE_HEALTH}", True, hp_color)
        self.screen.blit(base_text, (self.SCREEN_WIDTH - base_text.get_width() - 10, 5))

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg, color = ("YOU WIN!", self.COLOR_WIN) if self.wave_number > self.MAX_WAVES else ("GAME OVER", self.COLOR_LOSE)
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _create_particle(self, grid_x, grid_y, color, p_type, duration=15):
        px = self.GRID_OFFSET_X + grid_x * self.CELL_SIZE + self.CELL_SIZE / 2
        py = self.GRID_OFFSET_Y + grid_y * self.CELL_SIZE + self.CELL_SIZE / 2
        self.particles.append({
            "pos": [px, py], "color": color, "type": p_type,
            "timer": duration, "max_timer": duration
        })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['timer'] > 0]
        for p in self.particles:
            p['timer'] -= 1

    def _render_particles(self):
        for p in self.particles:
            progress = p['timer'] / p['max_timer']
            alpha = int(progress * 255)
            if p['type'] == 'hit':
                radius = int((1 - progress) * self.CELL_SIZE * 0.7)
                pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, p['color'] + (alpha,))
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, p['color'] + (alpha,))
            elif p['type'] == 'place':
                size = int((1 - progress**2) * self.CELL_SIZE)
                rect = pygame.Rect(p['pos'][0] - size/2, p['pos'][1] - size/2, size, size)
                place_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
                pygame.draw.rect(place_surf, p['color'] + (int(progress * 150),), (0,0,rect.width,rect.height), border_radius=4)
                self.screen.blit(place_surf, rect)
            elif p['type'] == 'fade':
                size = self.CELL_SIZE - 10
                rect = pygame.Rect(p['pos'][0] - size/2, p['pos'][1] - size/2, size, size)
                fade_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
                pygame.draw.rect(fade_surf, p['color'] + (alpha,), (0,0,rect.width, rect.height), border_radius=12)
                self.screen.blit(fade_surf, rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "blocks_remaining": self.blocks_remaining,
            "base_health": self.block_health.get(self.base_pos, 0),
        }

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Interactive Play ---
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Set up Pygame window for human play
    pygame.display.set_caption("Block Fortress Defense")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action.fill(0)

    print("\n" + "="*30)
    print("Block Fortress Defense")
    print(env.user_guide)
    print("="*30 + "\n")

    while not done:
        # --- Human Input ---
        movement, space, shift = 0, 0, 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = np.array([movement, space, shift])
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Render ---
        # The observation is already a rendered frame
        # We just need to convert it back to a Pygame surface to display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(15) # Control the speed of the game for human play

    print("Game Over!")
    print(f"Final Info: {info}")
    env.close()