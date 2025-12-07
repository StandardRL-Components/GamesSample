import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move the cursor. Space to place a reinforcement block."
    )

    game_description = (
        "Defend your fortress core against waves of enemies by strategically placing reinforcing blocks to alter their path."
    )

    auto_advance = False

    # --- Constants ---
    # Game world
    GRID_WIDTH = 32
    GRID_HEIGHT = 20
    CELL_SIZE = 20
    SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE
    SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE

    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_GRID = (40, 40, 50)
    COLOR_CORE = (0, 150, 255)
    COLOR_CORE_GLOW = (100, 200, 255)
    COLOR_BLOCK_FULL = (20, 200, 120)
    COLOR_BLOCK_DAMAGED = (255, 180, 0)
    COLOR_BLOCK_LOW = (220, 50, 50)
    COLOR_ENEMY = (255, 60, 60)
    COLOR_CURSOR = (255, 255, 100)
    COLOR_TEXT = (240, 240, 240)

    # Game rules
    MAX_STEPS = 2000
    MAX_WAVES = 10
    INITIAL_BLOCK_HEALTH = 3
    CORE_ID = 99

    # Rewards
    REWARD_WIN = 100.0
    REWARD_LOSS = -100.0
    REWARD_WAVE_COMPLETE = 10.0
    REWARD_ENEMY_DEFLECTED = 0.1
    REWARD_BLOCK_LOST = -1.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        # This can be uncommented to run validation on init
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave = 0
        self.blocks_to_place = 0

        # Grid and entities
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.int32)
        self.core_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.grid[self.core_pos[1], self.core_pos[0]] = self.CORE_ID

        self.cursor_pos = [self.core_pos[0], self.core_pos[1] - 3]

        self.enemies = []
        self.particles = []

        self._spawn_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # Seed is reset every step in the test, so we need to re-initialize
            # the environment state if we are already done.
            obs, info = self.reset(seed=self.np_random.integers(1_000_000))
            return obs, 0, True, False, info

        movement, space_held, _ = action
        reward = 0

        # --- Player Action Phase ---
        self._move_cursor(movement)
        if space_held:
            reward += self._place_block()

        # --- Enemy Action Phase ---
        reward += self._update_enemies()

        # --- Wave Management Phase ---
        if not self.enemies and not self.game_over:
            self.score += 100
            reward += self.REWARD_WAVE_COMPLETE
            if self.wave >= self.MAX_WAVES:
                self.game_over = True
                reward += self.REWARD_WIN
                self.score += 1000
            else:
                self._spawn_wave()

        # --- Finalization ---
        self._update_particles()
        self.steps += 1

        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated and reward < self.REWARD_WIN / 2:  # Don't overwrite win reward with loss
            reward += self.REWARD_LOSS
            self.score = max(0, self.score - 500)

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _move_cursor(self, movement):
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right

        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

    def _place_block(self):
        cx, cy = self.cursor_pos
        if self.blocks_to_place > 0 and self.grid[cy, cx] == 0:
            self.grid[cy, cx] = self.INITIAL_BLOCK_HEALTH
            self.blocks_to_place -= 1
            self._add_particles(cx, cy, 15, self.COLOR_BLOCK_FULL, 0.8)
            return 0.01  # Small reward for placing a block
        return 0

    def _update_enemies(self):
        reward = 0
        enemies_to_remove = []
        for i, enemy in enumerate(self.enemies):
            if self.game_over: break

            # Simple greedy pathfinding
            # Clamp the enemy's integer grid position to be within bounds. This prevents
            # using out-of-bounds indices (e.g., 20 for an axis of size 20) in the
            # pathfinding logic that follows, which was the cause of the IndexError.
            ex = np.clip(int(enemy['pos'][0]), 0, self.GRID_WIDTH - 1)
            ey = np.clip(int(enemy['pos'][1]), 0, self.GRID_HEIGHT - 1)
            cx, cy = self.core_pos

            dx, dy = cx - ex, cy - ey

            # Move towards core
            if abs(dx) > abs(dy):
                next_pos = (ex + np.sign(dx), ey)
            else:
                next_pos = (ex, ey + np.sign(dy))

            # Fallback if primary direction is blocked
            if self.grid[next_pos[1], next_pos[0]] > 0 and self.grid[next_pos[1], next_pos[0]] < self.CORE_ID:
                if abs(dx) <= abs(dy):
                    next_pos = (ex + np.sign(dx), ey) if dx != 0 else next_pos
                else:
                    next_pos = (ex, ey + np.sign(dy)) if dy != 0 else next_pos

            target_val = self.grid[next_pos[1], next_pos[0]]

            if target_val == self.CORE_ID:
                self.game_over = True
                self._add_particles(cx, cy, 100, self.COLOR_CORE_GLOW, 2.0, 40)
                break
            elif target_val > 0:  # Hit a block
                self.grid[next_pos[1], next_pos[0]] -= 1
                enemies_to_remove.append(i)
                reward += self.REWARD_ENEMY_DEFLECTED
                self.score += 10
                self._add_particles(next_pos[0], next_pos[1], 10, self.COLOR_ENEMY, 1.5)
                if self.grid[next_pos[1], next_pos[0]] == 0:
                    reward += self.REWARD_BLOCK_LOST
                    self._add_particles(next_pos[0], next_pos[1], 20, self.COLOR_GRID, 1.0)
            else:  # Move to empty space
                enemy['pos'][0] += np.sign(dx) * enemy['speed']
                enemy['pos'][1] += np.sign(dy) * enemy['speed']

        # Remove deflected enemies
        for i in sorted(enemies_to_remove, reverse=True):
            del self.enemies[i]

        return reward

    def _spawn_wave(self):
        self.wave += 1
        self.blocks_to_place = 5 + self.wave
        num_enemies = 5 + (self.wave - 1) * 2
        speed = 0.08 + (self.wave - 1) * 0.01

        for _ in range(num_enemies):
            side = self.np_random.integers(4)
            if side == 0:  # Top
                pos = [self.np_random.uniform(0, self.GRID_WIDTH), -1]
            elif side == 1:  # Bottom
                pos = [self.np_random.uniform(0, self.GRID_WIDTH), self.GRID_HEIGHT]
            elif side == 2:  # Left
                pos = [-1, self.np_random.uniform(0, self.GRID_HEIGHT)]
            else:  # Right
                pos = [self.GRID_WIDTH, self.np_random.uniform(0, self.GRID_HEIGHT)]

            self.enemies.append({'pos': pos, 'speed': speed})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw grid elements
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                val = self.grid[y, x]
                rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                if val == self.CORE_ID:
                    # Core breathing effect
                    pulse = (math.sin(self.steps * 0.1) + 1) / 2
                    glow_size = int(self.CELL_SIZE * (1.5 + pulse * 0.5))
                    glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
                    pygame.draw.circle(glow_surf, (*self.COLOR_CORE_GLOW, 50), (glow_size // 2, glow_size // 2),
                                       glow_size // 2)
                    self.screen.blit(glow_surf, glow_surf.get_rect(center=rect.center))
                    pygame.draw.rect(self.screen, self.COLOR_CORE, rect.inflate(-4, -4), border_radius=3)
                elif val > 0:
                    health_ratio = val / self.INITIAL_BLOCK_HEALTH
                    color = self._lerp_color(self.COLOR_BLOCK_LOW, self.COLOR_BLOCK_FULL, health_ratio)
                    pygame.draw.rect(self.screen, color, rect.inflate(-2, -2), border_radius=2)

        # Draw enemies
        for enemy in self.enemies:
            ex, ey = enemy['pos']
            rect = pygame.Rect(ex * self.CELL_SIZE, ey * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, rect.inflate(-4, -4), border_radius=4)

        # Draw particles
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            alpha_color = (*p['color'], p['alpha'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), alpha_color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(p['radius']), alpha_color)

        # Draw cursor
        if not self.game_over:
            cursor_rect = pygame.Rect(
                self.cursor_pos[0] * self.CELL_SIZE,
                self.cursor_pos[1] * self.CELL_SIZE,
                self.CELL_SIZE, self.CELL_SIZE
            )
            cursor_surf = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            alpha = 100 + 50 * math.sin(self.steps * 0.2)
            pygame.draw.rect(cursor_surf, (*self.COLOR_CURSOR, alpha), (0, 0, self.CELL_SIZE, self.CELL_SIZE), 3,
                             border_radius=3)
            self.screen.blit(cursor_surf, cursor_rect.topleft)

    def _render_ui(self):
        wave_text = self.font_main.render(f"WAVE: {self.wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))

        blocks_text = self.font_main.render(f"BLOCKS: {self.blocks_to_place}", True, self.COLOR_TEXT)
        self.screen.blit(blocks_text, (self.SCREEN_WIDTH - blocks_text.get_width() - 10, 10))

        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 40))

        if self.game_over:
            outcome_text_str = ""
            if self.wave > self.MAX_WAVES:
                outcome_text_str = "VICTORY"
            else:
                outcome_text_str = "CORE DESTROYED"

            outcome_text = self.font_main.render(outcome_text_str, True, self.COLOR_TEXT)
            text_rect = outcome_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(outcome_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "blocks_remaining": self.blocks_to_place,
            "enemies": len(self.enemies)
        }

    def _add_particles(self, grid_x, grid_y, count, color, speed_mult, life=20):
        px = (grid_x + 0.5) * self.CELL_SIZE
        py = (grid_y + 0.5) * self.CELL_SIZE
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': [px, py],
                'vel': vel,
                'radius': self.np_random.uniform(2, 5),
                'color': color,
                'life': life,
                'max_life': life,
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95  # friction
            p['vel'][1] *= 0.95
            p['life'] -= 1
            p['radius'] *= 0.97
            p['alpha'] = max(0, int(255 * (p['life'] / p['max_life'])))
            if p['life'] <= 0 or p['radius'] < 1:
                self.particles.remove(p)

    def _lerp_color(self, color1, color2, t):
        t = np.clip(t, 0, 1)
        return (
            int(color1[0] + (color2[0] - color1[0]) * t),
            int(color1[1] + (color2[1] - color1[1]) * t),
            int(color1[2] + (color2[2] - color1[2]) * t)
        )

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # To play, we need a display
    os.environ["SDL_VIDEODRIVER"] = "x11"
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Fortress Defense")
    clock = pygame.time.Clock()

    terminated = False
    truncated = False
    total_reward = 0

    # Mapping from Pygame keys to MultiDiscrete action components
    key_to_movement = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    print(GameEnv.game_description)
    print(GameEnv.user_guide)

    while not terminated and not truncated:
        movement_action = 0
        space_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            terminated = True

        for key, move_val in key_to_movement.items():
            if keys[key]:
                movement_action = move_val
                break  # Only one movement at a time

        if keys[pygame.K_SPACE]:
            space_action = 1

        # We need to step the environment to see the result of our action
        action = [movement_action, space_action, 0]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))

        pygame.display.flip()
        clock.tick(10)  # Control the speed of manual play

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait a bit before closing
            pygame.time.wait(3000)

    env.close()