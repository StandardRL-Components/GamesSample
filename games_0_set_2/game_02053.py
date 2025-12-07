import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the cursor. Press Space to place a block. Survive the waves."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from enemy waves by strategically placing blocks to alter their path."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.W, self.H = 640, 400
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()

        # Game constants
        self.GRID_W, self.GRID_H = 64, 40
        self.CELL_SIZE = 10
        self.MAX_WAVES = 10
        self.MAX_STEPS = 2000  # Increased to allow for longer games

        # Colors
        self.COLOR_BG = (15, 18, 28)
        self.COLOR_GRID = (30, 35, 50)
        self.COLOR_BASE = (60, 220, 140)
        self.COLOR_BASE_DMG = (255, 100, 100)
        self.COLOR_BLOCK = (60, 140, 220)
        self.COLOR_ENEMY = (255, 70, 70)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 220)

        # Fonts
        self.font_ui = pygame.font.Font(None, 24)
        self.font_game_over = pygame.font.Font(None, 64)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_pos = (0, 0)
        self.base_health = 0
        self.max_base_health = 100
        self.cursor_pos = [0, 0]
        self.blocks = set()
        self.enemies = []
        self.particles = []
        self.wave = 0
        self.prev_space_held = False
        self.distance_map = None
        self.pathfinding_dirty = True
        self.win_state = False

        # Initialize state variables
        # self.reset() is called by the gym wrapper, no need to call it here.
        # Calling validate_implementation() in __init__ is also problematic as it
        # runs a step before the environment is fully ready.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False

        self.base_pos = (self.GRID_W // 2, self.GRID_H // 2)
        self.base_health = self.max_base_health
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2 + 5]

        self.blocks = set()
        self.enemies = []
        self.particles = []
        self.wave = 0
        self.prev_space_held = False

        self.pathfinding_dirty = True
        self._start_next_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1

        # Handle player input
        self._handle_input(movement, space_held)

        # Update game logic
        self._update_game_state()

        self.steps += 1
        reward = self._calculate_reward()
        self.score += reward
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated or truncated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _handle_input(self, movement, space_held):
        # Move cursor
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1  # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1  # Right

        if dx != 0 or dy != 0:
            self.cursor_pos[0] = (self.cursor_pos[0] + dx) % self.GRID_W
            self.cursor_pos[1] = (self.cursor_pos[1] + dy) % self.GRID_H

        # Place block on key press (not hold)
        space_pressed = space_held and not self.prev_space_held
        if space_pressed:
            pos_tuple = tuple(self.cursor_pos)
            if pos_tuple != self.base_pos and pos_tuple not in self.blocks:
                self.blocks.add(pos_tuple)
                self.pathfinding_dirty = True
                # Sound: block_place.wav

        self.prev_space_held = space_held

    def _update_game_state(self):
        # Recalculate pathfinding if needed
        if self.pathfinding_dirty:
            self._calculate_pathfinding_map()
            self.pathfinding_dirty = False

        # Update enemies
        new_enemies = []
        for enemy in self.enemies:
            enemy_survived = self._update_enemy(enemy)
            if enemy_survived:
                new_enemies.append(enemy)
        self.enemies = new_enemies

        # Update particles
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1

        # Check for next wave
        if not self.enemies and not self.game_over:
            if self.wave >= self.MAX_WAVES:
                self.win_state = True
                self.game_over = True
            else:
                self._start_next_wave()

    def _update_enemy(self, enemy):
        grid_x, grid_y = int(enemy["pos"][0] / self.CELL_SIZE), int(
            enemy["pos"][1] / self.CELL_SIZE
        )

        # Clamp grid coordinates to be within the valid grid bounds to prevent IndexError.
        # This is necessary because enemies can spawn or move outside the screen area.
        grid_x = np.clip(grid_x, 0, self.GRID_W - 1)
        grid_y = np.clip(grid_y, 0, self.GRID_H - 1)

        # Check for collision with base
        if (grid_x, grid_y) == self.base_pos:
            self.base_health -= 10
            self.base_health = max(0, self.base_health)
            self._create_particles(enemy["pos"], self.COLOR_BASE_DMG)
            # Sound: base_hit.wav
            return False  # Enemy is destroyed

        # Move enemy along path
        if self.distance_map is not None:
            min_dist = self.distance_map[grid_y, grid_x]
            best_move = (0, 0)

            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = grid_x + dx, grid_y + dy
                if 0 <= nx < self.GRID_W and 0 <= ny < self.GRID_H:
                    if self.distance_map[ny, nx] < min_dist:
                        min_dist = self.distance_map[ny, nx]
                        best_move = (dx, dy)

            if best_move != (0, 0):
                target_x = (grid_x + best_move[0] + 0.5) * self.CELL_SIZE
                target_y = (grid_y + best_move[1] + 0.5) * self.CELL_SIZE

                vec_x = target_x - enemy["pos"][0]
                vec_y = target_y - enemy["pos"][1]
                dist = math.hypot(vec_x, vec_y)

                if dist > 0:
                    move_x = (vec_x / dist) * enemy["speed"]
                    move_y = (vec_y / dist) * enemy["speed"]
                    enemy["pos"][0] += move_x
                    enemy["pos"][1] += move_y
        return True

    def _start_next_wave(self):
        self.wave += 1
        num_enemies = 5 + self.wave * 2
        enemy_speed = 1.0 + self.wave * 0.05

        for _ in range(num_enemies):
            side = self.np_random.integers(4)
            if side == 0:  # Top
                x, y = self.np_random.uniform(0, self.W), -self.CELL_SIZE
            elif side == 1:  # Bottom
                x, y = self.np_random.uniform(0, self.W), self.H + self.CELL_SIZE
            elif side == 2:  # Left
                x, y = -self.CELL_SIZE, self.np_random.uniform(0, self.H)
            else:  # Right
                x, y = self.W + self.CELL_SIZE, self.np_random.uniform(0, self.H)

            self.enemies.append({"pos": [x, y], "speed": enemy_speed})
        # Sound: new_wave.wav
        self.pathfinding_dirty = True

    def _calculate_pathfinding_map(self):
        q = deque([self.base_pos])
        self.distance_map = np.full(
            (self.GRID_H, self.GRID_W), fill_value=np.inf, dtype=float
        )

        bx, by = self.base_pos
        self.distance_map[by, bx] = 0

        while q:
            cx, cy = q.popleft()
            current_dist = self.distance_map[cy, cx]

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy

                if 0 <= nx < self.GRID_W and 0 <= ny < self.GRID_H:
                    if self.distance_map[ny, nx] == np.inf and (nx, ny) not in self.blocks:
                        self.distance_map[ny, nx] = current_dist + 1
                        q.append((nx, ny))

    def _calculate_reward(self):
        if self.base_health <= 0:
            return -100.0
        if self.win_state:
            return 100.0
        return 0.1

    def _check_termination(self):
        return self.base_health <= 0 or self.win_state

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw pathfinding map (subtle heatmap)
        if self.distance_map is not None:
            max_dist = np.max(self.distance_map[self.distance_map != np.inf])
            if max_dist > 0:
                for y in range(self.GRID_H):
                    for x in range(self.GRID_W):
                        dist = self.distance_map[y, x]
                        if dist != np.inf:
                            alpha = int(30 * (1 - dist / max_dist))
                            if alpha > 0:
                                s = pygame.Surface(
                                    (self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA
                                )
                                s.fill((255, 255, 255, alpha))
                                self.screen.blit(s, (x * self.CELL_SIZE, y * self.CELL_SIZE))

        # Draw blocks
        for x, y in self.blocks:
            rect = (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_BLOCK, rect)
            pygame.draw.rect(self.screen, self.COLOR_BG, rect, 1)

        # Draw base with health bar
        base_rect = (
            self.base_pos[0] * self.CELL_SIZE,
            self.base_pos[1] * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE,
        )
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)
        health_percent = self.base_health / self.max_base_health
        health_bar_width = int(self.CELL_SIZE * health_percent)
        health_bar_rect = (base_rect[0], base_rect[1] - 4, health_bar_width, 2)
        pygame.draw.rect(self.screen, self.COLOR_BASE, health_bar_rect)

        # Draw enemies
        for enemy in self.enemies:
            x, y = int(enemy["pos"][0]), int(enemy["pos"][1])
            # Glow effect
            pygame.gfxdraw.aacircle(self.screen, x, y, 6, (*self.COLOR_ENEMY, 100))
            pygame.gfxdraw.filled_circle(self.screen, x, y, 6, (*self.COLOR_ENEMY, 100))
            # Core
            pygame.gfxdraw.aacircle(self.screen, x, y, 4, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_circle(self.screen, x, y, 4, self.COLOR_ENEMY)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / p["max_life"]))
            color = (*p["color"], alpha)
            pygame.gfxdraw.filled_circle(
                self.screen, int(p["pos"][0]), int(p["pos"][1]), int(p["life"] / 5), color
            )

        # Draw cursor
        cursor_rect = (
            self.cursor_pos[0] * self.CELL_SIZE,
            self.cursor_pos[1] * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE,
        )
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(s, (*self.COLOR_CURSOR, 100), (0, 0, self.CELL_SIZE, self.CELL_SIZE), 2)
        self.screen.blit(s, cursor_rect)

    def _render_ui(self):
        # Wave counter
        wave_text = self.font_ui.render(
            f"Wave: {self.wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT
        )
        self.screen.blit(wave_text, (10, 10))

        # Score
        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.W - score_text.get_width() - 10, 10))

        # Base Health
        health_text = self.font_ui.render(
            f"Base Health: {self.base_health}", True, self.COLOR_TEXT
        )
        self.screen.blit(health_text, (10, 35))

        # Game Over / Win message
        if self.game_over:
            msg = "YOU WON!" if self.win_state else "GAME OVER"
            color = self.COLOR_BASE if self.win_state else self.COLOR_ENEMY
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.W / 2, self.H / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "base_health": self.base_health,
            "enemies": len(self.enemies),
        }

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            life = self.np_random.integers(15, 30)
            self.particles.append(
                {
                    "pos": list(pos),
                    "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                    "life": life,
                    "max_life": life,
                    "color": color,
                }
            )

    def render(self):
        return self._get_observation()


if __name__ == "__main__":
    # To run and play the game manually
    # We need a real display for this part
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    screen = pygame.display.set_mode((env.W, env.H))
    pygame.display.set_caption("Block Defense")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0

    while running:
        movement, space, shift = 0, 0, 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            pygame.time.wait(3000)  # Pause for 3 seconds
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30)  # Run at 30 FPS

    pygame.quit()