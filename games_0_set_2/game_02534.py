import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Collect all 50 gems to win, but avoid the deadly traps!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated maze, collecting gems while avoiding deadly traps to achieve a high score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TILE_SIZE = 20
    GRID_WIDTH = SCREEN_WIDTH // TILE_SIZE
    GRID_HEIGHT = SCREEN_HEIGHT // TILE_SIZE

    MAX_STEPS = 1000
    GEM_TARGET = 50
    NUM_TRAPS = 40

    # --- Colors ---
    COLOR_BG = (15, 15, 25)
    COLOR_WALL = (40, 40, 60)
    COLOR_PLAYER = (255, 255, 0)
    COLOR_PLAYER_GLOW = (255, 255, 0, 50)
    COLOR_GEM = (0, 255, 255)
    COLOR_GEM_GLOW = (0, 255, 255, 60)
    COLOR_TRAP_PIT = (200, 100, 0)
    COLOR_TRAP_PIT_INNER = (50, 25, 0)
    COLOR_TRAP_SPIKE = (220, 20, 60)
    COLOR_TRAP_ELEC = (148, 0, 211)
    COLOR_TRAP_ELEC_GLOW = (148, 0, 211, 100)
    COLOR_UI_TEXT = (255, 255, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 72)

        # Etc...
        self.maze = None
        self.player_pos = None
        self.gems = None
        self.traps = None
        self.particles = None
        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.game_over = False
        self.victory = False
        self.np_random = None

        # The reset method is called here to initialize the state
        # but the test harness will call it again.
        # self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.game_over = False
        self.victory = False
        self.particles = []

        self._generate_maze()

        # np.argwhere returns (row, col) which is (y, x). We want to store as (x, y).
        empty_cells = np.argwhere(self.maze == 0).tolist()
        self.np_random.shuffle(empty_cells)

        # Place player
        y, x = empty_cells.pop()
        self.player_pos = (x, y)

        available_cells = min(len(empty_cells), self.GEM_TARGET + self.NUM_TRAPS)

        # Place gems
        self.gems = set()
        num_gems_to_place = min(self.GEM_TARGET, max(0, available_cells - self.NUM_TRAPS))
        for _ in range(num_gems_to_place):
            y, x = empty_cells.pop()
            self.gems.add((x, y))

        # Place traps
        self.traps = {}
        trap_types = ['pit', 'spike', 'elec']
        num_traps_to_place = min(self.NUM_TRAPS, len(empty_cells))
        for _ in range(num_traps_to_place):
            y, x = empty_cells.pop()
            pos = (x, y)
            trap_type = self.np_random.choice(trap_types)
            self.traps[pos] = trap_type

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over or self.victory:
            return self._get_observation(), 0, self._check_termination(), False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held = action[1] == 1  # Not used
        # shift_held = action[2] == 1  # Not used

        reward = 0
        old_pos = self.player_pos
        new_pos = list(self.player_pos)

        if movement == 1: new_pos[1] -= 1  # Up
        elif movement == 2: new_pos[1] += 1  # Down
        elif movement == 3: new_pos[0] -= 1  # Left
        elif movement == 4: new_pos[0] += 1  # Right

        new_pos = tuple(new_pos)

        dist_before = self._get_dist_to_nearest_gem(old_pos)

        # Wall collision check (including boundaries)
        if not (0 <= new_pos[0] < self.GRID_WIDTH and 0 <= new_pos[1] < self.GRID_HEIGHT) \
           or self.maze[new_pos[1], new_pos[0]] == 1:
            new_pos = old_pos  # Stay in place
            reward -= 0.2

        self.player_pos = new_pos
        dist_after = self._get_dist_to_nearest_gem(self.player_pos)

        # Proximity reward
        if dist_after < dist_before: reward += 1.0
        elif dist_after > dist_before: reward -= 0.1

        # Event handling
        gem_collected_this_step = False
        if self.player_pos in self.gems:
            gem_collected_this_step = True
            self.gems.remove(self.player_pos)
            self.score += 10
            reward += 10
            self.gems_collected += 1
            self._spawn_particles(self.player_pos, self.COLOR_GEM, 20)

            if self.gems_collected >= self.GEM_TARGET:
                self.victory = True
                self.score += 100
                reward += 100

        if self.player_pos in self.traps:
            self.game_over = True
            self.score -= 100
            reward -= 100
            self._spawn_particles(self.player_pos, self.COLOR_TRAP_SPIKE, 50, 3.0)

        # Risk/reward for being near a trap
        is_near_trap = False
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                neighbor = (self.player_pos[0] + dx, self.player_pos[1] + dy)
                if neighbor in self.traps:
                    is_near_trap = True
                    break
            if is_near_trap: break

        if is_near_trap:
            if gem_collected_this_step:
                reward += 5  # Bonus for risky gem collection
            else:
                reward -= 2  # Penalty for being near a trap without reward

        self.steps += 1
        terminated = self._check_termination()

        if self.steps >= self.MAX_STEPS and not (self.game_over or self.victory):
            reward -= 20

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_game()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gems_collected": self.gems_collected,
            "player_pos": self.player_pos,
            "victory": self.victory
        }

    def _check_termination(self):
        return self.game_over or self.victory or self.steps >= self.MAX_STEPS

    def _render_game(self):
        self._update_and_draw_particles()

        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.maze[y, x] == 1:
                    pygame.draw.rect(
                        self.screen, self.COLOR_WALL,
                        (x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                    )

        for pos, trap_type in self.traps.items():
            if trap_type == 'pit':
                pygame.draw.rect(self.screen, self.COLOR_TRAP_PIT, (pos[0] * self.TILE_SIZE, pos[1] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE))
                pygame.draw.rect(self.screen, self.COLOR_TRAP_PIT_INNER, (pos[0] * self.TILE_SIZE + 3, pos[1] * self.TILE_SIZE + 3, self.TILE_SIZE - 6, self.TILE_SIZE - 6))
            elif trap_type == 'spike':
                points = [
                    (pos[0] * self.TILE_SIZE + self.TILE_SIZE / 2, pos[1] * self.TILE_SIZE + 2),
                    (pos[0] * self.TILE_SIZE + 2, pos[1] * self.TILE_SIZE + self.TILE_SIZE - 2),
                    (pos[0] * self.TILE_SIZE + self.TILE_SIZE - 2, pos[1] * self.TILE_SIZE + self.TILE_SIZE - 2)
                ]
                pygame.gfxdraw.aapolygon(self.screen, [(int(p[0]), int(p[1])) for p in points], self.COLOR_TRAP_SPIKE)
                pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in points], self.COLOR_TRAP_SPIKE)
            elif trap_type == 'elec':
                self._draw_electric_trap(pos)

        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        gem_radius = int(self.TILE_SIZE * 0.3 + pulse * 2)
        glow_radius = int(gem_radius * 1.8)
        for pos in self.gems:
            cx, cy = int((pos[0] + 0.5) * self.TILE_SIZE), int((pos[1] + 0.5) * self.TILE_SIZE)
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, glow_radius, self.COLOR_GEM_GLOW)
            pygame.gfxdraw.aacircle(self.screen, cx, cy, gem_radius, self.COLOR_GEM)
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, gem_radius, self.COLOR_GEM)

        if not self.game_over:
            px, py = self.player_pos
            player_rect = pygame.Rect(px * self.TILE_SIZE, py * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            glow_radius = int(self.TILE_SIZE * 0.8)
            pygame.gfxdraw.filled_circle(self.screen, player_rect.centerx, player_rect.centery, glow_radius, self.COLOR_PLAYER_GLOW)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)

    def _draw_electric_trap(self, pos):
        trap_surf = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
        center = self.TILE_SIZE // 2

        glow_radius = int(self.TILE_SIZE * 0.7 + math.sin(self.steps * 0.5) * 3)
        pygame.gfxdraw.filled_circle(trap_surf, center, center, glow_radius, self.COLOR_TRAP_ELEC_GLOW)

        num_points = 5
        points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points + self.steps * 0.1
            flicker = self.np_random.uniform(0.5, 1.0) if (self.steps % 3 == 0) else 1.0
            radius = self.TILE_SIZE * 0.4 * flicker
            x = center + radius * math.cos(angle)
            y = center + radius * math.sin(angle)
            points.append((int(x), int(y)))

        if len(points) > 1:
            pygame.draw.lines(trap_surf, self.COLOR_UI_TEXT, True, points, 2)

        self.screen.blit(trap_surf, (pos[0] * self.TILE_SIZE, pos[1] * self.TILE_SIZE))

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        gems_text = self.font_ui.render(f"GEMS: {self.gems_collected} / {self.GEM_TARGET}", True, self.COLOR_UI_TEXT)
        self.screen.blit(gems_text, (10, 40))

        if self.game_over or self.victory:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            msg = "VICTORY!" if self.victory else "GAME OVER"
            color = self.COLOR_GEM if self.victory else self.COLOR_TRAP_SPIKE

            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _generate_maze(self):
        self.maze = np.ones((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.int8)
        stack = deque()
        start_x = self.np_random.integers(1, self.GRID_WIDTH // 2) * 2 - 1
        start_y = self.np_random.integers(1, self.GRID_HEIGHT // 2) * 2 - 1

        self.maze[start_y, start_x] = 0
        stack.append((start_x, start_y))

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < self.GRID_WIDTH - 1 and 0 < ny < self.GRID_HEIGHT - 1 and self.maze[ny, nx] == 1:
                    neighbors.append((nx, ny))

            if neighbors:
                nx, ny = neighbors[self.np_random.integers(len(neighbors))]
                self.maze[ny, nx] = 0
                self.maze[(cy + ny) // 2, (cx + nx) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()

    def _get_dist_to_nearest_gem(self, pos):
        if not self.gems: return 0
        return min(abs(pos[0] - g[0]) + abs(pos[1] - g[1]) for g in self.gems)

    def _spawn_particles(self, pos_grid, color, count, speed_mult=1.0):
        center_x = (pos_grid[0] + 0.5) * self.TILE_SIZE
        center_y = (pos_grid[1] + 0.5) * self.TILE_SIZE
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({'pos': [center_x, center_y], 'vel': vel, 'life': lifetime, 'color': color})

    def _update_and_draw_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 1]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            alpha = int(255 * (p['life'] / 30))
            color = (*p['color'], max(0, min(255, alpha)))
            try:
                particle_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
                pygame.draw.rect(particle_surf, color, (0, 0, 4, 4))
                self.screen.blit(particle_surf, (int(p['pos'][0]-2), int(p['pos'][1]-2)))
            except (ValueError, TypeError):
                pass

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # You might need to comment out the SDL_VIDEODRIVER line at the top
    # to see the pygame window.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Maze Gem Collector")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0
    last_action_time = 0
    ACTION_DELAY = 100 # milliseconds

    while running:
        current_time = pygame.time.get_ticks()
        action = [0, 0, 0] # Default action: no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # Continuous key presses with a delay
        if current_time - last_action_time > ACTION_DELAY:
            keys = pygame.key.get_pressed()
            moved = False
            if keys[pygame.K_UP]: action[0] = 1; moved = True
            elif keys[pygame.K_DOWN]: action[0] = 2; moved = True
            elif keys[pygame.K_LEFT]: action[0] = 3; moved = True
            elif keys[pygame.K_RIGHT]: action[0] = 4; moved = True

            if moved:
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                last_action_time = current_time

                if terminated or truncated:
                    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
                    frame = np.transpose(obs, (1, 0, 2))
                    surf = pygame.surfarray.make_surface(frame)
                    screen.blit(surf, (0, 0))
                    pygame.display.flip()
                    pygame.time.wait(2000)
                    obs, info = env.reset()
                    total_reward = 0

        # Always render the latest observation
        latest_obs = env._get_observation()
        # Pygame surfaces expect (W, H), but obs is (H, W)
        frame = np.transpose(latest_obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))

        pygame.display.flip()
        clock.tick(60)

    env.close()