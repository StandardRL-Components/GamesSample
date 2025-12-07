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


class Particle:
    def __init__(self, x, y, color, size, lifetime, dx, dy):
        self.x = x
        self.y = y
        self.color = color
        self.size = size
        self.lifetime = lifetime
        self.dx = dx
        self.dy = dy

    def update(self):
        self.x += self.dx
        self.y += self.dy
        self.size = max(0, self.size - 0.2)
        self.lifetime -= 1

    def draw(self, surface):
        if self.lifetime > 0 and self.size > 0:
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), int(self.size))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move your block one square at a time. "
        "Collect all the gems to win, but avoid the pits!"
    )

    game_description = (
        "Navigate a procedurally generated maze to collect gems and escape before time runs out. "
        "Each stage presents a new challenge. Collect 15 gems across 3 stages to win."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    MAZE_W, MAZE_H = 20, 12
    CELL_W = SCREEN_WIDTH // MAZE_W
    CELL_H = SCREEN_HEIGHT // MAZE_H

    GEMS_PER_STAGE = 5
    TOTAL_STAGES = 3
    PITS_PER_STAGE = 4

    MAX_EPISODE_STEPS = 1000
    TIME_LIMIT_SECONDS = 180

    # --- Colors ---
    COLOR_BG = (230, 230, 235)
    COLOR_WALL = (40, 40, 50)
    COLOR_PLAYER = (255, 200, 0)
    COLOR_PLAYER_GLOW = (255, 230, 100)
    COLOR_GEM = (0, 150, 255)
    COLOR_GEM_HIGHLIGHT = (200, 240, 255)
    COLOR_PIT = (220, 50, 50)
    COLOR_PIT_INNER = (180, 40, 40)
    COLOR_TEXT = (30, 30, 40)
    COLOR_TIMER_WARN = (200, 100, 0)
    COLOR_TIMER_CRITICAL = (220, 50, 50)

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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_stage = pygame.font.SysFont("monospace", 24, bold=True)

        self.player_pos = [0, 0]
        self.maze = []
        self.gems = []
        self.pits = []
        self.particles = []
        self.current_stage = 1
        self.stage_gems_collected = 0
        self.total_gems_collected = 0
        self.time_limit_frames = self.TIME_LIMIT_SECONDS * 30  # Assuming 30fps for time calc
        self.time_remaining = self.time_limit_frames
        self.steps = 0
        self.score = 0
        self.game_over = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_stage = 1
        self.total_gems_collected = 0
        self.time_remaining = self.time_limit_frames
        self.particles = []

        self._setup_stage()

        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        self.stage_gems_collected = 0
        self.maze = self._generate_maze(self.MAZE_W, self.MAZE_H)

        empty_cells = []
        for y in range(self.MAZE_H):
            for x in range(self.MAZE_W):
                empty_cells.append((x, y))

        # Use np_random for shuffling to ensure reproducibility
        self.np_random.shuffle(empty_cells)

        self.player_pos = list(empty_cells.pop())

        self.gems = [list(empty_cells.pop()) for _ in range(self.GEMS_PER_STAGE)]
        self.pits = [list(empty_cells.pop()) for _ in range(self.PITS_PER_STAGE)]

    def _generate_maze(self, w, h):
        maze = [[{'N': True, 'S': True, 'E': True, 'W': True} for _ in range(w)] for _ in range(h)]
        visited = [[False for _ in range(w)] for _ in range(h)]

        x, y = self.np_random.integers(0, w), self.np_random.integers(0, h)
        stack = [(x, y)]
        visited[y][x] = True

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            if cy > 0 and not visited[cy - 1][cx]: neighbors.append(('N', cx, cy - 1))
            if cy < h - 1 and not visited[cy + 1][cx]: neighbors.append(('S', cx, cy + 1))
            if cx < w - 1 and not visited[cy][cx + 1]: neighbors.append(('E', cx + 1, cy))
            if cx > 0 and not visited[cy][cx - 1]: neighbors.append(('W', cx - 1, cy))

            if neighbors:
                # FIX: self.np_random.choice on a list of tuples converts elements to strings.
                # Instead, we choose a random index and select the tuple from the list.
                choice_index = self.np_random.integers(len(neighbors))
                direction, nx, ny = neighbors[choice_index]

                if direction == 'N':
                    maze[cy][cx]['N'] = False
                    maze[ny][nx]['S'] = False
                elif direction == 'S':
                    maze[cy][cx]['S'] = False
                    maze[ny][nx]['N'] = False
                elif direction == 'E':
                    maze[cy][cx]['E'] = False
                    maze[ny][nx]['W'] = False
                elif direction == 'W':
                    maze[cy][cx]['W'] = False
                    maze[ny][nx]['E'] = False

                visited[ny][nx] = True
                stack.append((nx, ny))
            else:
                stack.pop()
        return maze

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = -0.01  # Small penalty for each step to encourage efficiency
        terminated = False
        truncated = False

        self.steps += 1
        self.time_remaining -= 1  # Each step consumes time

        old_pos = list(self.player_pos)
        px, py = self.player_pos

        # Calculate distance to nearest gem before move
        dist_before = self._get_dist_to_closest_gem()

        moved = False
        if movement != 0:
            if movement == 1 and py > 0 and not self.maze[py][px]['N']:  # Up
                self.player_pos[1] -= 1
                moved = True
            elif movement == 2 and py < self.MAZE_H - 1 and not self.maze[py][px]['S']:  # Down
                self.player_pos[1] += 1
                moved = True
            elif movement == 3 and px > 0 and not self.maze[py][px]['W']:  # Left
                self.player_pos[0] -= 1
                moved = True
            elif movement == 4 and px < self.MAZE_W - 1 and not self.maze[py][px]['E']:  # Right
                self.player_pos[0] += 1
                moved = True

        if moved:
            self._spawn_move_particles(old_pos)
            # Distance-based reward
            dist_after = self._get_dist_to_closest_gem()
            if dist_after < dist_before:
                reward += 1.0
            else:
                reward -= 0.1

        # Check for pitfall
        if self.player_pos in self.pits:
            reward -= 50
            self.game_over = True
            terminated = True
            # sfx: player_fall

        # Check for gem collection
        if self.player_pos in self.gems:
            self.gems.remove(self.player_pos)
            reward += 10
            self.stage_gems_collected += 1
            self.total_gems_collected += 1
            self.score += 10
            # sfx: gem_collect

            if self.stage_gems_collected == self.GEMS_PER_STAGE:
                if self.current_stage == self.TOTAL_STAGES:
                    reward += 50  # Win bonus
                    self.score += 50
                    self.game_over = True
                    terminated = True
                    # sfx: game_win
                else:
                    self.current_stage += 1
                    self._setup_stage()
                    # sfx: stage_clear

        # Check termination conditions
        if self.time_remaining <= 0:
            self.game_over = True
            terminated = True
        
        if self.steps >= self.MAX_EPISODE_STEPS:
            self.game_over = True
            truncated = True
        
        # In Gymnasium, `terminated` and `truncated` can both be true.
        # If the episode ends due to a time limit, it's a truncation.
        # If it ends due to a terminal state (win/loss), it's a termination.
        terminated = terminated or (self.game_over and not truncated)

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_dist_to_closest_gem(self):
        if not self.gems:
            return 0
        px, py = self.player_pos
        min_dist = float('inf')
        for gx, gy in self.gems:
            dist = abs(px - gx) + abs(py - gy)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _spawn_move_particles(self, old_grid_pos):
        px, py = (old_grid_pos[0] + 0.5) * self.CELL_W, (old_grid_pos[1] + 0.5) * self.CELL_H
        for _ in range(5):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            dx, dy = math.cos(angle) * speed, math.sin(angle) * speed
            size = self.np_random.uniform(2, 5)
            lifetime = self.np_random.integers(10, 20)
            self.particles.append(Particle(px, py, self.COLOR_PLAYER_GLOW, size, lifetime, dx, dy))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        # Pygame returns (width, height, 3), we need (height, width, 3)
        return np.transpose(arr, (1, 0, 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gems_collected": self.total_gems_collected,
            "stage": self.current_stage,
            "time_remaining_seconds": self.time_remaining / 30
        }

    def _render_game(self):
        # Update and draw particles
        for p in self.particles:
            p.update()
            p.draw(self.screen)
        self.particles = [p for p in self.particles if p.lifetime > 0]

        # Draw pits
        for x, y in self.pits:
            cx, cy = (x + 0.5) * self.CELL_W, (y + 0.5) * self.CELL_H
            size = self.CELL_W * 0.35
            p1 = (cx, cy - size)
            p2 = (cx - size, cy + size * 0.7)
            p3 = (cx + size, cy + size * 0.7)
            pygame.gfxdraw.filled_trigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]),
                                         int(p3[1]), self.COLOR_PIT)
            pygame.gfxdraw.aatrigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]),
                                    int(p3[1]), self.COLOR_PIT_INNER)

        # Draw gems
        pulse = (math.sin(self.steps * 0.2) + 1) / 2  # 0 to 1
        gem_size = int(self.CELL_W * 0.25 + pulse * 3)
        for x, y in self.gems:
            cx, cy = int((x + 0.5) * self.CELL_W), int((y + 0.5) * self.CELL_H)
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, gem_size, self.COLOR_GEM)
            pygame.gfxdraw.aacircle(self.screen, cx, cy, gem_size, self.COLOR_GEM_HIGHLIGHT)
            pygame.gfxdraw.filled_circle(self.screen, cx - gem_size // 3, cy - gem_size // 3, gem_size // 3,
                                         self.COLOR_GEM_HIGHLIGHT)

        # Draw player
        px, py = self.player_pos
        pcx, pcy = (px + 0.5) * self.CELL_W, (py + 0.5) * self.CELL_H
        player_size = self.CELL_W * 0.35
        glow_size = player_size + 3 + pulse * 3

        # Glow effect
        glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER_GLOW, 50), (glow_size, glow_size), glow_size)
        self.screen.blit(glow_surf, (pcx - glow_size, pcy - glow_size), special_flags=pygame.BLEND_RGBA_ADD)

        player_rect = pygame.Rect(pcx - player_size, pcy - player_size, player_size * 2, player_size * 2)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

        # Draw maze walls
        for y in range(self.MAZE_H):
            for x in range(self.MAZE_W):
                cell = self.maze[y][x]
                x1, y1 = x * self.CELL_W, y * self.CELL_H
                x2, y2 = (x + 1) * self.CELL_W, (y + 1) * self.CELL_H
                if cell['N']: pygame.draw.line(self.screen, self.COLOR_WALL, (x1, y1), (x2, y1), 3)
                if cell['S']: pygame.draw.line(self.screen, self.COLOR_WALL, (x1, y2), (x2, y2), 3)
                if cell['W']: pygame.draw.line(self.screen, self.COLOR_WALL, (x1, y1), (x1, y2), 3)
                if cell['E']: pygame.draw.line(self.screen, self.COLOR_WALL, (x2, y1), (x2, y2), 3)

    def _render_ui(self):
        # Gem count
        gem_text = f"Gems: {self.total_gems_collected}/{self.GEMS_PER_STAGE * self.TOTAL_STAGES}"
        gem_surf = self.font_ui.render(gem_text, True, self.COLOR_TEXT)
        self.screen.blit(gem_surf, (10, 10))

        # Timer
        time_sec = max(0, self.time_remaining / 30)
        time_str = f"{int(time_sec // 60):02d}:{int(time_sec % 60):02d}"
        time_color = self.COLOR_TEXT
        if time_sec < 10:
            time_color = self.COLOR_TIMER_CRITICAL
        elif time_sec < 30:
            time_color = self.COLOR_TIMER_WARN
        time_surf = self.font_ui.render(time_str, True, time_color)
        self.screen.blit(time_surf, (self.SCREEN_WIDTH - time_surf.get_width() - 10, 10))

        # Stage
        stage_text = f"Stage {self.current_stage}/{self.TOTAL_STAGES}"
        stage_surf = self.font_stage.render(stage_text, True, self.COLOR_TEXT)
        self.screen.blit(stage_surf, (self.SCREEN_WIDTH // 2 - stage_surf.get_width() // 2, 8))

    def render(self):
        return self._get_observation()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It requires a graphical display, so we unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    truncated = False

    # Use a separate display for human play
    pygame.display.init()
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Maze Runner")

    action = env.action_space.sample()
    action[0] = 0  # Start with no-op

    print(GameEnv.game_description)
    print(GameEnv.user_guide)

    # Game loop for human play
    running = True
    clock = pygame.time.Clock()

    while running:
        movement_action = 0  # Default to no-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                    truncated = False
                elif event.key == pygame.K_UP:
                    movement_action = 1
                elif event.key == pygame.K_DOWN:
                    movement_action = 2
                elif event.key == pygame.K_LEFT:
                    movement_action = 3
                elif event.key == pygame.K_RIGHT:
                    movement_action = 4

        # Step only if a move is made
        if movement_action != 0:
            action[0] = movement_action
            obs, reward, terminated, truncated, info = env.step(action)
            print(
                f"Step: {info['steps']}, Score: {info['score']:.1f}, Reward: {reward:.1f}, "
                f"Terminated: {terminated}, Truncated: {truncated}"
            )
            if terminated or truncated:
                print("Game Over! Press 'r' to reset.")

        # Update the display
        frame = env.render()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Control the frame rate for human play
        clock.tick(15)

    pygame.quit()