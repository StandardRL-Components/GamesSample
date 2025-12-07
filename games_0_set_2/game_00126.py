import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    user_guide = (
        "Controls: Arrow keys to move your robot. Press space to fire your laser. "
        "Reach the green exit with high health to win!"
    )

    game_description = (
        "Navigate a procedurally generated maze, blasting hostile robots with your laser. "
        "Reach the exit to win, but watch your health!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.CELL_SIZE = 40
        self.MAZE_WIDTH, self.MAZE_HEIGHT = 16, 10  # Adjusted to fit screen better
        self.WORLD_WIDTH = self.MAZE_WIDTH * self.CELL_SIZE
        self.WORLD_HEIGHT = self.MAZE_HEIGHT * self.CELL_SIZE
        self.MAX_STEPS = 2000
        self.WIN_HEALTH_THRESHOLD = 75

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_WALL = (50, 50, 70)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (0, 150, 255, 50)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_P_PROJ = (255, 255, 100)
        self.COLOR_E_PROJ = (255, 150, 0)
        self.COLOR_EXIT = (0, 255, 100)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_HEALTH_BAR_BG = (100, 0, 0)
        self.COLOR_HEALTH_BAR = (0, 200, 0)

        # Game parameters
        self.PLAYER_SIZE = 20
        self.PLAYER_SPEED = 4.0
        self.PLAYER_FIRE_RATE = 8  # steps between shots
        self.ENEMY_SIZE = 20
        self.ENEMY_SPEED = 1.5
        self.ENEMY_FIRE_RATE = 45
        self.PROJECTILE_SPEED = 10.0
        self.PROJECTILE_SIZE = 4

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # --- Game State (initialized in reset) ---
        self.np_random = None
        self.maze = None
        self.player = None
        self.exit_pos = None
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.camera_offset = pygame.Vector2(0, 0)

    def _generate_maze(self):
        w, h = self.MAZE_WIDTH, self.MAZE_HEIGHT
        maze = [[{'N': True, 'S': True, 'E': True, 'W': True} for _ in range(w)] for _ in range(h)]
        visited = [[False for _ in range(w)] for _ in range(h)]

        dx = {'E': 1, 'W': -1, 'N': 0, 'S': 0}
        dy = {'E': 0, 'W': 0, 'N': -1, 'S': 1}
        opposite = {'E': 'W', 'W': 'E', 'N': 'S', 'S': 'N'}

        stack = deque()
        start_x, start_y = self.np_random.integers(0, w), self.np_random.integers(0, h)
        stack.append((start_x, start_y))
        visited[start_y][start_x] = True

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for direction in ['N', 'S', 'E', 'W']:
                nx, ny = cx + dx[direction], cy + dy[direction]
                if 0 <= nx < w and 0 <= ny < h and not visited[ny][nx]:
                    neighbors.append((direction, nx, ny))

            if neighbors:
                # FIX: Use np_random.integers to pick an index, then select the neighbor tuple.
                # This avoids np.random.choice's type promotion which turns integers into strings.
                choice_index = self.np_random.integers(len(neighbors))
                direction, nx, ny = neighbors[choice_index]

                maze[cy][cx][direction] = False
                maze[ny][nx][opposite[direction]] = False
                visited[ny][nx] = True
                stack.append((nx, ny))
            else:
                stack.pop()
        return maze

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.maze = self._generate_maze()

        # Place player
        start_cell = (0, 0)
        self.player = {
            "pos": pygame.Vector2(start_cell[0] * self.CELL_SIZE + self.CELL_SIZE / 2,
                                  start_cell[1] * self.CELL_SIZE + self.CELL_SIZE / 2),
            "health": 100,
            "fire_cooldown": 0,
            "last_move_vec": pygame.Vector2(1, 0),
            "hit_timer": 0
        }

        # Place exit
        self.exit_pos = pygame.Vector2((self.MAZE_WIDTH - 0.5) * self.CELL_SIZE,
                                       (self.MAZE_HEIGHT - 0.5) * self.CELL_SIZE)

        # Place enemies
        self.enemies = []
        occupied_cells = {start_cell, (self.MAZE_WIDTH - 1, self.MAZE_HEIGHT - 1)}
        for _ in range(10):
            while True:
                ex, ey = self.np_random.integers(0, self.MAZE_WIDTH), self.np_random.integers(0, self.MAZE_HEIGHT)
                dist_to_player = math.hypot(ex - start_cell[0], ey - start_cell[1])
                if (ex, ey) not in occupied_cells and dist_to_player > 5:
                    self.enemies.append({
                        "pos": pygame.Vector2(ex * self.CELL_SIZE + self.CELL_SIZE / 2,
                                              ey * self.CELL_SIZE + self.CELL_SIZE / 2),
                        "fire_cooldown": self.np_random.integers(0, self.ENEMY_FIRE_RATE),
                        "move_dir": self.np_random.choice([-1, 1], 2).astype(float) * self.ENEMY_SPEED
                    })
                    occupied_cells.add((ex, ey))
                    break

        self.projectiles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False

        return self._get_observation(), self._get_info()

    def _get_cell(self, pos):
        x = int(pos.x // self.CELL_SIZE)
        y = int(pos.y // self.CELL_SIZE)
        x = np.clip(x, 0, self.MAZE_WIDTH - 1)
        y = np.clip(y, 0, self.MAZE_HEIGHT - 1)
        return x, y

    def _is_wall_collision(self, pos, size):
        points_to_check = [
            pos + pygame.Vector2(s_x, s_y) * size / 2
            for s_x in [-1, 1] for s_y in [-1, 1]
        ]
        for p in points_to_check:
            if not (0 <= p.x < self.WORLD_WIDTH and 0 <= p.y < self.WORLD_HEIGHT):
                return True  # Out of world bounds
        return False

    def _move_and_collide(self, pos, vel):
        new_pos = pos + vel

        # Maze wall collisions
        cx, cy = self._get_cell(pos)
        cell_x_start = cx * self.CELL_SIZE
        cell_y_start = cy * self.CELL_SIZE

        # Test new X position
        if vel.x > 0 and self.maze[cy][cx]['E'] and new_pos.x + self.PLAYER_SIZE/2 > cell_x_start + self.CELL_SIZE:
             new_pos.x = cell_x_start + self.CELL_SIZE - self.PLAYER_SIZE/2
        elif vel.x < 0 and self.maze[cy][cx]['W'] and new_pos.x - self.PLAYER_SIZE/2 < cell_x_start:
             new_pos.x = cell_x_start + self.PLAYER_SIZE/2

        # Test new Y position
        if vel.y > 0 and self.maze[cy][cx]['S'] and new_pos.y + self.PLAYER_SIZE/2 > cell_y_start + self.CELL_SIZE:
             new_pos.y = cell_y_start + self.CELL_SIZE - self.PLAYER_SIZE/2
        elif vel.y < 0 and self.maze[cy][cx]['N'] and new_pos.y - self.PLAYER_SIZE/2 < cell_y_start:
             new_pos.y = cell_y_start + self.PLAYER_SIZE/2
        
        pos.update(new_pos)
        pos.x = np.clip(pos.x, self.PLAYER_SIZE/2, self.WORLD_WIDTH - self.PLAYER_SIZE/2)
        pos.y = np.clip(pos.y, self.PLAYER_SIZE/2, self.WORLD_HEIGHT - self.PLAYER_SIZE/2)


    def _has_line_of_sight(self, p1, p2):
        c1x, c1y = self._get_cell(p1)
        c2x, c2y = self._get_cell(p2)

        if c1x == c2x:  # Vertical
            step = 1 if c2y > c1y else -1
            for y in range(c1y, c2y, step):
                if step == 1 and self.maze[y][c1x]['S']: return False
                if step == -1 and self.maze[y][c1x]['N']: return False
            return True
        elif c1y == c2y:  # Horizontal
            step = 1 if c2x > c1x else -1
            for x in range(c1x, c2x, step):
                if step == 1 and self.maze[c1y][x]['E']: return False
                if step == -1 and self.maze[c1y][x]['W']: return False
            return True
        return False

    def _create_explosion(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": self.np_random.integers(15, 30),
                "color": color,
                "radius": self.np_random.uniform(2, 5)
            })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01  # Time penalty
        self.steps += 1

        # --- Player Input ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        player_vel = pygame.Vector2(0, 0)

        if movement == 1: player_vel.y = -1
        elif movement == 2: player_vel.y = 1
        elif movement == 3: player_vel.x = -1
        elif movement == 4: player_vel.x = 1

        if player_vel.length() > 0:
            player_vel.normalize_ip()
            self.player["last_move_vec"] = player_vel.copy()

        self._move_and_collide(self.player["pos"], player_vel * self.PLAYER_SPEED)

        if self.player["fire_cooldown"] > 0: self.player["fire_cooldown"] -= 1
        if self.player["hit_timer"] > 0: self.player["hit_timer"] -= 1

        if space_held and self.player["fire_cooldown"] == 0:
            self.player["fire_cooldown"] = self.PLAYER_FIRE_RATE
            proj_vel = self.player["last_move_vec"] * self.PROJECTILE_SPEED
            self.projectiles.append({
                "pos": self.player["pos"].copy(), "vel": proj_vel, "owner": "player"
            })

        # --- Update Enemies ---
        for enemy in self.enemies:
            # Movement
            enemy["pos"] += enemy["move_dir"] * 0.2  # Slowed down for erratic pattern
            ecx, ecy = self._get_cell(enemy["pos"])
            if self.maze[ecy][ecx]['W'] and enemy["move_dir"][0] < 0: enemy["move_dir"][0] *= -1
            if self.maze[ecy][ecx]['E'] and enemy["move_dir"][0] > 0: enemy["move_dir"][0] *= -1
            if self.maze[ecy][ecx]['N'] and enemy["move_dir"][1] < 0: enemy["move_dir"][1] *= -1
            if self.maze[ecy][ecx]['S'] and enemy["move_dir"][1] > 0: enemy["move_dir"][1] *= -1
            enemy["pos"].x = np.clip(enemy["pos"].x, 0, self.WORLD_WIDTH)
            enemy["pos"].y = np.clip(enemy["pos"].y, 0, self.WORLD_HEIGHT)

            # Shooting
            enemy["fire_cooldown"] -= 1
            dist_to_player = self.player["pos"].distance_to(enemy["pos"])
            if enemy["fire_cooldown"] <= 0 and dist_to_player < 300 and self._has_line_of_sight(enemy["pos"], self.player["pos"]):
                enemy["fire_cooldown"] = self.ENEMY_FIRE_RATE
                direction = (self.player["pos"] - enemy["pos"]).normalize()
                self.projectiles.append({
                    "pos": enemy["pos"].copy(), "vel": direction * self.PROJECTILE_SPEED, "owner": "enemy"
                })

        # --- Update Projectiles & Collisions ---
        player_rect = pygame.Rect(self.player["pos"].x - self.PLAYER_SIZE / 2,
                                  self.player["pos"].y - self.PLAYER_SIZE / 2, self.PLAYER_SIZE, self.PLAYER_SIZE)

        projectiles_to_keep = []
        for p in self.projectiles:
            p["pos"] += p["vel"]
            pcx, pcy = self._get_cell(p["pos"])

            # Check wall collision
            if not (0 <= p["pos"].x < self.WORLD_WIDTH and 0 <= p["pos"].y < self.WORLD_HEIGHT):
                continue

            hit = False
            # Check entity collisions
            if p["owner"] == "player":
                for i in range(len(self.enemies) - 1, -1, -1):
                    enemy = self.enemies[i]
                    if (enemy["pos"] - p["pos"]).length() < (self.ENEMY_SIZE + self.PROJECTILE_SIZE) / 2:
                        reward += 1
                        self.score += 10
                        self._create_explosion(enemy["pos"], 30, self.COLOR_ENEMY)
                        self.enemies.pop(i)
                        hit = True
                        break
            elif p["owner"] == "enemy":
                if player_rect.collidepoint(p["pos"]):
                    reward -= 1
                    self.player["health"] -= 10
                    self.player["hit_timer"] = 5
                    hit = True

            if not hit:
                projectiles_to_keep.append(p)
        self.projectiles = projectiles_to_keep

        # --- Update Particles ---
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
            p["radius"] *= 0.95

        # --- Termination Check ---
        terminated = False
        truncated = False
        if self.player["health"] <= 0:
            reward -= 100
            self.game_over = True
            terminated = True

        if self.player["pos"].distance_to(self.exit_pos) < self.CELL_SIZE / 2:
            if self.player["health"] >= self.WIN_HEALTH_THRESHOLD:
                reward += 100
                self.game_won = True
            self.game_over = True
            terminated = True

        if self.steps >= self.MAX_STEPS:
            truncated = True # Use truncated for time limit
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _render_world(self):
        # Update camera
        self.camera_offset = self.player["pos"] - pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.camera_offset.x = np.clip(self.camera_offset.x, 0, self.WORLD_WIDTH - self.SCREEN_WIDTH)
        self.camera_offset.y = np.clip(self.camera_offset.y, 0, self.WORLD_HEIGHT - self.SCREEN_HEIGHT)

        # Render maze
        start_cx = int(self.camera_offset.x // self.CELL_SIZE)
        end_cx = int((self.camera_offset.x + self.SCREEN_WIDTH) // self.CELL_SIZE) + 1
        start_cy = int(self.camera_offset.y // self.CELL_SIZE)
        end_cy = int((self.camera_offset.y + self.SCREEN_HEIGHT) // self.CELL_SIZE) + 1

        for y in range(max(0, start_cy), min(self.MAZE_HEIGHT, end_cy)):
            for x in range(max(0, start_cx), min(self.MAZE_WIDTH, end_cx)):
                cell_x = x * self.CELL_SIZE - self.camera_offset.x
                cell_y = y * self.CELL_SIZE - self.camera_offset.y
                if self.maze[y][x]['N']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (cell_x, cell_y),
                                     (cell_x + self.CELL_SIZE, cell_y), 2)
                if self.maze[y][x]['S']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (cell_x, cell_y + self.CELL_SIZE),
                                     (cell_x + self.CELL_SIZE, cell_y + self.CELL_SIZE), 2)
                if self.maze[y][x]['W']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (cell_x, cell_y), (cell_x, cell_y + self.CELL_SIZE),
                                     2)
                if self.maze[y][x]['E']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (cell_x + self.CELL_SIZE, cell_y),
                                     (cell_x + self.CELL_SIZE, cell_y + self.CELL_SIZE), 2)

        # Render exit
        exit_screen_pos = self.exit_pos - self.camera_offset
        pygame.gfxdraw.filled_circle(self.screen, int(exit_screen_pos.x), int(exit_screen_pos.y),
                                     int(self.CELL_SIZE / 2.5), self.COLOR_EXIT)

        # Render projectiles
        for p in self.projectiles:
            start = p["pos"] - self.camera_offset
            end = p["pos"] - p["vel"] * 0.5 - self.camera_offset
            color = self.COLOR_P_PROJ if p["owner"] == "player" else self.COLOR_E_PROJ
            pygame.draw.line(self.screen, color, start, end, 4)

        # Render enemies
        for enemy in self.enemies:
            e_pos = enemy["pos"] - self.camera_offset
            points = [
                (e_pos.x, e_pos.y - self.ENEMY_SIZE / 2),
                (e_pos.x - self.ENEMY_SIZE / 2, e_pos.y + self.ENEMY_SIZE / 2),
                (e_pos.x + self.ENEMY_SIZE / 2, e_pos.y + self.ENEMY_SIZE / 2)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)

        # Render player
        p_pos = self.player["pos"] - self.camera_offset
        player_color = (255, 255, 255) if self.player["hit_timer"] > 0 else self.COLOR_PLAYER
        pygame.gfxdraw.filled_circle(self.screen, int(p_pos.x), int(p_pos.y), int(self.PLAYER_SIZE * 0.8),
                                     self.COLOR_PLAYER_GLOW)
        player_rect = pygame.Rect(p_pos.x - self.PLAYER_SIZE / 2, p_pos.y - self.PLAYER_SIZE / 2, self.PLAYER_SIZE,
                                  self.PLAYER_SIZE)
        pygame.draw.rect(self.screen, player_color, player_rect, border_radius=3)

        # Render particles
        for p in self.particles:
            part_pos = p["pos"] - self.camera_offset
            alpha = int(255 * (p["life"] / 30.0))
            color = (p["color"][0], p["color"][1], p["color"][2], alpha)
            if p["radius"] > 1:
                pygame.gfxdraw.filled_circle(self.screen, int(part_pos.x), int(part_pos.y), int(p["radius"]), color)

    def _render_ui(self):
        # Health bar
        health_pct = max(0, self.player["health"] / 100.0)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, 200 * health_pct, 20))
        health_text = self.font_ui.render(f"HP: {self.player['health']}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Game Over
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            msg = "VICTORY!" if self.game_won else "GAME OVER"
            color = self.COLOR_EXIT if self.game_won else self.COLOR_ENEMY

            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_world()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player["health"],
            "enemies_left": len(self.enemies)
        }

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game manually.
    # It will not be executed by the evaluation system.
    # To use, you might need to remove the dummy video driver environment variable.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()

    pygame.display.set_caption(env.game_description)
    render_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))

    total_reward = 0
    terminated = False
    truncated = False

    running = True
    while running:
        movement = 0  # none
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                 terminated, truncated = True, True # Force reset

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4

        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30)  # Run at 30 FPS

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            terminated = False
            truncated = False

    env.close()