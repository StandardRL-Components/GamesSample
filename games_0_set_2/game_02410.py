
# Generated: 2025-08-27T20:18:45.506952
# Source Brief: brief_02410.md
# Brief Index: 2410

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper class for particle effects, defined within the same file
class Particle:
    """A small, fading particle for visual effects."""
    def __init__(self, x, y, rng):
        self.x = x
        self.y = y
        self.rng = rng
        self.vx = self.rng.uniform(-0.5, 0.5)
        self.vy = self.rng.uniform(-0.5, 0.5)
        self.lifespan = self.rng.integers(15, 30)
        self.age = 0
        self.color = (150, 20, 20)
        self.start_radius = self.rng.integers(1, 3)

    def update(self):
        """Update particle position and age. Returns False if dead."""
        self.x += self.vx
        self.y += self.vy
        self.age += 1
        return self.age < self.lifespan

    def draw(self, surface):
        """Draw the particle on the given surface."""
        progress = self.age / self.lifespan
        # Use a non-linear fade for better effect
        alpha = int(200 * (1 - progress)**2)
        radius = int(self.start_radius * (1 - progress))
        if radius > 0:
            # Use standard draw circle with SRCALPHA for transparency
            s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.color, alpha), (radius, radius), radius)
            surface.blit(s, (int(self.x - radius), int(self.y - radius)))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to navigate the maze. Avoid the red enemies and reach the green exit."
    )

    game_description = (
        "A survival horror maze game. Navigate a procedurally generated labyrinth to find the exit before time runs out, all while being hunted by relentless enemies."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        # Game constants
        self.MAZE_WIDTH, self.MAZE_HEIGHT = 32, 20
        self.CELL_SIZE = 20
        self.NUM_ENEMIES = 10
        self.MAX_STAGES = 3
        self.TIME_PER_STAGE = 1800  # 60s @ 30fps equivalent steps

        # Visuals
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_WALL = (40, 40, 60)
        self.COLOR_PATH = (20, 20, 30)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (180, 180, 255)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_GLOW = (200, 0, 0)
        self.COLOR_EXIT = (50, 255, 50)
        self.COLOR_EXIT_GLOW = (0, 200, 0)
        self.COLOR_UI = (220, 220, 220)
        self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)

        # Persistent state (across episodes)
        self.total_steps_across_episodes = 0
        self.enemy_speed = 0.25

        # Episode-specific state variables
        self.maze = None
        self.player_pos = None
        self.exit_pos = None
        self.enemies = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.stage = 1
        self.timer = 0
        self.rng = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.stage = 1
        
        self._generate_new_stage()

        return self._get_observation(), self._get_info()

    def _generate_new_stage(self):
        """Resets the maze, player, and enemies for a new stage."""
        self.timer = self.TIME_PER_STAGE
        self.particles.clear()
        self.maze = self._generate_maze(self.MAZE_WIDTH, self.MAZE_HEIGHT)
        
        self.player_pos = [1, 1]
        self.exit_pos = [self.MAZE_WIDTH - 2, self.MAZE_HEIGHT - 2]
        self.maze[self.exit_pos[1], self.exit_pos[0]] = 0 # Ensure exit is reachable

        self._initialize_enemies()

    def _generate_maze(self, width, height):
        """Generates a maze using recursive backtracking."""
        maze = np.ones((height, width), dtype=np.uint8)
        visited = np.zeros_like(maze, dtype=bool)
        stack = [(1, 1)]
        visited[1, 1] = True
        maze[1, 1] = 0

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            # Check neighbors 2 cells away to create walls
            if cx > 1 and not visited[cy, cx - 2]: neighbors.append('W')
            if cx < width - 2 and not visited[cy, cx + 2]: neighbors.append('E')
            if cy > 1 and not visited[cy - 2, cx]: neighbors.append('N')
            if cy < height - 2 and not visited[cy + 2, cx]: neighbors.append('S')

            if neighbors:
                direction = self.rng.choice(neighbors)
                nx, ny = cx, cy
                if direction == 'N':
                    ny -= 2; maze[cy - 1, cx] = 0
                elif direction == 'S':
                    ny += 2; maze[cy + 1, cx] = 0
                elif direction == 'W':
                    nx -= 2; maze[cy, cx - 1] = 0
                elif direction == 'E':
                    nx += 2; maze[cy, cx + 1] = 0
                
                maze[ny, nx] = 0
                visited[ny, nx] = True
                stack.append((nx, ny))
            else:
                stack.pop()
        return maze

    def _initialize_enemies(self):
        """Finds valid patrol paths and places enemies on them."""
        self.enemies.clear()
        patrol_paths = self._find_valid_patrol_paths()
        self.rng.shuffle(patrol_paths)

        for i in range(self.NUM_ENEMIES):
            if not patrol_paths: break
            path = patrol_paths.pop(0)
            self.enemies.append({
                'path': path,
                'path_index': 0,
                'pos': list(path[0]),
                'target_node_idx': 1
            })

    def _find_valid_patrol_paths(self):
        """Identifies 2x2 open squares in the maze for enemy patrols."""
        paths = []
        for r in range(self.MAZE_HEIGHT - 1):
            for c in range(self.MAZE_WIDTH - 1):
                if (self.maze[r, c] == 0 and self.maze[r + 1, c] == 0 and
                    self.maze[r, c + 1] == 0 and self.maze[r + 1, c + 1] == 0 and
                    [c, r] != self.player_pos):
                    path = [(c, r), (c + 1, r), (c + 1, r + 1), (c, r + 1)]
                    paths.append(path)
        return paths

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0.1  # Survival reward for taking a step

        self.steps += 1
        self.total_steps_across_episodes += 1
        self.timer -= 1

        if self.total_steps_across_episodes > 0 and self.total_steps_across_episodes % 500 == 0:
            self.enemy_speed = min(self.enemy_speed + 0.05, 0.8)

        dist_before = abs(self.player_pos[0] - self.exit_pos[0]) + abs(self.player_pos[1] - self.exit_pos[1])
        moved = self._handle_player_movement(movement)
        if moved:
            dist_after = abs(self.player_pos[0] - self.exit_pos[0]) + abs(self.player_pos[1] - self.exit_pos[1])
            if dist_after > dist_before:
                reward -= 0.2 # Penalty for moving away from exit

        self._update_enemies()
        self._update_particles()

        terminated = False
        player_grid_pos = tuple(self.player_pos)
        for enemy in self.enemies:
            if player_grid_pos == (int(enemy['pos'][0]), int(enemy['pos'][1])):
                reward = -10.0; self.game_over = terminated = True; break
        
        if not terminated and tuple(self.player_pos) == tuple(self.exit_pos):
            reward += 10.0
            if self.stage >= self.MAX_STAGES:
                reward += 50.0; self.game_over = self.game_won = terminated = True
            else:
                self.stage += 1; self._generate_new_stage()

        if not terminated and self.timer <= 0:
            reward = -10.0; self.game_over = terminated = True

        self.score += reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_player_movement(self, movement):
        """Updates player position based on action, returns True if moved."""
        new_pos = list(self.player_pos)
        if movement == 1: new_pos[1] -= 1  # Up
        elif movement == 2: new_pos[1] += 1 # Down
        elif movement == 3: new_pos[0] -= 1 # Left
        elif movement == 4: new_pos[0] += 1 # Right
        else: return False

        if 0 <= new_pos[0] < self.MAZE_WIDTH and 0 <= new_pos[1] < self.MAZE_HEIGHT:
            if self.maze[new_pos[1], new_pos[0]] == 0:
                self.player_pos = new_pos
                return True
        return False

    def _update_enemies(self):
        """Moves enemies along their patrol paths."""
        for enemy in self.enemies:
            target_pos = enemy['path'][enemy['target_node_idx']]
            direction = [target_pos[0] - enemy['pos'][0], target_pos[1] - enemy['pos'][1]]
            dist = math.hypot(*direction)

            if dist < self.enemy_speed:
                enemy['pos'] = list(target_pos)
                enemy['target_node_idx'] = (enemy['target_node_idx'] + 1) % len(enemy['path'])
            else:
                norm_dir = [d / dist for d in direction]
                enemy['pos'][0] += norm_dir[0] * self.enemy_speed
                enemy['pos'][1] += norm_dir[1] * self.enemy_speed

            if self.rng.random() < 0.7:
                px, py = self._grid_to_pixel_center(enemy['pos'])
                self.particles.append(Particle(px, py, self.rng))

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over: self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "stage": self.stage}

    def _grid_to_pixel_center(self, grid_pos):
        x = (grid_pos[0] + 0.5) * self.CELL_SIZE
        y = (grid_pos[1] + 0.5) * self.CELL_SIZE
        return int(x), int(y)

    def _draw_glow(self, surface, color, center_px, max_radius):
        for r in range(max_radius, 0, -2):
            alpha = int(100 * (1 - r / max_radius)**1.5)
            if alpha > 0:
                pygame.gfxdraw.filled_circle(surface, center_px[0], center_px[1], r, (*color, alpha))

    def _render_game(self):
        for r in range(self.MAZE_HEIGHT):
            for c in range(self.MAZE_WIDTH):
                color = self.COLOR_WALL if self.maze[r, c] == 1 else self.COLOR_PATH
                pygame.draw.rect(self.screen, color, (c * self.CELL_SIZE, r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))

        exit_px = self._grid_to_pixel_center(self.exit_pos)
        self._draw_glow(self.screen, self.COLOR_EXIT_GLOW, exit_px, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, (self.exit_pos[0] * self.CELL_SIZE, self.exit_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))

        for p in self.particles: p.draw(self.screen)

        for enemy in self.enemies:
            enemy_px = self._grid_to_pixel_center(enemy['pos'])
            self._draw_glow(self.screen, self.COLOR_ENEMY_GLOW, enemy_px, int(self.CELL_SIZE * 0.8))
            p1 = (enemy_px[0], enemy_px[1] - self.CELL_SIZE // 3)
            p2 = (enemy_px[0] - self.CELL_SIZE // 3, enemy_px[1] + self.CELL_SIZE // 4)
            p3 = (enemy_px[0] + self.CELL_SIZE // 3, enemy_px[1] + self.CELL_SIZE // 4)
            pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_ENEMY)
            pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_ENEMY)

        player_px = self._grid_to_pixel_center(self.player_pos)
        self._draw_glow(self.screen, self.COLOR_PLAYER_GLOW, player_px, int(self.CELL_SIZE * 1.2))
        player_rect = pygame.Rect(
            self.player_pos[0] * self.CELL_SIZE + 4, self.player_pos[1] * self.CELL_SIZE + 4,
            self.CELL_SIZE - 8, self.CELL_SIZE - 8
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=2)

    def _render_ui(self):
        stage_text = self.font_ui.render(f"Stage: {self.stage}/{self.MAX_STAGES}", True, self.COLOR_UI)
        self.screen.blit(stage_text, (10, 10))

        timer_text = self.font_ui.render(f"Time: {self.timer}", True, self.COLOR_UI)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))
        
        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_UI)
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, 10))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        msg, color = ("YOU ESCAPED", self.COLOR_EXIT) if self.game_won else ("GAME OVER", self.COLOR_ENEMY)
        text = self.font_game_over.render(msg, True, color)
        overlay.blit(text, text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2)))
        self.screen.blit(overlay, (0, 0))

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3) and test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3) and isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3) and isinstance(reward, float)
        assert isinstance(term, bool) and not trunc and isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Maze Survival")
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    terminated = False
    while running:
        action_to_take = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if terminated and event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                    continue

                if not terminated:
                    current_action = [0, 0, 0]
                    if event.key == pygame.K_UP: current_action[0] = 1
                    elif event.key == pygame.K_DOWN: current_action[0] = 2
                    elif event.key == pygame.K_LEFT: current_action[0] = 3
                    elif event.key == pygame.K_RIGHT: current_action[0] = 4
                    
                    if current_action[0] != 0:
                        action_to_take = current_action

        if action_to_take and not terminated:
            obs, reward, term, truncated, info = env.step(action_to_take)
            terminated = term
            print(f"Action: {action_to_take}, Reward: {reward:.2f}, Terminated: {terminated}, Score: {info['score']:.1f}")
            if terminated:
                print("Game Over! Press 'R' to restart.")

        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30)

    pygame.quit()