# Generated: 2025-08-28T02:40:52.439118
# Source Brief: brief_01775.md
# Brief Index: 1775

        
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
        "Controls: Arrow keys to move. Avoid enemies and reach the green exit."
    )

    game_description = (
        "Navigate a procedurally generated maze, dodging enemies, to reach the exit as quickly as possible while minimizing damage."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and grid dimensions
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.CELL_SIZE = 20
        self.MAZE_WIDTH = self.SCREEN_WIDTH // self.CELL_SIZE
        self.MAZE_HEIGHT = self.SCREEN_HEIGHT // self.CELL_SIZE

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_WALL = (40, 50, 90)
        self.COLOR_PLAYER = (255, 215, 0)
        self.COLOR_EXIT = (0, 255, 127)
        self.COLOR_ENEMY_TRI = (255, 69, 0)
        self.COLOR_ENEMY_SQR = (255, 165, 0)
        self.COLOR_ENEMY_PEN = (148, 0, 211)
        self.COLOR_HEALTH_FG = (0, 200, 0)
        self.COLOR_HEALTH_BG = (80, 0, 0)
        self.COLOR_TEXT = (220, 220, 220)

        # Game constants
        self.MAX_STEPS = 1000
        self.INITIAL_HEALTH = 10
        self.NUM_ENEMIES_PER_TYPE = 2
        
        # All state variables are initialized in reset()
        self.maze = None
        self.player_pos = None
        self.player_health = None
        self.exit_pos = None
        self.enemies = None
        self.steps = None
        self.score = None
        self.particles = None
        self.last_player_dist_to_exit = None

        self.reset()
        # self.validate_implementation() # This is called for verification, not needed in final code

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.player_health = self.INITIAL_HEALTH
        self.particles = []

        self._generate_maze()
        self._place_entities()
        
        if self.player_pos and self.exit_pos:
            self.last_player_dist_to_exit = self._manhattan_distance(self.player_pos, self.exit_pos)
        else:
            self.last_player_dist_to_exit = 0


        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        # space_held and shift_held are unused per brief
        
        reward = -0.1  # Cost of time

        # --- Player Movement ---
        old_dist = self._manhattan_distance(self.player_pos, self.exit_pos)
        
        if movement != 0:
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            px, py = self.player_pos
            nx, ny = px + dx, py + dy
            
            # FIX: Add boundary check before accessing maze
            if 0 <= nx < self.MAZE_WIDTH and 0 <= ny < self.MAZE_HEIGHT and self.maze[ny][nx] == 0:
                self.player_pos = (nx, ny)

        new_dist = self._manhattan_distance(self.player_pos, self.exit_pos)
        
        if new_dist < old_dist:
            reward += 0.2
        elif new_dist > old_dist:
            reward -= 0.5
        
        # --- Enemy Movement ---
        self._move_enemies()
        
        # --- Collision and Termination Check ---
        collision_penalty = self._check_player_enemy_collisions()
        reward += collision_penalty
        
        terminated = False
        goal_reward = 0.0
        
        if self.player_health <= 0:
            terminated = True
        elif self.player_pos == self.exit_pos:
            terminated = True
            if self.player_health >= 5:
                goal_reward = 100.0
        
        truncated = self.steps >= self.MAX_STEPS - 1
        if truncated:
            terminated = True

        reward += goal_reward
        self.score += reward
        self.steps += 1
        
        self._update_particles()

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _generate_maze(self):
        w = self.MAZE_WIDTH // 2
        h = self.MAZE_HEIGHT // 2
        visited = np.zeros((h, w), dtype=bool)
        maze = np.ones((self.MAZE_HEIGHT, self.MAZE_WIDTH), dtype=np.uint8)

        def carve(x, y):
            visited[y, x] = True
            maze[2 * y + 1, 2 * x + 1] = 0
            
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            self.np_random.shuffle(directions)
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                    maze[2 * y + 1 + dy, 2 * x + 1 + dx] = 0
                    carve(nx, ny)

        start_x, start_y = self.np_random.integers(0, w), self.np_random.integers(0, h)
        carve(start_x, start_y)
        self.maze = maze

    def _place_entities(self):
        path_cells = np.argwhere(self.maze == 0).tolist()
        self.np_random.shuffle(path_cells)
        
        # Player start (top-left quadrant) and Exit (bottom-right quadrant)
        w, h = self.MAZE_WIDTH, self.MAZE_HEIGHT
        
        # Fallback if path_cells is empty
        if not path_cells:
            self.player_pos = (1, 1)
            self.exit_pos = (w - 2, h - 2)
            self.enemies = []
            return

        while True:
            if not path_cells: break # Avoid infinite loop
            start_idx = self.np_random.integers(len(path_cells))
            py, px = path_cells[start_idx]
            if px < w // 2 and py < h // 2:
                self.player_pos = (px, py)
                path_cells.pop(start_idx)
                break
        
        while True:
            if not path_cells: break # Avoid infinite loop
            exit_idx = self.np_random.integers(len(path_cells))
            ey, ex = path_cells[exit_idx]
            if ex > w // 2 and ey > h // 2 and self._manhattan_distance((ex, ey), self.player_pos) > (w+h)//4:
                self.exit_pos = (ex, ey)
                path_cells.pop(exit_idx)
                break
        
        # Enemies
        self.enemies = []
        enemy_types = ['tri', 'sqr', 'pen']
        for _ in range(self.NUM_ENEMIES_PER_TYPE):
            for etype in enemy_types:
                if not path_cells: break
                for i in range(len(path_cells) -1, -1, -1): # Iterate backwards for safe pop
                    pos = tuple(path_cells[i][::-1])
                    if self._manhattan_distance(pos, self.player_pos) > 5:
                        enemy_data = {'type': etype, 'pos': pos}
                        if etype == 'sqr':
                            enemy_data['patrol_dir'] = 0
                            enemy_data['patrol_path'] = [(0, -1), (1, 0), (0, 1), (-1, 0)]
                        self.enemies.append(enemy_data)
                        path_cells.pop(i)
                        break

    def _move_enemies(self):
        for enemy in self.enemies:
            if enemy['type'] == 'tri': self._move_hunter(enemy)
            elif enemy['type'] == 'sqr': self._move_patroller(enemy)
            elif enemy['type'] == 'pen': self._move_random(enemy)

    def _move_hunter(self, enemy):
        ex, ey = enemy['pos']
        px, py = self.player_pos
        dx, dy = px - ex, py - ey
        
        # Try primary direction
        move_x, move_y = 0, 0
        if abs(dx) > abs(dy):
            move_x = np.sign(dx)
        else:
            move_y = np.sign(dy)
        
        nx, ny = ex + move_x, ey + move_y
        # FIX: Add boundary check
        if 0 <= nx < self.MAZE_WIDTH and 0 <= ny < self.MAZE_HEIGHT and self.maze[ny][nx] == 0:
            enemy['pos'] = (nx, ny)
            return
            
        # Try secondary direction
        move_x, move_y = 0, 0
        if abs(dx) <= abs(dy):
            move_x = np.sign(dx)
        else:
            move_y = np.sign(dy)
            
        if move_x == 0 and move_y == 0: return

        nx, ny = ex + move_x, ey + move_y
        # FIX: Add boundary check
        if 0 <= nx < self.MAZE_WIDTH and 0 <= ny < self.MAZE_HEIGHT and self.maze[ny][nx] == 0:
            enemy['pos'] = (nx, ny)
    
    def _move_patroller(self, enemy):
        ex, ey = enemy['pos']
        path = enemy['patrol_path']
        current_dir_idx = enemy['patrol_dir']
        
        move_x, move_y = path[current_dir_idx]
        nx, ny = ex + move_x, ey + move_y
        
        # FIX: Add boundary check
        if 0 <= nx < self.MAZE_WIDTH and 0 <= ny < self.MAZE_HEIGHT and self.maze[ny][nx] == 0:
            enemy['pos'] = (nx, ny)
        
        # Always turn for the next step, preserving original logic
        enemy['patrol_dir'] = (current_dir_idx + 1) % len(path)
            
    def _move_random(self, enemy):
        ex, ey = enemy['pos']
        valid_moves = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = ex + dx, ey + dy
            # FIX: Add boundary check
            if 0 <= nx < self.MAZE_WIDTH and 0 <= ny < self.MAZE_HEIGHT and self.maze[ny][nx] == 0:
                valid_moves.append((nx, ny))
        
        if valid_moves:
            # FIX: Use integers to select index, ensuring tuple type is preserved
            idx = self.np_random.integers(len(valid_moves))
            enemy['pos'] = valid_moves[idx]

    def _check_player_enemy_collisions(self):
        collided = False
        for enemy in self.enemies:
            if enemy['pos'] == self.player_pos:
                collided = True
                break
        
        if collided:
            self.player_health = max(0, self.player_health - 1)
            # Add particle effect at player position
            px, py = self.player_pos
            center_x = int((px + 0.5) * self.CELL_SIZE)
            center_y = int((py + 0.5) * self.CELL_SIZE)
            for _ in range(20):
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                self.particles.append({
                    'pos': [center_x, center_y],
                    'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                    'life': 20,
                    'color': self.COLOR_ENEMY_TRI
                })
            return -10.0
        return 0.0

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "health": self.player_health}

    def _render_game(self):
        # Maze
        if self.maze is not None:
            for y in range(self.MAZE_HEIGHT):
                for x in range(self.MAZE_WIDTH):
                    if self.maze[y][x] == 1:
                        pygame.draw.rect(self.screen, self.COLOR_WALL, (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))
        
        # Exit
        if self.exit_pos:
            ex, ey = self.exit_pos
            pygame.draw.rect(self.screen, self.COLOR_EXIT, (ex * self.CELL_SIZE, ey * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))

        # Enemies
        if self.enemies:
            for enemy in self.enemies:
                ex, ey = enemy['pos']
                center_x = int((ex + 0.5) * self.CELL_SIZE)
                center_y = int((ey + 0.5) * self.CELL_SIZE)
                radius = int(self.CELL_SIZE * 0.4)
                
                if enemy['type'] == 'tri':
                    self._draw_polygon(self.screen, self.COLOR_ENEMY_TRI, 3, radius, center_x, center_y, math.pi / 2)
                elif enemy['type'] == 'sqr':
                     pygame.gfxdraw.box(self.screen, (center_x - radius, center_y - radius, radius*2, radius*2), self.COLOR_ENEMY_SQR)
                elif enemy['type'] == 'pen':
                    self._draw_polygon(self.screen, self.COLOR_ENEMY_PEN, 5, radius, center_x, center_y, math.pi / 2)

        # Player
        if self.player_pos:
            px, py = self.player_pos
            center_x = int((px + 0.5) * self.CELL_SIZE)
            center_y = int((py + 0.5) * self.CELL_SIZE)
            radius = int(self.CELL_SIZE * 0.4)
            
            # Glow effect
            glow_radius = int(radius * 1.8)
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, (*self.COLOR_PLAYER, 60))
            self.screen.blit(glow_surf, (center_x - glow_radius, center_y - glow_radius))

            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_PLAYER)

        # Particles
        for p in self.particles:
            alpha = max(0, int(255 * (p['life'] / 20)))
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 2, color)


    def _render_ui(self):
        # Health Bar
        health_ratio = max(0, self.player_health / self.INITIAL_HEALTH)
        bar_width = 150
        bar_height = 20
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (10, 10, int(bar_width * health_ratio), bar_height))
        
        health_text = self.font_small.render(f"HEALTH", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10 + bar_width + 10, 12))

        # Score and Steps
        score_text = self.font_large.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 40))

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _draw_polygon(self, surface, color, n_sides, radius, center_x, center_y, rotation=0):
        points = []
        for i in range(n_sides):
            angle = (2 * math.pi * i / n_sides) + rotation
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            points.append((int(x), int(y)))
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        
        # Test state guarantees
        assert self.player_health <= self.INITIAL_HEALTH
        assert self.player_health >= 0
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Create a display window
    pygame.display.set_caption("Maze Runner")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    total_reward = 0
    clock = pygame.time.Clock()
    
    while not terminated:
        action = [0, 0, 0] # Default action: no-op, no buttons
        
        move_made = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                else: continue
                move_made = True

        if terminated:
             break

        if move_made:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Terminated: {terminated}")
            if terminated:
                print("Game Over!")

        # Draw the current state
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate for manual play

    env.close()