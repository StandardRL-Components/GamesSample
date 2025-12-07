import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import heapq
import os
import os
import pygame


# Set Pygame to run headlessly
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to navigate the maze. Reach the green exit tile."
    )

    game_description = (
        "Navigate a procedurally generated minefield maze. Reach the exit for a high score, "
        "but beware of hidden mines. The closer you are to a mine, the more the screen "
        "will glow red. The number of mines increases with each successful run."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 10
        self.MAX_STEPS = 250
        self.INITIAL_MINES = 5
        self.MAX_MINES = 15

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # --- Colors ---
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_WALL = (40, 50, 60)
        self.COLOR_FLOOR = (25, 30, 40)
        self.COLOR_PLAYER = (255, 200, 0)
        self.COLOR_PLAYER_GLOW = (255, 200, 0, 50)
        self.COLOR_EXIT = (0, 255, 100)
        self.COLOR_MINE = (255, 50, 50)
        self.COLOR_MINE_GLOW = (255, 50, 50, 70)
        self.COLOR_DANGER_TINT = (200, 0, 0)
        self.COLOR_TEXT = (230, 230, 240)

        # --- Persistent State ---
        self.num_mines = self.INITIAL_MINES

        # --- Initialize State ---
        self.maze = None
        self.player_pos = None
        self.exit_pos = None
        self.mine_positions = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_player_pos = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.start_pos = (0, 0)
        self.exit_pos = (self.GRID_WIDTH - 1, self.GRID_HEIGHT - 1)
        self.player_pos = self.start_pos
        self.last_player_pos = self.player_pos

        self.maze = self._generate_maze(self.GRID_WIDTH, self.GRID_HEIGHT, self.start_pos)
        self.mine_positions = self._place_mines()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        self.last_player_pos = self.player_pos
        
        # --- Player Movement ---
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        reward = -0.1 # Base cost for taking a step

        if movement != 0:
            new_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
            if self._is_valid_move(self.player_pos, new_pos):
                self.player_pos = new_pos
        
        # --- Calculate Reward ---
        step_reward = self._calculate_reward()
        reward += step_reward
        self.score += reward
        
        # --- Check Termination ---
        terminated = False
        if self.player_pos in self.mine_positions:
            self.score -= 100.0 # Apply large penalty
            self.game_over = True
            terminated = True
        elif self.player_pos == self.exit_pos:
            self.score += 100.0 # Apply large reward
            self.game_over = True
            terminated = True
            self.num_mines = min(self.MAX_MINES, self.num_mines + 1)
        
        self.steps += 1
        truncated = False
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            truncated = True
            terminated = True # Per Gymnasium API, terminated should be True if truncated is

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _calculate_reward(self):
        reward = 0.0
        # Reward for movement relative to exit
        old_dist = self._a_star_dist(self.last_player_pos, self.exit_pos)
        new_dist = self._a_star_dist(self.player_pos, self.exit_pos)
        
        if new_dist is not None and old_dist is not None:
            if new_dist < old_dist:
                reward += 0.5  # Moved closer
            elif new_dist > old_dist:
                reward -= 0.2  # Moved further

        # Penalty for proximity to mines
        min_mine_dist = self._get_min_manhattan_dist(self.player_pos, self.mine_positions)
        if min_mine_dist is not None and min_mine_dist <= 1:
            reward -= 5.0
        
        # Reward for proximity to exit
        if self._manhattan_distance(self.player_pos, self.exit_pos) <= 1:
            reward += 10.0
            
        return reward

    def _get_observation(self):
        # --- Calculate Tile Size & Offsets for Centering ---
        game_area_height = self.HEIGHT - 60 # Reserve space for UI
        self.tile_size = min(self.WIDTH // self.GRID_WIDTH, game_area_height // self.GRID_HEIGHT)
        offset_x = (self.WIDTH - self.GRID_WIDTH * self.tile_size) // 2
        offset_y = ((game_area_height - self.GRID_HEIGHT * self.tile_size) // 2) + 60

        # --- Background and Danger Tint ---
        self.screen.fill(self.COLOR_BG)
        min_mine_dist = self._get_min_manhattan_dist(self.player_pos, self.mine_positions)
        if min_mine_dist is not None:
            danger_alpha = max(0, 255 * (1 - min_mine_dist / 5))
            if danger_alpha > 0:
                tint_surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
                pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2 # 0 to 1
                tint_surface.fill((*self.COLOR_DANGER_TINT, int(danger_alpha * (0.5 + pulse * 0.5))))
                self.screen.blit(tint_surface, (0, 0))

        # --- Render Game Elements ---
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = pygame.Rect(
                    offset_x + x * self.tile_size,
                    offset_y + y * self.tile_size,
                    self.tile_size, self.tile_size
                )
                
                # Draw floor everywhere
                pygame.draw.rect(self.screen, self.COLOR_FLOOR, rect)

                # Draw walls
                if not self.maze[y][x]['n']: pygame.draw.line(self.screen, self.COLOR_WALL, rect.topleft, rect.topright, 2)
                if not self.maze[y][x]['s']: pygame.draw.line(self.screen, self.COLOR_WALL, rect.bottomleft, rect.bottomright, 2)
                if not self.maze[y][x]['w']: pygame.draw.line(self.screen, self.COLOR_WALL, rect.topleft, rect.bottomleft, 2)
                if not self.maze[y][x]['e']: pygame.draw.line(self.screen, self.COLOR_WALL, rect.topright, rect.bottomright, 2)

        # Draw Exit
        exit_rect = pygame.Rect(offset_x + self.exit_pos[0] * self.tile_size, offset_y + self.exit_pos[1] * self.tile_size, self.tile_size, self.tile_size)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect.inflate(-self.tile_size*0.2, -self.tile_size*0.2))

        # Draw Mines (if game over)
        if self.game_over:
            for mx, my in self.mine_positions:
                center_x = int(offset_x + (mx + 0.5) * self.tile_size)
                center_y = int(offset_y + (my + 0.5) * self.tile_size)
                radius = int(self.tile_size * 0.3)
                pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_MINE)
                pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_MINE)

        # Draw Player
        player_center_x = int(offset_x + (self.player_pos[0] + 0.5) * self.tile_size)
        player_center_y = int(offset_y + (self.player_pos[1] + 0.5) * self.tile_size)
        player_radius = int(self.tile_size * 0.35)
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, player_center_x, player_center_y, int(player_radius * 1.5), self.COLOR_PLAYER_GLOW)
        # Body
        pygame.gfxdraw.filled_circle(self.screen, player_center_x, player_center_y, player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_center_x, player_center_y, player_radius, self.COLOR_PLAYER)

        # --- Render UI ---
        score_text = self.font_large.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 20, 10))

        steps_text = self.font_small.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (20, 10))
        
        mines_text = self.font_small.render(f"Mines: {self.num_mines}", True, self.COLOR_TEXT)
        self.screen.blit(mines_text, (20, 35))

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.player_pos,
            "mines_in_maze": self.num_mines,
        }

    # --- Helper Methods ---

    def _generate_maze(self, width, height, start_node):
        maze = [[{'n': False, 's': False, 'e': False, 'w': False, 'visited': False} for _ in range(width)] for _ in range(height)]
        
        stack = [start_node]
        maze[start_node[1]][start_node[0]]['visited'] = True

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            if cy > 0 and not maze[cy - 1][cx]['visited']: neighbors.append(('n', (cx, cy - 1)))
            if cy < height - 1 and not maze[cy + 1][cx]['visited']: neighbors.append(('s', (cx, cy + 1)))
            if cx < width - 1 and not maze[cy][cx + 1]['visited']: neighbors.append(('e', (cx + 1, cy)))
            if cx > 0 and not maze[cy][cx - 1]['visited']: neighbors.append(('w', (cx - 1, cy)))

            if neighbors:
                # FIX: np.random.choice fails on lists of non-uniform tuples.
                # Instead, we choose a random index.
                idx = self.np_random.integers(len(neighbors))
                direction, (nx, ny) = neighbors[idx]

                if direction == 'n':
                    maze[cy][cx]['n'] = True
                    maze[ny][nx]['s'] = True
                elif direction == 's':
                    maze[cy][cx]['s'] = True
                    maze[ny][nx]['n'] = True
                elif direction == 'e':
                    maze[cy][cx]['e'] = True
                    maze[ny][nx]['w'] = True
                elif direction == 'w':
                    maze[cy][cx]['w'] = True
                    maze[ny][nx]['e'] = True
                
                maze[ny][nx]['visited'] = True
                stack.append((nx, ny))
            else:
                stack.pop()
        return maze

    def _place_mines(self):
        solution_path = self._a_star_path(self.start_pos, self.exit_pos)
        if solution_path is None: solution_path = []
        
        banned_positions = set(solution_path)
        banned_positions.add(self.start_pos)
        banned_positions.add(self.exit_pos)

        available_positions = []
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if (x, y) not in banned_positions:
                    available_positions.append((x, y))
        
        if not available_positions:
            return []
            
        num_to_place = min(self.num_mines, len(available_positions))
        
        indices = self.np_random.choice(len(available_positions), num_to_place, replace=False)
        return [available_positions[i] for i in indices]

    def _is_valid_move(self, from_pos, to_pos):
        fx, fy = from_pos
        tx, ty = to_pos
        if not (0 <= tx < self.GRID_WIDTH and 0 <= ty < self.GRID_HEIGHT):
            return False
        if tx > fx: return self.maze[fy][fx]['e'] # Moving right
        if tx < fx: return self.maze[fy][fx]['w'] # Moving left
        if ty > fy: return self.maze[fy][fx]['s'] # Moving down
        if ty < fy: return self.maze[fy][fx]['n'] # Moving up
        return False

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _get_min_manhattan_dist(self, pos, targets):
        if not targets: return None
        return min(self._manhattan_distance(pos, t) for t in targets)
    
    def _get_neighbors(self, pos):
        x, y = pos
        neighbors = []
        if self.maze[y][x]['n']: neighbors.append((x, y - 1))
        if self.maze[y][x]['s']: neighbors.append((x, y + 1))
        if self.maze[y][x]['w']: neighbors.append((x - 1, y))
        if self.maze[y][x]['e']: neighbors.append((x + 1, y))
        return neighbors

    def _a_star_path(self, start, goal):
        pq = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}

        while pq:
            _, current = heapq.heappop(pq)
            if current == goal:
                break
            
            for neighbor in self._get_neighbors(current):
                new_cost = cost_so_far[current] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self._manhattan_distance(goal, neighbor)
                    heapq.heappush(pq, (priority, neighbor))
                    came_from[neighbor] = current
        else: # No path found
            return None
        
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from.get(current)
        return path[::-1]

    def _a_star_dist(self, start, goal):
        path = self._a_star_path(start, goal)
        return len(path) - 1 if path is not None else float('inf')

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    # This block is for demonstration and manual play.
    # It will not be executed by the verification script.
    
    # To run with a visible window, we need to unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset(seed=42)
    done = False
    
    pygame.display.set_caption("Minefield Maze")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    while not done:
        # --- Event Handling ---
        action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break
            
            # This is turn-based, so we only register an action on keydown
            if event.type == pygame.KEYDOWN:
                current_action = [0, 0, 0] # Reset action for this turn
                if event.key == pygame.K_UP:
                    current_action[0] = 1
                elif event.key == pygame.K_DOWN:
                    current_action[0] = 2
                elif event.key == pygame.K_LEFT:
                    current_action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    current_action[0] = 4
                elif event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset(seed=42)
                    print("Game Reset!")
                    continue

                # Only step if a movement key was pressed
                if current_action[0] != 0:
                    obs, reward, terminated, truncated, info = env.step(current_action)
                    print(f"Action: {current_action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")
                    if terminated:
                        action_taken = True # To trigger re-render
                        print("Game Over!")

        # --- Rendering ---
        frame = env._get_observation()
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()

        if env.game_over:
            pygame.time.wait(2000)
            obs, info = env.reset(seed=43) # Use a different seed for a new maze
            print("New Game Started...")


    env.close()