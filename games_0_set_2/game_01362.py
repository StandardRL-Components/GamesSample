import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set headless mode for Pygame
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move. Avoid the ghosts and reach the green exit before time runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated maze, evading spectral foes, to reach the exit before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAZE_W, self.MAZE_H = 20, 12 # Maze dimensions in cells
        self.CELL_W = self.WIDTH // self.MAZE_W
        self.CELL_H = self.HEIGHT // self.MAZE_H
        self.NUM_GHOSTS = 3
        self.MAX_STEPS = 600

        # --- Colors ---
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_WALL = (50, 50, 60)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_PLAYER_GLOW = (255, 255, 0, 50)
        self.COLOR_GHOST = (255, 0, 0)
        self.COLOR_EXIT = (0, 255, 0)
        self.COLOR_TEXT = (255, 255, 255)
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # Etc...        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0
        self.maze = {}
        self.player_pos = (0, 0)
        self.exit_pos = (0, 0)
        self.ghosts = []
        self.rng = None
        
        # Initialize state variables
        self.reset()
    
    def _generate_maze(self):
        maze = {(x, y): set() for x in range(self.MAZE_W) for y in range(self.MAZE_H)}
        stack = [(0, 0)]
        visited = set([(0, 0)])

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy, direction in [(0, -1, 'N'), (0, 1, 'S'), (-1, 0, 'W'), (1, 0, 'E')]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.MAZE_W and 0 <= ny < self.MAZE_H and (nx, ny) not in visited:
                    neighbors.append((nx, ny, direction))
            
            if neighbors:
                # FIX: np.random.choice converts mixed-type lists (int, str) to all strings.
                # This caused a KeyError because maze keys are (int, int) tuples.
                # Instead, we randomly choose an index from the list to preserve types.
                chosen_neighbor = neighbors[self.rng.integers(len(neighbors))]
                nx, ny, direction = chosen_neighbor
                
                opposite_dir = {'N': 'S', 'S': 'N', 'W': 'E', 'E': 'W'}
                maze[(cx, cy)].add(direction)
                maze[(nx, ny)].add(opposite_dir[direction])
                
                visited.add((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop()
        
        return maze

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()
        
        # Initialize all game state, for example:
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_STEPS
        
        self.maze = self._generate_maze()
        self.player_pos = (0, 0)
        self.exit_pos = (self.MAZE_W - 1, self.MAZE_H - 1)
        
        self.ghosts = []
        possible_starts = [(x,y) for x in range(self.MAZE_W) for y in range(self.MAZE_H) if math.dist((x,y), self.player_pos) > 5]
        
        if len(possible_starts) < self.NUM_GHOSTS:
            possible_starts = [(x,y) for x in range(self.MAZE_W) for y in range(self.MAZE_H) if (x,y) != self.player_pos]

        start_indices = self.rng.choice(len(possible_starts), self.NUM_GHOSTS, replace=False)

        for i in range(self.NUM_GHOSTS):
            pos = possible_starts[start_indices[i]]
            vel = tuple(self.rng.choice([(1,0), (-1,0), (0,1), (0,-1)]))
            self.ghosts.append({"pos": pos, "vel": vel})
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # Update game logic
        self.steps += 1
        self.time_remaining -= 1
        
        # 1. Update Player Position
        px, py = self.player_pos
        if movement == 1 and 'N' in self.maze.get((px, py), set()): # Up
            self.player_pos = (px, py - 1)
        elif movement == 2 and 'S' in self.maze.get((px, py), set()): # Down
            self.player_pos = (px, py + 1)
        elif movement == 3 and 'W' in self.maze.get((px, py), set()): # Left
            self.player_pos = (px - 1, py)
        elif movement == 4 and 'E' in self.maze.get((px, py), set()): # Right
            self.player_pos = (px + 1, py)
        
        # 2. Update Ghost Positions
        for ghost in self.ghosts:
            gx, gy = ghost["pos"]
            gvx, gvy = ghost["vel"]

            vel_to_dir = {(0,-1): 'N', (0,1): 'S', (-1,0): 'W', (1,0): 'E'}
            current_dir = vel_to_dir.get((gvx, gvy))

            if current_dir and current_dir in self.maze.get((gx, gy), set()):
                ghost["pos"] = (gx + gvx, gy + gvy)
            else: # Hit a wall, change direction
                possible_dirs = list(self.maze.get((gx, gy), set()))
                if possible_dirs:
                    new_dir_char = self.rng.choice(possible_dirs)
                    dir_to_vel = {'N': (0,-1), 'S': (0,1), 'W': (-1,0), 'E': (1,0)}
                    ghost["vel"] = dir_to_vel[new_dir_char]

        terminated = self._check_termination()
        reward = self._calculate_reward()

        self.score += reward
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _calculate_reward(self):
        reward = 0
        # Survival reward
        reward += 0.01

        # Proximity penalty
        for ghost in self.ghosts:
            if math.dist(self.player_pos, ghost["pos"]) < 1.5:
                reward -= 0.1
        
        # Terminal rewards
        if self.player_pos == self.exit_pos:
            reward += 100
        elif self.time_remaining <= 0:
            reward -= 10
        else:
            for ghost in self.ghosts:
                if self.player_pos == ghost["pos"]:
                    reward -= 100
                    break
        return reward
    
    def _check_termination(self):
        if self.player_pos == self.exit_pos:
            self.game_over = True
            return True
        if self.time_remaining <= 0:
            self.game_over = True
            return True
        for ghost in self.ghosts:
            if self.player_pos == ghost["pos"]:
                self.game_over = True
                return True
        return False

    def _to_pixels(self, pos):
        x, y = pos
        px = x * self.CELL_W + self.CELL_W // 2
        py = y * self.CELL_H + self.CELL_H // 2
        return int(px), int(py)

    def _render_game(self):
        # Draw maze walls
        for y in range(self.MAZE_H):
            for x in range(self.MAZE_W):
                px, py = x * self.CELL_W, y * self.CELL_H
                openings = self.maze.get((x, y), set())
                
                if 'N' not in openings:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py), (px + self.CELL_W, py), 3)
                if 'S' not in openings:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py + self.CELL_H), (px + self.CELL_W, py + self.CELL_H), 3)
                if 'W' not in openings:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py), (px, py + self.CELL_H), 3)
                if 'E' not in openings:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px + self.CELL_W, py), (px + self.CELL_W, py + self.CELL_H), 3)

        # Draw exit
        exit_px, exit_py = self._to_pixels(self.exit_pos)
        pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
        size = int(self.CELL_W * 0.4 + pulse * self.CELL_W * 0.3)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, (exit_px - size//2, exit_py - size//2, size, size))

        # Draw ghosts
        ghost_radius = int(min(self.CELL_W, self.CELL_H) * 0.35)
        for ghost in self.ghosts:
            gx, gy = self._to_pixels(ghost["pos"])
            alpha = self.rng.integers(100, 200)
            
            temp_surf = pygame.Surface((ghost_radius*2, ghost_radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, ghost_radius, ghost_radius, ghost_radius, (*self.COLOR_GHOST, alpha))
            pygame.gfxdraw.aacircle(temp_surf, ghost_radius, ghost_radius, ghost_radius, (*self.COLOR_GHOST, alpha))
            self.screen.blit(temp_surf, (gx - ghost_radius, gy - ghost_radius))

        # Draw player
        player_px, player_py = self._to_pixels(self.player_pos)
        player_radius = int(min(self.CELL_W, self.CELL_H) * 0.4)
        
        glow_radius = int(player_radius * 1.8)
        temp_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(temp_surf, glow_radius, glow_radius, glow_radius, self.COLOR_PLAYER_GLOW)
        self.screen.blit(temp_surf, (player_px - glow_radius, player_py - glow_radius))

        pygame.gfxdraw.filled_circle(self.screen, player_px, player_py, player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_px, player_py, player_radius, self.COLOR_PLAYER)

    def _render_ui(self):
        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        time_text = self.font_ui.render(f"Time: {self.time_remaining}", True, self.COLOR_TEXT)
        time_rect = time_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(time_text, time_rect)

        if self.game_over:
            msg = ""
            if self.player_pos == self.exit_pos:
                msg = "YOU ESCAPED!"
            elif self.time_remaining <= 0:
                msg = "TIME'S UP!"
            else:
                msg = "SPECTRALIZED!"
            
            over_text = self.font_game_over.render(msg, True, self.COLOR_TEXT)
            over_rect = over_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            
            bg_rect = over_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, bg_rect)
            self.screen.blit(over_text, over_rect)

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
        }

    def close(self):
        pygame.quit()

# Example of how to run the environment for human play
if __name__ == "__main__":
    # To display the game, comment out the os.environ line at the top of the file
    # and uncomment the line below.
    # os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    # --- Human Play Setup ---
    # Create a displayable screen if not running headless
    try:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Maze Evader")
        human_play = True
        print(env.user_guide)
    except pygame.error:
        human_play = False
        print("Pygame display not available. Running in headless mode.")

    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    running = True
    while running:
        if human_play:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    
                    action.fill(0) # Reset previous action
                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
                    elif event.key == pygame.K_r: # Reset on 'r'
                        obs, info = env.reset()
                        continue
                    
                    # Step the environment only on a key press for turn-based play
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    if done:
                        print(f"Game Over! Final Score: {info['score']:.1f}, Steps: {info['steps']}")
                        
                        # Render final frame
                        frame = np.transpose(obs, (1, 0, 2))
                        surf = pygame.surfarray.make_surface(frame)
                        screen.blit(surf, (0, 0))
                        pygame.display.flip()
                        
                        pygame.time.wait(2000)
                        obs, info = env.reset()

            # Render the observation to the display
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            env.clock.tick(30) # Limit FPS for human play
        
        else: # Headless random agent loop
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                print(f"Episode finished. Score: {info['score']:.1f}, Steps: {info['steps']}")
                obs, info = env.reset()
                # break after a while to prevent infinite loops
                if info['steps'] > 5000: 
                    running = False

    env.close()