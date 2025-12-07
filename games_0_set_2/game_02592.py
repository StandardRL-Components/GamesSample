import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to navigate the maze. "
        "Find the green exit before time runs out and avoid the red ghosts."
    )

    # Short, user-facing description of the game
    game_description = (
        "Navigate a procedurally generated, visually haunting maze while evading "
        "patrolling ghosts to find the exit before time runs out."
    )

    # Frames only advance when an action is received
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Critical: Action and Observation spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup (headless)
        pygame.init()
        pygame.font.init()
        self.screen_width, self.screen_height = 640, 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        # Game constants
        self.cell_size = 20
        self.maze_width = self.screen_width // self.cell_size
        self.maze_height = self.screen_height // self.cell_size
        self.max_steps_per_stage = 1000
        self.num_ghosts = 10
        self.num_stages = 3

        # Visuals
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        self.COLOR_BG = (5, 5, 16)
        self.COLOR_WALL = (0, 136, 255)
        self.COLOR_PLAYER = (0, 255, 255)
        self.COLOR_GHOST = (255, 0, 68)
        self.COLOR_EXIT = (0, 255, 136)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TIMER = (255, 60, 60)
        
        # Game state (persists across stages until full game over)
        self.stage = 1
        self.cumulative_score = 0
        self.full_game_over = True # Will be set to False in reset

        # Initialize all other state variables
        self.maze = None
        self.player_pos = None
        self.exit_pos = None
        self.ghosts = []
        self.steps = 0
        self.time_remaining = 0
        self.score = 0
        self.episode_terminated = False
        self.base_ghost_move_interval = 25

        # Initialize state for the first time
        # This is done here to ensure the environment is ready upon creation
        # The first call to reset() will re-initialize everything anyway
        # A seed is needed for the first call to _find_patrol_path
        self.reset(seed=0)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Handle full game reset vs. new stage
        if self.full_game_over:
            self.stage = 1
            self.cumulative_score = 0

        # Reset episode-specific state
        self.steps = 0
        self.score = 0
        self.episode_terminated = False
        self.full_game_over = False
        self.time_remaining = self.max_steps_per_stage

        # Generate maze and entities
        self._generate_maze()
        self.player_pos = np.array([1, 1], dtype=int)
        self.exit_pos = np.array([self.maze_width - 2, self.maze_height - 2], dtype=int)
        self._initialize_ghosts()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.episode_terminated:
            # If the episode is already over, subsequent steps do nothing
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.01  # Penalty for taking a step
        self.steps += 1
        self.time_remaining -= 1

        # 1. Unpack and process action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._move_player(movement)

        # 2. Update game state (ghosts)
        self._update_ghosts()

        # 3. Check for termination conditions
        terminated = False
        if tuple(self.player_pos) == tuple(self.exit_pos):
            # Reached exit
            reward += 5.0 + 50.0 # Exit reward + stage completion reward
            self.score += 55
            self.cumulative_score += 55
            self.stage += 1
            if self.stage > self.num_stages:
                self.full_game_over = True # Won the whole game
            terminated = True
        elif any(np.array_equal(self.player_pos, g['pos']) for g in self.ghosts):
            # Collided with a ghost
            reward -= 10.0
            self.score -= 10
            self.full_game_over = True # Lost the whole game
            terminated = True
        elif self.time_remaining <= 0 or self.steps >= self.max_steps_per_stage:
            # Out of time or steps
            self.full_game_over = True # Lost the whole game
            terminated = True
            
        self.episode_terminated = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _move_player(self, movement):
        # 0=none, 1=up, 2=down, 3=left, 4=right
        dx, dy = 0, 0
        if movement == 1: dy = -1
        elif movement == 2: dy = 1
        elif movement == 3: dx = -1
        elif movement == 4: dx = 1
        
        new_pos = self.player_pos + np.array([dx, dy])
        
        if self.maze[new_pos[1], new_pos[0]] == 0: # 0 is path, 1 is wall
            self.player_pos = new_pos

    def _update_ghosts(self):
        # Ghosts move at a speed determined by the current stage
        ghost_speed_modifier = (self.stage - 1) * 0.05
        # The brief says "increases by 0.05", which is a bit ambiguous for turn-based.
        # We interpret it as moving more frequently.
        # Let's say speed = 1 / interval. speed' = speed + 0.05. interval' = 1/(1/interval + 0.05)
        base_speed = 1.0 / self.base_ghost_move_interval
        current_speed = base_speed + ghost_speed_modifier
        move_interval = max(1, int(1.0 / current_speed))

        if self.steps % move_interval == 0:
            for ghost in self.ghosts:
                # Move along pre-defined patrol path
                ghost['path_idx'] += ghost['direction']
                if not (0 <= ghost['path_idx'] < len(ghost['path'])):
                    ghost['direction'] *= -1
                    ghost['path_idx'] += 2 * ghost['direction']
                
                # Ensure path index is valid after logic
                ghost['path_idx'] = np.clip(ghost['path_idx'], 0, len(ghost['path']) - 1)
                ghost['pos'] = ghost['path'][ghost['path_idx']]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "cumulative_score": self.cumulative_score,
            "steps": self.steps,
            "stage": self.stage,
            "time_remaining": self.time_remaining,
        }

    # --- Maze Generation and Entity Placement ---

    def _generate_maze(self):
        # Use a 2D array: 1 for wall, 0 for path
        self.maze = np.ones((self.maze_height, self.maze_width), dtype=int)
        
        # Recursive backtracking algorithm
        def carve_passages(cx, cy):
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            random.shuffle(directions)
            for dx, dy in directions:
                nx, ny = cx + dx * 2, cy + dy * 2
                if 0 < nx < self.maze_width and 0 < ny < self.maze_height:
                    if self.maze[ny, nx] == 1:
                        self.maze[ny - dy, nx - dx] = 0
                        self.maze[ny, nx] = 0
                        carve_passages(nx, ny)

        # Start carving from a random odd-numbered cell
        start_x, start_y = (1, 1)
        self.maze[start_y, start_x] = 0
        carve_passages(start_x, start_y)

    def _initialize_ghosts(self):
        self.ghosts = []
        occupied_cells = {tuple(self.player_pos), tuple(self.exit_pos)}
        
        for _ in range(self.num_ghosts):
            path = self._find_patrol_path(occupied_cells)
            if path:
                start_pos = path[0]
                ghost = {
                    'pos': start_pos,
                    'path': path,
                    'path_idx': 0,
                    'direction': 1,
                    'flicker': random.uniform(0.8, 1.0),
                    'flicker_speed': random.uniform(0.05, 0.15)
                }
                self.ghosts.append(ghost)
                # Add path cells to occupied to avoid clustering ghosts
                for p in path:
                    occupied_cells.add(tuple(p))

    def _find_patrol_path(self, occupied_cells):
        # Find a long, simple path (corridor) for a ghost to patrol
        for _ in range(50): # Try 50 times to find a good path
            start_x = self.np_random.integers(1, self.maze_width - 1)
            start_y = self.np_random.integers(1, self.maze_height - 1)
            start_node = (start_x, start_y)
            
            if self.maze[start_y, start_x] == 0 and start_node not in occupied_cells:
                # Use BFS to find the longest simple path from this point
                q = [(start_node, [start_node])]
                visited = {start_node}
                longest_path = [start_node]

                while q:
                    (vx, vy), path = q.pop(0)
                    
                    is_dead_end = True
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = vx + dx, vy + dy
                        # FIX: Added bounds check to prevent IndexError
                        if (0 <= ny < self.maze_height and 0 <= nx < self.maze_width and
                                self.maze[ny, nx] == 0 and (nx, ny) not in visited):
                            is_dead_end = False
                            visited.add((nx, ny))
                            new_path = path + [(nx, ny)]
                            q.append(((nx, ny), new_path))
                    
                    if is_dead_end and len(path) > len(longest_path):
                        longest_path = path

                if len(longest_path) > 4: # Only accept reasonably long paths
                    return [np.array(p) for p in longest_path]
        return []

    # --- Rendering ---

    def _render_game(self):
        # Render maze walls
        for y in range(self.maze_height):
            for x in range(self.maze_width):
                if self.maze[y, x] == 1:
                    self._draw_neon_rect(
                        (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size),
                        self.COLOR_WALL, 3
                    )
        
        # Render exit
        self._draw_neon_rect(
            (self.exit_pos[0] * self.cell_size, self.exit_pos[1] * self.cell_size, self.cell_size, self.cell_size),
            self.COLOR_EXIT, 5
        )

        # Render player
        self._draw_neon_rect(
            (self.player_pos[0] * self.cell_size, self.player_pos[1] * self.cell_size, self.cell_size, self.cell_size),
            self.COLOR_PLAYER, 4
        )
        
        # Render ghosts
        for ghost in self.ghosts:
            # Update flicker for unsettling effect
            ghost['flicker'] += ghost['flicker_speed']
            if ghost['flicker'] > 1.0 or ghost['flicker'] < 0.5:
                ghost['flicker_speed'] *= -1
            
            center_x = int(ghost['pos'][0] * self.cell_size + self.cell_size / 2)
            center_y = int(ghost['pos'][1] * self.cell_size + self.cell_size / 2)
            radius = int(self.cell_size / 2.5 * ghost['flicker'])
            self._draw_neon_circle(center_x, center_y, radius, self.COLOR_GHOST, 4)

    def _draw_neon_rect(self, rect, color, glow_layers):
        x, y, w, h = rect
        base_color = np.array(color)
        for i in range(glow_layers, 0, -1):
            glow_alpha = 150 / (i**1.5)
            glow_color = (*(base_color * (1 - i/glow_layers*0.5)).astype(int), glow_alpha)
            padding = i * 2
            pygame.draw.rect(
                self.screen,
                glow_color,
                (x - padding, y - padding, w + 2 * padding, h + 2 * padding),
                border_radius=int(padding/2)
            )
        pygame.draw.rect(self.screen, color, rect, border_radius=2)

    def _draw_neon_circle(self, x, y, radius, color, glow_layers):
        # gfxdraw doesn't support alpha, so we use a surface
        max_radius = radius + glow_layers * 2
        temp_surf = pygame.Surface((max_radius * 2, max_radius * 2), pygame.SRCALPHA)
        
        for i in range(glow_layers, 0, -1):
            glow_alpha = 100 / (i**1.2)
            glow_color = (*color, glow_alpha)
            pygame.gfxdraw.filled_circle(
                temp_surf, max_radius, max_radius, radius + i * 2, glow_color
            )
            pygame.gfxdraw.aacircle(
                temp_surf, max_radius, max_radius, radius + i * 2, glow_color
            )
        
        pygame.gfxdraw.filled_circle(temp_surf, max_radius, max_radius, radius, color)
        pygame.gfxdraw.aacircle(temp_surf, max_radius, max_radius, radius, color)

        self.screen.blit(temp_surf, (x - max_radius, y - max_radius))

    def _render_ui(self):
        # Stage Text
        stage_text = self.font_ui.render(f"Stage: {self.stage}/{self.num_stages}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (10, 10))

        # Time Text
        time_text = self.font_ui.render(f"Time: {self.time_remaining}", True, self.COLOR_TIMER)
        self.screen.blit(time_text, (self.screen_width - time_text.get_width() - 10, 10))

        # Game Over Text
        if self.episode_terminated and self.full_game_over:
            if self.stage > self.num_stages:
                msg = "YOU ESCAPED"
                color = self.COLOR_EXIT
            else:
                msg = "GAME OVER"
                color = self.COLOR_GHOST
            
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            game_over_text = self.font_game_over.render(msg, True, color)
            text_rect = game_over_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(game_over_text, text_rect)

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    # To do so, you need a display. Comment out the os.environ line at the top.
    # os.environ.pop("SDL_VIDEODRIVER", None) # Uncomment to run with display
    
    env = GameEnv()
    obs, info = env.reset()
    
    # Verify environment is loadable
    try:
        env_check_instance = GameEnv()
        print("✓ Environment created successfully.")
        obs, info = env_check_instance.reset()
        print("✓ Reset successful.")
        action = env_check_instance.action_space.sample()
        obs, reward, terminated, truncated, info = env_check_instance.step(action)
        print("✓ Step successful.")
        print("✓ Basic API checks passed.")
    except Exception as e:
        print(f"✗ Error during basic API checks: {e}")


    # Manual play block (requires a display)
    if "SDL_VIDEODRIVER" not in os.environ:
        running = True
        terminated = False
        
        # Create a window to display the game
        pygame.display.set_caption(env.game_description)
        display_screen = pygame.display.set_mode((env.screen_width, env.screen_height))
        
        action = env.action_space.sample()
        action[0] = 0 # Start with no-op
        action[1] = 0
        action[2] = 0
        
        print(env.user_guide)
        
        while running:
            # --- Human Input ---
            movement = 0 # No-op by default
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        obs, info = env.reset()
                        terminated = False
                    if terminated: continue
                    
                    if event.key == pygame.K_UP: movement = 1
                    elif event.key == pygame.K_DOWN: movement = 2
                    elif event.key == pygame.K_LEFT: movement = 3
                    elif event.key == pygame.K_RIGHT: movement = 4
                    
                    # Since auto_advance is False, we step on each key press
                    if movement != 0:
                        action[0] = movement
                        obs, reward, terminated, truncated, info = env.step(action)
                        print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")
                        if terminated:
                            print(f"Episode finished. Cumulative Score: {info['cumulative_score']}. Press 'R' to reset.")


            # --- Rendering ---
            # The observation is the rendered frame
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            env.clock.tick(30) # Limit FPS for human play

    env.close()