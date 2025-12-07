
# Generated: 2025-08-27T14:15:26.600953
# Source Brief: brief_00624.md
# Brief Index: 624

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the robot. Space to restart the current maze."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a robot through a procedurally generated grid maze to reach the green exit. Each success increases the number of obstacles. You have a limited number of moves per maze."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Colors and Visuals ---
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_OBSTACLE = (80, 80, 90)
    COLOR_ROBOT = (255, 0, 100)
    COLOR_ROBOT_GLOW = (255, 0, 100)
    COLOR_EXIT = (0, 255, 150)
    COLOR_TRAIL = (255, 0, 100)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SUCCESS = (0, 255, 150)
    COLOR_TEXT_FAIL = (255, 80, 80)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        # --- Game Configuration ---
        self.grid_size = 10
        self.max_moves_per_game = 50
        self.max_episode_steps = 500
        self.initial_obstacle_count = 5
        self.max_obstacles = 40

        # --- Grid Rendering ---
        self.grid_area_size = 400
        self.cell_size = self.grid_area_size // self.grid_size
        self.grid_offset_x = (self.screen.get_width() - self.grid_area_size) // 2
        self.grid_offset_y = (self.screen.get_height() - self.grid_area_size) // 2

        # Initialize state variables
        self.obstacle_count = self.initial_obstacle_count
        self.score = 0
        self.total_steps = 0
        self.robot_pos = [0, 0]
        self.exit_pos = [0, 0]
        self.obstacles = []
        self.moves_made = 0
        self.game_state = "playing" # "playing", "won", "lost_obstacle", "lost_moves"
        self.robot_trail = deque(maxlen=5)
        self.particles = []
        
        self.reset()
        
        self.validate_implementation()
    
    def _start_new_game(self):
        """Initializes a new maze, ensuring a path exists."""
        while True:
            # Reset game-specific state
            self.robot_pos = [0, 0]
            self.exit_pos = [self.grid_size - 1, self.grid_size - 1]
            
            # Generate potential obstacle positions
            all_cells = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
            all_cells.remove(tuple(self.robot_pos))
            all_cells.remove(tuple(self.exit_pos))
            random.shuffle(all_cells)
            
            self.obstacles = all_cells[:self.obstacle_count]
            
            # Verify path exists
            if self._is_path_available(self.robot_pos, self.exit_pos, self.obstacles):
                break
        
        self.moves_made = 0
        self.game_state = "playing"
        self.robot_trail.clear()
        self.particles.clear()

    def _is_path_available(self, start, end, obstacles):
        """Checks for a path using Breadth-First Search."""
        q = deque([start])
        visited = {tuple(start)}
        obstacle_set = {tuple(o) for o in obstacles}

        while q:
            x, y = q.popleft()

            if (x, y) == tuple(end):
                return True

            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if (nx, ny) not in visited and (nx, ny) not in obstacle_set:
                        visited.add((nx, ny))
                        q.append([nx, ny])
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset episode-level state
        self.total_steps = 0
        self.score = 0
        self.obstacle_count = self.initial_obstacle_count
        
        # Start the first game of the episode
        self._start_new_game()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        reward = 0
        terminated = False

        if self.game_state != "playing": # If game is already won/lost, action does nothing
             # This allows win/loss animations to play out until reset is called
            pass
        elif space_held:
            # Restart action
            reward = -1.0  # Penalty for giving up
            self._start_new_game()
        elif movement > 0:
            # Movement action
            self.moves_made += 1
            reward = -0.1  # Cost per move
            
            self.robot_trail.append(self.robot_pos[:])
            
            dx, dy = 0, 0
            if movement == 1: dy = -1  # Up
            elif movement == 2: dy = 1   # Down
            elif movement == 3: dx = -1  # Left
            elif movement == 4: dx = 1   # Right

            next_pos = [self.robot_pos[0] + dx, self.robot_pos[1] + dy]

            # Check boundaries
            if not (0 <= next_pos[0] < self.grid_size and 0 <= next_pos[1] < self.grid_size):
                pass # Hit a wall, no movement
            # Check obstacles
            elif tuple(next_pos) in [tuple(o) for o in self.obstacles]:
                self.game_state = "lost_obstacle"
                reward += -5.0
                self._create_particles(self.robot_pos, self.COLOR_ROBOT, 30)
                # Sound: Player_Hit_Obstacle.wav
            else:
                self.robot_pos = next_pos
                # Sound: Player_Move.wav

            # Check for win condition
            if self.robot_pos == self.exit_pos:
                self.game_state = "won"
                reward += 10.0
                self.obstacle_count = min(self.max_obstacles, self.obstacle_count + 1)
                self._create_particles(self.exit_pos, self.COLOR_EXIT, 50)
                # Sound: Level_Win.wav

        # Check for termination conditions
        if self.game_state in ["won", "lost_obstacle"]:
            terminated = True
        
        if self.moves_made >= self.max_moves_per_game and self.game_state == "playing":
            self.game_state = "lost_moves"
            self._create_particles(self.robot_pos, self.COLOR_TEXT_FAIL, 20)
            terminated = True
            # Sound: Out_Of_Time.wav

        self.total_steps += 1
        if self.total_steps >= self.max_episode_steps:
            terminated = True

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _create_particles(self, pos_grid, color, count):
        px, py = self._grid_to_pixel(pos_grid)
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = random.randint(20, 40)
            self.particles.append([px, py, vx, vy, lifetime, color])

    def _update_particles(self):
        new_particles = []
        for p in self.particles:
            p[0] += p[2]  # x += vx
            p[1] += p[3]  # y += vy
            p[4] -= 1     # lifetime--
            if p[4] > 0:
                new_particles.append(p)
        self.particles = new_particles

    def _grid_to_pixel(self, grid_pos):
        x = self.grid_offset_x + grid_pos[0] * self.cell_size + self.cell_size // 2
        y = self.grid_offset_y + grid_pos[1] * self.cell_size + self.cell_size // 2
        return x, y

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_game()
        self._update_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for i in range(self.grid_size + 1):
            # Vertical
            pygame.draw.line(self.screen, self.COLOR_GRID, 
                             (self.grid_offset_x + i * self.cell_size, self.grid_offset_y), 
                             (self.grid_offset_x + i * self.cell_size, self.grid_offset_y + self.grid_area_size))
            # Horizontal
            pygame.draw.line(self.screen, self.COLOR_GRID, 
                             (self.grid_offset_x, self.grid_offset_y + i * self.cell_size), 
                             (self.grid_offset_x + self.grid_area_size, self.grid_offset_y + i * self.cell_size))

        # Draw obstacles
        for obs in self.obstacles:
            rect = pygame.Rect(self.grid_offset_x + obs[0] * self.cell_size,
                               self.grid_offset_y + obs[1] * self.cell_size,
                               self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect)

        # Draw exit
        rect = pygame.Rect(self.grid_offset_x + self.exit_pos[0] * self.cell_size,
                           self.grid_offset_y + self.exit_pos[1] * self.cell_size,
                           self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, rect)

        # Draw robot trail
        for i, pos in enumerate(self.robot_trail):
            alpha = int(255 * (i + 1) / (len(self.robot_trail) + 1) * 0.4)
            trail_surf = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            trail_surf.fill((*self.COLOR_TRAIL, alpha))
            self.screen.blit(trail_surf, (self.grid_offset_x + pos[0] * self.cell_size,
                                          self.grid_offset_y + pos[1] * self.cell_size))

        # Draw robot
        if self.game_state not in ["lost_obstacle"]:
            rx_p, ry_p = self._grid_to_pixel(self.robot_pos)
            size = self.cell_size * 0.7
            glow_size = size * 2.5
            
            # Glow effect
            glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.COLOR_ROBOT_GLOW, 50), (glow_size // 2, glow_size // 2), glow_size // 2)
            self.screen.blit(glow_surf, (rx_p - glow_size // 2, ry_p - glow_size // 2))

            # Robot body
            robot_rect = pygame.Rect(rx_p - size // 2, ry_p - size // 2, size, size)
            pygame.draw.rect(self.screen, self.COLOR_ROBOT, robot_rect, border_radius=3)
        
        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p[4] / 40.0))))
            color = (*p[5], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), 2, color)

    def _render_ui(self):
        ui_x = 20
        ui_y = 20
        line_height = 25

        # Moves Left
        moves_left = self.max_moves_per_game - self.moves_made
        moves_text = self.font_main.render(f"Moves Left: {moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (ui_x, ui_y))

        # Obstacles
        obs_text = self.font_main.render(f"Obstacles:  {self.obstacle_count}", True, self.COLOR_TEXT)
        self.screen.blit(obs_text, (ui_x, ui_y + line_height))

        # Score
        score_text = self.font_main.render(f"Score:      {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (ui_x, ui_y + 2 * line_height))
        
        # Game Over Messages
        if self.game_state != "playing":
            msg = ""
            color = self.COLOR_TEXT
            if self.game_state == "won":
                msg = "SUCCESS"
                color = self.COLOR_TEXT_SUCCESS
            elif self.game_state == "lost_obstacle":
                msg = "GAME OVER"
                color = self.COLOR_TEXT_FAIL
            elif self.game_state == "lost_moves":
                msg = "OUT OF MOVES"
                color = self.COLOR_TEXT_FAIL

            if msg:
                text_surf = self.font_large.render(msg, True, color)
                text_rect = text_surf.get_rect(center=(self.screen.get_width() / 2, self.screen.get_height() / 2))
                self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.total_steps,
            "moves_made": self.moves_made,
            "obstacle_count": self.obstacle_count,
            "game_state": self.game_state,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Setup a window to display the game
    pygame.display.set_caption("Grid Maze Robot")
    screen = pygame.display.set_mode((640, 400))
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    print("\n" + "="*30)
    print(f"Game: {env.game_description}")
    print(f"Controls: {env.user_guide}")
    print("="*30 + "\n")

    while not done:
        # --- Human Input ---
        movement = 0 # no-op
        space = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Manual reset
                    obs, info = env.reset()
                    print("--- Environment Reset ---")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        
        # Only step if an action is taken
        if movement > 0 or space > 0:
            action = [movement, space, 0] # Shift is unused
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")

            if terminated:
                print("--- Game Over! Press 'R' to play again. ---")

        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate for human play

    env.close()