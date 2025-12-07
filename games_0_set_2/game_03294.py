
# Generated: 2025-08-27T22:56:15.063653
# Source Brief: brief_03294.md
# Brief Index: 3294

        
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
    """
    A Gymnasium environment for a grid-based maze puzzle game.

    The player controls a slime character and must navigate a procedurally
    generated maze to reach the exit within a limited number of steps.
    The environment is designed with a focus on visual quality and a clear,
    turn-based gameplay experience.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to move the slime one "
        "square at a time."
    )

    # Short, user-facing description of the game
    game_description = (
        "Guide the slime through the maze to the glowing exit. You have a "
        "limited number of moves, so plan your path carefully!"
    )

    # Frames only advance when an action is received
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        """
        Initializes the game environment.
        """
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 16, 10
        self.CELL_SIZE = self.SCREEN_HEIGHT // self.GRID_HEIGHT
        self.MAX_STEPS = 30
        self.BONUS_STEPS_THRESHOLD = 25

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_WALL = (40, 50, 80)
        self.COLOR_FLOOR = (30, 40, 60)
        self.COLOR_SLIME = (80, 220, 120)
        self.COLOR_SLIME_HIGHLIGHT = (200, 255, 220)
        self.COLOR_EXIT = (255, 200, 0)
        self.COLOR_TEXT = (220, 230, 240)
        self.COLOR_PARTICLE = (150, 255, 180)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)

        # --- Internal State ---
        # These are initialized in reset()
        self.maze = None
        self.slime_pos = None
        self.exit_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.particles = []

        # Initialize state variables
        self.reset()
        
        # Run validation check
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.
        """
        super().reset(seed=seed)

        self.maze = self._generate_maze(self.GRID_WIDTH, self.GRID_HEIGHT)
        self.slime_pos = np.array([0, 0])
        self.exit_pos = np.array([self.GRID_WIDTH - 1, self.GRID_HEIGHT - 1])

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        """
        Advances the environment by one timestep.
        """
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        self.steps += 1
        reward = -1  # Cost for taking a step

        # --- Movement Logic ---
        move_map = {1: [0, -1], 2: [0, 1], 3: [-1, 0], 4: [1, 0]} # up, down, left, right
        
        if movement in move_map:
            direction = np.array(move_map[movement])
            next_pos = self.slime_pos + direction
            
            # Check if move is valid (within bounds and not into a wall)
            if self._is_move_valid(self.slime_pos, direction):
                self._spawn_particles(self.slime_pos)
                self.slime_pos = next_pos
                # sfx: slime_move.wav
            else:
                # sfx: bump_wall.wav
                pass # Invalid move, slime stays put but step is consumed

        # --- Check for Termination ---
        terminated = False
        if np.array_equal(self.slime_pos, self.exit_pos):
            reward += 100  # Reached the exit
            self.win_message = "SUCCESS!"
            if self.steps < self.BONUS_STEPS_THRESHOLD:
                reward += 50 # Bonus for speed
                self.win_message = f"FAST! {self.steps} STEPS"
            # sfx: win_level.wav
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            self.win_message = "OUT OF MOVES"
            # sfx: lose_level.wav
            terminated = True
            self.game_over = True

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info(),
        )

    def _generate_maze(self, width, height):
        """
        Generates a maze using randomized Depth-First Search.
        Returns a grid where each cell is a dictionary of its walls.
        """
        maze = [[{'N': True, 'S': True, 'E': True, 'W': True} for _ in range(height)] for _ in range(width)]
        
        # Use the gym-provided RNG for reproducibility
        rng = self.np_random
        
        stack = deque()
        visited = set()

        start_cell = (0, 0)
        stack.append(start_cell)
        visited.add(start_cell)

        while stack:
            cx, cy = stack[-1]
            
            neighbors = []
            if cx > 0 and (cx - 1, cy) not in visited: neighbors.append((cx - 1, cy))
            if cx < width - 1 and (cx + 1, cy) not in visited: neighbors.append((cx + 1, cy))
            if cy > 0 and (cx, cy - 1) not in visited: neighbors.append((cx, cy - 1))
            if cy < height - 1 and (cx, cy + 1) not in visited: neighbors.append((cx, cy + 1))

            if neighbors:
                nx, ny = rng.choice(neighbors)
                
                if nx == cx - 1: # Move West
                    maze[cx][cy]['W'] = False
                    maze[nx][ny]['E'] = False
                elif nx == cx + 1: # Move East
                    maze[cx][cy]['E'] = False
                    maze[nx][ny]['W'] = False
                elif ny == cy - 1: # Move North
                    maze[cx][cy]['N'] = False
                    maze[nx][ny]['S'] = False
                elif ny == cy + 1: # Move South
                    maze[cx][cy]['S'] = False
                    maze[nx][ny]['N'] = False
                
                visited.add((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop()
        
        return maze

    def _is_move_valid(self, pos, direction):
        """Checks if a move from pos in direction is blocked by a wall."""
        x, y = pos
        dx, dy = direction
        
        if dx == 1 and not self.maze[x][y]['E']: return True
        if dx == -1 and not self.maze[x][y]['W']: return True
        if dy == 1 and not self.maze[x][y]['S']: return True
        if dy == -1 and not self.maze[x][y]['N']: return True
        
        return False

    def _get_info(self):
        """Returns a dictionary with auxiliary diagnostic information."""
        return {"score": self.score, "steps": self.steps}

    def _get_observation(self):
        """Renders the current game state to a numpy array."""
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders all game elements (maze, slime, exit, particles)."""
        grid_offset_x = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        
        self._render_maze(grid_offset_x)
        self._render_exit(grid_offset_x)
        self._update_and_render_particles(grid_offset_x)
        self._render_slime(grid_offset_x)

    def _render_maze(self, offset_x):
        """Renders the maze walls."""
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                px, py = x * self.CELL_SIZE + offset_x, y * self.CELL_SIZE
                
                # Draw floor
                pygame.draw.rect(self.screen, self.COLOR_FLOOR, (px, py, self.CELL_SIZE, self.CELL_SIZE))

                # Draw walls
                if self.maze[x][y]['N']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py), (px + self.CELL_SIZE, py), 2)
                if self.maze[x][y]['S']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py + self.CELL_SIZE), (px + self.CELL_SIZE, py + self.CELL_SIZE), 2)
                if self.maze[x][y]['W']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py), (px, py + self.CELL_SIZE), 2)
                if self.maze[x][y]['E']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px + self.CELL_SIZE, py), (px + self.CELL_SIZE, py + self.CELL_SIZE), 2)

    def _render_exit(self, offset_x):
        """Renders the glowing exit portal."""
        ex, ey = self.exit_pos
        px = ex * self.CELL_SIZE + self.CELL_SIZE // 2 + offset_x
        py = ey * self.CELL_SIZE + self.CELL_SIZE // 2

        # Glowing effect
        t = pygame.time.get_ticks() * 0.001
        for i in range(5):
            alpha = 150 - i * 30
            radius = self.CELL_SIZE // 3 + i * 2 + math.sin(t * 3 + i) * 2
            color = (*self.COLOR_EXIT, max(0, alpha))
            pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), int(radius), color)

    def _render_slime(self, offset_x):
        """Renders the animated slime character."""
        sx, sy = self.slime_pos
        px = sx * self.CELL_SIZE + self.CELL_SIZE // 2 + offset_x
        py = sy * self.CELL_SIZE + self.CELL_SIZE // 2

        # Pulsing animation
        t = pygame.time.get_ticks() * 0.005
        base_radius = self.CELL_SIZE * 0.35
        pulse = math.sin(t) * self.CELL_SIZE * 0.03
        radius = int(base_radius + pulse)

        # Body
        pygame.gfxdraw.filled_circle(self.screen, px, py, radius, self.COLOR_SLIME)
        pygame.gfxdraw.aacircle(self.screen, px, py, radius, self.COLOR_SLIME)

        # Highlight
        highlight_radius = int(radius * 0.4)
        highlight_offset_x = int(radius * 0.3)
        highlight_offset_y = -int(radius * 0.3)
        pygame.gfxdraw.filled_circle(self.screen, px + highlight_offset_x, py + highlight_offset_y, highlight_radius, self.COLOR_SLIME_HIGHLIGHT)

    def _spawn_particles(self, grid_pos):
        """Spawns particles at a given grid position."""
        px = grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        px += (self.SCREEN_WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        py = grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2

        for _ in range(10):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({'pos': [px, py], 'vel': vel, 'life': 15, 'radius': random.randint(2, 4)})
    
    def _update_and_render_particles(self, offset_x):
        """Updates particle physics and renders them."""
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] -= 0.1
            if p['life'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / 15))
                color = (*self.COLOR_PARTICLE, alpha)
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color)

    def _render_ui(self):
        """Renders the user interface elements."""
        # Step counter
        steps_left = self.MAX_STEPS - self.steps
        text_surf = self.font_large.render(f"Moves: {steps_left}", True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (15, 10))

        # Game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg_surf = self.font_large.render(self.win_message, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def close(self):
        """Cleans up Pygame resources."""
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode='rgb_array')
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    terminated = False
    
    print(env.user_guide)

    while running:
        action = np.array([0, 0, 0])  # Default to no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if terminated: # If game over, any key resets
                    obs, info = env.reset()
                    terminated = False
                    continue

                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Manual reset
                    obs, info = env.reset()
                    terminated = False
                
                # Space and Shift can be mapped here if needed
                # if event.key == pygame.K_SPACE: action[1] = 1
                # if event.key == pygame.K_LSHIFT: action[2] = 1

                if action[0] != 0: # If a move was made
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Step: {info['steps']}, Score: {info['score']:.0f}, Reward: {reward:.0f}, Terminated: {terminated}")

        # Render the observation to the display screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
    env.close()