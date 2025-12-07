
# Generated: 2025-08-27T16:57:01.453564
# Source Brief: brief_01378.md
# Brief Index: 1378

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import collections
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to move your robot one square at a time."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a robot through a deadly maze to find the exit. You only have 25 moves, so plan your path carefully!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and Grid Dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.CELL_SIZE = 40
        self.GRID_WIDTH = self.SCREEN_WIDTH // self.CELL_SIZE
        self.GRID_HEIGHT = self.SCREEN_HEIGHT // self.CELL_SIZE
        
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
        self.font_ui = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_msg = pygame.font.SysFont("Arial", 48, bold=True)
        
        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_ROBOT = (50, 150, 255)
        self.COLOR_EXIT = (50, 255, 150)
        self.COLOR_PIT = (255, 80, 80)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_WIN = (100, 255, 150)
        self.COLOR_LOSE = (255, 100, 100)
        
        # Game state variables (initialized in reset)
        self.robot_pos = None
        self.exit_pos = None
        self.pits = None
        self.steps_remaining = None
        self.steps_taken = None
        self.score = None
        self.game_over = None
        self.particles = []
        self.win_message = ""
        
        # Initialize state variables
        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def _generate_maze(self):
        """Generates a new maze layout, ensuring a path from start to exit exists."""
        max_retries = 100
        for _ in range(max_retries):
            # Place robot
            self.robot_pos = (
                self.np_random.integers(0, self.GRID_WIDTH),
                self.np_random.integers(0, self.GRID_HEIGHT)
            )

            # Place exit, ensuring it's not on the robot
            while True:
                self.exit_pos = (
                    self.np_random.integers(0, self.GRID_WIDTH),
                    self.np_random.integers(0, self.GRID_HEIGHT)
                )
                if self.exit_pos != self.robot_pos:
                    break
            
            # Generate pits
            self.pits = set()
            num_pits = self.np_random.integers(15, 30)
            
            # Define a safe zone around the robot's start
            safe_zone = set()
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    safe_zone.add((self.robot_pos[0] + dx, self.robot_pos[1] + dy))

            possible_pit_locations = [
                (x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)
                if (x, y) != self.robot_pos and (x, y) != self.exit_pos and (x, y) not in safe_zone
            ]
            
            if len(possible_pit_locations) > num_pits:
                pit_indices = self.np_random.choice(len(possible_pit_locations), num_pits, replace=False)
                self.pits = {possible_pit_locations[i] for i in pit_indices}

            # Check for a valid path
            if self._is_path_available():
                return
        
        # If we failed to generate a valid maze, create a trivial one
        self.pits = set()

    def _is_path_available(self):
        """Uses Breadth-First Search (BFS) to check if a path exists."""
        queue = collections.deque([self.robot_pos])
        visited = {self.robot_pos}
        
        while queue:
            x, y = queue.popleft()
            
            if (x, y) == self.exit_pos:
                return True
                
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                    neighbor = (nx, ny)
                    if neighbor not in visited and neighbor not in self.pits:
                        visited.add(neighbor)
                        queue.append(neighbor)
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps_taken = 0
        self.steps_remaining = 25
        self.score = 0
        self.game_over = False
        self.particles = []
        self.win_message = ""
        
        self._generate_maze()
        
        return self._get_observation(), self._get_info()
    
    def _spawn_particles(self, pos):
        # spawn particle effect on movement
        px, py = (pos[0] + 0.5) * self.CELL_SIZE, (pos[1] + 0.5) * self.CELL_SIZE
        for _ in range(10):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            radius = self.np_random.uniform(2, 5)
            self.particles.append({'pos': [px, py], 'vel': vel, 'life': life, 'max_life': life, 'radius': radius})

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held = action[1] == 1  # Boolean - not used
        # shift_held = action[2] == 1  # Boolean - not used
        
        old_pos = self.robot_pos
        new_pos = list(self.robot_pos)

        if movement == 1: # Up
            new_pos[1] -= 1
        elif movement == 2: # Down
            new_pos[1] += 1
        elif movement == 3: # Left
            new_pos[0] -= 1
        elif movement == 4: # Right
            new_pos[0] += 1
        
        # Check boundaries
        if 0 <= new_pos[0] < self.GRID_WIDTH and 0 <= new_pos[1] < self.GRID_HEIGHT:
            self.robot_pos = tuple(new_pos)
            if old_pos != self.robot_pos:
                # SFX: Robot move sound
                self._spawn_particles(old_pos)
        
        self.steps_taken += 1
        self.steps_remaining -= 1
        
        reward = -0.1  # Cost for taking a step
        terminated = False

        if self.robot_pos == self.exit_pos:
            reward += 10
            terminated = True
            self.win_message = "VICTORY!"
            # SFX: Win sound
        elif self.robot_pos in self.pits:
            reward -= 10
            terminated = True
            self.win_message = "GAME OVER"
            # SFX: Fall/Explosion sound
        
        if self.steps_remaining <= 0 and not terminated:
            terminated = True
            self.win_message = "OUT OF MOVES"
            # SFX: Time out sound

        self.score += reward
        if terminated:
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] > 0:
                active_particles.append(p)
                
                # Fade effect
                alpha = 255 * (p['life'] / p['max_life'])
                current_color = (*self.COLOR_ROBOT, alpha)
                
                # Draw particle
                temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, current_color, (p['radius'], p['radius']), p['radius'])
                self.screen.blit(temp_surf, (int(p['pos'][0] - p['radius']), int(p['pos'][1] - p['radius'])))
        self.particles = active_particles

    def _render_game(self):
        # Draw grid lines
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT), 1)
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y), 1)
            
        # Draw pits
        for x, y in self.pits:
            rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_PIT, rect, border_radius=4)
            
        # Draw exit
        ex, ey = self.exit_pos
        exit_rect = pygame.Rect(ex * self.CELL_SIZE, ey * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect, border_radius=4)
        
        # Draw robot
        rx, ry = self.robot_pos
        robot_rect = pygame.Rect(rx * self.CELL_SIZE + 4, ry * self.CELL_SIZE + 4, self.CELL_SIZE - 8, self.CELL_SIZE - 8)
        # Glow effect
        glow_rect = pygame.Rect(rx * self.CELL_SIZE, ry * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        glow_surf = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*self.COLOR_ROBOT, 60), glow_surf.get_rect(), border_radius=10)
        self.screen.blit(glow_surf, glow_rect.topleft)
        
        pygame.draw.rect(self.screen, self.COLOR_ROBOT, robot_rect, border_radius=6)

        # Draw particles
        self._update_and_draw_particles()

    def _render_ui(self):
        # Display steps remaining
        steps_text = f"Moves: {self.steps_remaining}"
        text_surf = self.font_ui.render(steps_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 10))
        
        # Display game over message
        if self.game_over:
            color = self.COLOR_WIN if self.win_message == "VICTORY!" else self.COLOR_LOSE
            msg_surf = self.font_msg.render(self.win_message, True, color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            # Text shadow
            shadow_surf = self.font_msg.render(self.win_message, True, (0,0,0))
            self.screen.blit(shadow_surf, (msg_rect.x + 3, msg_rect.y + 3))

            self.screen.blit(msg_surf, msg_rect)

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
            "steps": self.steps_taken,
            "steps_remaining": self.steps_remaining,
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
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Override screen for display
    env.screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Grid Maze Robot")

    print(GameEnv.game_description)
    print(GameEnv.user_guide)

    action = np.array([0, 0, 0]) # Start with a no-op
    
    while not done:
        # Human input
        human_action = 0 # No-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    human_action = 1
                elif event.key == pygame.K_DOWN:
                    human_action = 2
                elif event.key == pygame.K_LEFT:
                    human_action = 3
                elif event.key == pygame.K_RIGHT:
                    human_action = 4
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    print("--- Game Reset ---")
                elif event.key == pygame.K_q: # Quit
                    done = True

        if human_action != 0:
            action = np.array([human_action, 0, 0])
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated
            print(f"Step: {info['steps']}, Action: {action[0]}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Done: {done}")
        
        # Render the observation to the display
        rendered_obs = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(rendered_obs)
        env.screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS

    print(f"Final Score: {info['score']:.2f}")
    pygame.time.wait(2000) # Wait 2 seconds before closing
    env.close()