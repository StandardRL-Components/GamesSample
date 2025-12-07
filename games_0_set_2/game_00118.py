
# Generated: 2025-08-27T12:39:13.280090
# Source Brief: brief_00118.md
# Brief Index: 118

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move your robot one square at a time. Avoid the red lasers!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A turn-based puzzle game. Guide your robot through a maze of rotating lasers to reach the green exit. Each move counts!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
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
        
        # --- Game Constants ---
        self.SCREEN_W, self.SCREEN_H = 640, 400
        self.GRID_W, self.GRID_H = 20, 20
        self.CELL_SIZE = self.SCREEN_H // self.GRID_H
        self.GRID_PIXEL_W = self.GRID_W * self.CELL_SIZE
        self.GRID_PIXEL_H = self.GRID_H * self.CELL_SIZE
        self.OFFSET_X = (self.SCREEN_W - self.GRID_PIXEL_W) // 2

        # --- Visuals ---
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_GRID = (40, 45, 65)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (0, 150, 255, 50)
        self.COLOR_EXIT = (0, 255, 120)
        self.COLOR_LASER_SOURCE = (255, 100, 0)
        self.COLOR_LASER_BEAM = (255, 20, 20)
        self.COLOR_LASER_GLOW = (255, 20, 20, 100)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_OVERLAY = (15, 18, 32, 200)

        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Verdana", 40, bold=True)

        # --- Game State ---
        self.episode_count = 0
        self.base_laser_speed_deg = 10.0
        
        self.robot_pos = [0, 0]
        self.exit_pos = [0, 0]
        self.lasers = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.termination_reason = ""
        
        # Initialize state variables
        self.reset()

        # Run validation check
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.episode_count += 1
        
        # Increase difficulty every 500 episodes
        difficulty_tier = self.episode_count // 500
        current_max_speed = self.base_laser_speed_deg + 5 * difficulty_tier

        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.termination_reason = ""
        
        # Define the maze layout
        self.robot_pos = [1, self.GRID_H // 2]
        self.exit_pos = [self.GRID_W - 2, self.GRID_H // 2]
        
        self.lasers = []
        laser_positions = [
            (5, 5), (5, 14),
            (10, 2), (10, 10), (10, 17),
            (15, 5), (15, 14)
        ]
        
        for pos in laser_positions:
            self.lasers.append({
                'pos': pos,
                'angle': self.np_random.uniform(0, 360),
                'speed': self.np_random.uniform(current_max_speed * 0.75, current_max_speed * 1.25)
            })

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Update Game Logic ---
        # 1. Move Robot
        if movement == 1:  # Up
            self.robot_pos[1] -= 1
        elif movement == 2:  # Down
            self.robot_pos[1] += 1
        elif movement == 3:  # Left
            self.robot_pos[0] -= 1
        elif movement == 4:  # Right
            self.robot_pos[0] += 1
        
        # Clamp robot to grid boundaries
        self.robot_pos[0] = np.clip(self.robot_pos[0], 0, self.GRID_W - 1)
        self.robot_pos[1] = np.clip(self.robot_pos[1], 0, self.GRID_H - 1)

        # 2. Update Lasers
        for laser in self.lasers:
            laser['angle'] = (laser['angle'] + laser['speed']) % 360

        # 3. Calculate Reward and Termination
        self.steps += 1
        reward = -0.1  # Penalty for taking a step
        terminated = False

        if self._check_laser_collision():
            reward = -50.0
            self.game_over = True
            terminated = True
            self.termination_reason = "Hit by a laser!"
            # sound: player_zap.wav
        elif self.robot_pos[0] == self.exit_pos[0] and self.robot_pos[1] == self.exit_pos[1]:
            reward = 100.0
            self.score += 100
            self.game_over = True
            terminated = True
            self.termination_reason = "You escaped!"
            # sound: victory_chime.wav
        elif self.steps >= 250:
            reward = -10.0 # Small penalty for timing out
            self.game_over = True
            terminated = True
            self.termination_reason = "Out of time!"
            # sound: failure_buzzer.wav
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _grid_to_pixel(self, grid_pos, center=False):
        px = self.OFFSET_X + grid_pos[0] * self.CELL_SIZE
        py = grid_pos[1] * self.CELL_SIZE
        if center:
            px += self.CELL_SIZE // 2
            py += self.CELL_SIZE // 2
        return int(px), int(py)

    def _check_laser_collision(self):
        robot_rect_px = pygame.Rect(
            self._grid_to_pixel(self.robot_pos),
            (self.CELL_SIZE, self.CELL_SIZE)
        )
        for laser in self.lasers:
            source_px = self._grid_to_pixel(laser['pos'], center=True)
            angle_rad = math.radians(laser['angle'])
            # Calculate a point very far away to ensure the line spans the grid
            end_px = (
                source_px[0] + self.SCREEN_W * 2 * math.cos(angle_rad),
                source_px[1] + self.SCREEN_W * 2 * math.sin(angle_rad)
            )
            # clipline returns the clipped line segment if it intersects, or () if not
            if robot_rect_px.clipline(source_px, end_px):
                return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "robot_pos": self.robot_pos,
            "exit_pos": self.exit_pos,
        }

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_W + 1):
            start_pos = (self.OFFSET_X + x * self.CELL_SIZE, 0)
            end_pos = (self.OFFSET_X + x * self.CELL_SIZE, self.GRID_PIXEL_H)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)
        for y in range(self.GRID_H + 1):
            start_pos = (self.OFFSET_X, y * self.CELL_SIZE)
            end_pos = (self.OFFSET_X + self.GRID_PIXEL_W, y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)

        # Draw exit
        exit_px = self._grid_to_pixel(self.exit_pos)
        exit_rect = pygame.Rect(exit_px, (self.CELL_SIZE, self.CELL_SIZE))
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)

        # Draw lasers
        for laser in self.lasers:
            source_px = self._grid_to_pixel(laser['pos'], center=True)
            angle_rad = math.radians(laser['angle'])
            end_px = (
                source_px[0] + self.SCREEN_W * 2 * math.cos(angle_rad),
                source_px[1] + self.SCREEN_W * 2 * math.sin(angle_rad)
            )
            
            # Glow effect
            pygame.draw.aaline(self.screen, self.COLOR_LASER_GLOW, source_px, end_px, blend=1)
            # Core beam
            pygame.draw.aaline(self.screen, self.COLOR_LASER_BEAM, source_px, end_px)
            
            # Source orb
            pygame.gfxdraw.filled_circle(self.screen, source_px[0], source_px[1], 5, self.COLOR_LASER_SOURCE)
            pygame.gfxdraw.aacircle(self.screen, source_px[0], source_px[1], 5, self.COLOR_LASER_SOURCE)

        # Draw robot
        robot_px = self._grid_to_pixel(self.robot_pos)
        robot_rect = pygame.Rect(robot_px, (self.CELL_SIZE, self.CELL_SIZE))
        
        # Glow effect
        glow_rect = robot_rect.inflate(self.CELL_SIZE * 0.5, self.CELL_SIZE * 0.5)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_PLAYER_GLOW, glow_surf.get_rect(), border_radius=5)
        self.screen.blit(glow_surf, glow_rect.topleft)

        # Main body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, robot_rect, border_radius=3)
        # Inner detail
        pygame.draw.rect(self.screen, self.COLOR_BG, robot_rect.inflate(-6, -6), border_radius=2)


    def _render_ui(self):
        # UI Text
        steps_text = self.font_ui.render(f"STEPS: {self.steps}/250", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (10, 10))
        
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_W - 10, 10))
        self.screen.blit(score_text, score_rect)

        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_W, self.SCREEN_H), pygame.SRCALPHA)
            overlay.fill(self.COLOR_OVERLAY)
            self.screen.blit(overlay, (0, 0))

            end_text = self.font_game_over.render(self.termination_reason, True, self.COLOR_TEXT)
            end_rect = end_text.get_rect(center=(self.SCREEN_W // 2, self.SCREEN_H // 2))
            self.screen.blit(end_text, end_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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


# Example of how to run the environment
if __name__ == '__main__':
    # For interactive play
    import sys

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup interactive window
    pygame.display.set_caption("Laser Grid Maze")
    screen = pygame.display.set_mode((env.SCREEN_W, env.SCREEN_H))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = np.array([0, 0, 0])  # Default to no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                
                # We only process one key press to generate a single action for the step
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset on 'r' key
                    print("Resetting environment...")
                    obs, info = env.reset()
                    continue

                # Take a step in the environment
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")

                if terminated or truncated:
                    print("Episode finished. Press 'R' to reset.")


        # Get the observation from the environment (which is the rendered frame)
        frame = env._get_observation()
        # Pygame uses (width, height), numpy uses (height, width)
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit interactive frame rate

    env.close()
    pygame.quit()
    sys.exit()