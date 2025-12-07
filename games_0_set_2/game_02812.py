
# Generated: 2025-08-27T21:30:57.018161
# Source Brief: brief_02812.md
# Brief Index: 2812

        
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
    """
    An arcade puzzle game where the player navigates a pixel through a grid,
    dodging obstacles to reach a goal. The game consists of three stages of
    increasing difficulty, with a time limit (step limit) for each stage.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your pixel. Avoid red obstacles and reach the yellow goal."
    )

    # Short, user-facing description of the game
    game_description = (
        "Navigate a glowing pixel through a grid of deadly obstacles. Reach the goal in three "
        "increasingly difficult stages to win, all while racing against a step limit."
    )

    # The game state is static until an action is received.
    auto_advance = False

    class _Obstacle:
        """Helper class to manage individual obstacle state and movement."""
        def __init__(self, start_pos, pattern, speed, grid_size, np_random):
            self.pos = np.array(start_pos, dtype=float)
            self.start_pos = np.array(start_pos, dtype=float)
            self.pattern = pattern
            self.speed = speed
            self.grid_size = grid_size
            self.direction = 1
            self.angle = np_random.uniform(0, 2 * math.pi)
            self.radius = np_random.uniform(15, 30)

        def update(self):
            """Updates the obstacle's position based on its pattern."""
            if self.pattern == 'horizontal':
                self.pos[0] += self.speed * self.direction
                if not (0 < self.pos[0] < self.grid_size - 1):
                    self.direction *= -1
                    self.pos[0] = np.clip(self.pos[0], 0, self.grid_size - 1)
            elif self.pattern == 'vertical':
                self.pos[1] += self.speed * self.direction
                if not (0 < self.pos[1] < self.grid_size - 1):
                    self.direction *= -1
                    self.pos[1] = np.clip(self.pos[1], 0, self.grid_size - 1)
            elif self.pattern == 'circular':
                self.angle += self.speed / 10.0
                offset_x = self.radius * math.cos(self.angle)
                offset_y = self.radius * math.sin(self.angle)
                self.pos[0] = self.start_pos[0] + offset_x
                self.pos[1] = self.start_pos[1] + offset_y
            elif self.pattern == 'diagonal':
                self.pos[0] += self.speed * self.direction * 0.707
                self.pos[1] += self.speed * self.direction * 0.707
                if not (0 < self.pos[0] < self.grid_size - 1) or not (0 < self.pos[1] < self.grid_size - 1):
                    self.direction *= -1
            
            self.pos[0] = np.clip(self.pos[0], 0, self.grid_size - 1)
            self.pos[1] = np.clip(self.pos[1], 0, self.grid_size - 1)

        def get_int_pos(self):
            return (int(self.pos[0]), int(self.pos[1]))

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()

        # Game constants
        self.GRID_SIZE = 100
        self.MAX_STAGES = 3
        self.STEPS_PER_STAGE = 1200  # 60 seconds * 20 steps/sec
        self.NUM_OBSTACLES = 5

        # Visual constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GAME_AREA_SIZE = 400
        self.CELL_SIZE = self.GAME_AREA_SIZE // self.GRID_SIZE
        self.X_OFFSET = (self.SCREEN_WIDTH - self.GAME_AREA_SIZE) // 2

        self.COLOR_BG = (0, 0, 0)
        self.COLOR_GRID = (25, 25, 25)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_GLOW = (0, 100, 50)
        self.COLOR_OBSTACLE = (255, 50, 50)
        self.COLOR_GOAL = (255, 220, 0)
        self.COLOR_GOAL_GLOW = (120, 100, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_GAMEOVER = (200, 0, 0)
        self.COLOR_WIN = (0, 200, 0)

        self.font_s = pygame.font.SysFont("monospace", 16)
        self.font_m = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_l = pygame.font.SysFont("monospace", 48, bold=True)

        # Initialize state variables
        self.player_pos = [0, 0]
        self.goal_pos = [0, 0]
        self.obstacles = []
        self.current_stage = 1
        self.steps_remaining = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.np_random = None

        self.reset()
        
        # self.validate_implementation() # Uncomment for debugging

    def _setup_stage(self, stage_num):
        """Initializes the game state for a specific stage."""
        self.current_stage = stage_num
        self.steps_remaining = self.STEPS_PER_STAGE

        self.player_pos = [2, self.GRID_SIZE // 2]
        self.goal_pos = [self.GRID_SIZE - 3, self.GRID_SIZE // 2]

        self.obstacles = []
        obstacle_speed = 0.45 + (stage_num * 0.05)
        patterns = ['horizontal', 'vertical', 'circular', 'diagonal']
        
        occupied_coords = {tuple(self.player_pos), tuple(self.goal_pos)}

        for _ in range(self.NUM_OBSTACLES + stage_num * 2): # More obstacles on later stages
            while True:
                start_x = self.np_random.integers(10, self.GRID_SIZE - 10)
                start_y = self.np_random.integers(10, self.GRID_SIZE - 10)
                if (start_x, start_y) not in occupied_coords and math.dist((start_x, start_y), self.player_pos) > 15:
                    occupied_coords.add((start_x, start_y))
                    break
            
            pattern = self.np_random.choice(patterns)
            obs = self._Obstacle([start_x, start_y], pattern, obstacle_speed, self.GRID_SIZE, self.np_random)
            self.obstacles.append(obs)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self._setup_stage(1)
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement = action[0]
        
        # --- Calculate reward based on pre-move state ---
        old_dist_to_goal = math.dist(self.player_pos, self.goal_pos)
        old_min_dist_to_obs = self._get_min_dist_to_obstacle()

        # --- Update game state ---
        self._update_player(movement)
        for obs in self.obstacles:
            obs.update()

        self.steps += 1
        self.steps_remaining -= 1
        
        # --- Calculate reward based on post-move state ---
        reward = self._calculate_reward(old_dist_to_goal, old_min_dist_to_obs)
        
        # --- Check for events (goal, collision, timeout) ---
        terminated = False
        
        # Check goal collision
        if self.player_pos == self.goal_pos:
            # sfx: stage_clear.wav
            if self.current_stage < self.MAX_STAGES:
                reward += 10.0  # Stage completion bonus
                self.score += reward
                self._setup_stage(self.current_stage + 1)
            else:
                reward += 100.0  # Final win bonus
                self.score += reward
                self.game_won = True
                self.game_over = True
                terminated = True
        else:
            # Check obstacle collision
            player_tuple = tuple(self.player_pos)
            for obs in self.obstacles:
                if player_tuple == obs.get_int_pos():
                    # sfx: player_hit.wav
                    self.game_over = True
                    terminated = True
                    reward -= 20.0 # Collision penalty
                    break
        
        # Check timeout
        if self.steps_remaining <= 0 and not self.game_over:
            # sfx: timeout.wav
            self.game_over = True
            terminated = True

        if not self.game_over:
            self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self, movement):
        # sfx: move.wav (if movement != 0)
        if movement == 1:  # Up
            self.player_pos[1] -= 1
        elif movement == 2:  # Down
            self.player_pos[1] += 1
        elif movement == 3:  # Left
            self.player_pos[0] -= 1
        elif movement == 4:  # Right
            self.player_pos[0] += 1
        
        # Boundary checks
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.GRID_SIZE - 1)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.GRID_SIZE - 1)

    def _calculate_reward(self, old_dist_goal, old_dist_obs):
        reward = -0.1  # Time penalty

        new_dist_goal = math.dist(self.player_pos, self.goal_pos)
        if new_dist_goal < old_dist_goal:
            reward += 0.5  # Moved closer to goal
        
        new_dist_obs = self._get_min_dist_to_obstacle()
        if new_dist_obs < old_dist_obs and new_dist_obs < 10: # Only penalize getting close
             reward -= 0.5 # Moved closer to obstacle
        
        return reward

    def _get_min_dist_to_obstacle(self):
        if not self.obstacles:
            return float('inf')
        return min(math.dist(self.player_pos, obs.pos) for obs in self.obstacles)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            x = self.X_OFFSET + i * self.CELL_SIZE
            y = i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.GAME_AREA_SIZE))
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.X_OFFSET, y), (self.X_OFFSET + self.GAME_AREA_SIZE, y))

        # Helper to convert grid coords to screen coords
        def to_screen_coords(grid_pos):
            return (
                int(self.X_OFFSET + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2),
                int(grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2)
            )

        # Draw Goal
        goal_scr_pos = to_screen_coords(self.goal_pos)
        pygame.gfxdraw.filled_circle(self.screen, goal_scr_pos[0], goal_scr_pos[1], self.CELL_SIZE * 2, self.COLOR_GOAL_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, goal_scr_pos[0], goal_scr_pos[1], int(self.CELL_SIZE * 1.5), self.COLOR_GOAL)

        # Draw Obstacles
        for obs in self.obstacles:
            obs_scr_pos = to_screen_coords(obs.get_int_pos())
            obs_rect = pygame.Rect(obs_scr_pos[0] - self.CELL_SIZE // 2, obs_scr_pos[1] - self.CELL_SIZE // 2, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs_rect)

        # Draw Player
        player_scr_pos = to_screen_coords(self.player_pos)
        pygame.gfxdraw.filled_circle(self.screen, player_scr_pos[0], player_scr_pos[1], self.CELL_SIZE + 2, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, player_scr_pos[0], player_scr_pos[1], self.CELL_SIZE, self.COLOR_PLAYER)

    def _render_ui(self):
        # Stage Text
        stage_text = self.font_m.render(f"STAGE: {self.current_stage}/{self.MAX_STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (10, 10))

        # Timer/Steps Text
        time_text = self.font_m.render(f"STEPS LEFT: {self.steps_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))
        
        # Score Text
        score_text = self.font_m.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, ((self.SCREEN_WIDTH - score_text.get_width()) / 2, self.SCREEN_HEIGHT - 30))
        
        # Game Over / Win Message
        if self.game_over:
            if self.game_won:
                msg_text = self.font_l.render("YOU WIN!", True, self.COLOR_WIN)
            else:
                msg_text = self.font_l.render("GAME OVER", True, self.COLOR_GAMEOVER)
            
            # Create a semi-transparent overlay
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            text_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.current_stage,
            "steps_remaining": self.steps_remaining,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Use a dummy window to display the game
    pygame.display.set_caption("Pixel Grid Dodger")
    display_screen = pygame.display.set_mode((640, 400))
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        # --- Keyboard Controls for Human Player ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        current_action = np.array([movement, space_held, shift_held])
        
        # --- Step the environment ---
        # Since auto_advance is False, we only step when a key is pressed
        # To make it playable, we step continuously if a key is held down
        if movement != 0:
            obs, reward, terminated, truncated, info = env.step(current_action)
        
        # --- Render the game to the display window ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(15) # Control human play speed

    print(f"Game Over! Final Info: {info}")
    
    # Keep the final screen visible for a few seconds
    pygame.time.wait(3000)
    
    env.close()