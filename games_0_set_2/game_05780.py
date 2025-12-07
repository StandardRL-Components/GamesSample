
# Generated: 2025-08-28T06:04:34.518560
# Source Brief: brief_05780.md
# Brief Index: 5780

        
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
        "Controls: ←↑↓→ to move selected robot. Space to select next robot, Shift to select previous. Solve the puzzle before you run out of moves!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A turn-based puzzle game. Guide each robot to its matching goal by carefully planning your moves. Each move costs from a shared pool, so think ahead!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_WIDTH = 16
        self.GRID_HEIGHT = 10
        self.CELL_SIZE = 40
        self.MAX_EPISODE_STEPS = 1000
        
        # Colors
        self.COLOR_BG = (245, 245, 245)
        self.COLOR_GRID = (210, 210, 210)
        self.COLOR_TEXT = (50, 50, 50)
        self.COLOR_WALL = (100, 100, 100)
        self.COLOR_SELECT_PULSE = (0, 150, 255)
        self.ROBOT_COLORS = [
            (255, 87, 34),   # Bright Orange
            (3, 169, 244),    # Bright Blue
            (76, 175, 80),    # Bright Green
            (255, 235, 59),  # Bright Yellow
        ]
        self.GOAL_COLORS = [
            (239, 108, 0),    # Darker Orange
            (1, 87, 155),     # Darker Blue
            (27, 94, 32),     # Darker Green
            (245, 127, 23),   # Darker Yellow
        ]
        self.GOAL_FLASH_COLOR = (255, 255, 255)
        
        # Level Definitions
        self.LEVELS = [
            {
                "start_moves": 25,
                "robots": [(2, 2), (2, 7)],
                "goals": [(13, 7), (13, 2)],
                "walls": []
            },
            {
                "start_moves": 45,
                "robots": [(1, 1), (8, 8), (14, 1)],
                "goals": [(14, 8), (1, 8), (7, 1)],
                "walls": [(x, 4) for x in range(0, 7)] + [(x, 5) for x in range(9, 16)]
            },
            {
                "start_moves": 70,
                "robots": [(1, 1), (1, 8), (14, 1), (14, 8)],
                "goals": [(8, 5), (7, 5), (8, 4), (7, 4)],
                "walls": [(x, y) for x in range(16) for y in range(10) if (3 <= x <= 12 and (y == 2 or y == 7))] + \
                         [(x, y) for x in range(16) for y in range(10) if (x == 5 or x == 10) and (y < 3 or y > 6)]
            }
        ]
        
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
        try:
            self.font_ui = pygame.font.Font(pygame.font.get_default_font(), 20)
            self.font_feedback = pygame.font.Font(pygame.font.get_default_font(), 16)
        except IOError:
            self.font_ui = pygame.font.SysFont("arial", 20)
            self.font_feedback = pygame.font.SysFont("arial", 16)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.robots = []
        self.goals = []
        self.walls = []
        self.robots_on_goal = []
        self.moves_left = 0
        self.selected_robot_idx = 0
        self.goal_flash_timers = []
        self.feedback_message = ""
        self.feedback_timer = 0

        # self.validate_implementation() # For development
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Select and load a level
        level_idx = self.np_random.integers(0, len(self.LEVELS))
        level_data = self.LEVELS[level_idx]
        
        self.robots = [list(r) for r in level_data["robots"]]
        self.goals = [list(g) for g in level_data["goals"]]
        self.walls = [list(w) for w in level_data["walls"]]
        self.moves_left = level_data["start_moves"]
        
        num_robots = len(self.robots)
        self.robots_on_goal = [False] * num_robots
        self.goal_flash_timers = [0] * num_robots
        self.selected_robot_idx = 0
        self.feedback_message = "New Game Started!"
        self.feedback_timer = 60 # Show message for 60 steps

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = 0.0
        
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        action_taken = any([movement > 0, space_held, shift_held])
        if action_taken:
            reward -= 0.1 # Small cost for any action

        # --- Action Handling Priority: Selection > Movement ---
        
        # 1. Handle Robot Selection
        if space_held:
            # Select next available robot
            initial_idx = self.selected_robot_idx
            while True:
                self.selected_robot_idx = (self.selected_robot_idx + 1) % len(self.robots)
                if not self.robots_on_goal[self.selected_robot_idx] or self.selected_robot_idx == initial_idx:
                    break
            self.feedback_message = f"Selected Robot {self.selected_robot_idx + 1}"
            self.feedback_timer = 30
        
        elif shift_held:
            # Select previous available robot
            initial_idx = self.selected_robot_idx
            while True:
                self.selected_robot_idx = (self.selected_robot_idx - 1 + len(self.robots)) % len(self.robots)
                if not self.robots_on_goal[self.selected_robot_idx] or self.selected_robot_idx == initial_idx:
                    break
            self.feedback_message = f"Selected Robot {self.selected_robot_idx + 1}"
            self.feedback_timer = 30
        
        # 2. Handle Movement
        elif movement > 0:
            self.moves_left -= 1
            
            if self.robots_on_goal[self.selected_robot_idx]:
                self.feedback_message = "Robot is on goal and cannot move."
                self.feedback_timer = 30
            else:
                robot_pos = self.robots[self.selected_robot_idx]
                target_pos = list(robot_pos) # Create a copy

                if movement == 1: target_pos[1] -= 1 # Up
                elif movement == 2: target_pos[1] += 1 # Down
                elif movement == 3: target_pos[0] -= 1 # Left
                elif movement == 4: target_pos[0] += 1 # Right

                # --- Validate Move ---
                is_valid = True
                # Check bounds
                if not (0 <= target_pos[0] < self.GRID_WIDTH and 0 <= target_pos[1] < self.GRID_HEIGHT):
                    is_valid = False
                # Check wall collision
                if tuple(target_pos) in [tuple(w) for w in self.walls]:
                    is_valid = False
                # Check other robot collision
                for i, other_robot in enumerate(self.robots):
                    if i != self.selected_robot_idx and tuple(other_robot) == tuple(target_pos):
                        is_valid = False
                        break
                
                if is_valid:
                    # Execute move
                    self.robots[self.selected_robot_idx] = target_pos
                    # sfx: robot_move.wav
                    self.feedback_message = f"Robot {self.selected_robot_idx + 1} moved."
                    self.feedback_timer = 30
                    
                    # Check for goal achievement
                    if tuple(target_pos) == tuple(self.goals[self.selected_robot_idx]):
                        if not self.robots_on_goal[self.selected_robot_idx]:
                            self.robots_on_goal[self.selected_robot_idx] = True
                            reward += 10.0
                            self.goal_flash_timers[self.selected_robot_idx] = 15 # Flash for 15 steps
                            # sfx: goal_reached.wav
                            self.feedback_message = f"Robot {self.selected_robot_idx + 1} reached its goal!"
                            self.feedback_timer = 60
                else:
                    # sfx: invalid_move.wav
                    self.feedback_message = "Invalid Move!"
                    self.feedback_timer = 30

        # --- Update Game State & Check Termination ---
        self.score += reward
        
        win_condition = all(self.robots_on_goal)
        lose_condition = self.moves_left <= 0
        step_limit_reached = self.steps >= self.MAX_EPISODE_STEPS
        
        terminated = win_condition or lose_condition or step_limit_reached
        
        if terminated and not self.game_over:
            self.game_over = True
            if win_condition:
                reward += 50.0 # Victory bonus
                self.score += 50.0
                self.feedback_message = "Success! All robots on goals."
                # sfx: level_complete.wav
            elif lose_condition:
                self.feedback_message = "Out of moves! Game Over."
                # sfx: game_over.wav
            else:
                 self.feedback_message = "Step limit reached. Game Over."
            self.feedback_timer = 120

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "robots_on_goal": sum(self.robots_on_goal),
        }

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_WIDTH + 1):
            px = x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.SCREEN_HEIGHT), 1)
        for y in range(self.GRID_HEIGHT + 1):
            py = y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.SCREEN_WIDTH, py), 1)
            
        # Draw walls
        for wall in self.walls:
            rect = pygame.Rect(wall[0] * self.CELL_SIZE, wall[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
            
        # Draw goals
        for i, goal in enumerate(self.goals):
            rect = pygame.Rect(goal[0] * self.CELL_SIZE, goal[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            color = self.GOAL_COLORS[i]
            
            if self.goal_flash_timers[i] > 0:
                # Flash effect
                flash_alpha = (self.goal_flash_timers[i] / 15) * 255
                flash_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                flash_surface.fill((*self.GOAL_FLASH_COLOR, flash_alpha))
                self.screen.blit(flash_surface, rect.topleft)
                self.goal_flash_timers[i] -= 1
            
            pygame.draw.rect(self.screen, color, rect, border_radius=4)

        # Draw robots
        for i, robot in enumerate(self.robots):
            center_x = int((robot[0] + 0.5) * self.CELL_SIZE)
            center_y = int((robot[1] + 0.5) * self.CELL_SIZE)
            radius = int(self.CELL_SIZE * 0.35)
            
            # Selection highlight
            if i == self.selected_robot_idx and not self.robots_on_goal[i]:
                pulse_progress = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
                pulse_radius = radius + 4 + (pulse_progress * 4)
                pulse_alpha = 50 + (pulse_progress * 50)
                pygame.gfxdraw.aacircle(self.screen, center_x, center_y, int(pulse_radius), (*self.COLOR_SELECT_PULSE, int(pulse_alpha)))

            # Robot circle
            color = self.ROBOT_COLORS[i]
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
            
            # Draw a small number on the robot
            robot_num_text = self.font_feedback.render(str(i+1), True, self.COLOR_BG)
            text_rect = robot_num_text.get_rect(center=(center_x, center_y))
            self.screen.blit(robot_num_text, text_rect)
            
    def _render_ui(self):
        # Draw a semi-transparent panel at the top
        panel = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        panel.fill((255, 255, 255, 200))
        self.screen.blit(panel, (0, self.SCREEN_HEIGHT - 40))

        # Draw moves left
        moves_text = f"Moves Left: {self.moves_left}"
        self._draw_text(moves_text, self.font_ui, self.COLOR_TEXT, (10, self.SCREEN_HEIGHT - 30))

        # Draw score
        score_text = f"Score: {self.score:.1f}"
        self._draw_text(score_text, self.font_ui, self.COLOR_TEXT, (200, self.SCREEN_HEIGHT - 30))
        
        # Draw feedback message
        if self.feedback_timer > 0:
            alpha = min(255, int(255 * (self.feedback_timer / 30)))
            color = (*self.COLOR_TEXT, alpha)
            self._draw_text(self.feedback_message, self.font_ui, color, (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 30), center=True)
            self.feedback_timer -= 1
            
    def _draw_text(self, text, font, color, pos, center=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = (int(pos[0]), int(pos[1]))
        else:
            text_rect.topleft = (int(pos[0]), int(pos[1]))
        self.screen.blit(text_surface, text_rect)

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
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Robot Maze")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    terminated = False
    
    print(env.user_guide)
    
    while running:
        action = [0, 0, 0] # Default no-op action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    terminated = False
                    
        if not terminated:
            keys = pygame.key.get_pressed()
            
            # Map keys to MultiDiscrete action
            movement = 0
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            action = [movement, space_held, shift_held]
            
            # Step the environment only if an action is taken
            if any(action):
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")
        
        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(30) # Limit human play to 30 FPS

    env.close()