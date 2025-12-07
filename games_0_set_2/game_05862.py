
# Generated: 2025-08-28T06:19:12.448963
# Source Brief: brief_05862.md
# Brief Index: 5862

        
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
        "Controls: Arrow keys to move the worker. Push boxes (brown) onto targets (green)."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down puzzle game. Push all the boxes onto the target locations within the move limit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TILE_SIZE = 40
    GRID_WIDTH = 16
    GRID_HEIGHT = 10

    # Colors
    COLOR_BG = (26, 33, 41)
    COLOR_FLOOR = (43, 55, 69)
    COLOR_WALL = (69, 85, 102)
    COLOR_WALL_SHADOW = (55, 70, 85)
    COLOR_TARGET = (80, 158, 109)
    COLOR_TARGET_HIGHLIGHT = (100, 198, 136)
    COLOR_BOX = (156, 106, 84)
    COLOR_BOX_HIGHLIGHT = (181, 126, 102)
    COLOR_BOX_ON_TARGET = (130, 140, 95)
    COLOR_BOX_ON_TARGET_HIGHLIGHT = (155, 165, 115)
    COLOR_PLAYER = (255, 204, 0)
    COLOR_PLAYER_GLOW = (255, 204, 0, 50)
    COLOR_TEXT = (230, 230, 230)
    COLOR_WIN_TEXT = (130, 255, 160)
    COLOR_LOSE_TEXT = (255, 100, 100)
    
    LEVEL_LAYOUT = [
        "WWWWWWWWWWWWWWWW",
        "W T            W",
        "W W  P  B      W",
        "W W     W T    W",
        "W WWWWWWW W    W",
        "W W B B W W B  W",
        "W W     W W    W",
        "W W B B W   T  W",
        "W T     T      W",
        "WWWWWWWWWWWWWWWW",
    ]
    
    # Game parameters
    MAX_MOVES = 60
    WIN_BONUS_MOVES = 40

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("sans-serif", 50, bold=True)
        
        # State variables (initialized in reset)
        self.grid = []
        self.player_pos = None
        self.box_positions = None
        self.target_positions = None
        self.steps = 0
        self.moves = 0
        self.score = 0
        self.game_over = False
        self.win_condition_met = False
        
        # Initialize state
        self.reset()
        
        # Run validation check
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.moves = 0
        self.score = 0
        self.game_over = False
        self.win_condition_met = False
        
        self._parse_level()

        return self._get_observation(), self._get_info()
    
    def _parse_level(self):
        self.grid = [list(row) for row in self.LEVEL_LAYOUT]
        self.box_positions = set()
        self.target_positions = set()
        for r, row in enumerate(self.grid):
            for c, char in enumerate(row):
                if char == 'P':
                    self.player_pos = (r, c)
                    self.grid[r][c] = ' '
                elif char == 'B':
                    self.box_positions.add((r, c))
                    self.grid[r][c] = ' '
                elif char == 'T':
                    self.target_positions.add((r, c))
                    self.grid[r][c] = 'T' # Keep target marker on grid

    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        moved = False
        if movement > 0:
            dr = [-1, 1, 0, 0]
            dc = [0, 0, -1, 1]
            
            move_dr = dr[movement - 1]
            move_dc = dc[movement - 1]

            new_pos = (self.player_pos[0] + move_dr, self.player_pos[1] + move_dc)

            # Check if new position is a wall
            if self.grid[new_pos[0]][new_pos[1]] == 'W':
                pass # Can't move into a wall
            # Check if new position has a box
            elif new_pos in self.box_positions:
                box_new_pos = (new_pos[0] + move_dr, new_pos[1] + move_dc)
                # Check if box can be pushed
                if self.grid[box_new_pos[0]][box_new_pos[1]] != 'W' and box_new_pos not in self.box_positions:
                    # Successfully pushed a box
                    was_on_target = new_pos in self.target_positions
                    is_on_target = box_new_pos in self.target_positions

                    if is_on_target and not was_on_target:
                        reward += 1.0  # Box pushed onto a target
                    elif not is_on_target and was_on_target:
                        reward -= 1.0  # Box pushed off a target
                    
                    self.box_positions.remove(new_pos)
                    self.box_positions.add(box_new_pos)
                    self.player_pos = new_pos
                    moved = True
                    # sfx: box_push.wav
            else:
                # Normal move to an empty space
                self.player_pos = new_pos
                moved = True
                # sfx: footstep.wav

        if moved:
            self.moves += 1
            reward -= 0.1 # Cost per move

        self.score += reward
        terminated = self._check_termination()

        if terminated and not self.game_over:
            self.game_over = True
            if self.win_condition_met:
                # sfx: level_complete.wav
                completion_reward = 50.0
                if self.moves <= self.WIN_BONUS_MOVES:
                    completion_reward += 50.0 # Bonus for efficiency
                self.score += completion_reward
                reward += completion_reward
            else: # Lost by exceeding move limit
                # sfx: game_over.wav
                loss_penalty = -100.0
                self.score += loss_penalty
                reward += loss_penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _check_termination(self):
        # Win condition: all boxes are on targets
        if self.box_positions == self.target_positions:
            self.win_condition_met = True
            return True
        
        # Lose condition: exceeded max moves
        if self.moves >= self.MAX_MOVES:
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
            "moves": self.moves,
            "boxes_on_target": len(self.box_positions.intersection(self.target_positions))
        }

    def _render_game(self):
        shadow_offset = self.TILE_SIZE // 10
        
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                rect = pygame.Rect(c * self.TILE_SIZE, r * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                
                # Draw floor and walls
                if self.grid[r][c] == 'W':
                    shadow_rect = rect.copy()
                    shadow_rect.topleft = (rect.left + shadow_offset, rect.top + shadow_offset)
                    pygame.draw.rect(self.screen, self.COLOR_WALL_SHADOW, shadow_rect, border_radius=2)
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect, border_radius=2)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_FLOOR, rect)

                # Draw targets
                if (r, c) in self.target_positions:
                    target_rect = rect.inflate(-self.TILE_SIZE * 0.2, -self.TILE_SIZE * 0.2)
                    pygame.draw.rect(self.screen, self.COLOR_TARGET, target_rect, border_radius=4)
                    
        # Draw boxes
        for r, c in self.box_positions:
            rect = pygame.Rect(c * self.TILE_SIZE, r * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            box_rect = rect.inflate(-self.TILE_SIZE * 0.1, -self.TILE_SIZE * 0.1)
            
            on_target = (r, c) in self.target_positions
            
            color = self.COLOR_BOX_ON_TARGET if on_target else self.COLOR_BOX
            highlight_color = self.COLOR_BOX_ON_TARGET_HIGHLIGHT if on_target else self.COLOR_BOX_HIGHLIGHT
            
            # 3D effect
            pygame.draw.rect(self.screen, highlight_color, box_rect, border_top_left_radius=6, border_top_right_radius=6)
            bottom_part = pygame.Rect(box_rect.left, box_rect.centery, box_rect.width, box_rect.height / 2)
            pygame.draw.rect(self.screen, color, bottom_part, border_bottom_left_radius=6, border_bottom_right_radius=6)
            
        # Draw player
        player_r, player_c = self.player_pos
        player_center_x = int(player_c * self.TILE_SIZE + self.TILE_SIZE / 2)
        player_center_y = int(player_r * self.TILE_SIZE + self.TILE_SIZE / 2)
        player_radius = int(self.TILE_SIZE * 0.35)

        # Glow effect
        for i in range(player_radius, player_radius + 10, 2):
            alpha = self.COLOR_PLAYER_GLOW[3] * (1 - (i - player_radius) / 10)
            pygame.gfxdraw.aacircle(self.screen, player_center_x, player_center_y, i, (*self.COLOR_PLAYER_GLOW[:3], int(alpha)))
        
        # Player circle
        pygame.gfxdraw.aacircle(self.screen, player_center_x, player_center_y, player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, player_center_x, player_center_y, player_radius, self.COLOR_PLAYER)

    def _render_ui(self):
        # Display moves
        moves_left = max(0, self.MAX_MOVES - self.moves)
        moves_text = self.font_ui.render(f"MOVES LEFT: {moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (10, 10))
        
        # Display score
        score_text = self.font_ui.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            if self.win_condition_met:
                msg_text = self.font_game_over.render("PUZZLE SOLVED!", True, self.COLOR_WIN_TEXT)
            else:
                msg_text = self.font_game_over.render("OUT OF MOVES!", True, self.COLOR_LOSE_TEXT)

            msg_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_text, msg_rect)

    def close(self):
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

if __name__ == "__main__":
    # This block allows for interactive testing of the environment.
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Sokoban Warehouse")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = [0, 0, 0] # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
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
                elif event.key == pygame.K_q: # Quit on 'q'
                    running = False

        if any(a != 0 for a in action):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")

        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30)
        
    env.close()