
# Generated: 2025-08-27T20:16:37.601010
# Source Brief: brief_02406.md
# Brief Index: 2406

        
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
        "Controls: Use arrow keys to move your agent and push the red crates onto the green goals."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A timed puzzle game. Push all crates to their designated goals before the 45-step timer runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 16, 10
        self.TILE_SIZE = 40
        self.MAX_STEPS = 45

        # Colors
        self.COLOR_BG = (240, 240, 240)
        self.COLOR_WALL = (100, 100, 100)
        self.COLOR_GOAL_OFF = (180, 220, 180)
        self.COLOR_GOAL_ON = (100, 255, 100)
        self.COLOR_CRATE = (220, 50, 50)
        self.COLOR_PLAYER = (50, 100, 255)
        self.COLOR_SHADOW = (0, 0, 0, 50)
        self.COLOR_UI_TEXT = (20, 20, 20)
        self.COLOR_OVERLAY = (0, 0, 0, 150)
        
        # Action mapping
        self.ACTION_MAP = {
            0: (0, 0),   # none
            1: (0, -1),  # up
            2: (0, 1),   # down
            3: (-1, 0),  # left
            4: (1, 0),   # right
        }

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # Initialize state variables to avoid errors
        self.player_pos = (0, 0)
        self.crate_positions = []
        self.goal_positions = []
        self.walls = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.previous_crate_distances = []
        
        # Initialize state variables
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False

        # Define level layout
        self.player_pos = (8, 5)
        self.goal_positions = sorted([(2, 2), (2, 7), (13, 2), (13, 7)])
        # Crate positions are matched to goals by sorted order
        self.crate_positions = sorted([(6, 4), (6, 5), (9, 4), (9, 5)])
        
        self.walls = self._generate_walls()

        # Calculate initial distances for reward calculation
        self.previous_crate_distances = [
            self._manhattan_distance(c, g) for c, g in zip(self.crate_positions, self.goal_positions)
        ]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0
        
        dx, dy = self.ACTION_MAP[movement]
        
        # Only process movement if a direction is chosen
        if dx != 0 or dy != 0:
            target_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
            
            # Check for wall collision
            if target_pos in self.walls:
                pass # Player is blocked
            
            # Check for crate collision
            elif target_pos in self.crate_positions:
                crate_index = self.crate_positions.index(target_pos)
                push_pos = (target_pos[0] + dx, target_pos[1] + dy)
                
                # Check if crate push is blocked
                if push_pos in self.walls or push_pos in self.crate_positions:
                    pass # Crate is blocked
                else:
                    # Move crate and player
                    old_crate_pos = self.crate_positions[crate_index]
                    self.crate_positions[crate_index] = push_pos
                    self.player_pos = target_pos
                    
                    # Reward for moving crate towards/away from its goal
                    goal_pos = self.goal_positions[crate_index]
                    dist_after = self._manhattan_distance(push_pos, goal_pos)
                    dist_before = self.previous_crate_distances[crate_index]

                    if dist_after < dist_before:
                        reward += 1.0  # Closer to goal
                    elif dist_after > dist_before:
                        reward -= 0.1 # Further from goal
                    
                    self.previous_crate_distances[crate_index] = dist_after

                    # Reward for placing crate on goal
                    if push_pos == goal_pos and old_crate_pos != goal_pos:
                        reward += 10.0
            
            # Move into empty space
            else:
                self.player_pos = target_pos

        self.steps += 1
        self.score += reward
        
        terminated = self._check_termination()
        if terminated:
            if self.win_state:
                reward += 50.0
            else: # Timeout
                reward -= 50.0
            self.score += reward
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "crates_on_goal": self._count_crates_on_goals(),
        }

    def _render_game(self):
        # Draw goals
        for i, pos in enumerate(self.goal_positions):
            rect = self._get_tile_rect(pos)
            is_on_goal = pos in self.crate_positions
            color = self.COLOR_GOAL_ON if is_on_goal else self.COLOR_GOAL_OFF
            pygame.draw.rect(self.screen, color, rect)

        # Draw walls
        for pos in self.walls:
            rect = self._get_tile_rect(pos)
            pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
        
        # Draw shadows for crates
        for pos in self.crate_positions:
            self._draw_shadow(pos)

        # Draw shadows for player
        self._draw_shadow(self.player_pos)

        # Draw crates
        for i, pos in enumerate(self.crate_positions):
            rect = self._get_tile_rect(pos)
            pygame.draw.rect(self.screen, self.COLOR_CRATE, rect.inflate(-4, -4))
            if pos == self.goal_positions[i]:
                pygame.draw.rect(self.screen, (255, 255, 255), rect.inflate(-4, -4), 2)


        # Draw player
        center_x, center_y = self._get_tile_center(self.player_pos)
        radius = self.TILE_SIZE // 2 - 4
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_PLAYER)

    def _render_ui(self):
        time_text = self.font_ui.render(f"Time: {self.MAX_STEPS - self.steps}", True, self.COLOR_UI_TEXT)
        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        goals_text = self.font_ui.render(f"Goals: {self._count_crates_on_goals()}/4", True, self.COLOR_UI_TEXT)
        
        self.screen.blit(time_text, (10, 10))
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        self.screen.blit(goals_text, (self.WIDTH // 2 - goals_text.get_width() // 2, 10))
    
    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill(self.COLOR_OVERLAY)
        self.screen.blit(overlay, (0, 0))
        
        message = "YOU WIN!" if self.win_state else "TIME UP!"
        color = self.COLOR_GOAL_ON if self.win_state else self.COLOR_CRATE
        
        text = self.font_game_over.render(message, True, color)
        text_rect = text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
        self.screen.blit(text, text_rect)

    def _generate_walls(self):
        walls = set()
        for x in range(self.GRID_W):
            walls.add((x, 0))
            walls.add((x, self.GRID_H - 1))
        for y in range(self.GRID_H):
            walls.add((0, y))
            walls.add((self.GRID_W - 1, y))
        return list(walls)

    def _check_termination(self):
        self.win_state = self._count_crates_on_goals() == len(self.goal_positions)
        if self.win_state:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _count_crates_on_goals(self):
        count = 0
        for crate_pos, goal_pos in zip(self.crate_positions, self.goal_positions):
            if crate_pos == goal_pos:
                count += 1
        return count
    
    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_tile_rect(self, grid_pos):
        return pygame.Rect(grid_pos[0] * self.TILE_SIZE, grid_pos[1] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)

    def _get_tile_center(self, grid_pos):
        x = int(grid_pos[0] * self.TILE_SIZE + self.TILE_SIZE / 2)
        y = int(grid_pos[1] * self.TILE_SIZE + self.TILE_SIZE / 2)
        return x, y

    def _draw_shadow(self, grid_pos):
        center_x, center_y = self._get_tile_center(grid_pos)
        shadow_radius = self.TILE_SIZE // 2 - 2
        shadow_offset = 2
        
        # Use a surface for alpha blending
        shadow_surf = pygame.Surface((shadow_radius*2, shadow_radius*2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(shadow_surf, shadow_radius, shadow_radius, shadow_radius, self.COLOR_SHADOW)
        self.screen.blit(shadow_surf, (center_x - shadow_radius + shadow_offset, center_y - shadow_radius + shadow_offset))

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
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
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
    
    # Create a window to display the game
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Sokoban Puzzle")
    
    print("\n" + "="*30)
    print("MANUAL PLAY TEST")
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while not done:
        # Convert observation for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Get user input
        movement = 0 # Default to no-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    movement = 1
                elif event.key == pygame.K_DOWN:
                    movement = 2
                elif event.key == pygame.K_LEFT:
                    movement = 3
                elif event.key == pygame.K_RIGHT:
                    movement = 4
                elif event.key == pygame.K_r: # Press 'r' to reset
                    obs, info = env.reset()
                    print("--- Game Reset ---")
                elif event.key == pygame.K_q: # Press 'q' to quit
                    done = True

        if done:
            break
            
        # Only step if a key was pressed (or it's the first frame)
        if movement != 0:
            # Action is [movement, space, shift]
            action = [movement, 0, 0]
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated
            
            print(f"Step: {info['steps']}, Action: {action[0]}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Done: {terminated}")

            if terminated:
                # Show final frame
                surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                render_screen.blit(surf, (0, 0))
                pygame.display.flip()
                print("Game Over. Press 'r' to reset or 'q' to quit.")


    env.close()
    pygame.quit()