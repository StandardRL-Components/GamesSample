
# Generated: 2025-08-28T03:19:44.085109
# Source Brief: brief_04896.md
# Brief Index: 4896

        
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
        "Controls: Use arrow keys (↑, ↓, ←, →) to move your character (red square) and push boxes (blue squares)."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist puzzle game. Push all four boxes into the gray target zones within 15 steps to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_SIZE = 10
        self.TILE_SIZE = 40
        self.GRID_WIDTH = self.GRID_SIZE * self.TILE_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.TILE_SIZE
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.GRID_WIDTH) // 2
        self.GRID_OFFSET_Y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT) // 2
        self.MAX_STEPS = 15
        self.NUM_BOXES = 4
        self.NUM_ZONES = 4

        # Colors
        self.COLOR_BG = (255, 255, 255)
        self.COLOR_GRID = (220, 220, 220)
        self.COLOR_TEXT = (10, 10, 10)
        self.COLOR_PLAYER = (220, 50, 50)
        self.COLOR_BOX = (50, 100, 200)
        self.COLOR_BOX_IN_ZONE = (100, 150, 255)
        self.COLOR_ZONE = (240, 240, 240)
        self.COLOR_OVERLAY = (0, 0, 0, 180)

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
        
        # Fonts
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 20)
            self.font_title = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.Font(None, 24)
            self.font_title = pygame.font.Font(None, 30)
            self.font_game_over = pygame.font.Font(None, 60)
        
        # Initialize state variables to be set in reset()
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.player_pos = (0, 0)
        self.box_positions = []
        self.zone_positions = []
        self.boxes_in_zone_count = 0
        self.np_random = None

        self.reset()
        
    def _grid_to_pixel(self, pos):
        """Converts grid coordinates to pixel coordinates for rendering."""
        x, y = pos
        px = self.GRID_OFFSET_X + x * self.TILE_SIZE
        py = self.GRID_OFFSET_Y + y * self.TILE_SIZE
        return px, py

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        elif self.np_random is None:
            self.np_random = np.random.default_rng()

        # Generate a new puzzle layout
        all_coords = [(x, y) for x in range(self.GRID_SIZE) for y in range(self.GRID_SIZE)]
        self.np_random.shuffle(all_coords)
        
        num_entities = 1 + self.NUM_BOXES + self.NUM_ZONES
        chosen_coords = all_coords[:num_entities]
        
        self.player_pos = chosen_coords[0]
        self.box_positions = chosen_coords[1:1 + self.NUM_BOXES]
        self.zone_positions = chosen_coords[1 + self.NUM_BOXES:]

        # Initialize all game state
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.boxes_in_zone_count = self._count_boxes_in_zones()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = -1.0 # Cost for taking a step
        self.steps += 1
        
        # Handle movement
        if movement != 0:
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
            
            player_x, player_y = self.player_pos
            next_player_x, next_player_y = player_x + dx, player_y + dy
            
            # Check if next move is within bounds
            if 0 <= next_player_x < self.GRID_SIZE and 0 <= next_player_y < self.GRID_SIZE:
                
                # Check for box collision
                if (next_player_x, next_player_y) in self.box_positions:
                    box_idx = self.box_positions.index((next_player_x, next_player_y))
                    next_box_x, next_box_y = next_player_x + dx, next_player_y + dy
                    
                    # Check if box can be pushed
                    is_box_move_valid = (
                        0 <= next_box_x < self.GRID_SIZE and
                        0 <= next_box_y < self.GRID_SIZE and
                        (next_box_x, next_box_y) not in self.box_positions
                    )
                    
                    if is_box_move_valid:
                        # Push box and move player
                        self.box_positions[box_idx] = (next_box_x, next_box_y)
                        self.player_pos = (next_player_x, next_player_y)
                        # sfx: box_push.wav
                
                # No box, just move player
                else:
                    self.player_pos = (next_player_x, next_player_y)
                    # sfx: player_step.wav

        # Calculate reward for boxes in zones
        new_boxes_in_zone_count = self._count_boxes_in_zones()
        if new_boxes_in_zone_count > self.boxes_in_zone_count:
            # sfx: box_in_zone.wav
            reward += (new_boxes_in_zone_count - self.boxes_in_zone_count) * 10.0
        self.boxes_in_zone_count = new_boxes_in_zone_count
        
        self.score += reward

        # Check termination conditions
        terminated = False
        win = self.boxes_in_zone_count == self.NUM_ZONES
        loss = self.steps >= self.MAX_STEPS
        
        if win:
            reward += 100.0
            self.score += 100.0
            terminated = True
            self.game_over = True
            # sfx: win.wav
        elif loss:
            reward -= 100.0
            self.score -= 100.0
            terminated = True
            self.game_over = True
            # sfx: lose.wav

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _count_boxes_in_zones(self):
        return sum(1 for box_pos in self.box_positions if box_pos in self.zone_positions)

    def _render_grid(self):
        # Draw vertical lines
        for x in range(self.GRID_SIZE + 1):
            start_pos = (self.GRID_OFFSET_X + x * self.TILE_SIZE, self.GRID_OFFSET_Y)
            end_pos = (self.GRID_OFFSET_X + x * self.TILE_SIZE, self.GRID_OFFSET_Y + self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)
        # Draw horizontal lines
        for y in range(self.GRID_SIZE + 1):
            start_pos = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + y * self.TILE_SIZE)
            end_pos = (self.GRID_OFFSET_X + self.GRID_WIDTH, self.GRID_OFFSET_Y + y * self.TILE_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)

    def _render_game_elements(self):
        # Render zones
        for zone_pos in self.zone_positions:
            px, py = self._grid_to_pixel(zone_pos)
            pygame.draw.rect(self.screen, self.COLOR_ZONE, (px, py, self.TILE_SIZE, self.TILE_SIZE))

        # Render boxes
        gap = self.TILE_SIZE // 10
        box_size = self.TILE_SIZE - 2 * gap
        for box_pos in self.box_positions:
            px, py = self._grid_to_pixel(box_pos)
            color = self.COLOR_BOX_IN_ZONE if box_pos in self.zone_positions else self.COLOR_BOX
            pygame.draw.rect(self.screen, color, (px + gap, py + gap, box_size, box_size), border_radius=4)
            
        # Render player
        player_size = self.TILE_SIZE - 2 * gap
        px, py = self._grid_to_pixel(self.player_pos)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (px + gap, py + gap, player_size, player_size), border_radius=4)

    def _render_ui(self):
        # Title
        title_surf = self.font_title.render("PushBox Puzzle", True, self.COLOR_TEXT)
        self.screen.blit(title_surf, (self.GRID_OFFSET_X, 10))

        # Steps left
        steps_left = max(0, self.MAX_STEPS - self.steps)
        steps_surf = self.font_ui.render(f"Steps Left: {steps_left}", True, self.COLOR_TEXT)
        self.screen.blit(steps_surf, (20, 20))

        # Score
        score_surf = self.font_ui.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(score_surf, score_rect)

        # Game Over overlay
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_OVERLAY)
            self.screen.blit(overlay, (0, 0))
            
            win = self.boxes_in_zone_count == self.NUM_ZONES
            msg = "YOU WIN!" if win else "OUT OF STEPS"
            color = (100, 255, 100) if win else (255, 100, 100)
            
            msg_surf = self.font_game_over.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_grid()
        self._render_game_elements()
        
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    
    try:
        pygame.display.init()
        pygame.display.set_caption("PushBox Puzzle")
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        
        obs, info = env.reset()
        done = False
        
        print("\n--- Manual Play Instructions ---")
        print(env.user_guide)
        print("Press 'R' to reset. Press 'Q' or close window to quit.")

        while not done:
            frame = np.transpose(obs, (1, 0, 2))
            pygame_surface = pygame.surfarray.make_surface(frame)
            screen.blit(pygame_surface, (0, 0))
            pygame.display.flip()

            action = [0, 0, 0]
            
            event_happened = False
            while not event_happened:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                        event_happened = True
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            done = True
                        elif event.key == pygame.K_r:
                            obs, info = env.reset()
                        elif event.key == pygame.K_UP:
                            action[0] = 1
                        elif event.key == pygame.K_DOWN:
                            action[0] = 2
                        elif event.key == pygame.K_LEFT:
                            action[0] = 3
                        elif event.key == pygame.K_RIGHT:
                            action[0] = 4
                        
                        event_happened = True
                if done:
                    break

            if done:
                break
            
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")

            if terminated:
                frame = np.transpose(obs, (1, 0, 2))
                pygame_surface = pygame.surfarray.make_surface(frame)
                screen.blit(pygame_surface, (0, 0))
                pygame.display.flip()
                print("Game Over. Press 'R' to play again or 'Q' to quit.")
                
                wait_for_reset = True
                while wait_for_reset:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                            done = True
                            wait_for_reset = False
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                            obs, info = env.reset()
                            wait_for_reset = False

    except pygame.error as e:
        print(f"Pygame display could not be initialized. Manual play is disabled. ({e})")
    finally:
        env.close()