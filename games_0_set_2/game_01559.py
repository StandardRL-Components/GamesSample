
# Generated: 2025-08-28T00:48:36.894284
# Source Brief: brief_01559.md
# Brief Index: 1559

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


# Set a dummy video driver to run pygame headlessly
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    """
    A puzzle game where a robot must collect all gems on a grid within a limited number of moves.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Use arrow keys to move the robot. Collect all gems before you run out of moves."
    )

    # User-facing description of the game
    game_description = (
        "A puzzle game where you guide a robot to collect all gems on a grid within a limited number of moves."
    )

    # The game is turn-based, so it should only advance on action.
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_DIM = (20, 20)
        self.CELL_SIZE = 18
        self.GRID_WIDTH = self.GRID_DIM[0] * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_DIM[1] * self.CELL_SIZE
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.GRID_WIDTH) // 2
        self.GRID_OFFSET_Y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT) // 2
        
        self.NUM_GEMS = 25
        self.MAX_MOVES = 30
        self.MAX_STEPS = 1000 # Failsafe episode termination

        # --- Colors ---
        self.COLOR_BG = (13, 13, 13) # #0d0d0d
        self.COLOR_GRID = (51, 51, 51) # #333333
        self.COLOR_PLAYER = (51, 153, 255) # #3399FF
        self.COLOR_PLAYER_GLOW = (51, 153, 255, 100)
        self.COLOR_GEM = (255, 51, 51) # #FF3333
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_WIN_TEXT = (50, 205, 50)
        self.COLOR_LOSE_TEXT = (220, 20, 60)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_end = pygame.font.Font(None, 64)

        # --- Game State Variables (initialized in reset) ---
        self.robot_pos = None
        self.gems = None
        self.moves_remaining = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.win_state = None # "WIN", "LOSE", or None
        self.particles = None

        # Initialize state for the first time
        self.reset()
        
        # Run self-check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.moves_remaining = self.MAX_MOVES
        self.game_over = False
        self.win_state = None
        self.particles = []

        # Place robot in the center
        self.robot_pos = (self.GRID_DIM[0] // 2, self.GRID_DIM[1] // 2)

        # Place gems randomly
        self.gems = set()
        while len(self.gems) < self.NUM_GEMS:
            pos = (
                self.np_random.integers(0, self.GRID_DIM[0]),
                self.np_random.integers(0, self.GRID_DIM[1])
            )
            if pos != self.robot_pos:
                self.gems.add(pos)
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]  # 0-4: none/up/down/left/right
        reward = 0
        
        # Any action consumes a move and has a small time penalty
        self.moves_remaining -= 1
        reward -= 0.2

        # Process movement
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        if movement != 0:
            new_pos = (self.robot_pos[0] + dx, self.robot_pos[1] + dy)
            # Check boundaries
            if 0 <= new_pos[0] < self.GRID_DIM[0] and 0 <= new_pos[1] < self.GRID_DIM[1]:
                self.robot_pos = new_pos

        # Check for gem collection
        if self.robot_pos in self.gems:
            self.gems.remove(self.robot_pos)
            self.score += 1
            reward += 1  # Reward for collecting a gem
            self._spawn_particles(self.robot_pos)
            # sound effect: gem collect

        # Update step counter
        self.steps += 1

        # Check for termination conditions
        all_gems_collected = len(self.gems) == 0
        out_of_moves = self.moves_remaining <= 0
        max_steps_reached = self.steps >= self.MAX_STEPS

        terminated = False
        if all_gems_collected:
            terminated = True
            self.game_over = True
            self.win_state = "WIN"
            reward += 10  # Bonus for the last gem
            reward += 50  # Bonus for winning
        elif out_of_moves or max_steps_reached:
            terminated = True
            self.game_over = True
            self.win_state = "LOSE"
            reward -= 10 # Penalty for losing

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

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
            "moves_remaining": self.moves_remaining,
            "gems_remaining": len(self.gems),
        }

    def _grid_to_pixel(self, grid_pos):
        """Converts grid coordinates to pixel coordinates for the center of the cell."""
        px = self.GRID_OFFSET_X + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.GRID_OFFSET_Y + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return int(px), int(py)

    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_DIM[0] + 1):
            start_pos = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end_pos = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)
        for y in range(self.GRID_DIM[1] + 1):
            start_pos = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            end_pos = (self.GRID_OFFSET_X + self.GRID_WIDTH, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)

        # Draw gems
        gem_radius = self.CELL_SIZE // 3
        for gem_pos in self.gems:
            px, py = self._grid_to_pixel(gem_pos)
            pygame.gfxdraw.aacircle(self.screen, px, py, gem_radius, self.COLOR_GEM)
            pygame.gfxdraw.filled_circle(self.screen, px, py, gem_radius, self.COLOR_GEM)

        # Draw robot
        robot_px, robot_py = self._grid_to_pixel(self.robot_pos)
        robot_size = self.CELL_SIZE * 0.8
        robot_rect = pygame.Rect(
            robot_px - robot_size // 2,
            robot_py - robot_size // 2,
            robot_size,
            robot_size
        )
        # Glow effect for robot
        glow_surf = pygame.Surface((robot_size * 2, robot_size * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (robot_size, robot_size), robot_size)
        self.screen.blit(glow_surf, (robot_rect.centerx - robot_size, robot_rect.centery - robot_size), special_flags=pygame.BLEND_RGBA_ADD)
        
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, robot_rect, border_radius=2)
        
        # Update and draw particles
        self._update_and_draw_particles()

    def _render_ui(self):
        # Moves remaining
        moves_text = self.font_ui.render(f"Moves: {self.moves_remaining}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (20, 15))

        # Score
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 15))
        self.screen.blit(score_text, score_rect)

        # End game message
        if self.game_over:
            if self.win_state == "WIN":
                end_text_str = "YOU WIN!"
                end_color = self.COLOR_WIN_TEXT
            else:
                end_text_str = "GAME OVER"
                end_color = self.COLOR_LOSE_TEXT
            
            end_text = self.font_end.render(end_text_str, True, end_color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            
            # Draw a semi-transparent background for the text
            bg_rect = end_rect.inflate(40, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, bg_rect)
            
            self.screen.blit(end_text, end_rect)

    def _spawn_particles(self, grid_pos):
        px, py = self._grid_to_pixel(grid_pos)
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1.5, 3.0)
            velocity = (math.cos(angle) * speed, math.sin(angle) * speed)
            radius = self.np_random.uniform(2, 5)
            lifetime = self.np_random.integers(15, 30) # frames
            self.particles.append({
                'pos': [px, py],
                'vel': velocity,
                'radius': radius,
                'lifetime': lifetime,
                'max_lifetime': lifetime,
                'color': self.COLOR_GEM
            })

    def _update_and_draw_particles(self):
        remaining_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifetime'] -= 1
            
            if p['lifetime'] > 0:
                # Fade out effect
                alpha = int(255 * (p['lifetime'] / p['max_lifetime']))
                color = p['color'] + (alpha,)
                
                # Shrink effect
                current_radius = int(p['radius'] * (p['lifetime'] / p['max_lifetime']))
                if current_radius > 0:
                    # Use a temporary surface for alpha blending
                    particle_surf = pygame.Surface((current_radius*2, current_radius*2), pygame.SRCALPHA)
                    pygame.draw.circle(particle_surf, color, (current_radius, current_radius), current_radius)
                    self.screen.blit(particle_surf, (int(p['pos'][0]) - current_radius, int(p['pos'][1]) - current_radius))
                
                remaining_particles.append(p)
        self.particles = remaining_particles

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to run the file directly to test the environment
    # and play the game manually.
    
    # Re-enable video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Setup ---
    pygame.display.set_caption("Gem Collector")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    print("\n" + "="*30)
    print("      MANUAL PLAY MODE")
    print("="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    running = True
    while running:
        action = [0, 0, 0] # Default action is no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    done = False
                
                # Only register movement actions if the game is not over
                if not done:
                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
                    
                    # Take a step if a movement key was pressed
                    if action[0] != 0:
                        obs, reward, terminated, truncated, info = env.step(action)
                        done = terminated or truncated
                        print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")


        # Draw the observation from the environment to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(30) # Limit FPS for manual play

    env.close()