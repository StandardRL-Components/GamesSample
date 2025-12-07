
# Generated: 2025-08-27T18:11:58.186870
# Source Brief: brief_01760.md
# Brief Index: 1760

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move your character on the isometric grid. "
        "Collect all the gems before you run out of moves!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A turn-based puzzle game. Navigate an isometric grid, collecting all gems "
        "within a limited number of moves to progress through increasingly complex stages."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False # Turn-based game, so state changes only on action.

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

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

        # Colors
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID = (45, 52, 64)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_OUTLINE = (200, 200, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.GEM_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
        ]

        # Fonts
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_msg = pygame.font.SysFont("sans", 48, bold=True)

        # Stage configurations
        self.stage_configs = [
            {'grid_size': (8, 8), 'num_gems': 8, 'moves_limit': 12},
            {'grid_size': (10, 10), 'num_gems': 12, 'moves_limit': 16},
            {'grid_size': (12, 12), 'num_gems': 16, 'moves_limit': 20},
        ]
        self.max_steps = 1000

        # Initialize state variables
        self.player_pos = None
        self.gems = None
        self.current_stage = None
        self.moves_remaining = None
        self.grid_size = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.victory = None
        self.tile_width = None
        self.tile_height = None
        self.origin_x = None
        self.origin_y = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.current_stage = 0

        self._setup_stage(self.current_stage)

        return self._get_observation(), self._get_info()

    def _setup_stage(self, stage_index):
        config = self.stage_configs[stage_index]
        self.grid_size = config['grid_size']
        num_gems = config['num_gems']
        self.moves_remaining = config['moves_limit']

        # Center player
        self.player_pos = (self.grid_size[0] // 2, self.grid_size[1] // 2)

        # Generate unique gem positions
        self.gems = []
        possible_positions = set((x, y) for x in range(self.grid_size[0]) for y in range(self.grid_size[1]))
        possible_positions.discard(self.player_pos)
        
        # Use np_random for reproducibility
        shuffled_positions = list(possible_positions)
        self.np_random.shuffle(shuffled_positions)
        
        gem_positions = shuffled_positions[:num_gems]
        
        for i, pos in enumerate(gem_positions):
            self.gems.append({
                "pos": pos,
                "color": self.GEM_COLORS[i % len(self.GEM_COLORS)]
            })

        # Calculate rendering geometry for the new grid
        self.tile_width = 40
        self.tile_height = self.tile_width / 2
        self.origin_x = self.WIDTH // 2
        self.origin_y = self.HEIGHT // 2 - (self.grid_size[1] * self.tile_height) / 2 + 20

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        # space_held = action[1] == 1 # Unused
        # shift_held = action[2] == 1 # Unused

        reward = 0
        terminated = False

        if movement != 0: # 0 is no-op
            self.steps += 1
            reward -= 0.1 # Cost for making a move
            self.moves_remaining -= 1

            dx, dy = 0, 0
            if movement == 1: dy = -1  # Isometric Up
            elif movement == 2: dy = 1   # Isometric Down
            elif movement == 3: dx = -1  # Isometric Left
            elif movement == 4: dx = 1   # Isometric Right

            new_x = self.player_pos[0] + dx
            new_y = self.player_pos[1] + dy

            # Check boundaries
            if 0 <= new_x < self.grid_size[0] and 0 <= new_y < self.grid_size[1]:
                self.player_pos = (new_x, new_y)
                # # Sound effect placeholder: player move

        # Check for gem collection
        gem_to_remove = None
        for gem in self.gems:
            if gem["pos"] == self.player_pos:
                gem_to_remove = gem
                break
        
        if gem_to_remove:
            self.gems.remove(gem_to_remove)
            self.score += 1
            reward += 1
            # # Sound effect placeholder: gem collect

        # Check for stage completion
        if not self.gems:
            self.score += 10
            reward += 10
            self.current_stage += 1
            if self.current_stage >= len(self.stage_configs):
                # Game won
                self.game_over = True
                self.victory = True
                terminated = True
                self.score += 100
                reward += 100
                # # Sound effect placeholder: game win
            else:
                # Next stage
                self._setup_stage(self.current_stage)
                # # Sound effect placeholder: stage complete
        
        # Check for game over (out of moves)
        if self.moves_remaining <= 0 and not self.game_over and self.gems:
            self.game_over = True
            terminated = True
            self.score -= 10
            reward -= 10
            # # Sound effect placeholder: game over

        # Check for max steps termination
        if self.steps >= self.max_steps:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _iso_to_screen(self, grid_x, grid_y):
        screen_x = self.origin_x + (grid_x - grid_y) * (self.tile_width / 2)
        screen_y = self.origin_y + (grid_x + grid_y) * (self.tile_height / 2)
        return int(screen_x), int(screen_y)

    def _draw_iso_tile(self, surface, color, grid_pos, outline_color=None):
        x, y = grid_pos
        tile_w_half = self.tile_width / 2
        tile_h_half = self.tile_height / 2
        
        center_x, center_y = self._iso_to_screen(x, y)
        
        points = [
            (center_x, int(center_y - tile_h_half)),
            (int(center_x + tile_w_half), center_y),
            (center_x, int(center_y + tile_h_half)),
            (int(center_x - tile_w_half), center_y)
        ]
        
        # Use gfxdraw for anti-aliasing
        pygame.gfxdraw.filled_polygon(surface, points, color)
        if outline_color:
             pygame.gfxdraw.aapolygon(surface, points, outline_color)

    def _render_game(self):
        # Draw grid
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                points = [
                    self._iso_to_screen(r, c),
                    self._iso_to_screen(r + 1, c),
                    self._iso_to_screen(r + 1, c + 1),
                    self._iso_to_screen(r, c + 1)
                ]
                pygame.draw.aalines(self.screen, self.COLOR_GRID, True, points)

        # Draw gems
        for gem in self.gems:
            self._draw_iso_tile(self.screen, gem["color"], gem["pos"])

        # Draw player
        self._draw_iso_tile(self.screen, self.COLOR_PLAYER, self.player_pos, self.COLOR_PLAYER_OUTLINE)

    def _render_ui(self):
        # Moves remaining
        moves_text = self.font_ui.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (10, 10))

        # Stage number
        stage_text = self.font_ui.render(f"Stage: {self.current_stage + 1}/{len(self.stage_configs)}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.WIDTH - stage_text.get_width() - 10, 10))
        
        # Score
        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 30))

        # Game over / Victory message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            if self.victory:
                msg_text = self.font_msg.render("YOU WIN!", True, (100, 255, 100))
            else:
                msg_text = self.font_msg.render("GAME OVER", True, (255, 100, 100))
            
            text_rect = msg_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(msg_text, text_rect)

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
            "stage": self.current_stage + 1,
            "moves_remaining": self.moves_remaining,
            "gems_left": len(self.gems)
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

# Example of how to run the environment
if __name__ == '__main__':
    import os
    # Ensure the script can run in a headless environment
    try:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        
        env = GameEnv(render_mode="rgb_array")
        
        # Test reset
        obs, info = env.reset()
        print("Reset successful.")
        print("Initial info:", info)

        # Test a few random steps
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {i+1}: Action={action.tolist()}, Reward={reward:.1f}, Terminated={terminated}, Info={info}")
            if terminated:
                print("Episode finished.")
                obs, info = env.reset()
                print("Environment reset.")
        env.close()

        # Test rendering to a file to visually inspect
        try:
            # Re-init with a real video driver for saving an image
            pygame.quit()
            # This part is platform-dependent and might fail on some systems.
            # It's for visual verification by the user if they have a display server.
            os.environ["SDL_VIDEODRIVER"] = "x11"
            import sys
            if sys.platform == "win32":
                os.environ["SDL_VIDEODRIVER"] = "windows"
            elif sys.platform == "darwin":
                os.environ["SDL_VIDEODRIVER"] = "mac"
            
            env_vis = GameEnv(render_mode="rgb_array")
            obs, _ = env_vis.reset()
            
            # Move right to show a change
            action = np.array([4, 0, 0])
            obs, _, _, _, _ = env_vis.step(action)
            
            # Save the frame
            # The internal self.screen is already (W, H), so we can use it directly
            pygame.image.save(env_vis.screen, "gem_collector_frame.png")
            print("\nSaved a sample frame to gem_collector_frame.png")
            env_vis.close()
        except Exception as e:
            print(f"\nCould not save visual frame. This is expected on headless systems. Error: {e}")
            
    except Exception as e:
        print(f"An error occurred during environment execution: {e}")