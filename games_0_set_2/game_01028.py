
# Generated: 2025-08-27T15:37:08.306453
# Source Brief: brief_01028.md
# Brief Index: 1028

        
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
        "Controls: Arrow keys to move on the isometric grid. Collect all blue gems and avoid red traps."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic puzzle game. Navigate an isometric grid to collect gems while avoiding traps. Each move counts!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Game Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 16
    GRID_HEIGHT = 10
    NUM_GEMS = 25
    NUM_TRAPS = 30
    MAX_STEPS = 1000

    TILE_WIDTH = 40
    TILE_HEIGHT = 20
    TILE_WIDTH_HALF = TILE_WIDTH // 2
    TILE_HEIGHT_HALF = TILE_HEIGHT // 2

    # --- Colors ---
    COLOR_BG = (15, 18, 32)
    COLOR_GRID = (30, 35, 60)
    COLOR_PLAYER = (57, 255, 20)
    COLOR_PLAYER_GLOW = (57, 255, 20, 100) # Used for glow effect
    COLOR_GEM = (0, 191, 255)
    COLOR_GEM_SPARKLE = (200, 240, 255)
    COLOR_TRAP = (200, 30, 30)
    COLOR_TRAP_INNER = (120, 15, 15)
    COLOR_TEXT = (255, 255, 255)
    COLOR_UI_BG = (0, 0, 0, 128)

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
        self.font = pygame.font.Font(None, 28)
        
        # Calculate grid origin to center it
        self.origin_x = self.SCREEN_WIDTH // 2
        self.origin_y = self.SCREEN_HEIGHT // 2 - (self.GRID_HEIGHT * self.TILE_HEIGHT_HALF) // 2 + 20

        # Initialize state variables
        self.player_pos = (0, 0)
        self.gems = []
        self.traps = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Generate grid entities
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        # Place player near the center
        self.player_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)

        # Create a list of all possible coordinates
        all_coords = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        all_coords.remove(self.player_pos)

        # Define a safe zone around the player to guarantee a safe start
        safe_zone_radius = 2
        safe_coords = {
            (x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)
            if self._manhattan_distance(self.player_pos, (x, y)) <= safe_zone_radius
        }
        
        # Potential trap locations are outside the safe zone
        potential_trap_coords = [c for c in all_coords if c not in safe_coords]
        self.np_random.shuffle(potential_trap_coords)
        self.traps = potential_trap_coords[:min(self.NUM_TRAPS, len(potential_trap_coords))]
        
        # Potential gem locations are anywhere not occupied by a trap or player
        trap_set = set(self.traps)
        potential_gem_coords = [c for c in all_coords if c not in trap_set]
        self.np_random.shuffle(potential_gem_coords)
        self.gems = potential_gem_coords[:min(self.NUM_GEMS, len(potential_gem_coords))]
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0
        old_pos = self.player_pos
        dist_before = self._get_min_dist_to_gem(old_pos) if self.gems else 0

        # --- Update game logic ---
        dx, dy = 0, 0
        if movement == 1:  # Up (North-West)
            dy = -1
        elif movement == 2:  # Down (South-East)
            dy = 1
        elif movement == 3:  # Left (South-West)
            dx = -1
        elif movement == 4:  # Right (North-East)
            dx = 1
            
        new_pos = (old_pos[0] + dx, old_pos[1] + dy)

        # Boundary checks
        if 0 <= new_pos[0] < self.GRID_WIDTH and 0 <= new_pos[1] < self.GRID_HEIGHT:
            self.player_pos = new_pos
        else:
            # Penalize for hitting a wall
            reward -= 0.5 

        # --- Check for events and calculate rewards ---
        terminated = False

        # Gem collection
        if self.player_pos in self.gems:
            self.gems.remove(self.player_pos)
            self.score += 10
            reward += 10
            # sfx: gem_collect
            if not self.gems: # All gems collected
                self.score += 50
                reward += 50
                terminated = True
                self.game_over = True
                # sfx: level_complete

        # Trap collision
        elif self.player_pos in self.traps:
            self.score -= 100
            reward = -100 # Overwrite other rewards
            terminated = True
            self.game_over = True
            # sfx: player_die
        
        # Proximity-based reward if no major event occurred
        if not terminated and movement != 0:
            dist_after = self._get_min_dist_to_gem(self.player_pos) if self.gems else 0
            if dist_after < dist_before:
                reward += 1.0  # Moved closer
            else:
                reward -= 0.1  # Moved further or parallel

        # Step penalty
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid tiles
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                self._draw_iso_tile((x, y), self.COLOR_GRID)

        # Draw traps
        for trap_pos in self.traps:
            self._draw_iso_tile(trap_pos, self.COLOR_TRAP, inner_color=self.COLOR_TRAP_INNER)

        # Draw gems
        for gem_pos in self.gems:
            self._draw_gem(gem_pos)

        # Draw player
        self._draw_player()

    def _draw_iso_tile(self, pos, color, inner_color=None):
        center_x, center_y = self._grid_to_iso(pos)
        points = [
            (center_x, center_y - self.TILE_HEIGHT_HALF),
            (center_x + self.TILE_WIDTH_HALF, center_y),
            (center_x, center_y + self.TILE_HEIGHT_HALF),
            (center_x - self.TILE_WIDTH_HALF, center_y),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        
        if inner_color:
            inner_points = [
                (center_x, center_y - self.TILE_HEIGHT_HALF + 4),
                (center_x + self.TILE_WIDTH_HALF - 8, center_y),
                (center_x, center_y + self.TILE_HEIGHT_HALF - 4),
                (center_x - self.TILE_WIDTH_HALF + 8, center_y),
            ]
            pygame.gfxdraw.filled_polygon(self.screen, inner_points, inner_color)


    def _draw_gem(self, pos):
        center_x, center_y = self._grid_to_iso(pos)
        
        # Pulsing animation
        pulse = math.sin(self.steps * 0.2 + pos[0] + pos[1]) * 2
        
        width = self.TILE_WIDTH_HALF * 0.7 + pulse
        height = self.TILE_HEIGHT_HALF * 0.7 + pulse
        
        points = [
            (center_x, center_y - height),
            (center_x + width, center_y),
            (center_x, center_y + height),
            (center_x - width, center_y),
        ]
        
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_GEM)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GEM_SPARKLE)

    def _draw_player(self):
        iso_x, iso_y = self._grid_to_iso(self.player_pos)
        
        # Glow effect
        glow_radius = 12
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surface, (iso_x - glow_radius, iso_y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Player circle
        player_radius = 8
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, (iso_x, iso_y), player_radius)


    def _render_ui(self):
        # UI background
        ui_bg_surface = pygame.Surface((180, 40), pygame.SRCALPHA)
        ui_bg_surface.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_bg_surface, (10, 10))

        # Score display
        score_text = self.font.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 18))

        # Step display
        step_text = self.font.render(f"Moves: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        step_rect = step_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 18))
        
        ui_bg_surface_steps = pygame.Surface((step_rect.width + 20, 40), pygame.SRCALPHA)
        ui_bg_surface_steps.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_bg_surface_steps, (step_rect.left - 10, 10))
        self.screen.blit(step_text, step_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gems_left": len(self.gems),
            "player_pos": self.player_pos,
        }

    def _grid_to_iso(self, pos):
        """Converts grid coordinates (x, y) to screen coordinates."""
        x, y = pos
        iso_x = self.origin_x + (x - y) * self.TILE_WIDTH_HALF
        iso_y = self.origin_y + (x + y) * self.TILE_HEIGHT_HALF
        return int(iso_x), int(iso_y)

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _get_min_dist_to_gem(self, pos):
        if not self.gems:
            return 0
        return min(self._manhattan_distance(pos, gem_pos) for gem_pos in self.gems)

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

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Set up the display window
    pygame.display.set_caption(env.game_description)
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    action = [0, 0, 0] # No-op, no space, no shift
    
    print(env.user_guide)

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            
            # Reset action on key up
            if event.type == pygame.KEYUP:
                action = [0, 0, 0]

        # Get pressed keys for continuous movement
        keys = pygame.key.get_pressed()
        movement_action = 0 # No-op
        if keys[pygame.K_UP]:
            movement_action = 1
        elif keys[pygame.K_DOWN]:
            movement_action = 2
        elif keys[pygame.K_LEFT]:
            movement_action = 3
        elif keys[pygame.K_RIGHT]:
            movement_action = 4
        
        action[0] = movement_action
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        # Since auto_advance is False, we only step if an action is taken
        if any(keys):
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")

        # Render the environment to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(15) # Limit manual play speed

    env.close()
    print("Game Over!")