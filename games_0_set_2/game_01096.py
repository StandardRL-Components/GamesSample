
# Generated: 2025-08-27T16:01:23.551276
# Source Brief: brief_01096.md
# Brief Index: 1096

        
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

    user_guide = (
        "Use arrow keys to move. Each move consumes one turn. Land on a gem to collect it."
    )

    game_description = (
        "A strategic puzzle game. Collect 20 gems within 10 moves by planning your path on the isometric grid."
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 12, 12
        self.GEM_TARGET = 20
        self.MAX_MOVES = 10
        self.GEM_SPAWN_COUNT = 30

        # Visual constants
        self.TILE_W = 40
        self.TILE_H = 20
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 80

        # Colors
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID = (60, 65, 80)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TEXT_SHADOW = (10, 10, 10)
        self.GEM_COLORS = [
            (0, 255, 255),  # Cyan
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Yellow
        ]
        
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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # State variables (initialized in reset)
        self.player_pos = None
        self.gems = None
        self.score = None
        self.moves_left = None
        self.game_over = None
        self.win_condition_met = None
        self.last_gem_collect_pos = None
        
        # Initialize state
        self.reset()
        
        # Run validation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = np.array([self.GRID_W // 2, self.GRID_H // 2])
        
        # Generate unique gem positions
        self.gems = set()
        while len(self.gems) < self.GEM_SPAWN_COUNT:
            gx = self.np_random.integers(0, self.GRID_W)
            gy = self.np_random.integers(0, self.GRID_H)
            if (gx, gy) != tuple(self.player_pos):
                self.gems.add((gx, gy))

        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.win_condition_met = False
        self.last_gem_collect_pos = None

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # Every step consumes a move
        self.moves_left -= 1
        reward = 0
        self.last_gem_collect_pos = None

        # Update player position based on movement action
        if movement == 1:  # Up
            self.player_pos[1] -= 1
        elif movement == 2:  # Down
            self.player_pos[1] += 1
        elif movement == 3:  # Left
            self.player_pos[0] -= 1
        elif movement == 4:  # Right
            self.player_pos[0] += 1
        
        # Clamp player position to grid boundaries
        self.player_pos = np.clip(self.player_pos, [0, 0], [self.GRID_W - 1, self.GRID_H - 1])

        # Check for gem collection
        player_pos_tuple = tuple(self.player_pos)
        if player_pos_tuple in self.gems:
            self.gems.remove(player_pos_tuple)
            self.score += 1
            reward += 1
            self.last_gem_collect_pos = player_pos_tuple
            # SFX: Gem collect sound

        # Check for termination conditions
        terminated = False
        if self.score >= self.GEM_TARGET:
            self.win_condition_met = True
            terminated = True
            reward += 50  # Victory bonus
            # SFX: Win sound
        elif self.moves_left <= 0:
            terminated = True
            # SFX: Lose sound

        if terminated:
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _iso_to_cart(self, gx, gy):
        """Converts isometric grid coordinates to cartesian screen coordinates."""
        sx = self.ORIGIN_X + (gx - gy) * (self.TILE_W / 2)
        sy = self.ORIGIN_Y + (gx + gy) * (self.TILE_H / 2)
        return int(sx), int(sy)

    def _draw_iso_poly(self, gx, gy, color, surface):
        """Draws a filled isometric rhombus on the grid."""
        points = [
            self._iso_to_cart(gx, gy + 1),
            self._iso_to_cart(gx + 1, gy + 1),
            self._iso_to_cart(gx + 1, gy),
            self._iso_to_cart(gx, gy),
        ]
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)
        
    def _draw_text(self, text, pos, font, color, shadow_color=None):
        if shadow_color:
            text_surf_shadow = font.render(text, True, shadow_color)
            self.screen.blit(text_surf_shadow, (pos[0] + 2, pos[1] + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid and elements from back to front
        for gy in range(self.GRID_H):
            for gx in range(self.GRID_W):
                # Draw grid lines
                top_point = self._iso_to_cart(gx, gy)
                right_point = self._iso_to_cart(gx + 1, gy)
                bottom_point = self._iso_to_cart(gx + 1, gy + 1)
                left_point = self._iso_to_cart(gx, gy + 1)
                
                pygame.draw.line(self.screen, self.COLOR_GRID, top_point, right_point)
                pygame.draw.line(self.screen, self.COLOR_GRID, right_point, bottom_point)

                # Draw gems
                if (gx, gy) in self.gems:
                    gem_center_x, gem_center_y = self._iso_to_cart(gx + 0.5, gy + 0.5)
                    gem_color_index = (gx + gy) % len(self.GEM_COLORS)
                    gem_color = self.GEM_COLORS[gem_color_index]
                    
                    # Glow effect
                    glow_radius = 10
                    glow_color = (*gem_color, 50)
                    temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                    pygame.draw.circle(temp_surf, glow_color, (glow_radius, glow_radius), glow_radius)
                    self.screen.blit(temp_surf, (gem_center_x - glow_radius, gem_center_y - glow_radius))

                    # Gem shape
                    pygame.gfxdraw.aacircle(self.screen, gem_center_x, gem_center_y, 5, gem_color)
                    pygame.gfxdraw.filled_circle(self.screen, gem_center_x, gem_center_y, 5, gem_color)
        
        # Draw player
        px, py = self.player_pos
        player_center_x, player_center_y = self._iso_to_cart(px + 0.5, py + 0.5)
        
        # Player marker (a solid diamond)
        player_poly = [
            self._iso_to_cart(px, py + 0.5),
            self._iso_to_cart(px + 0.5, py + 1),
            self._iso_to_cart(px + 1, py + 0.5),
            self._iso_to_cart(px + 0.5, py),
        ]
        pygame.gfxdraw.aapolygon(self.screen, player_poly, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, player_poly, self.COLOR_PLAYER)

        # Draw gem collection effect (if any) for one frame
        if self.last_gem_collect_pos:
            gx, gy = self.last_gem_collect_pos
            center_x, center_y = self._iso_to_cart(gx + 0.5, gy + 0.5)
            gem_color_index = (gx + gy) % len(self.GEM_COLORS)
            color = self.GEM_COLORS[gem_color_index]
            # Draw a simple starburst
            for i in range(8):
                angle = i * (math.pi / 4)
                x1 = center_x + math.cos(angle) * 8
                y1 = center_y + math.sin(angle) * 8
                x2 = center_x + math.cos(angle) * 15
                y2 = center_y + math.sin(angle) * 15
                pygame.draw.line(self.screen, color, (x1, y1), (x2, y2), 2)
    
    def _render_ui(self):
        # Draw UI text
        self._draw_text(
            f"Gems: {self.score} / {self.GEM_TARGET}",
            (20, 20), self.font_main, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW
        )
        self._draw_text(
            f"Moves: {self.moves_left}",
            (self.WIDTH - 150, 20), self.font_main, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW
        )

        # Draw game over message
        if self.game_over:
            if self.win_condition_met:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 + 150))
            shadow_surf = self.font_large.render(msg, True, self.COLOR_TEXT_SHADOW)
            shadow_rect = shadow_surf.get_rect(center=(self.WIDTH / 2 + 3, self.HEIGHT / 2 + 153))
            
            self.screen.blit(shadow_surf, shadow_rect)
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "moves_left": self.moves_left,
            "player_pos": tuple(self.player_pos),
        }
        
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

if __name__ == "__main__":
    # To play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption("Gem Collector")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    print(env.game_description)
    print(env.user_guide)

    while not done:
        # Get action from keyboard
        movement = 0 # No-op
        space = 0
        shift = 0

        # This event loop is for manual play, not for the agent
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
                
                # We only want to step when a movement key is pressed
                if movement != 0:
                    action = [movement, space, shift]
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated
                    print(f"Action: {action}, Reward: {reward}, Info: {info}")

        # Render the observation to the display window
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if done:
            print("Game Over!")
            pygame.time.wait(3000) # Wait 3 seconds before closing

    env.close()