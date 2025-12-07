
# Generated: 2025-08-27T18:31:08.573761
# Source Brief: brief_01858.md
# Brief Index: 1858

        
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
        "Controls: Arrow keys to move your avatar on the isometric grid. Avoid traps and collect all the gems."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic puzzle game. Navigate an isometric grid, collecting gems while avoiding hidden traps to achieve the highest score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 10
        self.NUM_GEMS = 25
        self.NUM_TRAPS = 15
        self.MAX_STEPS = 1000

        # Visual constants
        self.TILE_WIDTH_HALF = 28
        self.TILE_HEIGHT_HALF = 14
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 100

        # Colors
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_TILE_TOP = (60, 65, 75)
        self.COLOR_TILE_SIDE = (50, 55, 65)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (255, 255, 255, 50)
        self.GEM_COLORS = [(255, 80, 80), (80, 120, 255), (80, 255, 120), (255, 255, 80)]
        self.COLOR_TRAP = (70, 75, 85)
        self.COLOR_TRAP_ACTIVE = (255, 50, 50)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TEXT_SHADOW = (10, 10, 10)
        
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
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Initialize state variables
        self.player_pos = [0, 0]
        self.gems = []
        self.traps = []
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.victory = False
        self.trap_activated_pos = None
        self.collected_gems_pos = []
        
        self.reset()
        self.validate_implementation()
    
    def _generate_level(self):
        """Generates positions for player, gems, and traps."""
        all_coords = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        self.np_random.shuffle(all_coords)

        # Player start position
        self.player_pos = list(all_coords.pop())
        
        # Define safe zone around player start
        safe_zone = set()
        px, py = self.player_pos
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if abs(x - px) + abs(y - py) <= 3:
                    safe_zone.add((x, y))
        
        # Filter out safe zone for trap placement
        possible_trap_coords = [c for c in all_coords if c not in safe_zone]
        
        # Place traps
        num_traps_to_place = min(self.NUM_TRAPS, len(possible_trap_coords))
        trap_indices = self.np_random.choice(len(possible_trap_coords), num_traps_to_place, replace=False)
        self.traps = [list(possible_trap_coords[i]) for i in trap_indices]
        
        # Place gems in remaining spots
        trap_set = set(map(tuple, self.traps))
        possible_gem_coords = [c for c in all_coords if c != tuple(self.player_pos) and c not in trap_set]
        num_gems_to_place = min(self.NUM_GEMS, len(possible_gem_coords))
        gem_indices = self.np_random.choice(len(possible_gem_coords), num_gems_to_place, replace=False)
        self.gems = [list(possible_gem_coords[i]) for i in gem_indices]
        self.total_gems_initial = len(self.gems)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_level()
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.victory = False
        self.trap_activated_pos = None
        self.collected_gems_pos = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement = action[0]
        
        reward = 0.0
        old_pos = list(self.player_pos)
        
        # Update player position
        if movement == 1: # Up
            self.player_pos[1] -= 1
        elif movement == 2: # Down
            self.player_pos[1] += 1
        elif movement == 3: # Left
            self.player_pos[0] -= 1
        elif movement == 4: # Right
            self.player_pos[0] += 1
        
        # Boundary checks
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.GRID_WIDTH - 1)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.GRID_HEIGHT - 1)
        
        moved = old_pos != self.player_pos
        
        # Calculate distance-based rewards
        if moved:
            reward += self._calculate_distance_reward(old_pos, self.player_pos)
        else: # Penalty for no-op or hitting a wall
            reward -= 0.05

        # Check for gem collection
        if self.player_pos in self.gems:
            self.gems.remove(self.player_pos)
            self.collected_gems_pos.append(self.player_pos)
            reward += 10.0  # Event-based reward for collecting a gem
            self.score += 10.0
            # // Sound: Gem collect sfx
            
        # Check for trap activation
        if self.player_pos in self.traps:
            self.game_over = True
            self.trap_activated_pos = self.player_pos
            reward -= 100.0  # Event-based penalty for falling into a trap
            self.score -= 100.0
            # // Sound: Trap spring sfx

        # Check for victory
        if not self.gems:
            self.victory = True
            self.game_over = True
            reward += 50.0  # Goal-oriented reward for winning
            self.score += 50.0
            # // Sound: Victory fanfare sfx

        self.steps += 1
        terminated = self.game_over or (self.steps >= self.MAX_STEPS)
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _calculate_distance_reward(self, old_pos, new_pos):
        reward = 0.0
        
        # Distance to nearest gem
        if self.gems:
            dist_old_gem = self._find_closest_entity_dist(old_pos, self.gems)
            dist_new_gem = self._find_closest_entity_dist(new_pos, self.gems)
            if dist_new_gem < dist_old_gem:
                reward += 1.0
        
        # Distance to nearest trap
        if self.traps:
            dist_old_trap = self._find_closest_entity_dist(old_pos, self.traps)
            dist_new_trap = self._find_closest_entity_dist(new_pos, self.traps)
            if dist_new_trap < dist_old_trap:
                reward -= 0.1
                
        return reward

    def _find_closest_entity_dist(self, pos, entity_list):
        if not entity_list:
            return float('inf')
        
        min_dist = float('inf')
        for entity_pos in entity_list:
            dist = abs(pos[0] - entity_pos[0]) + abs(pos[1] - entity_pos[1])
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _iso_to_screen(self, x, y):
        """Converts isometric grid coordinates to screen coordinates."""
        screen_x = self.ORIGIN_X + (x - y) * self.TILE_WIDTH_HALF
        screen_y = self.ORIGIN_Y + (x + y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _get_tile_poly(self, x, y, z_offset=0):
        """Gets the 4 vertices of the top face of a tile."""
        sx, sy = self._iso_to_screen(x, y)
        sy -= z_offset
        return [
            (sx, sy - self.TILE_HEIGHT_HALF),
            (sx + self.TILE_WIDTH_HALF, sy),
            (sx, sy + self.TILE_HEIGHT_HALF),
            (sx - self.TILE_WIDTH_HALF, sy),
        ]

    def _render_text(self, text, font, color, pos, shadow_color=None, shadow_offset=2):
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect(center=pos)
        if shadow_color:
            shadow_surf = font.render(text, True, shadow_color)
            shadow_rect = shadow_surf.get_rect(center=(pos[0] + shadow_offset, pos[1] + shadow_offset))
            self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

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
                poly_top = self._get_tile_poly(x, y)
                pygame.gfxdraw.filled_polygon(self.screen, poly_top, self.COLOR_TILE_TOP)
                
                # Draw side faces for a 3D effect
                bottom_y = poly_top[2][1] + 8
                poly_side_left = [poly_top[3], poly_top[2], (poly_top[2][0], bottom_y), (poly_top[3][0], bottom_y)]
                poly_side_right = [poly_top[1], poly_top[2], (poly_top[2][0], bottom_y), (poly_top[1][0], bottom_y)]
                pygame.gfxdraw.filled_polygon(self.screen, poly_side_left, self.COLOR_TILE_SIDE)
                pygame.gfxdraw.filled_polygon(self.screen, poly_side_right, self.COLOR_TILE_SIDE)

                # Outline the tiles
                pygame.gfxdraw.aapolygon(self.screen, poly_top, self.COLOR_GRID)

        # Draw traps
        for tx, ty in self.traps:
            poly = self._get_tile_poly(tx, ty)
            color = self.COLOR_TRAP_ACTIVE if (tx, ty) == self.trap_activated_pos else self.COLOR_TRAP
            pygame.gfxdraw.filled_polygon(self.screen, poly, color)
            pygame.gfxdraw.aapolygon(self.screen, poly, self.COLOR_GRID)

        # Draw gems
        for i, (gx, gy) in enumerate(self.gems):
            poly = self._get_tile_poly(gx, gy, z_offset=8)
            color = self.GEM_COLORS[i % len(self.GEM_COLORS)]
            pygame.gfxdraw.filled_polygon(self.screen, poly, color)
            pygame.gfxdraw.aapolygon(self.screen, poly, (255, 255, 255, 150))
        
        # Draw player
        px, py = self.player_pos
        player_poly = self._get_tile_poly(px, py, z_offset=12)
        
        # Glow effect
        glow_poly = self._get_tile_poly(px, py, z_offset=12)
        glow_rect = pygame.Rect(glow_poly[3][0]-5, glow_poly[0][1]-5, self.TILE_WIDTH_HALF*2+10, self.TILE_HEIGHT_HALF*2+10)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.ellipse(glow_surf, self.COLOR_PLAYER_GLOW, glow_surf.get_rect())
        self.screen.blit(glow_surf, glow_rect.topleft)

        # Player body
        pygame.gfxdraw.filled_polygon(self.screen, player_poly, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, player_poly, self.COLOR_GRID)

    def _render_ui(self):
        # Score display
        score_text = f"Score: {self.score:.1f}"
        self._render_text(score_text, self.font_ui, self.COLOR_TEXT, (85, 25), self.COLOR_TEXT_SHADOW)
        
        # Gems remaining display
        gems_text = f"Gems: {len(self.gems)}/{self.total_gems_initial}"
        self._render_text(gems_text, self.font_ui, self.COLOR_TEXT, (self.WIDTH - 85, 25), self.COLOR_TEXT_SHADOW)

        # Game over / Victory message
        if self.game_over:
            if self.victory:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            self._render_text(msg, self.font_msg, color, (self.WIDTH // 2, self.HEIGHT // 2), self.COLOR_TEXT_SHADOW, 4)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gems_remaining": len(self.gems),
            "player_pos": self.player_pos,
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
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Isometric Gem Collector")
    clock = pygame.time.Clock()
    
    terminated = False
    
    print(env.user_guide)

    while not terminated:
        action = [0, 0, 0]  # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
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
                    print("--- Game Reset ---")

        if action[0] != 0: # Only step if a move key was pressed
            obs, reward, term, trunc, info = env.step(action)
            terminated = term
            print(f"Step: {info['steps']}, Action: {action[0]}, Reward: {reward:.2f}, Score: {info['score']:.1f}, Terminated: {terminated}")

        # Render the observation to the display window
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit frame rate

    env.close()
    pygame.quit()