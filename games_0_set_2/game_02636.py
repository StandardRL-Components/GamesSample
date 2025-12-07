import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    An isometric grid-based puzzle game where the player collects gems while avoiding traps.
    The goal is to collect all gems to win. Hitting a trap or running out of steps ends the game.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to move your avatar one tile at a time on the grid."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate an isometric grid, strategically collecting gems while avoiding traps to reach the target score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    # Colors
    COLOR_BG = (25, 35, 45)
    COLOR_GRID_TOP = (70, 80, 90)
    COLOR_GRID_SIDE = (50, 60, 70)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (200, 200, 255, 60)
    COLOR_TRAP = (40, 40, 50)
    COLOR_TRAP_SPIKE = (200, 50, 50)
    COLOR_TEXT = (230, 230, 230)
    GEM_COLORS = [
        (50, 150, 255),  # Blue
        (50, 255, 150),  # Green
        (255, 200, 50),  # Yellow
        (255, 100, 100), # Red
    ]

    # Grid and Tile Dimensions
    GRID_WIDTH = 12
    GRID_HEIGHT = 12
    TILE_WIDTH = 48
    TILE_HEIGHT = 24
    TILE_DEPTH = 10

    # Game Parameters
    NUM_GEMS = 20
    NUM_TRAPS = 5
    MAX_STEPS = 1000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        # This is for seeding the random number generator
        self.np_random = None

        # State variables are initialized in reset()
        self.player_pos = None
        self.gems = None
        self.traps = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.win = None
        self.particles = None
        
        self.screen_center_x = self.screen.get_width() // 2
        self.screen_center_y = self.screen.get_height() // 2 - self.TILE_HEIGHT * 4

        # The reset call in __init__ is necessary to initialize the state
        # before any other method is called.
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.particles = []

        self._generate_level()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        """Generates a new random level layout."""
        all_coords = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        
        # Use the seeded random number generator
        self.np_random.shuffle(all_coords)

        self.player_pos = np.array(all_coords.pop())
        
        # Ensure the player doesn't spawn surrounded by traps
        # This simple check removes adjacent tiles from trap placement possibilities
        px, py = self.player_pos
        safe_zone = {(px, py), (px+1, py), (px-1, py), (px, py+1), (px, py-1)}
        placeable_coords = [c for c in all_coords if c not in safe_zone]

        self.traps = [np.array(pos) for pos in placeable_coords[:self.NUM_TRAPS]]
        self.gems = [
            {'pos': np.array(pos), 'color_idx': self.np_random.integers(0, len(self.GEM_COLORS))}
            for pos in placeable_coords[self.NUM_TRAPS : self.NUM_TRAPS + self.NUM_GEMS]
        ]

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        self.steps += 1
        
        # --- Calculate pre-move state for reward shaping ---
        dist_gem_before = self._find_closest_distance(self.player_pos, [g['pos'] for g in self.gems])
        dist_trap_before = self._find_closest_distance(self.player_pos, self.traps)

        # --- Process Movement ---
        new_pos = self.player_pos.copy()
        if movement == 1:  # Up
            new_pos[1] -= 1
        elif movement == 2:  # Down
            new_pos[1] += 1
        elif movement == 3:  # Left
            new_pos[0] -= 1
        elif movement == 4:  # Right
            new_pos[0] += 1
        
        # --- Boundary Checks ---
        if (0 <= new_pos[0] < self.GRID_WIDTH) and (0 <= new_pos[1] < self.GRID_HEIGHT):
            self.player_pos = new_pos
        else:
            reward -= 0.1 # Small penalty for bumping into walls

        # --- Calculate post-move state for reward shaping ---
        dist_gem_after = self._find_closest_distance(self.player_pos, [g['pos'] for g in self.gems])
        dist_trap_after = self._find_closest_distance(self.player_pos, self.traps)

        # Distance-based rewards
        if dist_gem_after < dist_gem_before:
            reward += 1.0
        if dist_trap_after < dist_trap_before:
            reward -= 1.0

        # --- Check for Interactions (Gems and Traps) ---
        # Gem Collection
        gem_to_remove = -1
        for i, gem in enumerate(self.gems):
            if np.array_equal(self.player_pos, gem['pos']):
                reward += 10.0
                self.score += 10
                self._spawn_particles(gem['pos'], self.GEM_COLORS[gem['color_idx']])
                gem_to_remove = i
                break
        if gem_to_remove != -1:
            self.gems.pop(gem_to_remove)

        # Trap Collision
        for trap_pos in self.traps:
            if np.array_equal(self.player_pos, trap_pos):
                reward = -100.0
                self.score -= 100
                self.game_over = True
                self.win = False
                break

        # --- Check Termination Conditions ---
        if not self.gems: # Win condition
            reward += 100.0
            self.score += 100
            self.game_over = True
            self.win = True

        if self.steps >= self.MAX_STEPS and not self.game_over:
            self.game_over = True # Ran out of time
            self.win = False

        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._draw_grid()
        self._update_and_draw_particles()
        self._draw_player()
        self._draw_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gems_remaining": len(self.gems),
            "player_pos": tuple(self.player_pos),
        }

    # --- Helper & Drawing Methods ---

    def _grid_to_screen(self, x, y):
        """Converts isometric grid coordinates to screen coordinates."""
        screen_x = self.screen_center_x + (x - y) * (self.TILE_WIDTH // 2)
        screen_y = self.screen_center_y + (x + y) * (self.TILE_HEIGHT // 2)
        return int(screen_x), int(screen_y)

    def _draw_iso_tile(self, surface, x, y, top_color, side_color, depth):
        """Draws a single isometric tile with 3D perspective."""
        sx, sy = self._grid_to_screen(x, y)
        hw, hh = self.TILE_WIDTH // 2, self.TILE_HEIGHT // 2
        
        points_top = [(sx, sy), (sx + hw, sy + hh), (sx, sy + self.TILE_HEIGHT), (sx - hw, sy + hh)]
        points_left = [(sx - hw, sy + hh), (sx, sy + self.TILE_HEIGHT), (sx, sy + self.TILE_HEIGHT + depth), (sx - hw, sy + hh + depth)]
        points_right = [(sx + hw, sy + hh), (sx, sy + self.TILE_HEIGHT), (sx, sy + self.TILE_HEIGHT + depth), (sx + hw, sy + hh + depth)]

        pygame.gfxdraw.filled_polygon(surface, points_left, side_color)
        pygame.gfxdraw.aapolygon(surface, points_left, side_color)
        pygame.gfxdraw.filled_polygon(surface, points_right, side_color)
        pygame.gfxdraw.aapolygon(surface, points_right, side_color)
        pygame.gfxdraw.filled_polygon(surface, points_top, top_color)
        pygame.gfxdraw.aapolygon(surface, points_top, top_color)

    def _draw_grid(self):
        """Draws the entire grid, including traps and gems."""
        # Draw from back to front for correct layering
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                self._draw_iso_tile(self.screen, x, y, self.COLOR_GRID_TOP, self.COLOR_GRID_SIDE, self.TILE_DEPTH)

        for trap_pos in self.traps:
            self._draw_trap(trap_pos)

        for gem in self.gems:
            self._draw_gem(gem)

    def _draw_trap(self, pos):
        """Draws a trap tile."""
        sx, sy = self._grid_to_screen(pos[0], pos[1])
        self._draw_iso_tile(self.screen, pos[0], pos[1], self.COLOR_TRAP, self.COLOR_TRAP, self.TILE_DEPTH)
        
        # Draw spikes
        spike_base_y = sy + self.TILE_HEIGHT // 2
        pygame.draw.line(self.screen, self.COLOR_TRAP_SPIKE, (sx - 8, spike_base_y), (sx, spike_base_y - 8), 2)
        pygame.draw.line(self.screen, self.COLOR_TRAP_SPIKE, (sx + 8, spike_base_y), (sx, spike_base_y - 8), 2)
        pygame.draw.line(self.screen, self.COLOR_TRAP_SPIKE, (sx, spike_base_y + 4), (sx, spike_base_y - 8), 2)

    def _draw_gem(self, gem):
        """Draws a gem with a bobbing animation."""
        pos = gem['pos']
        color = self.GEM_COLORS[gem['color_idx']]
        sx, sy = self._grid_to_screen(pos[0], pos[1])
        
        # Bobbing animation
        bob_offset = math.sin(self.steps * 0.1 + sx) * 4 - 8
        sy += int(bob_offset)

        # Diamond shape
        hw, hh = 8, 8
        points = [(sx, sy - hh), (sx + hw, sy), (sx, sy + hh), (sx - hw, sy)]
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, (255, 255, 255, 150))
        
    def _draw_player(self):
        """Draws the player avatar with a glow effect."""
        sx, sy = self._grid_to_screen(self.player_pos[0], self.player_pos[1])
        sy -= self.TILE_HEIGHT // 2 # Center on tile
        
        # Glow effect
        glow_radius = int(12 + math.sin(self.steps * 0.2) * 2)
        pygame.gfxdraw.filled_circle(self.screen, sx, sy, glow_radius, self.COLOR_PLAYER_GLOW)
        
        # Player circle
        pygame.gfxdraw.aacircle(self.screen, sx, sy, 8, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, sx, sy, 8, self.COLOR_PLAYER)

    def _draw_ui(self):
        """Draws the score and other UI elements."""
        score_text = self.font.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        gems_text = self.font.render(f"Gems: {len(self.gems)}/{self.NUM_GEMS}", True, self.COLOR_TEXT)
        self.screen.blit(gems_text, (self.screen.get_width() - gems_text.get_width() - 10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.screen.get_width(), self.screen.get_height()), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "YOU WIN!" if self.win else "GAME OVER"
            end_text = self.font.render(end_text_str, True, self.COLOR_PLAYER)
            text_rect = end_text.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2))
            self.screen.blit(end_text, text_rect)

    def _find_closest_distance(self, pos, entity_list):
        """Calculates Manhattan distance to the closest entity in a list."""
        if not entity_list:
            return float('inf')
        distances = [np.sum(np.abs(pos - entity_pos)) for entity_pos in entity_list]
        return min(distances)

    def _spawn_particles(self, grid_pos, color):
        """Spawns collection particles at a grid location."""
        sx, sy = self._grid_to_screen(grid_pos[0], grid_pos[1])
        for _ in range(10):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({'pos': [sx, sy], 'vel': vel, 'lifetime': lifetime, 'color': color})

    def _update_and_draw_particles(self):
        """Updates particle positions and lifetimes, and draws them."""
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['lifetime'] / 30))
                color = (*p['color'], alpha)
                temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
                pygame.draw.rect(temp_surf, color, (0, 0, 4, 4))
                self.screen.blit(temp_surf, (int(p['pos'][0]), int(p['pos'][1])))

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly.
    # It will not work with the default "dummy" video driver.
    # To run, you might need to unset the SDL_VIDEODRIVER environment variable.
    # E.g., in bash: `unset SDL_VIDEODRIVER; python your_script_name.py`
    
    # Re-enable display for interactive mode
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Or 'windows', 'cocoa' etc. depending on your OS

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Gem Collector")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = np.array([0, 0, 0]) # Default action: no-op
        
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
                    continue
                elif event.key == pygame.K_ESCAPE:
                    running = False
        
        # Only step if a move key was pressed
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")

        # Render the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print("Game Over. Press 'r' to restart.")
        
        clock.tick(30) # Limit FPS for human play

    env.close()