import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set up Pygame to run in a headless environment
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    An isometric puzzle game where the player collects gems and avoids traps.
    The game is presented in a clean, geometric visual style with a fixed camera.
    It's a turn-based environment, advancing state only upon receiving an action.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing strings
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to move one tile at a time on the grid."
    )
    game_description = (
        "Navigate an isometric grid, collecting all 10 gems while avoiding the deadly traps."
    )

    # Frame advance behavior
    auto_advance = False

    # Game constants
    GRID_WIDTH = 12
    GRID_HEIGHT = 12
    TILE_WIDTH_ISO = 48
    TILE_HEIGHT_ISO = 24
    NUM_GEMS = 10
    NUM_TRAPS = 5
    MAX_STEPS = 250

    # Color palette
    COLOR_BG = (25, 35, 45)
    COLOR_GRID = (60, 80, 100)
    COLOR_PLAYER = (50, 150, 255)
    COLOR_PLAYER_GLOW = (150, 200, 255, 50)
    COLOR_GEM = (255, 220, 50)
    COLOR_GEM_GLOW = (255, 240, 150, 70)
    COLOR_TRAP = (200, 50, 75)
    COLOR_TRAP_GLOW = (255, 100, 125, 60)
    COLOR_TEXT = (230, 240, 250)
    
    class Particle:
        """A simple particle class for visual effects."""
        def __init__(self, x, y, color, size, lifespan, angle, speed):
            self.x = x
            self.y = y
            self.color = color
            self.size = size
            self.lifespan = lifespan
            self.max_lifespan = lifespan
            self.vx = math.cos(angle) * speed
            self.vy = math.sin(angle) * speed
            self.gravity = 0.1

        def update(self):
            self.x += self.vx
            self.y += self.vy
            self.vy += self.gravity
            self.lifespan -= 1
            self.size = max(0, self.size - 0.1)

        def draw(self, surface):
            if self.lifespan > 0:
                alpha = int(255 * (self.lifespan / self.max_lifespan))
                temp_surf = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, (*self.color, alpha), (int(self.size), int(self.size)), int(self.size))
                surface.blit(temp_surf, (int(self.x - self.size), int(self.y - self.size)), special_flags=pygame.BLEND_RGBA_ADD)


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        # Calculate grid origin to center it
        self.origin_x = self.screen.get_width() // 2
        self.origin_y = (self.screen.get_height() // 2) - (self.GRID_HEIGHT * self.TILE_HEIGHT_ISO // 4) + 40

        # Initialize state variables
        self.player_pos = (0, 0)
        self.gem_locations = set()
        self.trap_locations = set()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.np_random = None

        # This will be properly initialized by the first call to reset()
        # self.reset() # Avoid calling reset in init, as per Gymnasium guidelines
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []

        # Place player
        self.player_pos = (self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT))

        # Generate unique locations for gems and traps
        possible_locations = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        possible_locations.remove(self.player_pos)
        
        random_indices = self.np_random.choice(len(possible_locations), size=self.NUM_GEMS + self.NUM_TRAPS, replace=False)
        chosen_locations = [possible_locations[i] for i in random_indices]

        self.gem_locations = set(chosen_locations[:self.NUM_GEMS])
        self.trap_locations = set(chosen_locations[self.NUM_GEMS:])
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        # space_held = action[1] == 1 # Unused
        # shift_held = action[2] == 1 # Unused

        # --- Calculate pre-move state for reward shaping ---
        old_dist_gem = self._get_distance_to_nearest(self.gem_locations)
        
        # --- Update player position ---
        px, py = self.player_pos
        if movement == 1: # Up
            py -= 1
        elif movement == 2: # Down
            py += 1
        elif movement == 3: # Left
            px -= 1
        elif movement == 4: # Right
            px += 1
        
        # Clamp to grid boundaries
        px = max(0, min(self.GRID_WIDTH - 1, px))
        py = max(0, min(self.GRID_HEIGHT - 1, py))
        self.player_pos = (px, py)
        
        # --- Calculate reward ---
        reward = 0
        
        # Reward for moving towards gem
        new_dist_gem = self._get_distance_to_nearest(self.gem_locations)

        if old_dist_gem is not None and new_dist_gem is not None:
            if new_dist_gem < old_dist_gem:
                reward += 1.0  # Moved closer to a gem
        
        # Small penalty for each step to encourage efficiency
        reward -= 0.01

        # --- Check for events ---
        if self.player_pos in self.gem_locations:
            # sfx: gem_collect.wav
            self.score += 1
            reward += 10.0
            self.gem_locations.remove(self.player_pos)
            self._create_particles(self.player_pos, self.COLOR_GEM, 20)
            if not self.gem_locations: # Win condition
                reward += 50.0
                self.game_over = True

        if self.player_pos in self.trap_locations:
            # sfx: trap_spring.wav
            reward -= 50.0
            self.game_over = True
            self._create_particles(self.player_pos, self.COLOR_TRAP, 30)

        # --- Update state and check for termination ---
        self.steps += 1
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True

        terminated = self.game_over and not truncated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
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
            "gems_remaining": len(self.gem_locations),
            "player_pos": self.player_pos,
        }

    def _grid_to_iso(self, x, y):
        """Converts grid coordinates to isometric screen coordinates."""
        iso_x = self.origin_x + (x - y) * (self.TILE_WIDTH_ISO / 2)
        iso_y = self.origin_y + (x + y) * (self.TILE_HEIGHT_ISO / 2)
        return int(iso_x), int(iso_y)

    def _render_game(self):
        # Update and draw particles
        for p in self.particles[:]:
            p.update()
            if p.lifespan <= 0:
                self.particles.remove(p)
            else:
                p.draw(self.screen)

        # Draw grid, traps, gems, and player in order
        entities_to_draw = []
        
        # Add grid tiles
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                entities_to_draw.append(('grid', (x, y)))

        # Add traps
        for pos in self.trap_locations:
            entities_to_draw.append(('trap', pos))

        # Add gems
        for pos in self.gem_locations:
            entities_to_draw.append(('gem', pos))
            
        # Add player
        entities_to_draw.append(('player', self.player_pos))
        
        # Sort by y then x to ensure correct isometric overlap
        entities_to_draw.sort(key=lambda e: (e[1][0] + e[1][1], e[1][1]))
        
        for entity_type, pos in entities_to_draw:
            screen_pos = self._grid_to_iso(pos[0], pos[1])
            if entity_type == 'grid':
                self._draw_iso_tile(screen_pos, self.COLOR_GRID)
            elif entity_type == 'trap':
                self._draw_iso_item(screen_pos, self.COLOR_TRAP, self.COLOR_TRAP_GLOW, 'circle')
            elif entity_type == 'gem':
                self._draw_iso_item(screen_pos, self.COLOR_GEM, self.COLOR_GEM_GLOW, 'diamond')
            elif entity_type == 'player':
                self._draw_iso_item(screen_pos, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW, 'player')

    def _draw_iso_tile(self, screen_pos, color):
        x, y = screen_pos
        w, h = self.TILE_WIDTH_ISO, self.TILE_HEIGHT_ISO
        points = [
            (x, y - h / 2),
            (x + w / 2, y),
            (x, y + h / 2),
            (x - w / 2, y)
        ]
        # FIX: The original code used `pygame.gfxdraw.aalines`, which does not exist.
        # The correct function to draw an anti-aliased polygon outline is `aapolygon`.
        pygame.gfxdraw.aapolygon(self.screen, points, color)

    def _draw_iso_item(self, screen_pos, color, glow_color, item_type):
        x, y = screen_pos
        bob = math.sin(self.steps * 0.15 + x + y) * 3
        y_offset = y - self.TILE_HEIGHT_ISO / 2 - 5 + bob

        # Draw glow
        glow_radius = int(self.TILE_WIDTH_ISO * 0.4)
        # Use a temporary surface for the glow to handle alpha correctly
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, glow_color, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(temp_surf, (x - glow_radius, int(y_offset) - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        
        if item_type == 'player':
            radius = int(self.TILE_WIDTH_ISO * 0.25)
            pygame.gfxdraw.filled_circle(self.screen, x, int(y_offset), radius, color)
            pygame.gfxdraw.aacircle(self.screen, x, int(y_offset), radius, color)
        elif item_type == 'gem':
            size = self.TILE_WIDTH_ISO * 0.2
            points = [
                (x, y_offset - size*0.7), (x + size*0.5, y_offset),
                (x, y_offset + size*0.7), (x - size*0.5, y_offset)
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, color)
        elif item_type == 'circle': # For traps
            radius = int(self.TILE_WIDTH_ISO * 0.2)
            pygame.gfxdraw.filled_circle(self.screen, x, int(y_offset), radius, color)
            pygame.gfxdraw.aacircle(self.screen, x, int(y_offset), radius, color)

    def _render_ui(self):
        score_text = self.font.render(f"Gems: {self.score} / {self.NUM_GEMS}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        steps_text = self.font.render(f"Steps: {self.steps} / {self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.screen.get_width() - steps_text.get_width() - 10, 10))

    def _create_particles(self, grid_pos, color, count):
        sx, sy = self._grid_to_iso(grid_pos[0], grid_pos[1])
        sy -= self.TILE_HEIGHT_ISO / 2
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            lifespan = self.np_random.integers(20, 40)
            size = self.np_random.uniform(3, 8)
            self.particles.append(self.Particle(sx, sy, color, size, lifespan, angle, speed))
            
    def _get_distance_to_nearest(self, target_list):
        """Calculates Manhattan distance to the nearest target."""
        if not target_list:
            return None
        px, py = self.player_pos
        min_dist = float('inf')
        for tx, ty in target_list:
            dist = abs(px - tx) + abs(py - ty)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def close(self):
        pygame.quit()
        
if __name__ == '__main__':
    # This block allows you to play the game directly, requires a display.
    # Re-enable the default video driver for direct play.
    os.environ['SDL_VIDEODRIVER'] = 'x11'
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a Pygame window to display the rendered frames
    pygame.display.set_caption("Isometric Gem Collector")
    screen = pygame.display.set_mode((640, 400))
    
    terminated = False
    truncated = False
    running = True
    clock = pygame.time.Clock()

    print(env.game_description)
    print(env.user_guide)

    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                
                if terminated or truncated: # If game is over, any key press can reset
                    obs, info = env.reset()
                    terminated = False
                    truncated = False
                    continue

                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    terminated = False
                    truncated = False
                
                # Only step if an action key was pressed
                if action[0] != 0:
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}, Truncated: {truncated}")
        
        # Draw the observation from the environment to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit FPS

    env.close()