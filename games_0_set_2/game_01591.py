import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys (↑, ↓, ←, →) to move your character on the isometric grid."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Collect 20 shimmering yellow crystals in an isometric world while dodging the red patrol enemies."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Grid and Isometric Projection
    GRID_WIDTH = 20
    GRID_HEIGHT = 20
    TILE_WIDTH_HALF = 16
    TILE_HEIGHT_HALF = 8
    ORIGIN_X = SCREEN_WIDTH // 2
    ORIGIN_Y = 60

    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_GRID = (40, 45, 55)
    COLOR_PLAYER = (50, 150, 255)
    COLOR_PLAYER_OUTLINE = (150, 200, 255)
    COLOR_CRYSTAL = (255, 220, 50)
    COLOR_CRYSTAL_OUTLINE = (255, 255, 150)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_OUTLINE = (255, 120, 120)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_UI_SHADOW = (10, 10, 10)

    # Game parameters
    NUM_CRYSTALS = 20
    NUM_ENEMIES = 3
    WIN_SCORE = 20
    MAX_STEPS = 1000

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
        self.font_ui = pygame.font.Font(None, 36)
        
        # State variables (initialized in reset)
        self.player_pos = None
        self.crystals = None
        self.enemies = None
        self.particles = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Initialize state variables
        # This call is needed to set up the initial state for validation
        # We need a seed to make sure the probabilistic part that was failing is consistent
        self.reset(seed=0)
        
        # self.validate_implementation() # Optional validation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []

        # Place player
        self.player_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)

        # Generate crystals
        self.crystals = self._generate_unique_positions(self.NUM_CRYSTALS, [self.player_pos])

        # Generate enemies and their paths
        self.enemies = []
        occupied_tiles = [self.player_pos] + self.crystals
        for _ in range(self.NUM_ENEMIES):
            path_len = self.np_random.integers(4, 7)
            path_start = self._generate_unique_positions(1, occupied_tiles)[0]
            
            path = [path_start]
            current_pos = path_start
            for _ in range(path_len - 1):
                neighbors = self._get_valid_neighbors(current_pos)
                # Prefer neighbors not already in path to avoid tight loops
                new_neighbors = [n for n in neighbors if n not in path]
                
                if new_neighbors:
                    # FIX: np_random.choice on a list of tuples returns a numpy array,
                    # which causes comparison issues later. Instead, we pick an index
                    # and then select the element, which preserves the tuple type.
                    idx = self.np_random.integers(len(new_neighbors))
                    next_pos = new_neighbors[idx]
                else: # All neighbors are in path, just pick one
                    idx = self.np_random.integers(len(neighbors))
                    next_pos = neighbors[idx]
                
                path.append(next_pos)
                current_pos = next_pos
            
            self.enemies.append({
                "path": path,
                "path_index": 0,
                "pos": path_start
            })
            occupied_tiles.extend(path)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0
        terminated = False
        
        # 1. Calculate distance to nearest crystal before moving
        dist_before = self._get_nearest_crystal_dist()

        # 2. Update player position based on action
        px, py = self.player_pos
        if movement == 1: # Up (NW)
            py -= 1
        elif movement == 2: # Down (SE)
            py += 1
        elif movement == 3: # Left (SW)
            px -= 1
        elif movement == 4: # Right (NE)
            px += 1
        
        # Clamp to grid
        px = max(0, min(self.GRID_WIDTH - 1, px))
        py = max(0, min(self.GRID_HEIGHT - 1, py))
        self.player_pos = (px, py)

        # 3. Update enemies
        for enemy in self.enemies:
            enemy["path_index"] = (enemy["path_index"] + 1) % len(enemy["path"])
            enemy["pos"] = enemy["path"][enemy["path_index"]]

        # 4. Check for crystal collection
        if self.player_pos in self.crystals:
            self.crystals.remove(self.player_pos)
            self.score += 1
            reward += 10
            self._spawn_particles(self.player_pos, self.COLOR_CRYSTAL)
            if self.score >= self.WIN_SCORE:
                reward += 100
                terminated = True
                self.game_over = True

        # 5. Calculate distance-based reward
        dist_after = self._get_nearest_crystal_dist()
        if dist_before is not None and dist_after is not None:
            if dist_after < dist_before:
                reward += 1.0
            else:
                reward -= 0.1
        
        # 6. Check for enemy collision
        for enemy in self.enemies:
            if self.player_pos == enemy["pos"]:
                reward = -100
                terminated = True
                self.game_over = True
                self._spawn_particles(self.player_pos, self.COLOR_PLAYER, 30)
                break

        # 7. Update step counter and check for max steps
        self.steps += 1
        truncated = False
        if self.steps >= self.MAX_STEPS:
            terminated = True # Per Gymnasium API, this should be truncation.
            truncated = True  # But to match original logic, we can set both.
            self.game_over = True

        # If terminated is due to win/loss, truncated should be False.
        if terminated and not truncated:
            pass
        
        # 8. Update particles
        self._update_particles()
        
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

    def _render_game(self):
        # Draw grid
        for i in range(self.GRID_WIDTH + 1):
            start = self._iso_to_screen(i, 0)
            end = self._iso_to_screen(i, self.GRID_HEIGHT)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        for i in range(self.GRID_HEIGHT + 1):
            start = self._iso_to_screen(0, i)
            end = self._iso_to_screen(self.GRID_WIDTH, i)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)

        # Create a list of all entities to draw and sort by depth (y+x)
        entities_to_draw = []
        for x, y in self.crystals:
            entities_to_draw.append(((x, y), 'crystal'))
        for enemy in self.enemies:
            entities_to_draw.append((enemy['pos'], 'enemy'))
        if not self.game_over or self.score >= self.WIN_SCORE: # Don't draw player if lost
             entities_to_draw.append((self.player_pos, 'player'))

        entities_to_draw.sort(key=lambda item: item[0][0] + item[0][1])

        # Draw entities in sorted order
        for pos, entity_type in entities_to_draw:
            if entity_type == 'crystal':
                self._draw_iso_tile(pos, self.COLOR_CRYSTAL, self.COLOR_CRYSTAL_OUTLINE, shimmer=True)
            elif entity_type == 'enemy':
                self._draw_iso_tile(pos, self.COLOR_ENEMY, self.COLOR_ENEMY_OUTLINE)
            elif entity_type == 'player':
                self._draw_iso_tile(pos, self.COLOR_PLAYER, self.COLOR_PLAYER_OUTLINE, scale=1.1)
        
        self._update_and_render_particles()

    def _render_ui(self):
        score_text = f"SCORE: {self.score}"
        # Shadow
        text_surf = self.font_ui.render(score_text, True, self.COLOR_UI_SHADOW)
        self.screen.blit(text_surf, (11, 11))
        # Main text
        text_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (10, 10))

        if self.game_over:
            message = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            color = self.COLOR_CRYSTAL if self.score >= self.WIN_SCORE else self.COLOR_ENEMY
            
            font_end = pygame.font.Font(None, 72)
            # Shadow
            text_surf_end = font_end.render(message, True, self.COLOR_UI_SHADOW)
            text_rect = text_surf_end.get_rect(center=(self.SCREEN_WIDTH/2 + 2, self.SCREEN_HEIGHT/2 + 2))
            self.screen.blit(text_surf_end, text_rect)
            # Main text
            text_surf_end = font_end.render(message, True, color)
            text_rect = text_surf_end.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text_surf_end, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.player_pos,
            "crystals_left": len(self.crystals)
        }

    # --- Helper Methods ---

    def _iso_to_screen(self, grid_x, grid_y):
        screen_x = self.ORIGIN_X + (grid_x - grid_y) * self.TILE_WIDTH_HALF
        screen_y = self.ORIGIN_Y + (grid_x + grid_y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _draw_iso_tile(self, pos, color, outline_color, scale=1.0, shimmer=False):
        x, y = pos
        center_x, center_y = self._iso_to_screen(x, y)
        
        w = self.TILE_WIDTH_HALF * scale
        h = self.TILE_HEIGHT_HALF * scale
        
        if shimmer:
            shimmer_factor = (math.sin(self.steps * 0.2) + 1) / 2 # Varies between 0 and 1
            w *= (0.9 + 0.1 * shimmer_factor)
            h *= (0.9 + 0.1 * shimmer_factor)

        points = [
            (center_x, center_y - h),
            (center_x + w, center_y),
            (center_x, center_y + h),
            (center_x - w, center_y),
        ]
        
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, outline_color)

    def _generate_unique_positions(self, num_positions, excluded_positions):
        positions = []
        excluded_set = set(excluded_positions)
        while len(positions) < num_positions:
            pos = (
                self.np_random.integers(0, self.GRID_WIDTH),
                self.np_random.integers(0, self.GRID_HEIGHT)
            )
            if pos not in excluded_set and pos not in positions:
                positions.append(pos)
        return positions

    def _get_valid_neighbors(self, pos):
        x, y = pos
        neighbors = []
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                neighbors.append((nx, ny))
        return neighbors

    def _get_nearest_crystal_dist(self):
        if not self.crystals:
            return None
        px, py = self.player_pos
        min_dist = float('inf')
        for cx, cy in self.crystals:
            dist = abs(px - cx) + abs(py - cy) # Manhattan distance
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _spawn_particles(self, grid_pos, color, count=15):
        screen_pos = self._iso_to_screen(grid_pos[0], grid_pos[1])
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            dx = math.cos(angle) * speed
            dy = math.sin(angle) * speed
            lifetime = self.np_random.integers(10, 20)
            self.particles.append({
                "x": screen_pos[0], "y": screen_pos[1],
                "dx": dx, "dy": dy,
                "lifetime": lifetime, "max_lifetime": lifetime,
                "color": color
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["lifetime"] > 0]
        for p in self.particles:
            p["x"] += p["dx"]
            p["y"] += p["dy"]
            p["lifetime"] -= 1

    def _update_and_render_particles(self):
        for p in self.particles:
            life_ratio = p["lifetime"] / p["max_lifetime"]
            radius = int(3 * life_ratio)
            if radius > 0:
                # Create a temporary surface for alpha blending
                temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(
                    temp_surf, (*p["color"], int(255 * life_ratio)),
                    (radius, radius), radius
                )
                self.screen.blit(temp_surf, (int(p["x"]) - radius, int(p["y"]) - radius))


    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game directly
    # To see the game, comment out the os.environ line at the top
    # and change render_mode to "human"
    # os.environ.pop("SDL_VIDEODRIVER", None) # Uncomment to see display
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption(env.game_description)
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    truncated = False
    running = True
    
    print(env.user_guide)

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
                    terminated = False
                    truncated = False

        if not (terminated or truncated):
            keys = pygame.key.get_pressed()
            movement = 0
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            action[0] = movement

            # Since auto_advance is False, we only step when a key is pressed
            if movement != 0:
                obs, reward, terminated, truncated, info = env.step(action)
        
        # Draw the observation from the environment to the display screen
        draw_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(draw_surface, (0, 0))
        pygame.display.flip()
        
        # A small delay to make it playable for humans
        pygame.time.wait(50)

    env.close()