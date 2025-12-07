
# Generated: 2025-08-28T06:52:40.557467
# Source Brief: brief_03069.md
# Brief Index: 3069

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    An isometric arcade game where the player collects crystals while avoiding enemies.
    The game is presented in a vibrant, high-contrast visual style with particle effects.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move on the isometric grid. Collect 20 crystals to win. Avoid enemies!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Collect shimmering crystals in a vibrant isometric world while dodging cunning enemies that patrol the grid."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_SIZE = 20
        self.NUM_ENEMIES = 3
        self.NUM_CRYSTALS_TO_WIN = 20
        self.MAX_ENEMY_TOUCHES = 5
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (15, 19, 41)
        self.COLOR_GRID = (40, 50, 80)
        self.COLOR_PLAYER = (0, 255, 255)
        self.COLOR_PLAYER_GLOW = (0, 150, 150)
        self.COLOR_CRYSTAL = (255, 255, 0)
        self.COLOR_CRYSTAL_GLOW = (150, 150, 0)
        self.COLOR_ENEMY = (255, 0, 128)
        self.COLOR_ENEMY_GLOW = (150, 0, 80)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_PARTICLE_IMPACT = (255, 50, 50)
        self.COLOR_PARTICLE_COLLECT = (255, 255, 150)

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
        try:
            self.font_ui = pygame.font.Font(pygame.font.get_default_font(), 18)
            self.font_game_over = pygame.font.Font(pygame.font.get_default_font(), 40)
        except IOError:
            self.font_ui = pygame.font.SysFont("Arial", 18)
            self.font_game_over = pygame.font.SysFont("Arial", 40)

        # --- Isometric Projection ---
        self.tile_width = 24
        self.tile_height = 12
        self.origin_x = self.SCREEN_WIDTH // 2
        self.origin_y = self.SCREEN_HEIGHT // 2 - self.GRID_SIZE * self.tile_height // 3

        # --- Game State (initialized in reset) ---
        self.player_pos = None
        self.crystal_pos = None
        self.enemies = []
        self.particles = []
        self.steps = 0
        self.crystals_collected = 0
        self.enemy_touches = 0
        self.game_over = False
        self.win_message = ""
        self.np_random = None

        self.validate_implementation()

    def _iso_to_screen(self, grid_x, grid_y):
        """Converts isometric grid coordinates to screen pixel coordinates."""
        screen_x = self.origin_x + (grid_x - grid_y) * self.tile_width / 2
        screen_y = self.origin_y + (grid_x + grid_y) * self.tile_height / 2
        return int(screen_x), int(screen_y)

    def _generate_enemy_path(self):
        """Generates a random, looping path for an enemy."""
        path_len = self.np_random.integers(8, 13)
        start_x, start_y = self.np_random.integers(0, self.GRID_SIZE, size=2)
        
        # Ensure enemy path doesn't start on player's initial spot
        while start_x == self.GRID_SIZE // 2 and start_y == self.GRID_SIZE // 2:
             start_x, start_y = self.np_random.integers(0, self.GRID_SIZE, size=2)

        path = [(start_x, start_y)]
        current_x, current_y = start_x, start_y

        for _ in range(path_len - 1):
            possible_moves = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = current_x + dx, current_y + dy
                if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and (nx, ny) not in path:
                    possible_moves.append((nx, ny))
            
            if not possible_moves: # If stuck, create a simple back-and-forth
                path.append((path[-2][0], path[-2][1]))
            else:
                current_x, current_y = random.choice(possible_moves)
                path.append((current_x, current_y))
        
        return [list(p) for p in path]

    def _spawn_crystal(self):
        """Spawns a crystal in a valid, unoccupied location."""
        occupied_tiles = [self.player_pos] + [e['pos'] for e in self.enemies]
        possible_tiles = []
        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE):
                if [x, y] not in occupied_tiles:
                    possible_tiles.append([x, y])
        
        if not possible_tiles: # Failsafe if grid is full
            self.crystal_pos = [self.np_random.integers(0, self.GRID_SIZE), self.np_random.integers(0, self.GRID_SIZE)]
        else:
            idx = self.np_random.integers(0, len(possible_tiles))
            self.crystal_pos = possible_tiles[idx]

    def _create_particles(self, pos, count, color, life_range, speed_range):
        """Creates a burst of particles."""
        screen_pos = self._iso_to_screen(*pos)
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(*speed_range)
            velocity = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(*life_range)
            self.particles.append({
                'pos': list(screen_pos),
                'vel': velocity,
                'life': lifespan,
                'max_life': lifespan,
                'color': color
            })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        
        self.enemies = []
        for _ in range(self.NUM_ENEMIES):
            path = self._generate_enemy_path()
            self.enemies.append({'path': path, 'path_idx': 0, 'pos': path[0]})
        
        self._spawn_crystal()

        self.particles = []
        self.steps = 0
        self.crystals_collected = 0
        self.enemy_touches = 0
        self.game_over = False
        self.win_message = ""

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0

        # --- 1. Update Player Position ---
        if movement != 0:
            prev_pos = list(self.player_pos)
            if movement == 1: self.player_pos[1] -= 1  # Up -> Up-Left
            elif movement == 2: self.player_pos[1] += 1 # Down -> Down-Right
            elif movement == 3: self.player_pos[0] -= 1 # Left -> Down-Left
            elif movement == 4: self.player_pos[0] += 1 # Right -> Up-Right
            
            # Clamp to grid
            self.player_pos[0] = max(0, min(self.GRID_SIZE - 1, self.player_pos[0]))
            self.player_pos[1] = max(0, min(self.GRID_SIZE - 1, self.player_pos[1]))
        else:
            reward -= 0.2 # Penalty for no-op

        # --- 2. Update Enemies ---
        for enemy in self.enemies:
            enemy['path_idx'] = (enemy['path_idx'] + 1) % len(enemy['path'])
            enemy['pos'] = enemy['path'][enemy['path_idx']]

        # --- 3. Check Collisions & Events ---
        # Player-Crystal
        if self.player_pos == self.crystal_pos:
            self.crystals_collected += 1
            reward += 10
            # sfx: crystal collect sound
            self._create_particles(self.crystal_pos, 30, self.COLOR_PARTICLE_COLLECT, (15, 30), (1, 3))
            self._spawn_crystal()

        # Player-Enemy
        for enemy in self.enemies:
            if self.player_pos == enemy['pos']:
                self.enemy_touches += 1
                reward -= 5
                # sfx: player damage sound
                self._create_particles(self.player_pos, 40, self.COLOR_PARTICLE_IMPACT, (20, 40), (2, 4))
                break # Only one hit per step

        # --- 4. Continuous Reward ---
        is_safe = True
        for enemy in self.enemies:
            dist = abs(self.player_pos[0] - enemy['pos'][0]) + abs(self.player_pos[1] - enemy['pos'][1])
            if dist <= 1:
                is_safe = False
                break
        if is_safe:
            reward += 0.1

        # --- 5. Update Particles ---
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        
        # --- 6. Update Step Counter and Check Termination ---
        self.steps += 1
        terminated = False
        
        if self.crystals_collected >= self.NUM_CRYSTALS_TO_WIN:
            reward += 100
            terminated = True
            self.game_over = True
            self.win_message = "YOU WIN!"
        elif self.enemy_touches >= self.MAX_ENEMY_TOUCHES:
            reward -= 100
            terminated = True
            self.game_over = True
            self.win_message = "GAME OVER"
        elif self.steps >= self.MAX_STEPS:
            reward -= 10
            terminated = True
            self.game_over = True
            self.win_message = "TIME UP"
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _render_game(self):
        # Draw grid
        for i in range(self.GRID_SIZE + 1):
            start_grid = self._iso_to_screen(i, 0)
            end_grid = self._iso_to_screen(i, self.GRID_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_grid, end_grid, 1)

            start_grid = self._iso_to_screen(0, i)
            end_grid = self._iso_to_screen(self.GRID_SIZE, i)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_grid, end_grid, 1)

        # Draw enemies
        for enemy in self.enemies:
            ex, ey = self._iso_to_screen(*enemy['pos'])
            pygame.gfxdraw.filled_circle(self.screen, ex, ey, 8, self.COLOR_ENEMY_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, ex, ey, 6, self.COLOR_ENEMY)

        # Draw crystal
        cx, cy = self._iso_to_screen(*self.crystal_pos)
        glow_size = 8 + int(math.sin(self.steps * 0.2) * 2)
        pygame.gfxdraw.filled_circle(self.screen, cx, cy, glow_size, self.COLOR_CRYSTAL_GLOW)
        
        crystal_poly = [
            (cx, cy - 6), (cx + 6, cy), (cx, cy + 6), (cx - 6, cy)
        ]
        pygame.gfxdraw.aapolygon(self.screen, crystal_poly, self.COLOR_CRYSTAL)
        pygame.gfxdraw.filled_polygon(self.screen, crystal_poly, self.COLOR_CRYSTAL)

        # Draw player
        px, py = self._iso_to_screen(*self.player_pos)
        pygame.gfxdraw.filled_circle(self.screen, px, py, 12, self.COLOR_PLAYER_GLOW)
        player_poly = [
            (px, py - 8), (px + 8, py), (px, py + 8), (px - 8, py)
        ]
        pygame.gfxdraw.aapolygon(self.screen, player_poly, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, player_poly, self.COLOR_PLAYER)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            size = int(3 * (p['life'] / p['max_life']))
            if size > 0:
                pygame.draw.circle(self.screen, color, (int(p['pos'][0]), int(p['pos'][1])), size)

    def _render_ui(self):
        # Crystal score
        crystal_text = self.font_ui.render(f"Crystals: {self.crystals_collected}/{self.NUM_CRYSTALS_TO_WIN}", True, self.COLOR_UI_TEXT)
        self.screen.blit(crystal_text, (10, 10))

        # Enemy touches
        touch_text = self.font_ui.render(f"Touches: {self.enemy_touches}/{self.MAX_ENEMY_TOUCHES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(touch_text, (10, 35))

        # Step count
        step_text = self.font_ui.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(step_text, (self.SCREEN_WIDTH - 120, 10))

        # Game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_game_over.render(self.win_message, True, self.COLOR_UI_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.crystals_collected,
            "steps": self.steps,
            "enemy_touches": self.enemy_touches
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset to get an observation
        obs, info = self.reset()
        
        # Test observation space  
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert self.observation_space.contains(obs)
        
        # Test reset return types
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert self.observation_space.contains(obs)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption("Isometric Crystal Collector")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    action = env.action_space.sample()
    action[0] = 0 # Start with no-op

    print("\n" + "="*30)
    print("      MANUAL PLAY TEST")
    print("="*30)
    print(env.game_description)
    print(env.user_guide)
    print("Press ESC or close window to quit.")
    print("="*30 + "\n")


    running = True
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                done = False

        # --- Action Polling ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space, shift]

        # --- Environment Step ---
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        # Pygame uses (width, height), numpy uses (height, width)
        # Transpose back for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Since auto_advance is False, we control the step rate here
        clock.tick(10) # Run at 10 steps per second for playability

    env.close()