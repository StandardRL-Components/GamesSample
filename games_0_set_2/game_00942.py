
# Generated: 2025-08-27T15:15:59.356099
# Source Brief: brief_00942.md
# Brief Index: 942

        
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
        "Controls: Arrow keys to move on the isometric grid (Up=NW, Down=SE, Left=SW, Right=NE)."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric puzzle game. Collect the glowing blue crystals while avoiding the pulsating red traps."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 12
    GRID_HEIGHT = 12
    TILE_WIDTH_HALF = 24
    TILE_HEIGHT_HALF = 12
    
    WIN_SCORE = 100
    MAX_STEPS = 1000
    INITIAL_TRAPS = 5
    INITIAL_CRYSTALS = 10

    # Colors
    COLOR_BG = (25, 35, 45)
    COLOR_GRID = (50, 65, 80)
    COLOR_PLAYER = (0, 255, 150)
    COLOR_CRYSTAL = (100, 150, 255)
    COLOR_TRAP = (255, 80, 80)
    COLOR_TEXT = (220, 220, 220)

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
        self.font = pygame.font.SysFont("Arial", 24)
        
        # Game state variables are initialized in reset()
        self.player_pos = None
        self.crystal_positions = None
        self.trap_positions = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.animation_timer = 0
        self.random = None

        # Center the grid
        self.origin_x = self.SCREEN_WIDTH // 2
        self.origin_y = self.SCREEN_HEIGHT // 2 - (self.GRID_HEIGHT * self.TILE_HEIGHT_HALF) // 2 + 30
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.random = random.Random(seed)
        else:
            # Fallback if seed is None
            if self.random is None:
                self.random = random.Random()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.animation_timer = 0
        
        self._place_entities()
        
        return self._get_observation(), self._get_info()

    def _place_entities(self):
        """Initializes positions of player, crystals, and traps."""
        self.player_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        
        occupied_tiles = {self.player_pos}
        all_tiles = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        
        # Place traps
        self.trap_positions = set()
        available_tiles = [t for t in all_tiles if t not in occupied_tiles]
        trap_count = self.INITIAL_TRAPS
        chosen_traps = self.random.sample(available_tiles, k=min(trap_count, len(available_tiles)))
        self.trap_positions.update(chosen_traps)
        occupied_tiles.update(chosen_traps)

        # Place crystals
        self.crystal_positions = set()
        available_tiles = [t for t in all_tiles if t not in occupied_tiles]
        crystal_count = self.INITIAL_CRYSTALS
        chosen_crystals = self.random.sample(available_tiles, k=min(crystal_count, len(available_tiles)))
        self.crystal_positions.update(chosen_crystals)

    def step(self, action):
        self.animation_timer += 1
        reward = 0
        
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        old_pos = self.player_pos
        new_pos = list(self.player_pos)

        # Apply movement based on isometric grid directions
        if movement == 1:  # Up (NW)
            new_pos[1] -= 1
        elif movement == 2:  # Down (SE)
            new_pos[1] += 1
        elif movement == 3:  # Left (SW)
            new_pos[0] -= 1
        elif movement == 4:  # Right (NE)
            new_pos[0] += 1
        
        new_pos = tuple(new_pos)

        # Boundary checks
        if 0 <= new_pos[0] < self.GRID_WIDTH and 0 <= new_pos[1] < self.GRID_HEIGHT:
            self.player_pos = new_pos
        else:
            new_pos = old_pos # Revert if out of bounds

        # Calculate distance-based rewards if player moved
        if old_pos != self.player_pos:
            dist_crystal_old = self._find_closest_entity_dist(old_pos, self.crystal_positions)
            dist_crystal_new = self._find_closest_entity_dist(self.player_pos, self.crystal_positions)
            if dist_crystal_new < dist_crystal_old:
                reward += 1.0 # Moved closer to a crystal

            dist_trap_old = self._find_closest_entity_dist(old_pos, self.trap_positions)
            dist_trap_new = self._find_closest_entity_dist(self.player_pos, self.trap_positions)
            if dist_trap_new < dist_trap_old:
                reward -= 0.1 # Moved closer to a trap
        
        # Check for interactions
        if self.player_pos in self.crystal_positions:
            self.score += 1
            reward += 10.0
            self.crystal_positions.remove(self.player_pos)
            self._spawn_entity(self.crystal_positions, {self.player_pos, *self.trap_positions})
            # SFX: Crystal collect sound

            # Difficulty scaling
            if self.score > 0 and self.score % 25 == 0:
                num_new_traps = self.score // 25
                current_traps = self.INITIAL_TRAPS + num_new_traps - 1
                if len(self.trap_positions) < current_traps + 1:
                   self._spawn_entity(self.trap_positions, {self.player_pos, *self.crystal_positions})
                   # SFX: Trap appears sound

        if self.player_pos in self.trap_positions:
            reward -= 100.0
            self.game_over = True
            # SFX: Player death/trap sound

        self.steps += 1
        
        # Check termination conditions
        terminated = self.game_over or self.steps >= self.MAX_STEPS or self.score >= self.WIN_SCORE
        if terminated and self.score >= self.WIN_SCORE:
            reward += 100.0 # Victory bonus
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _spawn_entity(self, entity_set, occupied_tiles):
        """Spawns a single entity in a random, unoccupied tile."""
        all_tiles = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        available_tiles = [t for t in all_tiles if t not in occupied_tiles]
        if available_tiles:
            pos = self.random.choice(available_tiles)
            entity_set.add(pos)

    def _grid_to_screen(self, x, y):
        """Converts isometric grid coordinates to screen coordinates."""
        screen_x = self.origin_x + (x - y) * self.TILE_WIDTH_HALF
        screen_y = self.origin_y + (x + y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _find_closest_entity_dist(self, pos, entity_set):
        """Calculates Manhattan distance to the closest entity in a set."""
        if not entity_set:
            return float('inf')
        closest = min(entity_set, key=lambda e: abs(e[0] - pos[0]) + abs(e[1] - pos[1]))
        return abs(closest[0] - pos[0]) + abs(closest[1] - pos[1])

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders the grid and all entities."""
        # Draw entities from back to front for correct isometric layering
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                # Draw grid cell
                sx, sy = self._grid_to_screen(x, y)
                points = [
                    (sx, sy - self.TILE_HEIGHT_HALF),
                    (sx + self.TILE_WIDTH_HALF, sy),
                    (sx, sy + self.TILE_HEIGHT_HALF),
                    (sx - self.TILE_WIDTH_HALF, sy)
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

                # Draw entities in the cell
                pos = (x, y)
                if pos in self.crystal_positions:
                    self._draw_pulsating_entity(sx, sy, self.COLOR_CRYSTAL, 0.8, 1.2, 0.05)
                if pos in self.trap_positions:
                    self._draw_pulsating_entity(sx, sy, self.COLOR_TRAP, 1.0, 1.3, 0.08)
                if pos == self.player_pos:
                    self._draw_player(sx, sy)

    def _draw_player(self, sx, sy):
        """Draws the player with a glow effect."""
        color = self.COLOR_PLAYER
        # Glow effect
        for i in range(4, 0, -1):
            alpha = 80 - i * 20
            glow_color = (*color, alpha)
            pygame.gfxdraw.filled_circle(self.screen, sx, sy, 8 + i * 2, glow_color)
        
        # Core shape (diamond)
        points = [
            (sx, sy - 8),
            (sx + 6, sy),
            (sx, sy + 8),
            (sx - 6, sy)
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)


    def _draw_pulsating_entity(self, sx, sy, color, min_scale, max_scale, speed):
        """Draws an entity with a pulsating size and glow."""
        pulse = (math.sin(self.animation_timer * speed) + 1) / 2
        scale = min_scale + (max_scale - min_scale) * pulse
        radius = int(6 * scale)
        
        # Glow effect
        for i in range(4, 0, -1):
            alpha = int(60 * scale - i * 15)
            if alpha > 0:
                glow_color = (*color, alpha)
                pygame.gfxdraw.filled_circle(self.screen, sx, sy, radius + i * 2, glow_color)
        
        # Core circle
        pygame.gfxdraw.aacircle(self.screen, sx, sy, radius, color)
        pygame.gfxdraw.filled_circle(self.screen, sx, sy, radius, color)

    def _render_ui(self):
        """Renders the score display."""
        score_text = f"Crystals: {self.score} / {self.WIN_SCORE}"
        text_surface = self.font.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (10, 10))

        steps_text = f"Steps: {self.steps} / {self.MAX_STEPS}"
        text_surface_steps = self.font.render(steps_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface_steps, (self.SCREEN_WIDTH - text_surface_steps.get_width() - 10, 10))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.player_pos,
            "crystals_remaining": len(self.crystal_positions),
            "traps_active": len(self.trap_positions)
        }
        
    def close(self):
        pygame.font.quit()
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

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Isometric Crystal Collector")
    clock = pygame.time.Clock()
    
    terminated = False
    
    print(env.user_guide)
    print(env.game_description)

    while not terminated:
        movement = 0 # No-op by default
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    movement = 1
                elif event.key == pygame.K_DOWN:
                    movement = 2
                elif event.key == pygame.K_LEFT:
                    movement = 3
                elif event.key == pygame.K_RIGHT:
                    movement = 4
                elif event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    print("--- Game Reset ---")
                    
        # Only step if a key was pressed
        if movement != 0:
            action = [movement, 0, 0] # space and shift are not used
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {movement}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")
        
        # Always render the current state
        frame = env._get_observation()
        # The observation is (H, W, C), but pygame needs (W, H, C) for surfarray
        # So we need to transpose it back
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS for the interactive loop

    print("Game Over!")
    env.close()