
# Generated: 2025-08-27T18:34:56.251049
# Source Brief: brief_01876.md
# Brief Index: 1876

        
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
    """
    An isometric puzzle game where the player collects crystals while avoiding traps.
    The goal is to collect a target number of crystals to win.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move on the isometric grid. "
        "Your goal is to collect all the blue crystals."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate an isometric grid, strategically collecting crystals while avoiding traps to amass a high score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_WIDTH = 12
        self.GRID_HEIGHT = 12
        self.TILE_WIDTH_HALF = 24
        self.TILE_HEIGHT_HALF = 12
        self.NUM_CRYSTALS_TO_WIN = 20
        self.NUM_INITIAL_CRYSTALS = 25
        self.NUM_TRAPS = 15
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_PLAYER = (50, 220, 50)
        self.COLOR_PLAYER_SHADOW = (40, 180, 40)
        self.COLOR_CRYSTAL = (80, 150, 255)
        self.COLOR_CRYSTAL_SHADOW = (60, 120, 225)
        self.COLOR_TRAP = (200, 50, 50)
        self.COLOR_TRAP_SHADOW = (160, 40, 40)
        self.COLOR_TRAP_ACTIVE = (255, 100, 100)
        self.COLOR_UI_TEXT = (230, 230, 230)
        self.COLOR_PLAYER_GLOW = (150, 255, 150)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Center the grid
        self.origin_x = self.SCREEN_WIDTH / 2
        self.origin_y = self.SCREEN_HEIGHT / 2 - (self.GRID_HEIGHT * self.TILE_HEIGHT_HALF) / 2 + 30

        # --- State Variables ---
        # These are initialized properly in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.player_pos = [0, 0]
        self.crystal_locations = []
        self.trap_locations = []
        self.crystals_collected = 0
        self.active_trap_pos = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.crystals_collected = 0
        self.active_trap_pos = None

        self.player_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]

        # Generate entity positions
        all_coords = list(np.ndindex((self.GRID_WIDTH, self.GRID_HEIGHT)))
        self.np_random.shuffle(all_coords)

        # Player start position is invalid for traps/crystals
        safe_zone = self._get_adjacent(self.player_pos)
        safe_zone.add(tuple(self.player_pos))
        
        valid_coords = [c for c in all_coords if c not in safe_zone]

        self.crystal_locations = set(valid_coords[:self.NUM_INITIAL_CRYSTALS])
        
        # Ensure traps don't overlap with crystals
        trap_candidate_coords = [c for c in valid_coords if c not in self.crystal_locations]
        self.trap_locations = set(trap_candidate_coords[:self.NUM_TRAPS])

        return self._get_observation(), self._get_info()
    
    def _get_adjacent(self, pos):
        adj = set()
        px, py = pos
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = px + dx, py + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                    adj.add((nx, ny))
        return adj

    def step(self, action):
        reward = 0.0
        terminated = False
        
        if self.game_over or self.victory:
            return self._get_observation(), 0.0, True, False, self._get_info()

        movement = action[0]

        dist_before = self._find_nearest_crystal_dist()

        # --- Handle Movement ---
        if movement != 0:
            original_pos = list(self.player_pos)
            # 1=up(NW), 2=down(SE), 3=left(SW), 4=right(NE)
            if movement == 1: self.player_pos[1] -= 1 # NW
            elif movement == 2: self.player_pos[1] += 1 # SE
            elif movement == 3: self.player_pos[0] -= 1 # SW
            elif movement == 4: self.player_pos[0] += 1 # NE
            
            # Boundary checks
            if not (0 <= self.player_pos[0] < self.GRID_WIDTH and 0 <= self.player_pos[1] < self.GRID_HEIGHT):
                self.player_pos = original_pos # Revert if out of bounds

        dist_after = self._find_nearest_crystal_dist()

        # --- Movement Reward ---
        if dist_after < dist_before:
            reward += 1.0
        elif dist_after > dist_before:
            reward -= 0.1
        
        # --- Collision & Event Checks ---
        player_pos_tuple = tuple(self.player_pos)

        if player_pos_tuple in self.crystal_locations:
            # sfx: crystal_collect.wav
            self.crystal_locations.remove(player_pos_tuple)
            self.crystals_collected += 1
            self.score += 10
            reward += 10.0

        if player_pos_tuple in self.trap_locations:
            # sfx: trap_spring.wav
            self.game_over = True
            self.active_trap_pos = player_pos_tuple
            self.score = max(0, self.score - 100) # Score can't go below 0
            reward = -100.0
            # sfx: game_over.wav
        
        # --- Termination Checks ---
        self.steps += 1
        if self.game_over:
            terminated = True
        elif self.crystals_collected >= self.NUM_CRYSTALS_TO_WIN:
            # sfx: victory.wav
            self.victory = True
            self.score += 100
            reward = 100.0
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _find_nearest_crystal_dist(self):
        if not self.crystal_locations:
            return float('inf')
        
        px, py = self.player_pos
        min_dist = float('inf')
        for cx, cy in self.crystal_locations:
            dist = abs(px - cx) + abs(py - cy) # Manhattan distance
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _world_to_screen(self, grid_x, grid_y):
        screen_x = self.origin_x + (grid_x - grid_y) * self.TILE_WIDTH_HALF
        screen_y = self.origin_y + (grid_x + grid_y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _draw_iso_cube(self, surface, pos, color, shadow_color, height):
        sx, sy = pos
        tw_half, th_half = self.TILE_WIDTH_HALF, self.TILE_HEIGHT_HALF
        
        # Vertices of the top face
        top_points = [
            (sx, sy - th_half),
            (sx + tw_half, sy),
            (sx, sy + th_half),
            (sx - tw_half, sy)
        ]
        
        # Vertices for the two visible sides
        side_l_points = [(sx - tw_half, sy), (sx, sy + th_half), (sx, sy + th_half + height), (sx - tw_half, sy + height)]
        side_r_points = [(sx + tw_half, sy), (sx, sy + th_half), (sx, sy + th_half + height), (sx + tw_half, sy + height)]
        
        # Draw sides first (darker color)
        pygame.gfxdraw.filled_polygon(surface, side_l_points, shadow_color)
        pygame.gfxdraw.aapolygon(surface, side_l_points, shadow_color)
        pygame.gfxdraw.filled_polygon(surface, side_r_points, shadow_color)
        pygame.gfxdraw.aapolygon(surface, side_r_points, shadow_color)
        
        # Draw top face (brighter color)
        pygame.gfxdraw.filled_polygon(surface, top_points, color)
        pygame.gfxdraw.aapolygon(surface, top_points, color)

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_WIDTH + 1):
            start = self._world_to_screen(i, 0)
            end = self._world_to_screen(i, self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for i in range(self.GRID_HEIGHT + 1):
            start = self._world_to_screen(0, i)
            end = self._world_to_screen(self.GRID_WIDTH, i)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

        # Draw traps
        for tx, ty in self.trap_locations:
            pos = self._world_to_screen(tx, ty)
            color = self.COLOR_TRAP_ACTIVE if (tx, ty) == self.active_trap_pos else self.COLOR_TRAP
            self._draw_iso_cube(self.screen, pos, color, self.COLOR_TRAP_SHADOW, 4)

        # Draw crystals
        for cx, cy in self.crystal_locations:
            pos = self._world_to_screen(cx, cy)
            self._draw_iso_cube(self.screen, pos, self.COLOR_CRYSTAL, self.COLOR_CRYSTAL_SHADOW, 8)

        # Draw player
        player_screen_pos = self._world_to_screen(self.player_pos[0], self.player_pos[1])
        # Glow effect
        glow_radius = int(self.TILE_WIDTH_HALF * 1.5)
        glow_center = (player_screen_pos[0], player_screen_pos[1] + 10)
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, self.COLOR_PLAYER_GLOW + (50,), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(temp_surf, (glow_center[0] - glow_radius, glow_center[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        self._draw_iso_cube(self.screen, player_screen_pos, self.COLOR_PLAYER, self.COLOR_PLAYER_SHADOW, 16)


    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        crystal_text = self.font_ui.render(f"CRYSTALS: {self.crystals_collected}/{self.NUM_CRYSTALS_TO_WIN}", True, self.COLOR_UI_TEXT)
        self.screen.blit(crystal_text, (10, 35))
        
        steps_text = self.font_ui.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 10))

        if self.game_over:
            msg = "GAME OVER"
            color = self.COLOR_TRAP_ACTIVE
        elif self.victory:
            msg = "VICTORY!"
            color = self.COLOR_PLAYER
        else:
            return

        end_text = self.font_game_over.render(msg, True, color)
        text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        
        # Draw a semi-transparent background for the text
        bg_rect = text_rect.inflate(20, 20)
        bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        bg_surf.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 180))
        self.screen.blit(bg_surf, bg_rect)
        self.screen.blit(end_text, text_rect)


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
            "crystals_collected": self.crystals_collected,
            "player_pos": tuple(self.player_pos)
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
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Override screen to be a display surface for human play
    env.screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Crystal Collector")

    done = False
    action = [0, 0, 0] # No-op, no space, no shift

    print("--- Crystal Collector ---")
    print(env.user_guide)

    while not done:
        # Human controls
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1
                
                # Since it's turn-based, we step on keydown
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Done: {done}")

                # Reset action after step
                action = [0, 0, 0]

        # Render the current state to the display
        env.screen.blit(pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2))), (0, 0))
        pygame.display.flip()
        env.clock.tick(30) # Limit FPS

    print("Game Over!")
    pygame.time.wait(2000) # Wait 2 seconds before closing
    env.close()