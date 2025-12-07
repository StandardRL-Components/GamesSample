
# Generated: 2025-08-28T01:54:08.208929
# Source Brief: brief_04272.md
# Brief Index: 4272

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move on the isometric grid. Collect all gems to win, avoid traps."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate an isometric grid, collecting gems while dodging traps to achieve a high score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 10
    MAX_STEPS = 1000
    NUM_GEMS_START = 10
    NUM_TRAPS_START = 3

    # Colors
    COLOR_BG = (15, 20, 40)
    COLOR_GRID = (30, 40, 70)
    COLOR_PLAYER = (0, 200, 255)
    COLOR_PLAYER_SIDE = (0, 150, 200)
    COLOR_GEM = (255, 220, 0)
    COLOR_GEM_HIGHLIGHT = (255, 255, 150)
    COLOR_TRAP = (50, 50, 60)
    COLOR_TRAP_ACTIVE_1 = (255, 0, 0)
    COLOR_TRAP_ACTIVE_2 = (180, 0, 0)
    COLOR_TEXT = (230, 230, 240)
    COLOR_UI_BG = (25, 30, 55, 180)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.ui_font = pygame.font.Font(None, 28)
        self.game_over_font = pygame.font.Font(None, 72)

        # Game state variables are initialized in reset()
        self.player_pos = None
        self.gem_locations = None
        self.trap_locations = None
        self.score = None
        self.steps = None
        self.stage = None
        self.game_over = None
        self.game_won = None
        self.trap_activated_pos = None

        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.stage = options.get("stage", 1) if options else 1
        self.game_over = False
        self.game_won = False
        self.trap_activated_pos = None
        
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        """Generates a new level, ensuring all gems are reachable."""
        num_traps = self.NUM_TRAPS_START + self.stage - 1
        
        while True:
            all_coords = [(x, y) for x in range(self.GRID_SIZE) for y in range(self.GRID_SIZE)]
            self.player_pos = (self.GRID_SIZE // 2, self.GRID_SIZE // 2)
            all_coords.remove(self.player_pos)

            # Ensure we don't try to place more items than available tiles
            num_gems = min(self.NUM_GEMS_START, len(all_coords))
            self.gem_locations = self.np_random.choice(len(all_coords), num_gems, replace=False)
            self.gem_locations = [all_coords[i] for i in self.gem_locations]
            
            available_for_traps = [c for c in all_coords if c not in self.gem_locations]
            num_traps_to_place = min(num_traps, len(available_for_traps))
            trap_indices = self.np_random.choice(len(available_for_traps), num_traps_to_place, replace=False)
            self.trap_locations = [available_for_traps[i] for i in trap_indices]
            
            # Validate that all gems are reachable
            all_gems_reachable = True
            for gem_pos in self.gem_locations:
                if self._bfs(self.player_pos, gem_pos, self.trap_locations) is None:
                    all_gems_reachable = False
                    break
            
            if all_gems_reachable:
                break

    def _bfs(self, start, end, blocked):
        """Breadth-First Search to find a path."""
        q = deque([(start, [start])])
        visited = {start}
        
        while q:
            (node, path) = q.popleft()
            if node == end:
                return path
            
            x, y = node
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]: # Using cardinal directions for pathfinding
                neighbor = (x + dx, y + dy)
                if (0 <= neighbor[0] < self.GRID_SIZE and 0 <= neighbor[1] < self.GRID_SIZE and
                        neighbor not in visited and neighbor not in blocked):
                    visited.add(neighbor)
                    q.append((neighbor, path + [neighbor]))
        return None

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement = action[0]
        self.steps += 1
        
        reward, terminated = self._update_game_state(movement)
        
        self.score += reward
        self.game_over = terminated
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_game_state(self, movement):
        """Handles game logic updates for a single step."""
        old_pos = self.player_pos
        
        # --- Handle Movement ---
        if movement != 0:
            px, py = self.player_pos
            if movement == 1:  # Up -> Up-Left
                py -= 1
            elif movement == 2:  # Down -> Down-Right
                py += 1
            elif movement == 3:  # Left -> Down-Left
                px -= 1
            elif movement == 4:  # Right -> Up-Right
                px += 1
            
            # Clamp to grid boundaries
            px = max(0, min(self.GRID_SIZE - 1, px))
            py = max(0, min(self.GRID_SIZE - 1, py))
            self.player_pos = (px, py)
        
        # --- Check for Events & Calculate Rewards ---
        reward = 0
        terminated = False
        
        # Check for trap
        if self.player_pos in self.trap_locations:
            # SFX: trap_spring
            reward = -10
            terminated = True
            self.trap_activated_pos = self.player_pos
            return reward, terminated
        
        # Check for gem collection
        if self.player_pos in self.gem_locations:
            # SFX: gem_collect
            reward += 10
            self.gem_locations.remove(self.player_pos)
            if not self.gem_locations:
                # SFX: stage_clear
                reward += 50
                terminated = True
                self.game_won = True
                self.stage += 1
        
        # Step limit termination
        if self.steps >= self.MAX_STEPS:
            terminated = True
            
        # Continuous movement reward if a move was made
        if old_pos != self.player_pos:
            dist = lambda p1, p2: abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
            
            if self.gem_locations:
                closest_gem_dist_old = min(dist(old_pos, g) for g in self.gem_locations)
                closest_gem_dist_new = min(dist(self.player_pos, g) for g in self.gem_locations)
                moved_towards_gem = closest_gem_dist_new < closest_gem_dist_old
            else: # No gems left, neutral reward for movement
                moved_towards_gem = False

            if self.trap_locations:
                furthest_trap_dist_old = max(dist(old_pos, t) for t in self.trap_locations)
                furthest_trap_dist_new = max(dist(self.player_pos, t) for t in self.trap_locations)
                moved_away_from_trap = furthest_trap_dist_new > furthest_trap_dist_old
            else: # No traps, always "safe"
                moved_away_from_trap = True

            if moved_towards_gem and moved_away_from_trap:
                reward += 0.2  # Risky step
            else:
                reward -= 0.1  # Safe/neutral step
        else: # No-op or bumped into wall
            reward -= 0.1

        return reward, terminated

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Isometric projection constants
        tile_w = (self.SCREEN_WIDTH * 0.8) / self.GRID_SIZE
        tile_h = tile_w * 0.5
        offset_x = self.SCREEN_WIDTH / 2
        offset_y = self.SCREEN_HEIGHT * 0.25

        def iso_to_cart(iso_x, iso_y):
            cart_x = (iso_x - iso_y) * (tile_w / 2) + offset_x
            cart_y = (iso_x + iso_y) * (tile_h / 2) + offset_y
            return int(cart_x), int(cart_y)

        def draw_iso_poly(points, color, border_color=None):
            if border_color:
                pygame.gfxdraw.aapolygon(self.screen, points, border_color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

        # Draw grid, traps, and gems
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                cx, cy = iso_to_cart(x, y)
                points = [
                    (cx, cy),
                    (cx + tile_w / 2, cy + tile_h / 2),
                    (cx, cy + tile_h),
                    (cx - tile_w / 2, cy + tile_h / 2)
                ]
                draw_iso_poly(points, self.COLOR_GRID)

                if (x, y) in self.trap_locations:
                    if (x, y) == self.trap_activated_pos:
                        # Flash red when activated
                        flash_color = self.COLOR_TRAP_ACTIVE_1 if (self.steps // 2) % 2 == 0 else self.COLOR_TRAP_ACTIVE_2
                        draw_iso_poly(points, flash_color)
                    else:
                        draw_iso_poly(points, self.COLOR_TRAP)

                if (x, y) in self.gem_locations:
                    gem_points = [
                        (cx, cy + tile_h * 0.2),
                        (cx + tile_w * 0.3, cy + tile_h * 0.5),
                        (cx, cy + tile_h * 0.8),
                        (cx - tile_w * 0.3, cy + tile_h * 0.5)
                    ]
                    highlight_points = [
                        (cx, cy + tile_h * 0.2),
                        (cx + tile_w * 0.15, cy + tile_h * 0.35),
                        (cx, cy + tile_h * 0.5),
                        (cx - tile_w * 0.15, cy + tile_h * 0.35)
                    ]
                    draw_iso_poly(gem_points, self.COLOR_GEM, (255, 255, 255))
                    draw_iso_poly(highlight_points, self.COLOR_GEM_HIGHLIGHT)

        # Draw player
        px, py = self.player_pos
        pcx, pcy = iso_to_cart(px, py)
        player_height = tile_h * 0.7
        top_points = [
            (pcx, pcy - player_height),
            (pcx + tile_w / 2, pcy - player_height + tile_h / 2),
            (pcx, pcy - player_height + tile_h),
            (pcx - tile_w / 2, pcy - player_height + tile_h / 2)
        ]
        side1_points = [
            (pcx, pcy), (pcx, pcy - player_height + tile_h),
            (pcx - tile_w / 2, pcy - player_height + tile_h / 2), (pcx - tile_w / 2, pcy + tile_h / 2)
        ]
        side2_points = [
            (pcx, pcy), (pcx, pcy - player_height + tile_h),
            (pcx + tile_w / 2, pcy - player_height + tile_h / 2), (pcx + tile_w / 2, pcy + tile_h / 2)
        ]
        draw_iso_poly(side1_points, self.COLOR_PLAYER_SIDE)
        draw_iso_poly(side2_points, self.COLOR_PLAYER_SIDE)
        draw_iso_poly(top_points, self.COLOR_PLAYER)

    def _render_ui(self):
        ui_surface = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        ui_surface.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_surface, (0, 0))

        score_text = self.ui_font.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        stage_text = self.ui_font.render(f"STAGE: {self.stage}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (200, 10))
        
        gems_text = self.ui_font.render(f"GEMS: {len(self.gem_locations)}/{self.NUM_GEMS_START}", True, self.COLOR_TEXT)
        self.screen.blit(gems_text, (350, 10))

        steps_text = self.ui_font.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - 150, 10))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.game_won:
                msg = "STAGE CLEAR"
                color = self.COLOR_GEM
            else:
                msg = "GAME OVER"
                color = self.COLOR_TRAP_ACTIVE_1
                
            text_surf = self.game_over_font.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "gems_remaining": len(self.gem_locations),
        }

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert not trunc
        assert isinstance(info, dict)
        
        # Test game logic assertions
        px, py = self.player_pos
        assert 0 <= px < self.GRID_SIZE and 0 <= py < self.GRID_SIZE
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Set up Pygame window for human play
    pygame.display.set_caption("Isometric Gem Collector")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    print(GameEnv.game_description)
    print(GameEnv.user_guide)

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
                elif event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    print("--- Game Reset ---")
                    continue
        
        if movement != 0:
            action = [movement, 0, 0] # space and shift are unused
            obs, reward, terminated, _, info = env.step(action)
            print(f"Step: {info['steps']}, Action: {action[0]}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")
        
        # Render the observation to the display window
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print("--- Episode Finished ---")
            print(f"Final Score: {info['score']:.2f}, Total Steps: {info['steps']}")
            pygame.time.wait(2000) # Wait 2 seconds before resetting
            obs, info = env.reset()
            terminated = False

    env.close()